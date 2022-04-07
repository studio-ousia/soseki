import csv
import gzip
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


@dataclass
class Passage:
    id: int
    title: Optional[str] = None
    text: Optional[str] = None
    dataset: Optional[str] = None


@dataclass
class RetrievedPassage(Passage):
    score: Optional[float] = None
    has_answer: Optional[bool] = None


@dataclass
class RetrieverDatasetExample:
    index: int
    dataset: str
    question: str
    answers: List[str]
    positive_passages: List[Passage]
    hard_negative_passages: List[Passage]
    normal_negative_passages: List[Passage]
    metadata: Optional[Any]


class MultipleKeyDataset(Dataset):
    dataset_dict: Dict[str, Dataset]
    dataset_keys: List[str]
    dataset_sizes: Dict[str, int]
    total_dataset_size: int

    def __init__(self, dataset_dict: Dict[str, Dataset]) -> None:
        super(MultipleKeyDataset, self).__init__()
        self.dataset_dict = dataset_dict
        self.dataset_keys = list(dataset_dict.keys())
        self.dataset_sizes = {key: len(dataset) for key, dataset in dataset_dict.items()}
        self.total_dataset_size = sum(self.dataset_sizes.values())

    def __len__(self) -> int:
        return self.total_dataset_size

    def __getitem__(self, key_idx_tuple: Tuple[str, int]) -> Any:
        dataset_key, sample_idx = key_idx_tuple
        return self.dataset_dict[dataset_key][sample_idx]


class MultipleKeyDatasetBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        multiple_key_dataset: MultipleKeyDataset,
        weights: Optional[Dict[str, float]] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if distributed:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")

                num_replicas = dist.get_world_size()

            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")

                rank = dist.get_rank()

            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1)
                )
        else:
            num_replicas = 1
            rank = 0

        if weights is None:
            weights = dict()

        for key in multiple_key_dataset.dataset_dict.keys():
            weights.setdefault(key, 1.0)

        self.multiple_key_dataset = multiple_key_dataset
        self.weights = weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.distributed = distributed
        self.num_replicas = num_replicas
        self.rank = rank

        self.chunk_size = self.batch_size * self.num_replicas
        self.num_chunks = sum(
            int(len(dataset) * self.weights[key]) // self.chunk_size
            for key, dataset in self.multiple_key_dataset.dataset_dict.items()
        )
        self.total_size = self.num_chunks * self.chunk_size
        self.epoch = 0

    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
        else:
            generator = None

        chunks: List[List[int]] = []
        for key in self.multiple_key_dataset.dataset_keys:
            weight = self.weights[key]
            dataset_size = self.multiple_key_dataset.dataset_sizes[key]

            sample_idxs = []
            for _ in range(int(weight) + 1):
                if self.shuffle:
                    sample_idxs.extend(torch.randperm(dataset_size, generator=generator).tolist())
                else:
                    sample_idxs.extend(list(range(dataset_size)))

            sample_idxs = sample_idxs[: int(dataset_size * weight)]

            for i in range(0, len(sample_idxs), self.chunk_size):
                chunk = [(key, idx) for idx in sample_idxs[i : i + self.chunk_size]]
                if len(chunk) < self.chunk_size:
                    break

                chunks.append(chunk)

        if self.shuffle:
            chunk_idxs = torch.randperm(len(chunks), generator=generator).tolist()
        else:
            chunk_idxs = list(range(len(chunks)))

        for chunk_idx in chunk_idxs:
            chunk = chunks[chunk_idx]
            batch = chunk[self.rank : self.chunk_size : self.num_replicas]
            assert len(batch) == self.batch_size
            yield batch

    def __len__(self) -> int:
        return self.num_chunks

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


@dataclass
class ReaderDatasetExample:
    index: int
    question: str
    answers: List[str]
    passages: List[RetrievedPassage]
    positive_passage_idxs: List[int]
    negative_passage_idxs: List[int]


@dataclass
class AnswerSpan:
    start: int
    end: int
    start_logit: float
    end_logit: float


@dataclass
class AnswerCandidate:
    answer_text: str
    passage_text: str
    score: float
    passage_score: float
    span_score: float


def readitem_json(input_file: str, read_json_lines: Optional[bool] = None):
    # if `read_json_lines` is not specified, it is determined by the input file's extension
    if read_json_lines is None:
        read_json_lines = input_file.endswith(".jsonl") or input_file.endswith(".jsonl.gz")

    if read_json_lines:
        with gzip.open(input_file, "rt") if input_file.endswith(".gz") else open(input_file) as f:
            for line in f:
                yield json.loads(line)
    else:
        with gzip.open(input_file, "rt") if input_file.endswith(".gz") else open(input_file) as f:
            for item in json.load(f):
                yield item


def readitem_tsv(input_file: str, fieldnames: Optional[List[str]] = None):
    with gzip.open(input_file, "rt") if input_file.endswith(".gz") else open(input_file) as f:
        for row in csv.DictReader(f, fieldnames=fieldnames, delimiter="\t"):
            yield row


def writeitem_json(items: Iterable[Any], output_file: str, write_json_lines: Optional[bool] = None):
    # if `write_json_lines` is not specified, it is determined by the output file's extension
    if write_json_lines is None:
        write_json_lines = output_file.endswith(".jsonl") or output_file.endswith(".jsonl.gz")

    if write_json_lines:
        with gzip.open(output_file, "wt") if output_file.endswith(".gz") else open(output_file, "w") as fo:
            for item in items:
                print(json.dumps(item, ensure_ascii=False), file=fo)
    else:
        with gzip.open(output_file, "wt") if output_file.endswith(".gz") else open(output_file, "w") as fo:
            json.dump(items, fo, ensure_ascii=False, indent=2)


def batch_iter(iterable: Iterable, batch_size: int) -> Iterable:
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    else:
        if len(batch) > 0:
            yield batch


def find_sublist_slices(
    small_list: List, large_list: List, start: int = 0, end: Optional[int] = None
) -> List[Tuple[int, int]]:
    if end is None:
        end = len(large_list)

    slices = []
    for i in range(start, end - len(small_list) + 1):
        if large_list[i : i + len(small_list)] == small_list:
            slices.append((i, i + len(small_list)))

    return slices
