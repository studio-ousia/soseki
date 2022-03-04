import gzip
import json
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Sampler


@dataclass
class Passage:
    id: int
    title: Optional[str] = None
    text:  Optional[str] = None


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


class WeightedConcatDatasetSampler(Sampler):
    def __init__(
        self,
        concat_dataset: ConcatDataset,
        weights: Optional[List[float]] = None,
        chunk_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")

            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")

            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))

        if chunk_size % num_replicas != 0:
            raise ValueError(
                "The chunk_size should be divisible by num_replicas so that each chunk is evenly split across GPUs"
            )

        if weights is None or len(weights) == 0:
            weights = [1.0] * len(concat_dataset.datasets)

        if len(concat_dataset.datasets) != len(weights):
            raise ValueError("The length of weights must be the same as the number of datasets in concat_dataset")

        self.concat_dataset = concat_dataset
        self.weights = weights
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_samples = sum(
            int(len(dataset) * weight) // self.chunk_size * self.chunk_size
            for dataset, weight in zip(self.concat_dataset.datasets, self.weights)
        ) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
        else:
            generator = None

        sample_idx_chunks: List[List[int]] = []
        cumulative_size = 0
        for dataset, weight in zip(self.concat_dataset.datasets, self.weights):
            dataset_size = len(dataset)

            sample_idxs = []
            for _ in range(int(weight) + 1):
                if self.shuffle:
                    sample_idxs.extend(
                        (torch.randperm(dataset_size, generator=generator) + cumulative_size).tolist()
                    )
                else:
                    sample_idxs.extend(list(range(cumulative_size, cumulative_size + dataset_size)))

            sample_idxs = sample_idxs[:int(dataset_size * weight)]

            for i in range(0, len(sample_idxs), self.chunk_size):
                chunk = sample_idxs[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    break

                sample_idx_chunks.append(chunk)

            cumulative_size += dataset_size

        if self.shuffle:
            chunk_idxs = torch.randperm(len(sample_idx_chunks), generator=generator).tolist()
        else:
            chunk_idxs = list(range(len(sample_idx_chunks)))

        flattened_sample_idxs = [sample_idx for chunk_idx in chunk_idxs for sample_idx in sample_idx_chunks[chunk_idx]]

        # subsample
        indices = flattened_sample_idxs[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

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
    small_list: List,
    large_list: List,
    start: int = 0,
    end: Optional[int] = None
) -> List[Tuple[int, int]]:
    if end is None:
        end = len(large_list)

    slices = []
    for i in range(start, end - len(small_list) + 1):
        if large_list[i:i + len(small_list)] == small_list:
            slices.append((i, i + len(small_list)))

    return slices
