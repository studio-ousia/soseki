import gzip
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple


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
    question: str
    answers: List[str]
    positive_passages: List[Passage]
    hard_negative_passages: List[Passage]
    normal_negative_passages: List[Passage]
    metadata: Optional[Any]


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
