import csv
import gzip
from typing import Iterator

from tqdm import tqdm

from ..utils.data_utils import Passage


class InMemoryPassageDB:
    def __init__(self, passage_file: str, skip_header: bool = True):
        self.data = dict()
        with gzip.open(passage_file, "rt") if passage_file.endswith(".gz") else open(passage_file) as f:
            tsv_reader = csv.reader(f, doublequote=False, delimiter="\t")
            for i, row in tqdm(enumerate(tsv_reader)):
                if i == 0 and skip_header:
                    continue

                passage_id, text, title = row[0], row[1], row[2]
                self.data[int(passage_id)] = (text, title)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, passage_id: int) -> Passage:
        text, title = self.data[passage_id]
        return Passage(passage_id, title, text)

    def __iter__(self) -> Iterator[Passage]:
        for passage_id, (text, title) in self.data.items():
            yield Passage(passage_id, title, text)
