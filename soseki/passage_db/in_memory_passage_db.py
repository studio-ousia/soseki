from typing import Iterator

from tqdm import tqdm

from ..utils.data_utils import Passage, readitem_tsv


class InMemoryPassageDB:
    def __init__(self, passage_file: str, skip_header: bool = True):
        self.data = dict()
        for row in tqdm(readitem_tsv(passage_file, skip_header=skip_header)):
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
