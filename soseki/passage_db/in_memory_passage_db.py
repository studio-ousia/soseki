from typing import Iterator

from tqdm import tqdm

from ..utils.data_utils import Passage, readitem_tsv


class InMemoryPassageDB:
    def __init__(self, passage_file: str):
        self.data = dict()
        for row in tqdm(readitem_tsv(passage_file)):
            self.data[int(row["id"])] = (row["title"], row["text"], row.get("dataset", None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, passage_id: int) -> Passage:
        title, text, dataset = self.data[passage_id]
        return Passage(passage_id, title, text, dataset=dataset)

    def __iter__(self) -> Iterator[Passage]:
        for passage_id, (text, title, dataset) in self.data.items():
            yield Passage(passage_id, title, text, dataset=dataset)
