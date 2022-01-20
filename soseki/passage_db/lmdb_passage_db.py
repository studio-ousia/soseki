import csv
import gzip
import json
from typing import Iterator

import lmdb
from tqdm import tqdm

from ..utils.data_utils import Passage


class LMDBPassageDB:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db = lmdb.open(db_path, subdir=False, readonly=True)

    def __reduce__(self):
        return (self.__class__, (self._db_path,))

    def __len__(self):
        return self._db.stat()["entries"]

    def __getitem__(self, passage_id: int) -> Passage:
        with self._db.begin() as txn:
            dumped_passage = txn.get(str(passage_id).encode("utf-8"))
            if dumped_passage is None:
                raise KeyError("Invalid passage_id: " + str(passage_id))

            title, text = json.loads(dumped_passage)
            return Passage(passage_id, title, text)

    def __iter__(self) -> Iterator[Passage]:
        with self._db.begin() as txn:
            cursor = txn.cursor()
            for passage_id, dumped_passage in cursor:
                title, text = json.loads(dumped_passage.decode("utf-8"))
                yield Passage(int(passage_id.decode("utf-8")), title, text)

    @classmethod
    def from_passage_file(
        cls,
        passage_file: str,
        db_file: str,
        db_map_size: int = 2147483648,  # 2GB
        skip_header: bool = True,
        chunk_size: int = 1024,
    ):
        db = lmdb.open(db_file, map_size=db_map_size, subdir=False)
        with db.begin(write=True) as txn:
            with gzip.open(passage_file, "rt") if passage_file.endswith(".gz") else open(passage_file) as f:
                tsv_reader = csv.reader(f, doublequote=False, delimiter="\t")
                buffer = []
                for i, row in tqdm(enumerate(tsv_reader)):
                    if i == 0 and skip_header:
                        continue

                    passage_id, text, title = row[0], row[1], row[2]
                    dumped_passage = json.dumps([title, text])
                    buffer.append((passage_id.encode("utf-8"), dumped_passage.encode("utf-8")))
                    if len(buffer) >= chunk_size:
                        txn.cursor().putmulti(buffer)
                        buffer = []

                if buffer:
                    txn.cursor().putmulti(buffer)

        return cls(db_file)
