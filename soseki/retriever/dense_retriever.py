from typing import List, Optional

import faiss
import numpy as np

from ..utils.data_utils import RetrievedPassage


class DenseRetriever:
    def __init__(
        self,
        faiss_index_file: str,
        passage_db_file: str,
        passage_file: Optional[str] = None,
    ) -> None:
        self.faiss_index: faiss.IndexIDMap2 = faiss.read_index(faiss_index_file)
        if passage_db_file is not None:
            from ..passage_db.lmdb_passage_db import LMDBPassageDB
            self.passage_db = LMDBPassageDB(passage_db_file)
        elif passage_file is not None:
            from ..passage_db.in_memory_passage_db import InMemoryPassageDB
            self.passage_db = InMemoryPassageDB(passage_file)
        else:
            raise ValueError("Either of passage_db_file or passage_file must be specified.")

    def retrieve_top_k_passages(
        self,
        encoded_questions: np.ndarray,
        k: int = 10
    ) -> List[List[RetrievedPassage]]:
        num_questions, _ = encoded_questions.shape

        scores_array, passage_ids_array = self.index.search(encoded_questions, k)
        assert scores_array.shape == (num_questions, k)
        assert passage_ids_array.shape == (num_questions, k)

        outputs = []
        for scores, passage_ids in zip(scores_array, passage_ids_array):
            retrieved_passages = []
            for score, passage_id in zip(scores, passage_ids):
                passage = self.passage_db[int(passage_id)]
                retrieved_passages.append(
                    RetrievedPassage(id=passage.id, title=passage.title, text=passage.text, score=float(score))
                )

            outputs.append(sorted(retrieved_passages, key=lambda p: p.score, reverse=True))

        assert len(outputs[0]) <= k
        assert len(outputs) == num_questions

        return outputs
