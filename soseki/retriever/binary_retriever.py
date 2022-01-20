from typing import List, Optional

import faiss
import numpy as np

from ..utils.data_utils import RetrievedPassage


class BinaryRetriever:
    def __init__(
        self,
        faiss_index_file: str,
        passage_db_file: Optional[str] = None,
        passage_file: Optional[str] = None,
    ) -> None:
        self.faiss_index: faiss.IndexBinaryIDMap2 = faiss.read_index_binary(faiss_index_file)
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
        k: int = 10,
        binary_k: int = 2048,
    ) -> List[List[RetrievedPassage]]:
        num_questions, question_dim = encoded_questions.shape

        binary_encoded_questions = np.packbits(np.where(encoded_questions < 0, 0, 1), axis=-1)
        assert binary_encoded_questions.shape == (num_questions, question_dim // 8)

        _, passage_ids_array = self.faiss_index.search(binary_encoded_questions, binary_k)
        assert passage_ids_array.shape == (num_questions, binary_k)

        encoded_passages = np.vstack(
            [np.unpackbits(self.faiss_index.reconstruct(int(pi))) for pi in passage_ids_array.flatten()]
        ).reshape(num_questions, binary_k, question_dim).astype(np.float32)
        encoded_passages = encoded_passages * 2 - 1
        assert encoded_passages.shape == (num_questions, binary_k, question_dim)

        scores_array = np.einsum("ijk,ik->ij", encoded_passages, encoded_questions)
        assert scores_array.shape == (num_questions, binary_k)

        outputs = []
        for scores, passage_ids in zip(scores_array, passage_ids_array):
            retrieved_passages = []
            for score, passage_id in zip(scores, passage_ids):
                passage = self.passage_db[int(passage_id)]
                retrieved_passages.append(
                    RetrievedPassage(id=passage.id, title=passage.title, text=passage.text, score=float(score))
                )

            outputs.append(sorted(retrieved_passages, key=lambda p: p.score, reverse=True)[:k])

        assert len(outputs[0]) <= k
        assert len(outputs) == num_questions

        return outputs
