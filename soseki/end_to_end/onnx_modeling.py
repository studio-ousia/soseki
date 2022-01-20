import json
import os
from typing import List, Optional, Union

import numpy as np
import onnxruntime as ort

from ..biencoder.tokenization import EncoderTokenization
from ..retriever.binary_retriever import BinaryRetriever
from ..retriever.dense_retriever import DenseRetriever
from ..reader.tokenization import ReaderTokenization
from ..utils.data_utils import AnswerCandidate, RetrievedPassage


class OnnxEndToEndQuestionAnswering():
    def __init__(
        self,
        onnx_model_dir: str,
        passage_embeddings_file: str,
        passage_db_file: Optional[str] = None,
        passage_file: Optional[str] = None,
    ):
        question_encoder_onnx_file = os.path.join(onnx_model_dir, "question_encoder.onnx")
        reader_onnx_file = os.path.join(onnx_model_dir, "reader.onnx")

        biencoder_hparams_file = os.path.join(onnx_model_dir, "biencoder_hparams.json")
        biencoder_hparams = json.load(open(biencoder_hparams_file))

        reader_hparams_file = os.path.join(onnx_model_dir, "reader_hparams.json")
        reader_hparams = json.load(open(reader_hparams_file))

        self.question_encoder_session = ort.InferenceSession(question_encoder_onnx_file)
        self.reader_session = ort.InferenceSession(reader_onnx_file)

        self.encoder_tokenization = EncoderTokenization(
            base_model_name=biencoder_hparams["base_pretrained_model"],
        )
        self.reader_tokenization = ReaderTokenization(
            base_model_name=reader_hparams["base_pretrained_model"],
            include_title_in_passage=reader_hparams["include_title_in_passage"],
            answer_normalization_type=reader_hparams["answer_normalization_type"],
        )

        self.binary = biencoder_hparams["binary"]
        if self.binary:
            self.retriever = BinaryRetriever(
                passage_embeddings_file, passage_db_file=passage_db_file, passage_file=passage_file
            )
        else:
            self.retriever = DenseRetriever(
                passage_embeddings_file, passage_db_file=passage_db_file, passage_file=passage_file
            )

        self.max_question_length = biencoder_hparams["max_question_length"]
        self.max_reader_input_length = reader_hparams["max_input_length"]

    def retrieve_top_k_passages(
        self,
        questions: Union[str, List[str]],
        k: int = 10,
        binary_k: int = 2048,
    ) -> List[List[RetrievedPassage]]:
        encoder_inputs = self.encoder_tokenization.tokenize_questions(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="np",
        )
        encoded_questions = self.question_encoder_session.run(None, dict(encoder_inputs))[0]

        if self.binary:
            return self.retriever.retrieve_top_k_passages(encoded_questions, k=k, binary_k=binary_k)
        else:
            return self.retriever.retrieve_top_k_passages(encoded_questions, k=k)

    def _generate_answer_candidates(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray,
        classifier_logits: np.ndarray,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        num_reading_passages: int = 1,
        num_answer_candidates_per_passage: int = 1,
    ) -> List[List[AnswerCandidate]]:
        # obtain passage indices with highest classifier logit
        top_passage_idxs = classifier_logits.argsort()[::-1][:num_reading_passages]

        answer_candidates: List[AnswerCandidate] = []
        for pi in top_passage_idxs.tolist():
            answer_spans = self.reader_tokenization._compute_best_answer_spans(
                input_ids=input_ids[pi].tolist(),
                attention_mask=attention_mask[pi].tolist(),
                token_type_ids=token_type_ids[pi].tolist(),
                start_logits=start_logits[pi].tolist(),
                end_logits=end_logits[pi].tolist(),
                num_answer_spans=num_answer_candidates_per_passage,
            )
            for answer_span in answer_spans:
                answer_text, passage_text = self.reader_tokenization._get_answer_passage_texts_from_input_ids(
                    input_ids=input_ids[pi].tolist(),
                    attention_mask=attention_mask[pi].tolist(),
                    token_type_ids=token_type_ids[pi].tolist(),
                    answer_span=answer_span,
                )
                passage_score = classifier_logits[pi].item()
                span_score = answer_span.start_logit + answer_span.end_logit
                answer_score = passage_score + span_score

                answer_candidates.append(
                    AnswerCandidate(
                        answer_text=answer_text,
                        passage_text=passage_text,
                        score=answer_score,
                        passage_score=passage_score,
                        span_score=span_score,
                    )
                )

        answer_candidates = sorted(answer_candidates, key=lambda x: x.score, reverse=True)
        return answer_candidates

    def answer_question(
        self,
        question: str,
        num_retrieval_passages: int = 10,
        num_reading_passages: int = 1,
        num_answer_candidates_per_passage: int = 1,
    ):
        retrieved_passages = self.retrieve_top_k_passages(question, k=num_retrieval_passages)[0]

        input_ids = []
        attention_mask = []
        token_type_ids = []
        for passage in retrieved_passages:
            encoded_question_and_passage = self.reader_tokenization.tokenize_input(
                question,
                passage.title,
                passage.text,
                padding="max_length",
                truncation="only_second",
                max_length=self.max_reader_input_length,
            )
            input_ids.append(encoded_question_and_passage["input_ids"])
            attention_mask.append(encoded_question_and_passage["attention_mask"])
            token_type_ids.append(encoded_question_and_passage["token_type_ids"])

        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)
        token_type_ids = np.array(token_type_ids)

        classifier_logits, start_logits, end_logits = self.reader_session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
        )
        answer_candidates = self._generate_answer_candidates(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            classifier_logits=classifier_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            num_reading_passages=num_reading_passages,
            num_answer_candidates_per_passage=num_answer_candidates_per_passage,
        )
        return answer_candidates
