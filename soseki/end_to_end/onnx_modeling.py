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


class OnnxEndToEndQuestionAnswering:
    def __init__(
        self,
        onnx_model_dir: str,
        passage_embeddings_file: str,
        passage_db_file: Optional[str] = None,
        passage_file: Optional[str] = None,
    ) -> None:
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
        classifier_logits: np.ndarray,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        num_passages_to_read: int = 1,
        num_answer_candidates_per_passage: int = 1,
        max_answer_length: int = 10,
    ) -> List[AnswerCandidate]:
        num_passages, max_input_length = input_ids.shape

        # Check the shapes of the input arrays.
        assert input_ids.shape == (num_passages, max_input_length)
        assert classifier_logits.shape == (num_passages,)
        assert start_logits.shape == (num_passages, max_input_length)
        assert end_logits.shape == (num_passages, max_input_length)

        # Obtain `num_passages_to_read` passage indices with highest classifier logits.
        top_passage_idxs = classifier_logits.argsort()[::-1][:num_passages_to_read]
        assert top_passage_idxs.shape == (min(num_passages, num_passages_to_read),)

        # Generate answer candidates for each passage of the question.
        answer_candidates = []
        for pi in top_passage_idxs.tolist():
            answer_spans = self.reader_tokenization._compute_best_answer_spans(
                input_ids=input_ids[pi].tolist(),
                start_logits=start_logits[pi].tolist(),
                end_logits=end_logits[pi].tolist(),
                num_answer_spans=num_answer_candidates_per_passage,
                max_answer_length=max_answer_length,
            )
            for answer_span in answer_spans:
                input_text, answer_text, answer_text_span = self.reader_tokenization._get_input_and_answer_texts(
                    input_ids=input_ids[pi].tolist(),
                    answer_span=answer_span,
                )
                classifier_score = classifier_logits[pi].item()
                span_score = answer_span.start_logit + answer_span.end_logit
                answer_score = classifier_score + span_score

                answer_candidates.append(
                    AnswerCandidate(
                        input_text=input_text,
                        answer_text=answer_text,
                        answer_text_span=answer_text_span,
                        score=answer_score,
                        classifier_score=classifier_score,
                        span_score=span_score,
                    )
                )

        # The answer candidates are sorted by their scores in descending order.
        answer_candidates = sorted(answer_candidates, key=lambda x: x.score, reverse=True)

        return answer_candidates

    def answer_question(
        self,
        question: str,
        num_retrieval_passages: int = 10,
        num_passages_to_read: int = 1,
        num_answer_candidates_per_passage: int = 1,
    ) -> List[AnswerCandidate]:
        # Retrieve passages for the question.
        retrieved_passages = self.retrieve_top_k_passages(question, k=num_retrieval_passages)[0]

        # Tokenize pairs of the question and a passage, which make the inputs to the reader.
        reader_inputs = []
        for passage in retrieved_passages:
            reader_input = self.reader_tokenization.tokenize_input(
                question,
                passage.title,
                passage.text,
                padding="max_length",
                truncation="only_second",
                max_length=self.max_reader_input_length,
            )
            reader_inputs.append(reader_input)

        # Tensorize the reader inputs.
        reader_inputs = {key: np.array([x[key] for x in reader_inputs]) for key in reader_inputs[0].keys()}

        # Apply the reader's inference to the input.
        classifier_logits, start_logits, end_logits = self.reader_session.run(None, dict(reader_inputs))

        # Generate answer candidates from the reader's inference.
        answer_candidates = self._generate_answer_candidates(
            input_ids=reader_inputs["input_ids"],
            classifier_logits=classifier_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            num_passages_to_read=num_passages_to_read,
            num_answer_candidates_per_passage=num_answer_candidates_per_passage,
        )
        return answer_candidates
