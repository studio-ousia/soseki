from typing import List, Optional, Union

import torch

from ..biencoder.modeling import BiencoderLightningModule
from ..retriever.binary_retriever import BinaryRetriever
from ..retriever.dense_retriever import DenseRetriever
from ..reader.modeling import ReaderLightningModule
from ..utils.data_utils import AnswerCandidate, RetrievedPassage


class EndToEndQuestionAnswering:
    def __init__(
        self,
        biencoder_ckpt_file: str,
        reader_ckpt_file: str,
        passage_embeddings_file: str,
        passage_db_file: Optional[str] = None,
        passage_file: Optional[str] = None,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        self.biencoder = BiencoderLightningModule.load_from_checkpoint(biencoder_ckpt_file, map_location="cpu")
        self.reader = ReaderLightningModule.load_from_checkpoint(reader_ckpt_file, map_location="cpu")
        self.biencoder.freeze()
        self.reader.freeze()

        self.encoder_tokenization = self.biencoder.tokenization
        self.reader_tokenization = self.reader.tokenization

        self.max_question_length = self.biencoder.hparams.max_question_length
        self.max_reader_input_length = self.reader.hparams.max_input_length

        self.binary = self.biencoder.hparams.binary
        if self.binary:
            self.retriever = BinaryRetriever(
                passage_embeddings_file, passage_db_file=passage_db_file, passage_file=passage_file
            )
        else:
            self.retriever = DenseRetriever(
                passage_embeddings_file, passage_db_file=passage_db_file, passage_file=passage_file
            )

        self.device = device
        self.biencoder.to(self.device)
        self.reader.to(self.device)

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
            return_tensors="pt",
        )
        encoder_inputs = {key: tensor.to(self.device) for key, tensor in encoder_inputs.items()}
        encoded_questions = self.biencoder.question_encoder(encoder_inputs).cpu().numpy()

        if self.binary:
            return self.retriever.retrieve_top_k_passages(encoded_questions, k=k, binary_k=binary_k)
        else:
            return self.retriever.retrieve_top_k_passages(encoded_questions, k=k)

    def answer_question(
        self,
        question: str,
        num_retrieval_passages: int = 10,
        num_passages_to_read: int = 1,
        num_answer_candidates_per_passage: int = 1,
    ) -> List[AnswerCandidate]:
        with torch.no_grad():
            # Retrieve passages for the question.
            retrieved_passages = self.retrieve_top_k_passages(question, k=num_retrieval_passages)[0]

            # Tokenize pairs of the question and a passage, which make the inputs to the reader.
            reader_inputs = []
            for passage in retrieved_passages:
                reader_input = self.reader.tokenization.tokenize_input(
                    question,
                    passage.title,
                    passage.text,
                    padding="max_length",
                    truncation="only_second",
                    max_length=self.max_reader_input_length,
                )
                reader_inputs.append(reader_input)

            # Tensorize the reader inputs.
            reader_inputs = {
                key: torch.tensor([x[key] for x in reader_inputs]).to(self.device) for key in reader_inputs[0].keys()
            }

            # Apply the reader's inference to the input.
            classifier_logits, start_logits, end_logits = self.reader.forward(reader_inputs)

            # Generate answer candidates from the reader's inference.
            answer_candidates = self.reader.generate_answer_candidates(
                input_ids=reader_inputs["input_ids"],
                classifier_logits=classifier_logits,
                start_logits=start_logits,
                end_logits=end_logits,
                num_passages_to_read=num_passages_to_read,
                num_answer_candidates_per_passage=num_answer_candidates_per_passage,
            )
            return answer_candidates
