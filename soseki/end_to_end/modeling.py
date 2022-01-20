from typing import Optional

import torch

from ..biencoder.modeling import BiencoderLightningModule
from ..retriever.binary_retriever import BinaryRetriever
from ..retriever.dense_retriever import DenseRetriever
from ..reader.modeling import ReaderLightningModule


class EndToEndQuestionAnswering():
    def __init__(
        self,
        biencoder_ckpt_file: str,
        reader_ckpt_file: str,
        passage_embeddings_file: str,
        passage_db_file: Optional[str] = None,
        passage_file: Optional[str] = None,
        device: Optional[torch.device] = "cpu",
    ):
        self.biencoder = BiencoderLightningModule.load_from_checkpoint(biencoder_ckpt_file, map_location="cpu")
        self.reader = ReaderLightningModule.load_from_checkpoint(reader_ckpt_file, map_location="cpu")
        self.biencoder.freeze()
        self.reader.freeze()

        self.binary = self.biencoder.hparams["binary"]
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

    def answer_question(
        self,
        question: str,
        num_retrieval_passages: int = 10,
        num_reading_passages: int = 1,
        num_answer_candidates_per_passage: int = 1,
    ):
        with torch.no_grad():
            encoder_inputs = dict(self.biencoder.tokenization.tokenize_questions(
                question,
                padding=True,
                truncation=True,
                max_length=self.biencoder.hparams.max_question_length,
                return_tensors="pt",
            ))
            encoder_inputs = {key: tensor.to(self.device) for key, tensor in encoder_inputs.items()}
            encoded_questions = self.biencoder.question_encoder(**encoder_inputs).cpu().numpy()

            retrieved_passages = self.retriever.retrieve_top_k_passages(encoded_questions, k=num_retrieval_passages)[0]

            input_ids = []
            attention_mask = []
            token_type_ids = []
            for passage in retrieved_passages:
                encoded_question_and_passage = self.reader.tokenization.tokenize_input(
                    question,
                    passage.title,
                    passage.text,
                    padding="max_length",
                    truncation="only_second",
                    max_length=self.reader.hparams.max_input_length,
                )
                input_ids.append(encoded_question_and_passage["input_ids"])
                attention_mask.append(encoded_question_and_passage["attention_mask"])
                token_type_ids.append(encoded_question_and_passage["token_type_ids"])

            input_ids = torch.tensor([input_ids]).to(self.device)
            attention_mask = torch.tensor([attention_mask]).to(self.device)
            token_type_ids = torch.tensor([token_type_ids]).to(self.device)

            classifier_logits, start_logits, end_logits = self.reader.forward(
                {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
            )
            answer_candidates = self.reader.generate_answer_candidates(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                classifier_logits=classifier_logits,
                start_logits=start_logits,
                end_logits=end_logits,
                num_reading_passages=num_reading_passages,
                num_answer_candidates_per_passage=num_answer_candidates_per_passage,
            )[0]
            return answer_candidates
