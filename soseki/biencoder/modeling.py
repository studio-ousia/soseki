import math
import random
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.distributed import distributed_available, rank_zero_info
from transformers import AutoModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from .tokenization import EncoderTokenization
from ..passage_db.lmdb_passage_db import LMDBPassageDB
from ..utils.data_utils import Passage, RetrieverDatasetExample, readitem_json


class EncoderModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        pooling_index: int = 0,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False, return_dict=False)
        self._pooling_index = pooling_index
        self._projection_dim = projection_dim

        if self._projection_dim is not None:
            self.projection = nn.Linear(self.encoder.config.hidden_size, projection_dim)
        else:
            self.projection = None

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Dict[str, Tensor]:
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = encoder_output[0][:, self._pooling_index, :]

        if self.projection is not None:
            pooled_output = self.projection(pooled_output)

        return pooled_output


class BiencoderLightningModule(LightningModule):
    def __init__(self, hparams: Optional[Namespace] = None, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.question_encoder = EncoderModel(
            self.hparams.base_pretrained_model,
            pooling_index=self.hparams.encoder_pooling_index,
            projection_dim=self.hparams.encoder_projection_dim,
        )
        self.passage_encoder = EncoderModel(
            self.hparams.base_pretrained_model,
            pooling_index=self.hparams.encoder_pooling_index,
            projection_dim=self.hparams.encoder_projection_dim,
        )

        self.tokenization = EncoderTokenization(self.hparams.base_pretrained_model)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Biencoder")

        parser.add_argument("--train_file", type=str, required=True)
        parser.add_argument("--dev_file", type=str, required=True)
        parser.add_argument("--passage_db_file", type=str)

        parser.add_argument("--train_max_load_positive_passages", type=int)
        parser.add_argument("--train_max_load_hard_negative_passages", type=int)
        parser.add_argument("--train_max_load_normal_negative_passages", type=int)
        parser.add_argument("--dev_max_load_positive_passages", type=int)
        parser.add_argument("--dev_max_load_hard_negative_passages", type=int)
        parser.add_argument("--dev_max_load_normal_negative_passages", type=int)

        parser.add_argument("--max_question_length", type=int, default=256)
        parser.add_argument("--max_passage_length", type=int, default=256)
        parser.add_argument("--num_negative_passages", type=int, default=1)
        parser.add_argument("--shuffle_positive_passages", action="store_true")
        parser.add_argument("--shuffle_hard_negative_passages", action="store_true")
        parser.add_argument("--shuffle_normal_negative_passages", action="store_true")
        parser.add_argument("--max_hard_negative_passages", type=int)
        parser.add_argument("--max_normal_negative_passages", type=int)
        parser.add_argument("--num_dataloader_workers", type=int, default=4)

        parser.add_argument("--base_pretrained_model", type=str, default="bert-base-uncased")
        parser.add_argument("--encoder_pooling_index", type=int, default=0)
        parser.add_argument("--encoder_projection_dim", type=int)

        parser.add_argument("--binary", action="store_true")
        parser.add_argument("--use_ste", action="store_true")
        parser.add_argument("--hashnet_gamma", type=float, default=0.1)
        parser.add_argument("--use_binary_cross_entropy_loss", action="store_true")
        parser.add_argument("--binary_ranking_loss_margin", type=float, default=0.1)
        parser.add_argument("--no_dense_loss", action="store_true")
        parser.add_argument("--no_binary_loss", action="store_true")

        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--warmup_proportion", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--eval_rank_start_epoch", type=int, default=0)

        return parent_parser

    def setup(self, stage: str) -> None:
        if stage == "fit":
            rank_zero_info("Loading training data")
            self._train_dataset = self._load_dataset(
                self.hparams.train_file,
                max_load_positive_passages=self.hparams.train_max_load_positive_passages,
                max_load_hard_negative_passages=self.hparams.train_max_load_hard_negative_passages,
                max_load_normal_negative_passages=self.hparams.train_max_load_normal_negative_passages,
            )
            rank_zero_info(f"The number of examples in training data: {len(self._train_dataset)}")

        if stage in ("fit", "validate"):
            rank_zero_info("Loading development data")
            self._dev_dataset = self._load_dataset(
                self.hparams.dev_file,
                max_load_positive_passages=self.hparams.dev_max_load_positive_passages,
                max_load_hard_negative_passages=self.hparams.dev_max_load_hard_negative_passages,
                max_load_normal_negative_passages=self.hparams.dev_max_load_normal_negative_passages,
            )
            rank_zero_info(f"The number of examples in development data: {len(self._dev_dataset)}")

    def _load_dataset(
        self,
        dataset_file: str,
        max_load_positive_passages: Optional[int] = None,
        max_load_hard_negative_passages: Optional[int] = None,
        max_load_normal_negative_passages: Optional[int] = None,
    ) -> List[RetrieverDatasetExample]:
        dataset_iterator = readitem_json(dataset_file)
        if self.global_rank == 0:
            dataset_iterator = tqdm(dataset_iterator)

        dataset = []
        for item in dataset_iterator:
            index = len(dataset)
            example = self._create_dataset_example(
                index,
                item,
                max_load_positive_passages=max_load_positive_passages,
                max_load_hard_negative_passages=max_load_hard_negative_passages,
                max_load_normal_negative_passages=max_load_normal_negative_passages,
            )
            if len(example.positive_passages) == 0:
                continue
            if len(example.hard_negative_passages) + len(example.normal_negative_passages) == 0:
                continue

            dataset.append(example)

        return dataset

    def _create_dataset_example(
        self,
        index: int,
        item: Dict[str, Any],
        max_load_positive_passages: Optional[int] = None,
        max_load_hard_negative_passages: Optional[int] = None,
        max_load_normal_negative_passages: Optional[int] = None,
    ) -> RetrieverDatasetExample:
        question = item.pop("question")
        answers = item.pop("answers")

        positive_passages = [
            Passage(id=ctx.get("passage_id"), title=ctx["title"], text=ctx["text"])
            for ctx in item.pop("positive_ctxs")
        ]
        if max_load_positive_passages is not None:
            positive_passages = positive_passages[:max_load_positive_passages]

        hard_negative_passages = [
            Passage(id=ctx.get("passage_id"), title=ctx["title"], text=ctx["text"])
            for ctx in item.pop("hard_negative_ctxs")
        ]
        if max_load_hard_negative_passages is not None:
            hard_negative_passages = hard_negative_passages[:max_load_hard_negative_passages]

        normal_negative_passages = [
            Passage(id=ctx.get("passage_id"), title=ctx["title"], text=ctx["text"])
            for ctx in item.pop("negative_ctxs")
        ]
        if max_load_normal_negative_passages is not None:
            normal_negative_passages = normal_negative_passages[:max_load_normal_negative_passages]

        metadata = item

        example = RetrieverDatasetExample(
            index=index,
            question=question,
            answers=answers,
            positive_passages=positive_passages,
            hard_negative_passages=hard_negative_passages,
            normal_negative_passages=normal_negative_passages,
            metadata=metadata,
        )
        return example

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            dataset=self._train_dataset,
            training=True,
            batch_size=self.hparams.train_batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            dataset=self._dev_dataset,
            training=False,
            batch_size=self.hparams.eval_batch_size,
        )

    def _get_dataloader(self, dataset: Dataset, training: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            num_workers=self.hparams.num_dataloader_workers,
            collate_fn=self._get_collate_fn(training),
            pin_memory=True,
        )

    def _get_collate_fn(self, training: bool) -> Callable:
        def collate_fn(batch_examples: List[RetrieverDatasetExample]) -> Dict[str, Tensor]:
            # passage db is used when the loaded dataset does not have the passage text fields
            if self.hparams.passage_db_file is not None:
                passage_db = LMDBPassageDB(self.hparams.passage_db_file)
            else:
                passage_db = None

            tokenized_questions = []
            tokenized_passages = []
            labels = []

            for example in batch_examples:
                # tokenize the question
                tokenized_question = self.tokenization.tokenize_questions(
                    example.question,
                    padding="max_length",
                    truncation=True,
                    max_length=self.hparams.max_question_length,
                )
                tokenized_questions.append(tokenized_question)

                # select positive passage
                if training and self.hparams.shuffle_positive_passages:
                    positive_passage = random.choice(example.positive_passages)
                else:
                    positive_passage = example.positive_passages[0]

                # select hard negative passages
                hard_negative_passages = example.hard_negative_passages
                if training and self.hparams.shuffle_hard_negative_passages:
                    random.shuffle(hard_negative_passages)

                if self.hparams.max_hard_negative_passages is not None:
                    hard_negative_passages = hard_negative_passages[:self.hparams.max_hard_negative_passages]

                # select normal negative passages
                normal_negative_passages = example.normal_negative_passages
                if training and self.hparams.shuffle_normal_negative_passages:
                    random.shuffle(normal_negative_passages)

                if self.hparams.max_normal_negative_passages is not None:
                    normal_negative_passages = normal_negative_passages[:self.hparams.max_normal_negative_passages]

                # concatenate negative passages; hard negative passages are prioritized
                negative_passages = (hard_negative_passages + normal_negative_passages)
                negative_passages = negative_passages[:self.hparams.num_negative_passages]
                if len(negative_passages) < self.hparams.num_negative_passages:
                    raise ValueError("--num_negative_passages is larger than retrieved negative passages.")

                # tokenize the passages
                for i, passage in enumerate([positive_passage] + negative_passages):
                    passage_text = passage.text
                    if passage_text is None:
                        if passage_db is not None:
                            # fetch passage text from passage db
                            passage_text = passage_db[passage.id].text
                        else:
                            raise KeyError("--passage_db_file must be specified if the dataset have no passage texts")

                    tokenized_passage = self.tokenization.tokenize_passages(
                        passage.title,
                        passage_text,
                        padding="max_length",
                        truncation="only_second",
                        max_length=self.hparams.max_passage_length,
                    )
                    tokenized_passages.append(tokenized_passage)

                    labels.append(int(i == 0))  # positive passage is always indexed 0

            # tensorize the lists of integers
            question_input_ids = torch.tensor([q["input_ids"] for q in tokenized_questions])
            question_attention_mask = torch.tensor([q["attention_mask"] for q in tokenized_questions])
            question_token_type_ids = torch.tensor([q["token_type_ids"] for q in tokenized_questions])
            passage_input_ids = torch.tensor([p["input_ids"] for p in tokenized_passages])
            passage_attention_mask = torch.tensor([p["attention_mask"] for p in tokenized_passages])
            passage_token_type_ids = torch.tensor([p["token_type_ids"] for p in tokenized_passages])
            passage_label = torch.tensor(labels)

            # check dimensionality of the tensors
            Q = len(batch_examples)                     # the number of questions in the batch
            P = 1 + self.hparams.num_negative_passages  # the number of passages per question
            Lq = self.hparams.max_question_length       # the sequence length of questions
            Lp = self.hparams.max_passage_length        # the sequence length of passages
            assert question_input_ids.size()      == (Q, Lq)
            assert question_attention_mask.size() == (Q, Lq)
            assert question_token_type_ids.size() == (Q, Lq)
            assert passage_input_ids.size()       == (Q * P, Lp)
            assert passage_attention_mask.size()  == (Q * P, Lp)
            assert passage_token_type_ids.size()  == (Q * P, Lp)
            assert passage_label.size()           == (Q * P,)

            batch_tensors = {
                "question_input_ids": question_input_ids,
                "question_attention_mask": question_attention_mask,
                "question_token_type_ids": question_token_type_ids,
                "passage_input_ids": passage_input_ids,
                "passage_attention_mask": passage_attention_mask,
                "passage_token_type_ids": passage_token_type_ids,
                "passage_label": passage_label,
            }
            return batch_tensors

        return collate_fn

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        encoded_question = self.question_encoder(
            input_ids=batch["question_input_ids"],
            attention_mask=batch["question_attention_mask"],
            token_type_ids=batch["question_token_type_ids"],
        )
        encoded_passage = self.passage_encoder(
            input_ids=batch["passage_input_ids"],
            attention_mask=batch["passage_attention_mask"],
            token_type_ids=batch["passage_token_type_ids"],
        )
        return encoded_question, encoded_passage

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tuple[Tensor, Tensor]:
        encoded_question, encoded_passage = self.forward(batch)
        return encoded_question, encoded_passage

    def training_step_end(self, batch_parts_outputs: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        encoded_question, encoded_passage = batch_parts_outputs

        if distributed_available():
            encoded_question = self.all_gather(encoded_question, sync_grads=True).reshape(-1, encoded_question.size(-1))
            encoded_passage = self.all_gather(encoded_passage, sync_grads=True).reshape(-1, encoded_passage.size(-1))

        output = self._compute_loss(encoded_question, encoded_passage)
        for key, value in output.items():
            self.log(key, value)

        return output

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_question, encoded_passage = self.forward(batch)
        return encoded_question, encoded_passage, batch["passage_label"]

    def validation_step_end(self, batch_parts_outputs: Tuple[Tensor, Tensor, Tensor]) -> Dict[str, Tensor]:
        encoded_question, encoded_passage, labels = batch_parts_outputs

        if distributed_available():
            encoded_question = self.all_gather(encoded_question, sync_grads=True).reshape(-1, encoded_question.size(-1))
            encoded_passage = self.all_gather(encoded_passage, sync_grads=True).reshape(-1, encoded_passage.size(-1))
            labels = self.all_gather(labels, sync_grads=True).reshape(-1)

        output = self._compute_loss(encoded_question, encoded_passage)
        output["ranks"] = self._compute_ranks(encoded_question, encoded_passage, labels)

        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        ave_loss = torch.stack([output["loss"] for output in outputs]).mean()
        self.log("val_loss", ave_loss)

        if self.current_epoch >= self.hparams.eval_rank_start_epoch:
            avg_rank = torch.cat([output["ranks"] for output in outputs], dim=0).mean()
        else:
            avg_rank = 10000000.0

        self.log("val_avg_rank", avg_rank)

    def _convert_to_binary_code(self, input_tensor: Tensor) -> Tensor:
        if self.training:
            if self.hparams.use_ste:
                hard_input_tensor = input_tensor.new_ones(input_tensor.size()).masked_fill_(input_tensor < 0, -1.0)
                input_tensor = torch.tanh(input_tensor)
                return hard_input_tensor + input_tensor - input_tensor.detach()
            else:
                # https://github.com/thuml/HashNet/blob/55bcaaa0bbaf0c404ca7a071b47d6287dc95e81d/pytorch/src/network.py#L40
                scale = math.pow((1.0 + self.global_step * self.hparams.hashnet_gamma), 0.5)
                return torch.tanh(input_tensor * scale)
        else:
            return input_tensor.new_ones(input_tensor.size()).masked_fill_(input_tensor < 0, -1.0)

    def _compute_loss(self, encoded_question: Tensor, encoded_passage: Tensor) -> Dict[str, Tensor]:
        # The i-th (zero-based) question's positive passage label is i * (num of passages per question)
        # For example, if the number of passages per question is 2 (default), the positive passage label will be
        # label = [0, 2, 4, ..., (num of questions - 1) * 2]
        num_passages_per_question = encoded_passage.size(0) // encoded_question.size(0)
        label = torch.arange(0, encoded_passage.size(0), num_passages_per_question).to(encoded_question.device)

        output = {}

        if self.hparams.binary:
            binary_encoded_passage = self._convert_to_binary_code(encoded_passage)
            dense_scores = torch.matmul(encoded_question, binary_encoded_passage.transpose(0, 1))
            dense_loss = F.cross_entropy(dense_scores, label)

            output["binary_loss"] = dense_loss.detach()

            binary_encoded_question = self._convert_to_binary_code(encoded_question)
            binary_scores = torch.matmul(binary_encoded_question, binary_encoded_passage.transpose(0, 1))
            if self.hparams.use_binary_cross_entropy_loss:
                binary_loss = F.cross_entropy(binary_scores, label)
            else:
                positive_mask = binary_scores.new_zeros(binary_scores.size(), dtype=torch.bool)
                for i, label in enumerate(label):
                    positive_mask[i, label] = True

                positive_binary_scores = torch.masked_select(binary_scores, positive_mask)
                positive_binary_scores = positive_binary_scores.repeat_interleave(binary_encoded_passage.size(0) - 1)
                negative_binary_scores = torch.masked_select(binary_scores, torch.logical_not(positive_mask))
                binary_labels = positive_binary_scores.new_ones(positive_binary_scores.size(), dtype=torch.long)
                binary_loss = F.margin_ranking_loss(
                    positive_binary_scores,
                    negative_binary_scores,
                    binary_labels,
                    self.hparams.binary_ranking_loss_margin,
                )

            output["binary_loss"] = binary_loss.detach()

            if self.hparams.no_binary_loss:
                output["loss"] = dense_loss
            elif self.hparams.no_dense_loss:
                output["loss"] = binary_loss
            else:
                output["loss"] = binary_loss + dense_loss
        else:
            scores = torch.matmul(encoded_question, encoded_passage.transpose(0, 1))
            loss = F.cross_entropy(scores, label)
            output["loss"] = loss

        return output

    def _compute_ranks(self, encoded_question: Tensor, encoded_passage: Tensor, labels: Tensor) -> Tensor:
        scores = torch.matmul(encoded_question, encoded_passage.transpose(0, 1))
        gold_positions = labels.nonzero(as_tuple=False).reshape(-1)
        gold_scores = scores.gather(1, gold_positions.unsqueeze(1))
        ranks = (scores > gold_scores).sum(1).float() + 1.0

        return ranks

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        # set up the optimizer
        optimizer_parameters = [
            {"params": [], "weight_decay": 0.0},
            {"params": [], "weight_decay": self.hparams.weight_decay}
        ]
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            elif "bias" in name or "LayerNorm.weight" in name:
                optimizer_parameters[0]["params"].append(param)
            else:
                optimizer_parameters[1]["params"].append(param)

        optimizer = AdamW(optimizer_parameters, lr=self.hparams.learning_rate)

        # set up the learning rate scheduler
        num_training_steps = int(
            len(self._train_dataset)
            // (self.hparams.train_batch_size * max(self.trainer.num_gpus, 1))
            // self.trainer.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        num_warmup_steps = int(self.hparams.warmup_proportion * num_training_steps)
        rank_zero_info("The total number of training steps: %d", num_training_steps)
        rank_zero_info("The total number of warmup steps: %d", num_warmup_steps)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
