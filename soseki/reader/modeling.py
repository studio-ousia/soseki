import json
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

from .tokenization import ReaderTokenization
from ..utils.data_utils import ReaderDatasetExample, RetrievedPassage, AnswerCandidate, readitem_json


class ReaderModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        pooling_index: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False, return_dict=False)

        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

        self._pooling_index = pooling_index

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor) -> Tuple[Tensor]:
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = encoder_output[0]

        pooled_output = sequence_output[:, self._pooling_index, :]
        classifier_logits = self.qa_classifier(pooled_output).squeeze(-1)

        start_logits, end_logits = self.qa_outputs(sequence_output).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return classifier_logits, start_logits, end_logits


class ReaderLightningModule(LightningModule):
    def __init__(self, hparams: Optional[Namespace] = None, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)

        self.reader = ReaderModel(
            self.hparams.base_pretrained_model,
            pooling_index=self.hparams.reader_pooling_index,
        )

        self.tokenization = ReaderTokenization(
            self.hparams.base_pretrained_model,
            include_title_in_passage=self.hparams.include_title_in_passage,
            answer_normalization_type=self.hparams.answer_normalization_type,
        )

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Reader")

        parser.add_argument("--train_file", type=str, required=True)
        parser.add_argument("--dev_file", type=str, required=True)
        parser.add_argument("--test_file", type=str)
        parser.add_argument("--train_gold_passages_file", type=str)
        parser.add_argument("--dev_gold_passages_file", type=str)
        parser.add_argument("--test_gold_passages_file", type=str)

        parser.add_argument("--train_max_load_passages", type=int)
        parser.add_argument("--train_max_load_positive_passages", type=int)
        parser.add_argument("--train_max_load_negative_passages", type=int)
        parser.add_argument("--eval_max_load_passages", type=int)

        parser.add_argument("--train_num_passages", type=int, default=24)
        parser.add_argument("--eval_num_passages", type=int, default=50)
        parser.add_argument("--max_answer_spans", type=int, default=10)
        parser.add_argument("--max_input_length", type=int, default=350)
        parser.add_argument("--max_answer_length", type=int, default=10)
        parser.add_argument("--include_title_in_passage", action="store_true")
        parser.add_argument("--shuffle_positive_passages", action="store_true")
        parser.add_argument("--shuffle_negative_passages", action="store_true")
        parser.add_argument("--num_dataloader_workers", type=int, default=4)

        parser.add_argument("--base_pretrained_model", type=str, default="bert-base-uncased")
        parser.add_argument("--reader_pooling_index", type=int, default=0)

        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--warmup_proportion", type=float, default=0.0)
        parser.add_argument("--weight_decay", type=float, default=0.0)

        parser.add_argument("--answer_normalization_type", default="dpr")

        return parent_parser

    def setup(self, stage: str) -> None:
        if stage == "fit":
            rank_zero_info("Loading training data")
            self._train_dataset = self._load_dataset(
                self.hparams.train_file,
                self.hparams.train_gold_passages_file,
                training=True,
            )
            rank_zero_info(f"The number of examples in training data: {len(self._train_dataset)}")

        if stage in ("fit", "validate"):
            rank_zero_info("Loading development data")
            self._dev_dataset = self._load_dataset(
                self.hparams.dev_file,
                self.hparams.dev_gold_passages_file,
                training=False,
            )
            rank_zero_info(f"The number of examples in development data: {len(self._dev_dataset)}")

        if stage == "test":
            rank_zero_info("Loading test data")
            self._test_dataset = self._load_dataset(
                self.hparams.test_file,
                self.hparams.test_gold_passages_file,
                training=False,
            )
            rank_zero_info(f"The number of examples in test data: {len(self._test_dataset)}")

    def _load_dataset(
        self,
        dataset_file: str,
        gold_passages_file: Optional[str],
        training: bool,
    ) -> List[ReaderDatasetExample]:
        if gold_passages_file is not None:
            gold_passage_info, canonical_question_info = self._load_gold_passages_file(gold_passages_file)
        else:
            gold_passage_info, canonical_question_info = {}, {}

        dataset_iterator = readitem_json(dataset_file)
        if self.global_rank == 0:
            dataset_iterator = tqdm(dataset_iterator)

        dataset = []
        for item in dataset_iterator:
            index = len(dataset)
            example = self._create_dataset_example(
                index,
                item,
                gold_passage_info,
                canonical_question_info,
                training,
            )
            if training and len(example.positive_passage_idxs) == 0:
                continue

            dataset.append(example)

        return dataset

    def _load_gold_passages_file(file_path: str) -> Tuple[Dict[str, RetrievedPassage], Dict[str, str]]:
        gold_passage_info = {}        # question or question_tokens -> passage
        canonical_question_info = {}  # question_tokens -> question

        with open(file_path) as f:
            items = json.load(f)["data"]

        for item in items:
            question: str = item["question"]
            question_tokens: str = item.get("question_tokens", question)  # an alternate version of the question
            canonical_question_info[question_tokens] = question

            passage = RetrievedPassage(item["example_id"], title=item["title"], text=item["context"])
            gold_passage_info[question] = passage
            gold_passage_info[question_tokens] = passage

        return gold_passage_info, canonical_question_info

    def _create_dataset_example(
        self,
        index: int,
        item: Dict[str, Any],
        gold_passage_info: Dict[str, RetrievedPassage],
        canonical_question_info: Dict[str, str],
        training: bool,
        include_gold_passage: bool = False,
        gold_page_only_positives: bool = True,
    ) -> ReaderDatasetExample:
        question = item["question"]
        if question in canonical_question_info:
            question = canonical_question_info[question]

        answers = item["answers"]

        def has_any_answer_span(passage: RetrievedPassage):
            _, answer_spans = self.tokenization.tokenize_input_with_answers(
                question,
                passage.title,
                passage.text,
                answers,
                padding="max_length",
                truncation="only_second",
                max_length=self.hparams.max_input_length,
            )
            return len(answer_spans) > 0

        passages = [RetrievedPassage(**ctx) for ctx in item["ctxs"]]
        gold_passage = gold_passage_info.get(question)  # not necessarily included in the retrieved passages

        if training:
            if self.hparams.train_max_load_passages is not None:
                passages = passages[:self.hparams.train_max_load_passages]

            positive_passage_idxs = [idx for idx, p in enumerate(passages) if p.has_answer and has_any_answer_span(p)]
            negative_passage_idxs = [idx for idx, p in enumerate(passages) if not p.has_answer]

            if gold_passage is not None:
                gold_positive_passage_idxs = [
                    idx for idx in positive_passage_idxs if passages[idx].title.lower() == gold_passage.title.lower()
                ]
            else:
                gold_positive_passage_idxs = []

            # if specified so and possible, take the gold passages as the positive passages
            if gold_page_only_positives and len(gold_positive_passage_idxs) > 0:
                positive_passage_idxs = gold_positive_passage_idxs
            else:
                if self.hparams.train_max_load_positive_passages is not None:
                    positive_passage_idxs = positive_passage_idxs[:self.hparams.train_max_load_positive_passages]

            # if specified so and possible, include the question's gold passage in the positive passages
            if include_gold_passage and gold_passage is not None:
                if not any(passages[i].id == gold_passage.id for i in positive_passage_idxs):
                    if has_any_answer_span(gold_passage):
                        passages.append(gold_passage)
                        positive_passage_idxs.append(len(passages) - 1)

            if self.hparams.train_max_load_negative_passages is not None:
                negative_passage_idxs = negative_passage_idxs[:self.hparams.train_max_load_negative_passages]

        else:
            if self.hparams.eval_max_load_passages is not None:
                passages = passages[:self.hparams.eval_max_load_passages]

            positive_passage_idxs = [i for i, p in enumerate(passages) if p.has_answer and has_any_answer_span(p)]
            negative_passage_idxs = [i for i, p in enumerate(passages) if not p.has_answer]

        return ReaderDatasetExample(index, question, answers, passages, positive_passage_idxs, negative_passage_idxs)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            dataset=self._train_dataset,
            training=True,
            batch_size=self.hparams.train_batch_size,
            shuffle_dataset=True,
            num_passages_per_question=self.hparams.train_num_passages,
            shuffle_positive_passages=self.hparams.shuffle_positive_passages,
            shuffle_negative_passages=self.hparams.shuffle_negative_passages,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return self._get_dataloader(
            dataset=self._dev_dataset,
            training=False,
            batch_size=self.hparams.eval_batch_size,
            shuffle_dataset=False,
            num_passages_per_question=self.hparams.eval_num_passages,
            shuffle_positive_passages=False,
            shuffle_negative_passages=False,
        )

    def test_dataloader(self) -> List[DataLoader]:
        return self._get_dataloader(
            dataset=self._test_dataset,
            training=False,
            batch_size=self.hparams.eval_batch_size,
            shuffle_dataset=False,
            num_passages_per_question=self.hparams.eval_num_passages,
            shuffle_positive_passages=False,
            shuffle_negative_passages=False,
        )

    def _get_dataloader(
        self,
        dataset: Dataset,
        training: bool,
        batch_size: int,
        shuffle_dataset: bool,
        num_passages_per_question: int,
        shuffle_positive_passages: bool = False,
        shuffle_negative_passages: bool = False,
    ) -> DataLoader:
        collate_fn = self._get_collate_fn(
            training=training,
            num_passages_per_question=num_passages_per_question,
            shuffle_positive_passages=shuffle_positive_passages,
            shuffle_negative_passages=shuffle_negative_passages,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_dataset,
            num_workers=self.hparams.num_dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _get_collate_fn(
        self,
        training: bool,
        num_passages_per_question: int,
        shuffle_positive_passages: bool = False,
        shuffle_negative_passages: bool = False,
    ) -> Callable:
        def collate_fn(batch_examples: List[ReaderDatasetExample]) -> Dict[str, Tensor]:
            input_ids: List[List[int]] = []
            attention_mask: List[List[int]] = []
            token_type_ids: List[List[int]] = []
            start_positions: List[List[int]] = []
            end_positions: List[List[int]] = []
            answer_mask: List[List[int]] = []
            example_idx: List[int] = []
            passage_idxs: List[List[int]] = []

            for example in batch_examples:
                example_idx.append(example.index)
                passage_count = 0

                if training:
                    # positive passage
                    if shuffle_positive_passages:
                        positive_passage_idx = random.choice(example.positive_passage_idxs)
                    else:
                        positive_passage_idx = example.positive_passage_idxs[0]

                    positive_passage = example.passages[positive_passage_idx]

                    positive_input, answer_spans = self.tokenization.tokenize_input_with_answers(
                        example.question,
                        positive_passage.title,
                        positive_passage.text,
                        example.answers,
                        padding="max_length",
                        truncation="only_second",
                        max_length=self.hparams.max_input_length,
                    )
                    answer_spans = answer_spans[:self.hparams.max_answer_spans]
                    answers_padding_length = self.hparams.max_answer_spans - len(answer_spans)
                    answer_starts = [span[0] for span in answer_spans] + [0] * answers_padding_length
                    answer_ends = [span[1] for span in answer_spans] + [0] * answers_padding_length
                    answer_masks = [1] * len(answer_spans) + [0] * answers_padding_length

                    input_ids.append(positive_input["input_ids"])
                    attention_mask.append(positive_input["attention_mask"])
                    token_type_ids.append(positive_input["token_type_ids"])
                    passage_idxs.append(positive_passage_idx)

                    start_positions.append(answer_starts)
                    end_positions.append(answer_ends)
                    answer_mask.append(answer_masks)

                    passage_count += 1

                    # negative passages
                    negative_passage_idxs = example.negative_passage_idxs
                    if shuffle_negative_passages:
                        random.shuffle(negative_passage_idxs)

                    max_negative_passages_per_question = num_passages_per_question - passage_count
                    negative_passage_idxs = negative_passage_idxs[:max_negative_passages_per_question]

                    for negative_passage_idx in negative_passage_idxs:
                        negative_passage = example.passages[negative_passage_idx]
                        negative_input = self.tokenization.tokenize_input(
                            example.question,
                            negative_passage.title,
                            negative_passage.text,
                            padding="max_length",
                            truncation="only_second",
                            max_length=self.hparams.max_input_length,
                        )

                        input_ids.append(negative_input["input_ids"])
                        attention_mask.append(negative_input["attention_mask"])
                        token_type_ids.append(negative_input["token_type_ids"])
                        passage_idxs.append(negative_passage_idx)

                        passage_count += 1

                else:
                    passages = example.passages[:num_passages_per_question]
                    for passage_idx, passage in enumerate(passages):
                        passage_input = self.tokenization.tokenize_input(
                            example.question,
                            passage.title,
                            passage.text,
                            padding="max_length",
                            truncation="only_second",
                            max_length=self.hparams.max_input_length,
                        )

                        input_ids.append(passage_input["input_ids"])
                        attention_mask.append(passage_input["attention_mask"])
                        token_type_ids.append(passage_input["token_type_ids"])
                        passage_idxs.append(passage_idx)

                        passage_count += 1

                # supply empty passages if necessary
                while passage_count < num_passages_per_question:
                    input_ids.append([self.tokenization.tokenizer.pad_token_id] * self.hparams.max_input_length)
                    attention_mask.append([0] * self.hparams.max_input_length)
                    token_type_ids.append([0] * self.hparams.max_input_length)
                    passage_idxs.append(-1)

                    passage_count += 1

            # check and modify dimentionality of the tensors
            Q = len(batch_examples)            # the number of questions in the batch
            P = num_passages_per_question      # the number of passages per question
            L = self.hparams.max_input_length  # the sequence length of inputs (question + passage)
            A = self.hparams.max_answer_spans  # the number of answer spans per question

            input_ids = torch.tensor(input_ids).reshape(Q, P, L)
            attention_mask = torch.tensor(attention_mask).reshape(Q, P, L)
            token_type_ids = torch.tensor(token_type_ids).reshape(Q, P, L)
            example_idx = torch.tensor(example_idx)
            passage_idxs = torch.tensor(passage_idxs).reshape(Q, P)
            assert input_ids.size()      == (Q, P, L)
            assert attention_mask.size() == (Q, P, L)
            assert token_type_ids.size() == (Q, P, L)
            assert example_idx.size()    == (Q,)
            assert passage_idxs.size()   == (Q, P)

            if training:
                start_positions = torch.tensor(start_positions)
                end_positions = torch.tensor(end_positions)
                answer_mask = torch.tensor(answer_mask)
                assert start_positions.size() == (Q, A)
                assert end_positions.size()   == (Q, A)
                assert answer_mask.size()     == (Q, A)
            else:
                start_positions = None
                end_positions = None
                answer_mask = None

            batch_tensors = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "example_idx": example_idx,
                "passage_idxs": passage_idxs,
                "start_positions": start_positions,
                "end_positions": end_positions,
                "answer_mask": answer_mask,
            }
            return batch_tensors

        return collate_fn

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        # check the tensor sizes
        Q, P, L = input_ids.size()
        assert input_ids.size()      == (Q, P, L)
        assert attention_mask.size() == (Q, P, L)
        assert token_type_ids.size() == (Q, P, L)

        input_ids = batch["input_ids"].reshape(Q * P, L)
        attention_mask = batch["attention_mask"].reshape(Q * P, L)
        token_type_ids = batch["token_type_ids"].reshape(Q * P, L)

        # input the tensors to the reader model
        classifier_logits, start_logits, end_logits = self.reader(input_ids, attention_mask, token_type_ids)

        classifier_logits = classifier_logits.reshape(Q, P)
        start_logits = start_logits.reshape(Q, P, L)
        end_logits = end_logits.reshape(Q, P, L)

        return classifier_logits, start_logits, end_logits

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        classifier_logits, start_logits, end_logits = self.forward(batch)

        # check the tensor sizes
        Q, P = classifier_logits.size()
        _, _, L = start_logits.size()
        assert classifier_logits.size() == (Q, P)
        assert start_logits.size()      == (Q, P, L)
        assert end_logits.size()        == (Q, P, L)

        # classifier loss
        classifier_label = classifier_logits.new_zeros(Q, dtype=torch.long)  # positive passage is always at 0th
        passage_mask = batch["passage_idxs"] != -1
        classifier_loss = self._compute_classifier_loss(
            classifier_logits=classifier_logits,
            classifier_label=classifier_label,
            passage_mask=passage_mask,
        )

        # answer span loss
        start_logits = start_logits[:, 0, :]  # since we compute answer span losses of the positive passages,
        end_logits = end_logits[:, 0, :]      # we take logits for the 0th passages only
        answer_span_loss = self._compute_answer_span_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
            answer_mask=batch["answer_mask"],
        )

        loss = classifier_loss + answer_span_loss
        self.log("loss", loss)
        self.log("classifier_loss", classifier_loss.detach())    # detaching is necessary for logging
        self.log("answer_span_loss", answer_span_loss.detach())

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._eval_step(batch, batch_idx, testing=False)

    def validation_step_end(self, batch_parts_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._eval_step_end(batch_parts_outputs, testing=False)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        return self._eval_epoch_end(outputs, testing=False)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._eval_step(batch, batch_idx, testing=True)

    def test_step_end(self, batch_parts_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._eval_step_end(batch_parts_outputs, testing=True)

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        return self._eval_epoch_end(outputs, testing=True)

    def _eval_step(self, batch: Dict[str, Tensor], batch_idx: int, testing: bool = False) -> Dict[str, Tensor]:
        eval_dataset = self._test_dataset if testing else self._dev_dataset

        classifier_logits, start_logits, end_logits = self.forward(batch)

        # check the tensor sizes
        Q, P = classifier_logits.size()
        _, _, L = start_logits.size()
        assert classifier_logits.size() == (Q, P)
        assert start_logits.size()      == (Q, P, L)
        assert end_logits.size()        == (Q, P, L)

        # classifier precision
        classifier_precision = self._compute_classifier_precision(
            classifier_logits=classifier_logits,
            example_idx=batch["example_idx"],
            passage_idxs=batch["passage_idxs"],
            examples=eval_dataset,
        )

        # answer accuracy
        answer_accuracy = self._compute_answer_accuracy(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            classifier_logits=classifier_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            example_idx=batch["example_idx"],
            examples=eval_dataset,
        )

        output = {
            "classifier_precision": classifier_precision,
            "answer_accuracy": answer_accuracy,
            "example_idx": batch["example_idx"],
        }
        return output

    def _eval_step_end(self, batch_parts_outputs: Dict[str, Tensor], testing: bool = False) -> Dict[str, Tensor]:
        return batch_parts_outputs

    def _eval_epoch_end(self, outputs: List[Dict[str, Tensor]], testing: bool = False) -> None:
        eval_dataset = self._test_dataset if testing else self._dev_dataset
        prefix = "test" if testing else "val"

        # accumurate the metrics
        classifier_precision = torch.cat([x["classifier_precision"] for x in outputs])
        answer_accuracy = torch.cat([x["answer_accuracy"] for x in outputs])
        example_idx = torch.cat([x["example_idx"] for x in outputs])

        if distributed_available():
            # we need to ensure that there is no replication of datapoints
            # cf. https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-in-distributed-data-parallel-ddp-mode
            classifier_precision = self.all_gather(classifier_precision).transpose(0, 1).flatten()[:len(eval_dataset)]
            answer_accuracy = self.all_gather(answer_accuracy).transpose(0, 1).flatten()[:len(eval_dataset)]
            example_idx = self.all_gather(example_idx).transpose(0, 1).flatten()[:len(eval_dataset)]
            assert all(example_idx == torch.arange(example_idx.size(0)).to(example_idx.device))

        # aggregate and log the metrics
        classifier_precision = classifier_precision.sum() / len(eval_dataset)
        self.log(f"{prefix}_classifier_precision", classifier_precision)
        answer_accuracy = answer_accuracy.sum() / len(eval_dataset)
        self.log(f"{prefix}_answer_accuracy", answer_accuracy)

    def _compute_classifier_loss(
        self,
        classifier_logits: Tensor,
        classifier_label: Tensor,
        passage_mask: Tensor,
    ) -> Tensor:
        # check the tensor sizes
        Q, P = classifier_logits.size()
        assert classifier_logits.size() == (Q, P)
        assert classifier_label.size()  == (Q,)
        assert passage_mask.size()   == (Q, P)

        mask_value = -1e4 if classifier_logits.dtype == torch.float16 else -1e8

        classifier_logits = classifier_logits.masked_fill(~passage_mask, mask_value)
        classifier_loss = nn.CrossEntropyLoss(reduction="sum")(classifier_logits, classifier_label)

        return classifier_loss

    def _compute_answer_span_loss(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        start_positions: Tensor,
        end_positions: Tensor,
        answer_mask: Tensor,
    ) -> Tensor:
        # check the tensor sizes
        Q, L = start_logits.size()
        _, A = start_positions.size()
        assert start_logits.size()      == (Q, L)
        assert end_logits.size()        == (Q, L)
        assert start_positions.size()   == (Q, A)
        assert end_positions.size()     == (Q, A)
        assert answer_mask.size()       == (Q, A)

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)

        start_labeled_position_probs = start_probs.gather(dim=1, index=start_positions) * answer_mask.float()
        end_labeled_position_probs = end_probs.gather(dim=1, index=end_positions) * answer_mask.float()
        labeled_span_probs = start_labeled_position_probs * end_labeled_position_probs
        assert labeled_span_probs.size() == (Q, A)

        log_sum_labeled_span_probs = labeled_span_probs.sum(dim=1).log()
        assert log_sum_labeled_span_probs.size() == (Q,)

        span_loss = -log_sum_labeled_span_probs.sum()

        return span_loss

    def _compute_classifier_precision(
        self,
        classifier_logits: Tensor,
        example_idx: Tensor,
        passage_idxs: Tensor,
        examples: List[ReaderDatasetExample],
    ) -> Tensor:
        # check the tensor sizes
        Q, P = classifier_logits.size()
        assert classifier_logits.size() == (Q, P)
        assert example_idx.size()     == (Q,)
        assert passage_idxs.size()      == (Q, P)

        classifier_precision = classifier_logits.new_zeros(Q)

        for qi, pi in enumerate(classifier_logits.argmax(1).tolist()):
            example = examples[example_idx[qi]]
            pred_passage_idx = passage_idxs[qi, pi]

            if pred_passage_idx in example.positive_passage_idxs:
                classifier_precision[qi] = 1.0

        return classifier_precision

    def _compute_answer_accuracy(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        classifier_logits: Tensor,
        start_logits: Tensor,
        end_logits: Tensor,
        example_idx: Tensor,
        examples: List[ReaderDatasetExample],
    ) -> Tensor:
        # check the tensor sizes
        Q, P, L = start_logits.size()
        assert input_ids.size()         == (Q, P, L)
        assert attention_mask.size()    == (Q, P, L)
        assert token_type_ids.size()    == (Q, P, L)
        assert classifier_logits.size() == (Q, P)
        assert start_logits.size()      == (Q, P, L)
        assert end_logits.size()        == (Q, P, L)
        assert example_idx.size()     == (Q,)

        answer_precision = classifier_logits.new_zeros(Q)

        batch_answer_candidates = self.generate_answer_candidates(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            classifier_logits=classifier_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            num_reading_passages=1,
            num_answer_candidates_per_passage=1,
            max_answer_length=self.hparams.max_answer_length,
        )
        for qi, answer_candidates in enumerate(batch_answer_candidates):
            example = examples[example_idx[qi]]
            gold_answers = [self.tokenization._normalize_answer(a) for a in example.answers]

            best_answer_candidate = answer_candidates[0]
            pred_answer = self.tokenization._normalize_answer(best_answer_candidate.answer_text)

            if pred_answer in gold_answers:
                answer_precision[qi] = 1.0

        return answer_precision

    def generate_answer_candidates(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        classifier_logits: Tensor,
        start_logits: Tensor,
        end_logits: Tensor,
        num_reading_passages: int = 1,
        num_answer_candidates_per_passage: int = 1,
        max_answer_length: int = 10,
    ) -> List[List[AnswerCandidate]]:
        # check the tensor sizes
        Q, P, L = input_ids.size()
        assert input_ids.size()         == (Q, P, L)
        assert attention_mask.size()    == (Q, P, L)
        assert token_type_ids.size()    == (Q, P, L)
        assert classifier_logits.size() == (Q, P)
        assert start_logits.size()      == (Q, P, L)
        assert end_logits.size()        == (Q, P, L)

        # obtain passage indices with highest classifier logits for each question
        top_classifier_logits, top_passage_idxs = classifier_logits.topk(num_reading_passages, dim=1)
        assert top_classifier_logits.size() == (Q, min(P, num_reading_passages))
        assert top_passage_idxs.size()   == (Q, min(P, num_reading_passages))

        batch_answer_candidates: List[List[AnswerCandidate]] = []
        for qi in range(Q):
            answer_candidates: List[AnswerCandidate] = []
            for pi, classifier_logit in zip(top_passage_idxs[qi].tolist(), top_classifier_logits[qi].tolist()):
                answer_spans = self.tokenization._compute_best_answer_spans(
                    input_ids=input_ids[qi, pi].tolist(),
                    attention_mask=attention_mask[qi, pi].tolist(),
                    token_type_ids=token_type_ids[qi, pi].tolist(),
                    start_logits=start_logits[qi, pi].tolist(),
                    end_logits=end_logits[qi, pi].tolist(),
                    num_answer_spans=num_answer_candidates_per_passage,
                    max_answer_length=max_answer_length,
                )
                for answer_span in answer_spans:
                    answer_text, input_text = self.tokenization._get_answer_passage_texts_from_input_ids(
                        input_ids=input_ids[qi, pi].tolist(),
                        attention_mask=attention_mask[qi, pi].tolist(),
                        token_type_ids=token_type_ids[qi, pi].tolist(),
                        answer_span=answer_span,
                    )
                    passage_score = classifier_logit
                    span_score = answer_span.start_logit + answer_span.end_logit
                    answer_score = passage_score + span_score

                    answer_candidates.append(
                        AnswerCandidate(
                            answer_text=answer_text,
                            passage_text=input_text,
                            score=answer_score,
                            passage_score=passage_score,
                            span_score=span_score,
                        )
                    )

            answer_candidates = sorted(answer_candidates, key=lambda x: x.score, reverse=True)
            batch_answer_candidates.append(answer_candidates)

        return batch_answer_candidates

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
            // self.hparams.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        num_warmup_steps = int(self.hparams.warmup_proportion * num_training_steps)
        rank_zero_info("The total number of training steps: %d", num_training_steps)
        rank_zero_info("The total number of warmup steps: %d", num_warmup_steps)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
