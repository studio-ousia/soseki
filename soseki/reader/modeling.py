import json
import random
from argparse import ArgumentParser, Namespace
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from transformers import AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

from .tokenization import ReaderTokenization
from ..utils.data_utils import ReaderDatasetExample, RetrievedPassage, AnswerCandidate, readitem_json


class ReaderModel(nn.Module):
    def __init__(self, base_model_name: str, pooling_index: int = 0) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False, return_dict=False)
        self.pooling_index = pooling_index

        self.qa_outputs = nn.Linear(self.encoder.config.hidden_size, 2)
        self.qa_classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_tensors: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        encoder_output = self.encoder(**input_tensors)
        sequence_output = encoder_output[0]

        pooled_output = sequence_output[:, self.pooling_index, :]
        classifier_logits = self.qa_classifier(pooled_output).squeeze(-1)

        start_logits, end_logits = self.qa_outputs(sequence_output).split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return classifier_logits, start_logits, end_logits


class ReaderLightningModule(LightningModule):
    def __init__(self, hparams: Optional[Namespace] = None, **kwargs) -> None:
        super().__init__()
        # Set arguments to `hparams` attribute.
        self.save_hyperparameters(hparams)

        # Initialize the reader model.
        self.reader = ReaderModel(
            self.hparams.base_pretrained_model,
            pooling_index=self.hparams.reader_pooling_index,
        )

        # Initialize the tokenizer.
        self.tokenization = ReaderTokenization(
            self.hparams.base_pretrained_model,
            answer_normalization_type=self.hparams.answer_normalization_type,
        )

        # Set attributes used for checking tensor shapes.
        self.train_num_passages = self.hparams.train_num_passages
        self.eval_num_passages = self.hparams.eval_num_passages
        self.max_input_length = self.hparams.max_input_length
        self.num_answers = self.hparams.max_answer_spans

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Reader")

        parser.add_argument("--train_file", type=str, nargs="+", required=True)
        parser.add_argument("--val_file", type=str, nargs="+", required=True)
        parser.add_argument("--test_file", type=str, nargs="+")
        parser.add_argument("--train_gold_passages_file", type=str)
        parser.add_argument("--val_gold_passages_file", type=str)
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
        parser.add_argument("--shuffle_positive_passages", action="store_true")
        parser.add_argument("--shuffle_negative_passages", action="store_true")
        parser.add_argument("--num_dataloader_workers", type=int, default=4)

        parser.add_argument("--base_pretrained_model", type=str, default="bert-base-uncased")
        parser.add_argument("--reader_pooling_index", type=int, default=0)
        parser.add_argument("--answer_normalization_type", default="dpr")

        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--learning_rate", type=float, default=1e-5)
        parser.add_argument("--warmup_proportion", type=float, default=0.0)
        parser.add_argument("--weight_decay", type=float, default=0.0)

        return parent_parser

    def setup(self, stage: str) -> None:
        # Load the training dataset.
        if stage == "fit":
            rank_zero_info("Loading the training dataset")
            self._train_dataset = self._load_dataset(
                dataset_files=self.hparams.train_file,
                gold_passages_file=self.hparams.train_gold_passages_file,
                training=True,
            )
            rank_zero_info(f"The number of examples in training data: {len(self._train_dataset)}")

        # Load the validation dataset.
        if stage in ("fit", "validate"):
            rank_zero_info("Loading the validation dataset")
            self._val_dataset = self._load_dataset(
                dataset_files=self.hparams.val_file,
                gold_passages_file=self.hparams.val_gold_passages_file,
                training=False,
            )
            rank_zero_info(f"The number of examples in validation data: {len(self._val_dataset)}")

        # Load the test dataset.
        if stage == "test":
            rank_zero_info("Loading the test dataset")
            self._test_dataset = self._load_dataset(
                dataset_files=self.hparams.test_file,
                gold_passages_file=self.hparams.test_gold_passages_file,
                training=False,
            )
            rank_zero_info(f"The number of examples in test data: {len(self._test_dataset)}")

    def _load_dataset(
        self,
        dataset_files: List[str],
        gold_passages_file: Optional[str],
        training: bool,
    ) -> Dataset:
        # Load a gold passage file if provided.
        if gold_passages_file is not None:
            gold_passage_info, canonical_question_info = self._load_gold_passages_file(gold_passages_file)
        else:
            gold_passage_info, canonical_question_info = {}, {}

        # Initialize a dataset iterator for reading JSON (JSON Lines) files.
        dataset_iterator = chain.from_iterable(map(readitem_json, dataset_files))
        if self.global_rank == 0:
            dataset_iterator = tqdm(dataset_iterator)

        dataset = []

        # Load the dataset.
        index = 0
        for item in dataset_iterator:
            example = self._create_dataset_example(
                index,
                item,
                gold_passage_info,
                canonical_question_info,
                training=training,
            )

            # For training data, skip items with no postive passages.
            if training and len(example.positive_passage_idxs) == 0:
                continue

            dataset.append(example)
            index += 1

        return dataset

    def _load_gold_passages_file(file_path: str) -> Tuple[Dict[str, RetrievedPassage], Dict[str, str]]:
        gold_passage_info = {}  # question or question_tokens -> passage
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
        # Pop the question and answers.
        question = item.pop("question")
        if question in canonical_question_info:
            question = canonical_question_info[question]

        answers = item.pop("answers")

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

        # Create `RetrievedPassage` objects from the passages.
        passages = [RetrievedPassage(**ctx) for ctx in item.pop("ctxs")]
        gold_passage = gold_passage_info.get(question)  # not necessarily included in the retrieved passages

        # Make lists of positive/negative passage indices.
        if training:
            if self.hparams.train_max_load_passages is not None:
                passages = passages[: self.hparams.train_max_load_passages]

            positive_passage_idxs = [idx for idx, p in enumerate(passages) if p.has_answer and has_any_answer_span(p)]
            negative_passage_idxs = [idx for idx, p in enumerate(passages) if not p.has_answer]

            # Get the gold passage indices if they exist.
            if gold_passage is not None:
                gold_positive_passage_idxs = [
                    idx for idx in positive_passage_idxs if passages[idx].title.lower() == gold_passage.title.lower()
                ]
            else:
                gold_positive_passage_idxs = []

            # If specified so and possible, take the gold passages as the positive passages.
            if gold_page_only_positives and len(gold_positive_passage_idxs) > 0:
                positive_passage_idxs = gold_positive_passage_idxs
            else:
                if self.hparams.train_max_load_positive_passages is not None:
                    positive_passage_idxs = positive_passage_idxs[: self.hparams.train_max_load_positive_passages]

            # If specified so and possible, include the question's gold passage in the positive passages.
            if include_gold_passage and gold_passage is not None:
                if not any(passages[i].id == gold_passage.id for i in positive_passage_idxs):
                    if has_any_answer_span(gold_passage):
                        passages.append(gold_passage)
                        positive_passage_idxs.append(len(passages) - 1)

            if self.hparams.train_max_load_negative_passages is not None:
                negative_passage_idxs = negative_passage_idxs[: self.hparams.train_max_load_negative_passages]

        else:
            if self.hparams.eval_max_load_passages is not None:
                passages = passages[: self.hparams.eval_max_load_passages]

            positive_passage_idxs = [i for i, p in enumerate(passages) if p.has_answer and has_any_answer_span(p)]
            negative_passage_idxs = [i for i, p in enumerate(passages) if not p.has_answer]

        return ReaderDatasetExample(index, question, answers, passages, positive_passage_idxs, negative_passage_idxs)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self._train_dataset, training=True)

    def val_dataloader(self) -> List[DataLoader]:
        return self._get_dataloader(self._val_dataset, training=False)

    def test_dataloader(self) -> List[DataLoader]:
        return self._get_dataloader(self._test_dataset, training=False)

    def _get_dataloader(self, dataset: Dataset, training: bool) -> DataLoader:
        if training:
            batch_size = self.hparams.train_batch_size
            shuffle = True
        else:
            batch_size = self.hparams.eval_batch_size
            shuffle = False

        # Get a batch collation function.
        collate_fn = self._get_collate_fn(training=training)

        # Initialize a `DataLoader` and return it.
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_dataloader_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _get_collate_fn(self, training: bool) -> Callable:
        def collate_fn(
            batch_examples: List[ReaderDatasetExample],
        ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor]:
            if training:
                num_passages = self.hparams.train_num_passages
                shuffle_positive_passages = True
                shuffle_negative_passages = True
            else:
                num_passages = self.hparams.eval_num_passages
                shuffle_positive_passages = False
                shuffle_negative_passages = False

            num_questions = len(batch_examples)

            tokenized_inputs = []

            start_positions = []
            end_positions = []
            answer_mask = []

            passage_idxs = []
            example_idxs = []

            for example in batch_examples:
                example_passage_idxs = []

                if training:
                    # Process the positive passage.
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
                    answer_spans = answer_spans[: self.hparams.max_answer_spans]
                    answers_padding_length = self.hparams.max_answer_spans - len(answer_spans)
                    answer_starts = [span[0] for span in answer_spans] + [0] * answers_padding_length
                    answer_ends = [span[1] for span in answer_spans] + [0] * answers_padding_length
                    answer_masks = [1] * len(answer_spans) + [0] * answers_padding_length

                    tokenized_inputs.append(positive_input)
                    example_passage_idxs.append(positive_passage_idx)

                    start_positions.append(answer_starts)
                    end_positions.append(answer_ends)
                    answer_mask.append(answer_masks)

                    # Process the negative passages.
                    negative_passage_idxs = example.negative_passage_idxs
                    if shuffle_negative_passages:
                        random.shuffle(negative_passage_idxs)

                    max_negative_passages_per_question = num_passages - len(example_passage_idxs)
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

                        tokenized_inputs.append(negative_input)
                        example_passage_idxs.append(negative_passage_idx)

                else:
                    # In evaluation mode, there are no distinction in positive/negative passages.
                    passages = example.passages[:num_passages]
                    for passage_idx, passage in enumerate(passages):
                        passage_input = self.tokenization.tokenize_input(
                            example.question,
                            passage.title,
                            passage.text,
                            padding="max_length",
                            truncation="only_second",
                            max_length=self.hparams.max_input_length,
                        )

                        tokenized_inputs.append(passage_input)
                        example_passage_idxs.append(passage_idx)

                # Supply empty passages if there are not enough passages.
                while len(example_passage_idxs) < num_passages:
                    dummy_passage_input = self.tokenization.tokenize_input(
                        "",
                        "",
                        "",
                        padding="max_length",
                        truncation="only_second",
                        max_length=self.hparams.max_input_length,
                    )

                    tokenized_inputs.append(dummy_passage_input)
                    example_passage_idxs.append(-1)

                passage_idxs.append(example_passage_idxs)
                example_idxs.append(example.index)

            # Tensorize the lists of integers.
            tokenized_inputs = {
                key: torch.tensor([x[key] for x in tokenized_inputs]) for key in tokenized_inputs[0].keys()
            }
            passage_idxs = torch.tensor(passage_idxs)
            example_idxs = torch.tensor(example_idxs)

            # Check the shapes of the tensors.
            for tensor in tokenized_inputs.values():
                assert tensor.size() == (num_questions * num_passages, self.max_input_length)

            assert passage_idxs.size() == (num_questions, num_passages)
            assert example_idxs.size() == (num_questions,)

            if training:
                start_positions = torch.tensor(start_positions)
                end_positions = torch.tensor(end_positions)
                answer_mask = torch.tensor(answer_mask)

                assert start_positions.size() == (num_questions, self.num_answers)
                assert end_positions.size() == (num_questions, self.num_answers)
                assert answer_mask.size() == (num_questions, self.num_answers)
            else:
                start_positions = None
                end_positions = None
                answer_mask = None

            return (tokenized_inputs, passage_idxs, example_idxs, start_positions, end_positions, answer_mask)

        return collate_fn

    def forward(self, tokenized_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        batch_size = tokenized_inputs["input_ids"].size(0)

        # Apply the reader's inference to the input.
        classifier_logits, start_logits, end_logits = self.reader(tokenized_inputs)

        # Check the shapes of the output tensors.
        assert classifier_logits.size() == (batch_size,)
        assert start_logits.size() == (batch_size, self.max_input_length)
        assert end_logits.size() == (batch_size, self.max_input_length)

        return classifier_logits, start_logits, end_logits

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        # Unpack the tuple of the batch tensors.
        tokenized_inputs, passage_idxs, _, start_positions, end_positions, answer_mask = batch

        num_questions = tokenized_inputs["input_ids"].size(0) // self.train_num_passages

        # Check the shapes of the input tensors.
        assert passage_idxs.size() == (num_questions, self.train_num_passages)
        assert start_positions.size() == (num_questions, self.num_answers)
        assert end_positions.size() == (num_questions, self.num_answers)
        assert answer_mask.size() == (num_questions, self.num_answers)

        # Apply the reader's inference to the input.
        classifier_logits, start_logits, end_logits = self.forward(tokenized_inputs)

        # Check the shapes of the reader's output tensors.
        assert classifier_logits.size() == (num_questions * self.train_num_passages,)
        assert start_logits.size() == (num_questions * self.train_num_passages, self.max_input_length)
        assert end_logits.size() == (num_questions * self.train_num_passages, self.max_input_length)

        # Reshape the reader's output tensors and take the start/end logits of the positive passage.
        classifier_logits = classifier_logits.view(num_questions, self.train_num_passages)
        start_logits = start_logits.view(num_questions, self.train_num_passages, self.max_input_length)[:, 0, :]
        end_logits = end_logits.view(num_questions, self.train_num_passages, self.max_input_length)[:, 0, :]

        # Create labels of the positive passages, which are always indexed zero.
        classifier_label = classifier_logits.new_zeros(num_questions, dtype=torch.long)

        # `passage_mask` masks out the passage indices of dummy inputs.
        passage_mask = passage_idxs != -1

        # Compute the classifier loss.
        classifier_loss = self._compute_classifier_loss(
            classifier_logits=classifier_logits,
            classifier_label=classifier_label,
            passage_mask=passage_mask,
        )

        # Compute the answer span loss.
        answer_span_loss = self._compute_answer_span_loss(
            start_logits=start_logits,
            end_logits=end_logits,
            start_positions=start_positions,
            end_positions=end_positions,
            answer_mask=answer_mask,
        )

        loss = classifier_loss + answer_span_loss
        self.log("loss", loss)
        self.log("classifier_loss", classifier_loss.detach())  # detaching is necessary for logging
        self.log("answer_span_loss", answer_span_loss.detach())

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._eval_step(batch, batch_idx, testing=False)

    def validation_step_end(self, batch_parts_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._eval_step_end(batch_parts_outputs)

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        return self._eval_epoch_end(outputs, testing=False)

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._eval_step(batch, batch_idx, testing=True)

    def test_step_end(self, batch_parts_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self._eval_step_end(batch_parts_outputs)

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        return self._eval_epoch_end(outputs, testing=True)

    def _eval_step(
        self,
        batch: Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
        testing: bool = False,
    ) -> Dict[str, Tensor]:
        eval_dataset = self._test_dataset if testing else self._val_dataset

        # Unpack the tuple of the batch tensors.
        tokenized_inputs, passage_idxs, example_idxs, _, _, _ = batch

        num_questions = tokenized_inputs["input_ids"].size(0) // self.eval_num_passages

        # Check the shapes of the input tensors.
        assert passage_idxs.size() == (num_questions, self.eval_num_passages)
        assert example_idxs.size() == (num_questions,)

        # Apply the reader's inference to the input.
        classifier_logits, start_logits, end_logits = self.forward(tokenized_inputs)

        # Check the shapes of the reader's output tensors.
        assert classifier_logits.size() == (num_questions * self.eval_num_passages,)
        assert start_logits.size() == (num_questions * self.eval_num_passages, self.max_input_length)
        assert end_logits.size() == (num_questions * self.eval_num_passages, self.max_input_length)

        # Reshape the reader's output tensors.
        input_ids = tokenized_inputs["input_ids"].view(num_questions, self.eval_num_passages, self.max_input_length)
        classifier_logits = classifier_logits.view(num_questions, self.eval_num_passages)
        start_logits = start_logits.view(num_questions, self.eval_num_passages, self.max_input_length)
        end_logits = end_logits.view(num_questions, self.eval_num_passages, self.max_input_length)

        # Compute the classifier precision.
        classifier_precision = self._compute_classifier_precision(
            classifier_logits=classifier_logits,
            passage_idxs=passage_idxs,
            example_idxs=example_idxs,
            examples=eval_dataset,
        )
        assert classifier_precision.size() == (num_questions,)

        # Compute the answer accuracy.
        answer_accuracy = self._compute_answer_accuracy(
            input_ids=input_ids,
            classifier_logits=classifier_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            example_idxs=example_idxs,
            examples=eval_dataset,
        )
        assert answer_accuracy.size() == (num_questions,)

        output = {
            "classifier_precision": classifier_precision,
            "answer_accuracy": answer_accuracy,
            "example_idxs": example_idxs,
        }

        return output

    def _eval_step_end(self, batch_parts_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return batch_parts_outputs

    def _eval_epoch_end(self, outputs: List[Dict[str, Tensor]], testing: bool = False) -> None:
        prefix = "test" if testing else "val"

        # Accumurate the metrics.
        classifier_precision = torch.cat([x["classifier_precision"] for x in outputs])
        answer_accuracy = torch.cat([x["answer_accuracy"] for x in outputs])
        example_idxs = torch.cat([x["example_idxs"] for x in outputs])

        # Gather tensors distributed to multiple devices.
        if distributed_available():
            # Note that the gathered tensors may contain duplicated data points when the number of examples in
            # the evaluation dataset is not equally divisible by batch_size * num_processors.
            # For example, when the dataset size is 50 and the number of GPUs is 4, the example indices should
            # have been arranged in the distributed tensors by PyTorch's `DistributedSampler` as follows:
            #     (gpu 0): [0, 4, 8, ..., 48]
            #     (gpu 1): [1, 5, 9, ..., 49]
            #     (gpu 2): [2, 6, 10, ..., 0]
            #     (gpu 3): [3, 7, 11, ..., 1]
            classifier_precision = self.all_gather(classifier_precision)
            answer_accuracy = self.all_gather(answer_accuracy)
            example_idxs = self.all_gather(example_idxs)

            # Reshape the tensors so that the elements of the tensors are sorted by the example indices.
            classifier_precision = classifier_precision.transpose(0, 1).reshape(-1)
            answer_accuracy = answer_accuracy.transpose(0, 1).reshape(-1)
            example_idxs = example_idxs.transpose(0, 1).reshape(-1)

        # Get the number of examples in the evaluation dataset.
        dataset_size = len(self._test_dataset) if testing else len(self._val_dataset)
        if self.trainer.sanity_checking:
            # Avoid possible errors when conducting the PyTorch Lightning's sanity check where
            # the number of examples in the batches is much smaller than the dataset size.
            dataset_size = example_idxs.numel()

        # Slice the tensors to make sure each tensor contains `dataset_size` elements without dupricated data points.
        classifier_precision = classifier_precision[:dataset_size]
        answer_accuracy = answer_accuracy[:dataset_size]
        example_idxs = example_idxs[:dataset_size]
        assert all(example_idxs == torch.arange(dataset_size).to(example_idxs.device))

        # Check the shapes of the gathered tensors.
        assert classifier_precision.size() == (dataset_size,)
        assert answer_accuracy.size() == (dataset_size,)
        assert example_idxs.size() == (dataset_size,)

        # Aggregate and log the metrics.
        classifier_precision = classifier_precision.sum() / dataset_size
        answer_accuracy = answer_accuracy.sum() / dataset_size

        self.log(f"{prefix}_classifier_precision", classifier_precision)
        self.log(f"{prefix}_answer_accuracy", answer_accuracy)

    def _compute_classifier_loss(
        self,
        classifier_logits: Tensor,
        classifier_label: Tensor,
        passage_mask: Tensor,
    ) -> Tensor:
        num_questions = classifier_logits.size(0)
        num_passages = self.train_num_passages if self.training else self.eval_num_passages

        # Check the shapes of the input tensors.
        assert classifier_logits.size() == (num_questions, num_passages)
        assert classifier_label.size() == (num_questions,)
        assert passage_mask.size() == (num_questions, num_passages)

        # Mask the classifier logits with a large negative value where the passage is empty.
        mask_value = -1e4 if classifier_logits.dtype == torch.float16 else -1e8
        classifier_logits = classifier_logits.masked_fill(~passage_mask, mask_value)

        # Compute the classifier loss.
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
        num_questions = start_logits.size(0)

        # Check the shapes of the input tensors.
        assert start_logits.size() == (num_questions, self.max_input_length)
        assert end_logits.size() == (num_questions, self.max_input_length)
        assert start_positions.size() == (num_questions, self.num_answers)
        assert end_positions.size() == (num_questions, self.num_answers)
        assert answer_mask.size() == (num_questions, self.num_answers)

        # Compute the probability distributions of the predicted start/end positions.
        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)
        assert start_probs.size() == (num_questions, self.max_input_length)
        assert end_probs.size() == (num_questions, self.max_input_length)

        # Aggregate the probabilities of the labeled start/end positions to compute the span probabilities.
        start_labeled_position_probs = start_probs.gather(dim=1, index=start_positions) * answer_mask.float()
        end_labeled_position_probs = end_probs.gather(dim=1, index=end_positions) * answer_mask.float()
        labeled_span_probs = start_labeled_position_probs * end_labeled_position_probs
        assert labeled_span_probs.size() == (num_questions, self.num_answers)

        # Compute the answer span loss.
        log_sum_labeled_span_probs = labeled_span_probs.sum(dim=1).log()
        assert log_sum_labeled_span_probs.size() == (num_questions,)

        span_loss = -log_sum_labeled_span_probs.sum()

        return span_loss

    def _compute_classifier_precision(
        self,
        classifier_logits: Tensor,
        passage_idxs: Tensor,
        example_idxs: Tensor,
        examples: List[ReaderDatasetExample],
    ) -> Tensor:
        num_questions = classifier_logits.size(0)
        num_passages = self.train_num_passages if self.training else self.eval_num_passages

        # Check the shapes of the input tensors.
        assert classifier_logits.size() == (num_questions, num_passages)
        assert passage_idxs.size() == (num_questions, num_passages)
        assert example_idxs.size() == (num_questions,)

        # Compute the classifier's precision.
        classifier_precision = classifier_logits.new_zeros(num_questions)
        for qi, pi in enumerate(classifier_logits.argmax(dim=1).tolist()):
            example = examples[example_idxs[qi]]
            pred_passage_idx = passage_idxs[qi, pi]

            # A predicted passage is regarded correct if it is one of the example's postive passages.
            if pred_passage_idx in example.positive_passage_idxs:
                classifier_precision[qi] = 1.0

        assert classifier_precision.size() == (num_questions,)

        return classifier_precision

    def _compute_answer_accuracy(
        self,
        input_ids: Tensor,
        classifier_logits: Tensor,
        start_logits: Tensor,
        end_logits: Tensor,
        example_idxs: Tensor,
        examples: List[ReaderDatasetExample],
    ) -> Tensor:
        num_questions = classifier_logits.size(0)
        num_passages = self.train_num_passages if self.training else self.eval_num_passages

        # Check the shapes of the input tensors.
        assert input_ids.size() == (num_questions, num_passages, self.max_input_length)
        assert classifier_logits.size() == (num_questions, num_passages)
        assert start_logits.size() == (num_questions, num_passages, self.max_input_length)
        assert end_logits.size() == (num_questions, num_passages, self.max_input_length)
        assert example_idxs.size() == (num_questions,)

        answer_precision = classifier_logits.new_zeros(num_questions)
        for qi in range(num_questions):
            # Take the question's gold answers
            example = examples[example_idxs[qi]]
            gold_answers = [self.tokenization._normalize_answer(a) for a in example.answers]

            # Generate the question's answer candidates
            answer_candidates = self.generate_answer_candidates(
                input_ids=input_ids[qi],
                classifier_logits=classifier_logits[qi],
                start_logits=start_logits[qi],
                end_logits=end_logits[qi],
                num_passages_to_read=1,
                num_answer_candidates_per_passage=1,
                max_answer_length=self.hparams.max_answer_length,
            )
            best_answer_candidate = answer_candidates[0]
            pred_answer = self.tokenization._normalize_answer(best_answer_candidate.answer_text)

            # A predicted answer is regarded correct if its normalized form matches to any of
            # the normalized form of the gold answers.
            if pred_answer in gold_answers:
                answer_precision[qi] = 1.0

        return answer_precision

    def generate_answer_candidates(
        self,
        input_ids: Tensor,
        classifier_logits: Tensor,
        start_logits: Tensor,
        end_logits: Tensor,
        num_passages_to_read: int = 1,
        num_answer_candidates_per_passage: int = 1,
        max_answer_length: int = 10,
    ) -> List[AnswerCandidate]:
        num_passages, max_input_length = input_ids.size()

        # Check the shapes of the input tensors.
        # Here we do not use `self.max_input_length` since this function can be called outside the trainier.
        assert input_ids.size() == (num_passages, max_input_length)
        assert classifier_logits.size() == (num_passages,)
        assert start_logits.size() == (num_passages, max_input_length)
        assert end_logits.size() == (num_passages, max_input_length)

        # Obtain `num_passages_to_read` passage indices with highest classifier logits.
        top_passage_idxs = classifier_logits.argsort(descending=True)[:num_passages_to_read]
        assert top_passage_idxs.size() == (min(num_passages, num_passages_to_read),)

        # Generate answer candidates for each passage of the question.
        answer_candidates = []
        for pi in top_passage_idxs.tolist():
            answer_spans = self.tokenization._compute_best_answer_spans(
                input_ids=input_ids[pi].tolist(),
                start_logits=start_logits[pi].tolist(),
                end_logits=end_logits[pi].tolist(),
                num_answer_spans=num_answer_candidates_per_passage,
                max_answer_length=max_answer_length,
            )
            for answer_span in answer_spans:
                input_text, answer_text, answer_text_span = self.tokenization._get_input_and_answer_texts(
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

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        # Set up the optimizer.
        optimizer_parameters = [
            {"params": [], "weight_decay": 0.0},
            {"params": [], "weight_decay": self.hparams.weight_decay},
        ]
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            elif "bias" in name or "LayerNorm.weight" in name:
                optimizer_parameters[0]["params"].append(param)
            else:
                optimizer_parameters[1]["params"].append(param)

        optimizer = AdamW(optimizer_parameters, lr=self.hparams.learning_rate)

        # Set up the learning rate scheduler.
        num_training_steps = int(
            len(self.train_dataloader().batch_sampler)
            // self.trainer.accumulate_grad_batches
            * float(self.hparams.max_epochs)
        )
        num_warmup_steps = int(self.hparams.warmup_proportion * num_training_steps)
        rank_zero_info("The total number of training steps: %d", num_training_steps)
        rank_zero_info("The total number of warmup steps: %d", num_warmup_steps)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
