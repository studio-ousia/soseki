import math
import random
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler
from tqdm import tqdm
from pytorch_lightning.core import LightningModule
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from transformers import AutoModel
from transformers.optimization import get_linear_schedule_with_warmup

from .tokenization import EncoderTokenization
from ..passage_db.lmdb_passage_db import LMDBPassageDB
from ..utils.data_utils import Passage, RetrieverDatasetExample, readitem_json


class MultipleKeyDataset(Dataset):
    dataset_dict: Dict[str, Dataset]
    dataset_keys: List[str]
    dataset_sizes: Dict[str, int]
    total_dataset_size: int

    def __init__(self, dataset_dict: Dict[str, Dataset]) -> None:
        super(MultipleKeyDataset, self).__init__()
        self.dataset_dict = dataset_dict
        self.dataset_keys = list(dataset_dict.keys())
        self.dataset_sizes = {key: len(dataset) for key, dataset in dataset_dict.items()}
        self.total_dataset_size = sum(self.dataset_sizes.values())

    def __len__(self) -> int:
        return self.total_dataset_size

    def __getitem__(self, key_idx_tuple: Tuple[str, int]) -> Any:
        dataset_key, sample_idx = key_idx_tuple
        return self.dataset_dict[dataset_key][sample_idx]


class MultipleKeyDatasetBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        multiple_key_dataset: MultipleKeyDataset,
        weights: Optional[Dict[str, float]] = None,
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        if distributed:
            if num_replicas is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")

                num_replicas = dist.get_world_size()

            if rank is None:
                if not dist.is_available():
                    raise RuntimeError("Requires distributed package to be available")

                rank = dist.get_rank()

            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1)
                )
        else:
            num_replicas = 1
            rank = 0

        if weights is None:
            weights = dict()

        for key in multiple_key_dataset.dataset_dict.keys():
            weights.setdefault(key, 1.0)

        self.multiple_key_dataset = multiple_key_dataset
        self.weights = weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.distributed = distributed
        self.num_replicas = num_replicas
        self.rank = rank

        self.chunk_size = self.batch_size * self.num_replicas
        self.num_chunks = sum(
            int(len(dataset) * self.weights[key]) // self.chunk_size
            for key, dataset in self.multiple_key_dataset.dataset_dict.items()
        )
        self.total_size = self.num_chunks * self.chunk_size
        self.epoch = 0

    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
        else:
            generator = None

        chunks: List[List[int]] = []
        for key in self.multiple_key_dataset.dataset_keys:
            weight = self.weights[key]
            dataset_size = self.multiple_key_dataset.dataset_sizes[key]

            sample_idxs = []
            for _ in range(int(weight) + 1):
                if self.shuffle:
                    sample_idxs.extend(torch.randperm(dataset_size, generator=generator).tolist())
                else:
                    sample_idxs.extend(list(range(dataset_size)))

            sample_idxs = sample_idxs[: int(dataset_size * weight)]

            for i in range(0, len(sample_idxs), self.chunk_size):
                chunk = [(key, idx) for idx in sample_idxs[i : i + self.chunk_size]]
                if len(chunk) < self.chunk_size:
                    break

                chunks.append(chunk)

        if self.shuffle:
            chunk_idxs = torch.randperm(len(chunks), generator=generator).tolist()
        else:
            chunk_idxs = list(range(len(chunks)))

        for chunk_idx in chunk_idxs:
            chunk = chunks[chunk_idx]
            batch = chunk[self.rank : self.chunk_size : self.num_replicas]
            assert len(batch) == self.batch_size
            yield batch

    def __len__(self) -> int:
        return self.num_chunks

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class EncoderModel(nn.Module):
    def __init__(self, base_model_name: str, pooling_index: int = 0, projection_dim: Optional[int] = None) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, add_pooling_layer=False, return_dict=False)
        self.pooling_index = pooling_index

        if projection_dim is not None:
            self.projection_dense = nn.Linear(self.encoder.config.hidden_size, projection_dim)
            self.projection_layer_norm = torch.nn.LayerNorm(projection_dim, eps=self.encoder.config.layer_norm_eps)
            self.output_dim = projection_dim
        else:
            self.output_dim = self.encoder.config.hidden_size

    def forward(self, input_tensors: Dict[str, Tensor]) -> Tensor:
        encoder_output = self.encoder(**input_tensors)
        sequence_output = encoder_output[0]

        pooled_output = sequence_output[:, self.pooling_index, :].contiguous()

        if hasattr(self, "projection_dense"):
            pooled_output = self.projection_dense(pooled_output)
        if hasattr(self, "projection_layer_norm"):
            pooled_output = self.projection_layer_norm(pooled_output)

        return pooled_output


class BiencoderLightningModule(LightningModule):
    def __init__(self, hparams: Optional[Namespace] = None, **kwargs) -> None:
        super().__init__()
        # Set arguments to `hparams` attribute.
        self.save_hyperparameters(hparams)

        # Initialize the question encoder.
        self.question_encoder = EncoderModel(
            self.hparams.base_pretrained_model,
            pooling_index=self.hparams.encoder_pooling_index,
            projection_dim=self.hparams.encoder_projection_dim,
        )

        # Initialize the passage encoder.
        if self.hparams.share_encoders:
            self.passage_encoder = self.question_encoder
        else:
            self.passage_encoder = EncoderModel(
                self.hparams.base_pretrained_model,
                pooling_index=self.hparams.encoder_pooling_index,
                projection_dim=self.hparams.encoder_projection_dim,
            )

        # Initialize the tokenizer.
        self.tokenization = EncoderTokenization(self.hparams.base_pretrained_model)

        # Set attributes used for checking tensor shapes.
        self.num_passages = 1 + self.hparams.num_negative_passages
        self.max_question_length = self.hparams.max_question_length
        self.max_passage_length = self.hparams.max_passage_length
        self.embed_size = self.question_encoder.output_dim

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Biencoder")

        parser.add_argument("--train_file", type=str, required=True)
        parser.add_argument("--val_file", type=str, required=True)
        parser.add_argument("--passage_db_file", type=str)

        parser.add_argument("--train_use_multiple_key_dataset", action="store_true")
        parser.add_argument("--train_dataset_keys", nargs="*", type=str)
        parser.add_argument("--train_dataset_weights", nargs="*", type=float)
        parser.add_argument("--val_use_multiple_key_dataset", action="store_true")
        parser.add_argument("--val_dataset_keys", nargs="*", type=str)
        parser.add_argument("--val_dataset_weights", nargs="*", type=float)

        parser.add_argument("--train_max_load_positive_passages", type=int)
        parser.add_argument("--train_max_load_hard_negative_passages", type=int)
        parser.add_argument("--train_max_load_normal_negative_passages", type=int)
        parser.add_argument("--eval_max_load_positive_passages", type=int)
        parser.add_argument("--eval_max_load_hard_negative_passages", type=int)
        parser.add_argument("--eval_max_load_normal_negative_passages", type=int)

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
        parser.add_argument("--share_encoders", action="store_true")

        parser.add_argument("--binary", action="store_true")
        parser.add_argument("--use_ste", action="store_true")
        parser.add_argument("--hashnet_gamma", type=float, default=0.1)
        parser.add_argument("--cand_loss_use_cross_entropy", action="store_true")
        parser.add_argument("--cand_loss_margin", type=float, default=0.1)
        parser.add_argument("--no_cand_loss", action="store_true")
        parser.add_argument("--no_rerank_loss", action="store_true")

        parser.add_argument("--train_batch_size", type=int, default=16)
        parser.add_argument("--eval_batch_size", type=int, default=16)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--warmup_proportion", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--eval_rank_start_epoch", type=int, default=0)

        # Temporary params.
        parser.add_argument("--use_pl_all_gather", action="store_true")

        return parent_parser

    def setup(self, stage: str) -> None:
        # Load the training dataset.
        if stage == "fit":
            rank_zero_info("Loading the training dataset")
            self._train_dataset = self._load_dataset(
                dataset_file=self.hparams.train_file,
                use_multiple_key_dataset=self.hparams.train_use_multiple_key_dataset,
                dataset_keys=self.hparams.train_dataset_keys,
                training=True,
            )
            rank_zero_info(f"The number of examples in training data: {len(self._train_dataset)}")

        # Load the validation dataset.
        if stage in ("fit", "validate"):
            rank_zero_info("Loading the validation dataset")
            self._val_dataset = self._load_dataset(
                dataset_file=self.hparams.val_file,
                use_multiple_key_dataset=self.hparams.val_use_multiple_key_dataset,
                dataset_keys=self.hparams.val_dataset_keys,
                training=False,
            )
            rank_zero_info(f"The number of examples in validation data: {len(self._val_dataset)}")

    def _load_dataset(
        self,
        dataset_file: str,
        use_multiple_key_dataset: bool,
        dataset_keys: List[str],
        training: bool,
    ) -> Dataset:
        # Initialize a dataset iterator for reading a JSON (JSON Lines) file.
        dataset_iterator = readitem_json(dataset_file)
        if self.global_rank == 0:
            dataset_iterator = tqdm(dataset_iterator)

        if use_multiple_key_dataset:
            dataset_dict = defaultdict(list)
        else:
            dataset = []

        # Load the dataset.
        index = 0
        for item in dataset_iterator:
            example = self._create_dataset_example(index, item, training=training)

            # Skip items with no postive passages.
            if len(example.positive_passages) == 0:
                continue
            # Skip items with no negative passages.
            if len(example.hard_negative_passages) + len(example.normal_negative_passages) == 0:
                continue

            if use_multiple_key_dataset:
                if dataset_keys is not None and example.dataset not in dataset_keys:
                    continue

                dataset_dict[example.dataset].append(example)
            else:
                dataset.append(example)

            index += 1

        if use_multiple_key_dataset:
            dataset = MultipleKeyDataset(dataset_dict)

        return dataset

    def _create_dataset_example(self, index: int, item: Dict[str, Any], training: bool) -> RetrieverDatasetExample:
        if training:
            max_load_positive_passages = self.hparams.train_max_load_positive_passages
            max_load_hard_negative_passages = self.hparams.train_max_load_hard_negative_passages
            max_load_normal_negative_passages = self.hparams.train_max_load_normal_negative_passages
        else:
            max_load_positive_passages = self.hparams.eval_max_load_positive_passages
            max_load_hard_negative_passages = self.hparams.eval_max_load_hard_negative_passages
            max_load_normal_negative_passages = self.hparams.eval_max_load_normal_negative_passages

        # Pop the question and answers.
        question = item.pop("question")
        answers = item.pop("answers")
        dataset_name = item.pop("dataset", "")  # e.g., "nq_train_psgs_w100" in the DPR's nq-train.json file

        # Create `Passage` objects from the positive/negative passages.
        positive_passages = [
            Passage(id=ctx.get("passage_id"), title=ctx["title"], text=ctx["text"]) for ctx in item.pop("positive_ctxs")
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
            Passage(id=ctx.get("passage_id"), title=ctx["title"], text=ctx["text"]) for ctx in item.pop("negative_ctxs")
        ]
        if max_load_normal_negative_passages is not None:
            normal_negative_passages = normal_negative_passages[:max_load_normal_negative_passages]

        # Metadata is the original JSON object with the popped items removed.
        metadata = item

        # Create a `RetrieverDatasetExample` object.
        example = RetrieverDatasetExample(
            index=index,
            dataset=dataset_name,
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
            use_multiple_key_dataset=self.hparams.train_use_multiple_key_dataset,
            dataset_keys=self.hparams.train_dataset_keys,
            dataset_weights=self.hparams.train_dataset_weights,
            training=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(
            dataset=self._val_dataset,
            use_multiple_key_dataset=self.hparams.val_use_multiple_key_dataset,
            dataset_keys=self.hparams.val_dataset_keys,
            dataset_weights=self.hparams.val_dataset_weights,
            training=False,
        )

    def _get_dataloader(
        self,
        dataset: Dataset,
        use_multiple_key_dataset: bool,
        dataset_keys: List[str],
        dataset_weights: List[float],
        training: bool,
    ) -> DataLoader:
        if training:
            batch_size = self.hparams.train_batch_size
            shuffle = True
        else:
            batch_size = self.hparams.eval_batch_size
            shuffle = False

        # Get a batch collation function.
        collate_fn = self._get_collate_fn(training)

        # Initialize a `DataLoader` and return it.
        if use_multiple_key_dataset:
            if dataset_keys is not None and dataset_weights is not None:
                weights = {key: weight for key, weight in zip(dataset_keys, dataset_weights)}
            else:
                weights = None

            batch_sampler = MultipleKeyDatasetBatchSampler(
                dataset,
                weights=weights,
                batch_size=batch_size,
                shuffle=shuffle,
                distributed=distributed_available(),
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_dataloader_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
        else:
            if distributed_available():
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    sampler=DistributedSampler(dataset, shuffle=shuffle),
                    num_workers=self.hparams.num_dataloader_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )
            else:
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=self.hparams.num_dataloader_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                )

    def _get_collate_fn(self, training: bool) -> Callable:
        def collate_fn(batch_examples: List[RetrieverDatasetExample]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            # The passage DB is used when the loaded dataset does not have the passage text fields.
            if self.hparams.passage_db_file is not None:
                passage_db = LMDBPassageDB(self.hparams.passage_db_file)
            else:
                passage_db = None

            def _get_passage_title(passage: Passage) -> str:
                if passage.title is not None:
                    return passage.title
                elif passage_db is not None:
                    # Fetch the passage title from the passage DB.
                    return passage_db[passage.id].title
                else:
                    raise KeyError("--passage_db_file must be specified if the dataset has no passage title fields")

            def _get_passage_text(passage: Passage) -> str:
                if passage.text is not None:
                    return passage.text
                elif passage_db is not None:
                    # Fetch the passage text from the passage DB.
                    return passage_db[passage.id].text
                else:
                    raise KeyError("--passage_db_file must be specified if the dataset has no passage text fields")

            num_questions = len(batch_examples)

            tokenized_questions = []
            tokenized_passages = []

            for example in batch_examples:
                # Tokenize the question.
                tokenized_question = self.tokenization.tokenize_questions(
                    example.question,
                    padding="max_length",
                    truncation=True,
                    max_length=self.hparams.max_question_length,
                )
                tokenized_questions.append(tokenized_question)

                # Select a positive passage.
                if training and self.hparams.shuffle_positive_passages:
                    positive_passage = random.choice(example.positive_passages)
                else:
                    positive_passage = example.positive_passages[0]

                # Select hard negative passages.
                hard_negative_passages = example.hard_negative_passages
                if training and self.hparams.shuffle_hard_negative_passages:
                    random.shuffle(hard_negative_passages)

                if self.hparams.max_hard_negative_passages is not None:
                    hard_negative_passages = hard_negative_passages[: self.hparams.max_hard_negative_passages]

                # Select normal negative passages.
                normal_negative_passages = example.normal_negative_passages
                if training and self.hparams.shuffle_normal_negative_passages:
                    random.shuffle(normal_negative_passages)

                if self.hparams.max_normal_negative_passages is not None:
                    normal_negative_passages = normal_negative_passages[: self.hparams.max_normal_negative_passages]

                # Concatenate the negative passages; hard negative passages are prioritized.
                negative_passages = hard_negative_passages + normal_negative_passages
                negative_passages = negative_passages[: self.hparams.num_negative_passages]
                if len(negative_passages) < self.hparams.num_negative_passages:
                    raise ValueError("--num_negative_passages is larger than retrieved negative passages")

                # Tokenize the passages.
                for passage in [positive_passage] + negative_passages:
                    tokenized_passage = self.tokenization.tokenize_passages(
                        _get_passage_title(passage),
                        _get_passage_text(passage),
                        padding="max_length",
                        truncation="only_second",
                        max_length=self.hparams.max_passage_length,
                    )
                    tokenized_passages.append(tokenized_passage)

            # Tensorize the lists of integers.
            tokenized_questions = {
                key: torch.tensor([q[key] for q in tokenized_questions]) for key in tokenized_questions[0].keys()
            }
            tokenized_passages = {
                key: torch.tensor([p[key] for p in tokenized_passages]) for key in tokenized_passages[0].keys()
            }

            # Check the shapes of the tensors.
            for tensor in tokenized_questions.values():
                assert tensor.size() == (num_questions, self.max_question_length)

            for tensor in tokenized_passages.values():
                assert tensor.size() == (num_questions * self.num_passages, self.max_passage_length)

            return tokenized_questions, tokenized_passages

        return collate_fn

    def forward(
        self,
        tokenized_questions: Dict[str, Tensor],
        tokenized_passages: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        question_batch_size = tokenized_questions["input_ids"].size(0)
        passage_batch_size = tokenized_passages["input_ids"].size(0)

        # Encode the questions and passages into tensors.
        encoded_question = self.question_encoder(tokenized_questions)
        encoded_passage = self.passage_encoder(tokenized_passages)

        # Check the shapes of the output tensors.
        assert encoded_question.size() == (question_batch_size, self.embed_size)
        assert encoded_passage.size() == (passage_batch_size, self.embed_size)

        return encoded_question, encoded_passage

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int,
    ) -> Tuple[Tensor, Tensor]:
        # Unpack the tuple of the batch tensors.
        tokenized_questions, tokenized_passages = batch

        num_questions = tokenized_questions["input_ids"].size(0)

        # Encode the questions and passages into tensors.
        encoded_question, encoded_passage = self.forward(tokenized_questions, tokenized_passages)

        # Check the shapes of the output tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        return encoded_question, encoded_passage

    def training_step_end(self, batch_parts_outputs: Tuple[Tensor, Tensor]) -> Tensor:
        # Unpack the tuple of the batch tensors.
        encoded_question, encoded_passage = batch_parts_outputs

        # Gather tensors distributed to multiple devices.
        if distributed_available():
            if self.hparams.use_pl_all_gather:
                encoded_question = self.all_gather(encoded_question, sync_grads=True)
                encoded_question = encoded_question.view(-1, self.question_encoder.output_dim)
                encoded_passage = self.all_gather(encoded_passage, sync_grads=True)
                encoded_passage = encoded_passage.view(-1, self.passage_encoder.output_dim)
            else:
                encoded_question = self._gather_distributed_tensors(encoded_question)
                encoded_passage = self._gather_distributed_tensors(encoded_passage)

        num_questions = encoded_question.size(0)

        # Check the shapes of the gathered tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        # Compute the losses.
        loss, cand_loss, rerank_loss = self._compute_loss(encoded_question, encoded_passage)

        self.log("loss", loss.detach())
        if cand_loss is not None:
            self.log("cand_loss", cand_loss.detach())
        if rerank_loss is not None:
            self.log("rerank_loss", rerank_loss.detach())

        return loss

    def validation_step(
        self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int
    ) -> Tuple[Tensor, Tensor]:
        # Unpack the tuple of the batch tensors.
        tokenized_questions, tokenized_passages = batch

        num_questions = tokenized_questions["input_ids"].size(0)

        # Encode the questions and passages into tensors.
        encoded_question, encoded_passage = self.forward(tokenized_questions, tokenized_passages)

        # Check the shapes of the output tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        return encoded_question, encoded_passage

    def validation_step_end(self, batch_parts_outputs: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        # Unpack the tuple of the batch tensors.
        encoded_question, encoded_passage = batch_parts_outputs

        # Gather tensors distributed to multiple devices.
        if distributed_available():
            if self.hparams.use_pl_all_gather:
                encoded_question = self.all_gather(encoded_question)
                encoded_question = encoded_question.view(-1, self.question_encoder.output_dim)
                encoded_passage = self.all_gather(encoded_passage)
                encoded_passage = encoded_passage.view(-1, self.passage_encoder.output_dim)
            else:
                encoded_question = self._gather_distributed_tensors(encoded_question)
                encoded_passage = self._gather_distributed_tensors(encoded_passage)

        num_questions = encoded_question.size(0)

        # Check the shapes of the gathered tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        # Compute the loss and ranks of predicted passages.
        loss, _, _ = self._compute_loss(encoded_question, encoded_passage)
        ranks = self._compute_ranks(encoded_question, encoded_passage)

        output = {"loss": loss, "ranks": ranks}

        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        # Compute the avarage of the loss.
        ave_loss = torch.stack([output["loss"] for output in outputs]).mean()
        self.log("val_loss", ave_loss)

        # Compute the avarage of the ranks of predicted passages.
        if self.current_epoch >= self.hparams.eval_rank_start_epoch:
            avg_rank = torch.cat([output["ranks"] for output in outputs], dim=0).mean()
        else:
            avg_rank = 10000000.0

        self.log("val_avg_rank", avg_rank)

    def _gather_distributed_tensors(self, input_tensor: Tensor) -> Tensor:
        tensor_list = [torch.empty_like(input_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, input_tensor)
        # overwrite a portion of the list with a gradient-preserved tensor
        tensor_list[dist.get_rank()] = input_tensor

        return torch.cat(tensor_list, dim=0)

    def _convert_to_binary_code(self, input_tensor: Tensor) -> Tensor:
        if self.training:
            if self.hparams.use_ste:
                # Use the hashing method of straight-through estimator (STE).
                hard_input_tensor = input_tensor.new_ones(input_tensor.size()).masked_fill_(input_tensor < 0, -1.0)
                input_tensor = torch.tanh(input_tensor)
                return hard_input_tensor + input_tensor - input_tensor.detach()
            else:
                # Use the hashing method of HashNet.
                # https://github.com/thuml/HashNet/blob/55bcaaa0bbaf0c404ca7a071b47d6287dc95e81d/pytorch/src/network.py#L40
                scale = math.pow((1.0 + self.global_step * self.hparams.hashnet_gamma), 0.5)
                return torch.tanh(input_tensor * scale)
        else:
            return input_tensor.new_ones(input_tensor.size()).masked_fill_(input_tensor < 0, -1.0)

    def _compute_loss(
        self, encoded_question: Tensor, encoded_passage: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        num_questions = encoded_question.size(0)

        # Check the shapes of the input tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        if self.hparams.binary:
            # Compute the BPR loss.
            cand_loss, rerank_loss = self._compute_bpr_loss(
                encoded_question,
                encoded_passage,
                no_cand_loss=self.hparams.no_cand_loss,
                no_rerank_loss=self.hparams.no_rerank_loss,
            )
            loss = sum(loss for loss in [cand_loss, rerank_loss] if loss is not None)
        else:
            # Compute the DPR loss.
            loss = self._compute_dpr_loss(encoded_question, encoded_passage)
            cand_loss = None
            rerank_loss = None

        return loss, cand_loss, rerank_loss

    def _compute_dpr_loss(self, encoded_question: Tensor, encoded_passage: Tensor) -> Tensor:
        num_questions = encoded_question.size(0)

        # Check the shapes of the input tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        # Compute the question-passsage similarity scores.
        scores = torch.matmul(encoded_question, encoded_passage.transpose(0, 1))
        assert scores.size() == (num_questions, num_questions * self.num_passages)

        # Create the labels of the positive passages for each of the questions.
        # The index of the positive passage for the i-th question is i * `num_passages`.
        # For example, when `num_questions` is 8 and `num_passages` is 2, `passage_labels` will be [0, 2, 4, ..., 14].
        passage_labels = torch.arange(0, num_questions * self.num_passages, self.num_passages).to(
            encoded_question.device
        )
        assert passage_labels.size() == (num_questions,)

        # Compute the cross entropy loss.
        loss = F.cross_entropy(scores, passage_labels)

        return loss

    def _compute_bpr_loss(
        self,
        encoded_question: Tensor,
        encoded_passage: Tensor,
        no_cand_loss: bool = False,
        no_rerank_loss: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        num_questions = encoded_question.size(0)

        # Check the shapes of the input tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        # Create the labels of the positive passages for each of the questions.
        # The index of the positive passage for the i-th question is i * `num_passages`.
        # For example, when `num_questions` is 8 and `num_passages` is 2, `passage_labels` will be [0, 2, 4, ..., 14].
        passage_labels = torch.arange(0, num_questions * self.num_passages, self.num_passages).to(
            encoded_question.device
        )
        assert passage_labels.size() == (num_questions,)

        # Binary-encode the questions and passages.
        binary_encoded_question = self._convert_to_binary_code(encoded_question)
        assert binary_encoded_question.size() == (num_questions, self.embed_size)
        binary_encoded_passage = self._convert_to_binary_code(encoded_passage)
        assert binary_encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        # Compute losses of two tasks (see the BPR paper for details).
        # Task #1: compute the candidate generation loss.
        if not no_cand_loss:
            binary_scores = torch.matmul(binary_encoded_question, binary_encoded_passage.transpose(0, 1))
            assert binary_scores.size() == (num_questions, num_questions * self.num_passages)

            if self.hparams.cand_loss_use_cross_entropy:
                # Compute the loss using a cross entropy loss
                cand_loss = F.cross_entropy(binary_scores, passage_labels)
            else:
                # Compute the loss using a ranking loss as described in the BPR paper
                positive_mask = binary_scores.new_zeros(binary_scores.size(), dtype=torch.bool)
                for i, label in enumerate(passage_labels):
                    positive_mask[i, label] = True

                positive_binary_scores = torch.masked_select(binary_scores, positive_mask)
                assert positive_binary_scores.size() == (num_questions,)
                positive_binary_scores = positive_binary_scores.repeat_interleave(num_questions * self.num_passages - 1)
                assert positive_binary_scores.size() == (num_questions * (num_questions * self.num_passages - 1),)

                negative_binary_scores = torch.masked_select(binary_scores, ~positive_mask)
                assert positive_binary_scores.size() == (num_questions * (num_questions * self.num_passages - 1),)

                ranking_loss_labels = torch.ones_like(positive_binary_scores, dtype=torch.long)
                cand_loss = F.margin_ranking_loss(
                    positive_binary_scores,
                    negative_binary_scores,
                    ranking_loss_labels,
                    self.hparams.cand_loss_margin,
                )
        else:
            cand_loss = None

        # Task #2: compute the reranking loss.
        if not no_rerank_loss:
            dense_scores = torch.matmul(encoded_question, binary_encoded_passage.transpose(0, 1))
            assert dense_scores.size() == (num_questions, num_questions * self.num_passages)
            rerank_loss = F.cross_entropy(dense_scores, passage_labels)
        else:
            rerank_loss = None

        return cand_loss, rerank_loss

    def _compute_ranks(self, encoded_question: Tensor, encoded_passage: Tensor) -> Tensor:
        num_questions = encoded_question.size(0)

        # Check the shapes of the input tensors.
        assert encoded_question.size() == (num_questions, self.embed_size)
        assert encoded_passage.size() == (num_questions * self.num_passages, self.embed_size)

        # Compute the question-passsage similarity scores.
        scores = torch.matmul(encoded_question, encoded_passage.transpose(0, 1))
        assert scores.size() == (num_questions, num_questions * self.num_passages)

        # Create the labels of the positive passages for each of the questions.
        # The index of the positive passage for the i-th question is i * `num_passages`.
        # For example, when `num_questions` is 8 and `num_passages` is 2, `passage_labels` will be [0, 2, 4, ..., 14].
        passage_labels = torch.arange(0, num_questions * self.num_passages, self.num_passages).to(
            encoded_question.device
        )
        assert passage_labels.size() == (num_questions,)

        # Aggregate the scores of the labeled passages.
        positive_passage_scores = scores.gather(dim=1, index=passage_labels.unsqueeze(1))
        assert positive_passage_scores.size() == (num_questions, 1)

        # Compute the ranks of the labeled passages.
        ranks = (scores > positive_passage_scores).sum(dim=1).float() + 1.0
        assert ranks.size() == (num_questions,)

        return ranks

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

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        grad = self.passage_encoder.encoder.encoder.layer[-1].output.dense.weight.grad
        if grad is not None:
            self.log("grad", torch.linalg.norm(grad).detach())

        optimizer.step(closure=optimizer_closure)
