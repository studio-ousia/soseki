# Sōseki

Sōseki is an implementation of an end-to-end question answering (QA) system.

Currently, Sōseki makes use of [Binary Passage Retriever (BPR)](https://github.com/studio-ousia/bpr), an efficient passages retrieval model for a large collection of documents.
BPR was originally developed to achieve high computational efficiency of the QA system submitted to the [Systems under 6GB track](https://ai.google.com/research/NaturalQuestions/efficientqa) in the [NeurIPS 2020 EfficientQA competition](https://efficientqa.github.io/).

## Installation

```sh
# Before installation, upgrade pip and setuptools
$ pip install -U pip setuptools

# For reproducing experiments, install dependencies with sepcific versions beforehand
$ pip install -r requirements.txt

# Install the soseki package
$ pip install .
# If you want to install it in editable mode
$ pip install -e .
```

**Note:** If you are using a GPU Environment different from CUDA 10.2, you may need to reinstall PyTorch according to [the official documentation](https://pytorch.org/get-started/locally/).

## Example Usage

Before you start, you need to download the datasets available on the
[DPR repository](https://github.com/facebookresearch/DPR) into `<DPR_DATASET_DIR>`.

We used 4 GPUs with 12GB memory each for the experiments.

**1. Build passage database**

```sh
$ python build_passage_db.py \
    --passage_file <DPR_DATASET_DIR>/wikipedia_split/psgs_w100.tsv \
    --db_file <WORK_DIR>/passages.db \
    --db_map_size 21000000000 \
    --skip_header
```

**2. Train a biencoder**

```sh
$ python train_biencoder.py \
    --train_file <DPR_DATASET_DIR>/retriever/nq-train.json \
    --val_file <DPR_DATASET_DIR>/retriever/nq-dev.json \
    --output_dir <WORK_DIR>/biencoder \
    --max_question_length 64 \
    --max_passage_length 192 \
    --num_negative_passages 1 \
    --shuffle_hard_negative_passages \
    --shuffle_normal_negative_passages \
    --base_pretrained_model bert-base-uncased \
    --binary \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --gradient_clip_val 2.0 \
    --max_epochs 40 \
    --gpus 4 \
    --precision 16 \
    --strategy ddp
```

**3. Build passage embeddings**

```sh
$ python build_passage_embeddings.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --output_file <WORK_DIR>/passage_embeddings.idx \
    --max_passage_length 192 \
    --batch_size 2048 \
    --device_ids 0 1 2 3
```

**4. Evaluate the retriever and create datasets for reader**

```sh
$ mkdir <WORK_DIR>/reader_data

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DPR_DATASET_DIR>/retriever/qas/nq-train.csv \
    --output_file <WORK_DIR>/reader_data/nq_train.jsonl \
    --batch_size 64 \
    --max_question_length 64 \
    --top_k 1 2 5 10 20 50 100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type dpr_string \
    --include_title_in_passage \
    --device_ids 0 1 2 3
# The result should be logged as follows:
# Recall at 1: 0.4966 (39317/79168)
# Recall at 2: 0.6142 (48625/79168)
# Recall at 5: 0.7327 (58007/79168)
# Recall at 10: 0.7898 (62523/79168)
# Recall at 20: 0.8270 (65474/79168)
# Recall at 50: 0.8594 (68040/79168)
# Recall at 100: 0.8748 (69253/79168)

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DPR_DATASET_DIR>/retriever/qas/nq-dev.csv \
    --output_file <WORK_DIR>/reader_data/nq_dev.jsonl \
    --batch_size 64 \
    --max_question_length 64 \
    --top_k 1 2 5 10 20 50 100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type dpr_string \
    --include_title_in_passage \
    --device_ids 0 1 2 3
# The result should be logged as follows:
# Recall at 1: 0.4086 (3578/8757)
# Recall at 2: 0.5147 (4507/8757)
# Recall at 5: 0.6359 (5569/8757)
# Recall at 10: 0.7082 (6202/8757)
# Recall at 20: 0.7605 (6660/8757)
# Recall at 50: 0.8129 (7119/8757)
# Recall at 100: 0.8388 (7345/8757)

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DPR_DATASET_DIR>/retriever/qas/nq-test.csv \
    --output_file <WORK_DIR>/reader_data/nq_test.jsonl \
    --batch_size 64 \
    --max_question_length 64 \
    --top_k 1 2 5 10 20 50 100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type dpr_string \
    --include_title_in_passage \
    --device_ids 0 1 2 3
# The result should be logged as follows:
# Recall at 1: 0.4194 (1514/3610)
# Recall at 2: 0.5338 (1927/3610)
# Recall at 5: 0.6560 (2368/3610)
# Recall at 10: 0.7235 (2612/3610)
# Recall at 20: 0.7720 (2787/3610)
# Recall at 50: 0.8186 (2955/3610)
# Recall at 100: 0.8501 (3069/3610)
```

**5. Train a reader**

```sh
$ python train_reader.py \
    --train_file <WORK_DIR>/reader_data/nq_train.jsonl \
    --val_file <WORK_DIR>/reader_data/nq_dev.jsonl \
    --output_dir <WORK_DIR>/reader \
    --train_num_passages 24 \
    --eval_num_passages 100 \
    --max_input_length 256 \
    --shuffle_positive_passage \
    --shuffle_negative_passage \
    --num_dataloader_workers 1 \
    --base_pretrained_model bert-base-uncased \
    --answer_normalization_type dpr \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --accumulate_grad_batches 4 \
    --gradient_clip_val 2.0 \
    --max_epochs 20 \
    --gpus 4 \
    --precision 16 \
    --strategy ddp
```

**6. Evaluate the reader**

```sh
$ python evaluate_reader.py \
    --reader_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --test_file <WORK_DIR>/reader_data/nq_dev.jsonl \
    --test_num_passages 100 \
    --test_max_load_passages 100 \
    --test_batch_size 4 \
    --gpus 4 \
    --strategy ddp
# The result should be printed as follows:
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │   test_answer_accuracy    │     0.39237180352211      │
# │ test_classifier_precision │    0.5836473703384399     │
# └───────────────────────────┴───────────────────────────┘
```

```sh
$ python evaluate_reader.py \
    --reader_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --test_file <WORK_DIR>/reader_data/nq_test.jsonl \
    --test_num_passages 100 \
    --test_max_load_passages 100 \
    --test_batch_size 4 \
    --gpus 4 \
    --strategy ddp
# The result should be printed as follows:
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │   test_answer_accuracy    │    0.3891966640949249     │
# │ test_classifier_precision │    0.5764542818069458     │
# └───────────────────────────┴───────────────────────────┘
```

**7. (optional) Convert the trained models into ONNX format**

```sh
$ python convert_models_to_onnx.py \
    --biencoder_ckpt_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --reader_ckpt_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --output_dir <WORK_DIR>/onnx
```

**8. Run demo**

```sh
$ streamlit run demo.py --browser.serverAddress localhost --browser.serverPort 8501 -- \
    --biencoder_ckpt_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --reader_ckpt_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --device cuda:0
```

or if you have exported the models to ONNX format:

```sh
$ streamlit run demo.py --browser.serverAddress localhost --browser.serverPort 8501 -- \
    --onnx_model_dir <WORK_DIR>/onnx \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx
```

Then open http://localhost:8501.

The demo can also be launched with Docker:

```sh
$ docker build -t soseki --build-arg TRANSFORMERS_BASE_MODEL_NAME='bert-base-uncased' .
$ docker run --rm -v $(realpath <WORK_DIR>):/app/model -p 8501:8501 -it soseki \
    streamlit run demo.py --browser.serverAddress localhost --browser.serverPort 8501 -- \
        --onnx_model_dir /app/model/onnx \
        --passage_db_file /app/model/passages.db \
        --passage_embeddings_file /app/model/passage_embeddings.idx
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This
work is licensed under a
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative
Commons Attribution-NonCommercial 4.0 International License</a>.

## Citation

If you find this work useful, please cite the following paper:

[Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2106.00882)

```
@inproceedings{yamada2021bpr,
  title={Efficient Passage Retrieval with Hashing for Open-domain Question Answering},
  author={Ikuya Yamada and Akari Asai and Hannaneh Hajishirzi},
  booktitle={ACL},
  year={2021}
}
```
