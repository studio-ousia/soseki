# Sōseki

Sōseki is an implementation of an end-to-end question answering (QA) system.

Currently, Sōseki makes use of [Binary Passage Retriever (BPR)](https://github.com/studio-ousia/bpr), an efficient passages retrieval model for a large collection of documents.
BPR was originally developed to achieve high computational efficiency of the QA system submitted to the [Systems under 6GB track](https://ai.google.com/research/NaturalQuestions/efficientqa) in the [NeurIPS 2020 EfficientQA competition](https://efficientqa.github.io/).

## Installation

You can install the required libraries using pip:

```sh
$ pip install -r requirements.txt
```

**Note:** If you are using a GPU Environment different from CUDA 10.2, you may need to reinstall PyTorch according to [the official documentation](https://pytorch.org/get-started/locally/).

## Example Usage

Before you start, you need to download the datasets available on the
[DPR repository](https://github.com/facebookresearch/DPR) into `<DPR_DATASET_DIR>`.

We used a server with 4 GeForce RTX 2080 GPUs with 11GB memory for the experiments.

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
    --dev_file <DPR_DATASET_DIR>/retriever/nq-dev.json \
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
    --accelerator ddp
```

**3. Build passage embeddings**

```sh
$ python build_passage_embeddings.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/best.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --output_file <WORK_DIR>/passage_embeddings.idx \
    --max_passage_length 192 \
    --batch_size 2048 \
    --device_ids 0,1,2,3
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
    --top_k 1,2,5,10,20,50,100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type dpr_string \
    --include_title_in_passage \
    --device_ids 0,1,2,3
# The result should be logged as follows:
# Recall at 1: 0.4867 (38529/79168)
# Recall at 2: 0.6059 (47964/79168)
# Recall at 5: 0.7275 (57592/79168)
# Recall at 10: 0.7862 (62245/79168)
# Recall at 20: 0.8251 (65319/79168)
# Recall at 50: 0.8588 (67991/79168)
# Recall at 100: 0.8748 (69255/79168)

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DPR_DATASET_DIR>/retriever/qas/nq-dev.csv \
    --output_file <WORK_DIR>/reader_data/nq_dev.jsonl \
    --batch_size 64 \
    --max_question_length 64 \
    --top_k 1,2,5,10,20,50,100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type dpr_string \
    --include_title_in_passage \
    --device_ids 0,1,2,3
# The result should be logged as follows:
# Recall at 1: 0.3984 (3489/8757)
# Recall at 2: 0.5085 (4453/8757)
# Recall at 5: 0.6377 (5584/8757)
# Recall at 10: 0.7075 (6196/8757)
# Recall at 20: 0.7588 (6645/8757)
# Recall at 50: 0.8122 (7112/8757)
# Recall at 100: 0.8409 (7364/8757)

$ python evaluate_retriever.py \
    --biencoder_file <WORK_DIR>/biencoder/lightning_logs/version_0/checkpoints/last.ckpt \
    --passage_db_file <WORK_DIR>/passages.db \
    --passage_embeddings_file <WORK_DIR>/passage_embeddings.idx \
    --qa_file <DPR_DATASET_DIR>/retriever/qas/nq-test.csv \
    --output_file <WORK_DIR>/reader_data/nq_test.jsonl \
    --batch_size 64 \
    --max_question_length 64 \
    --top_k 1,2,5,10,20,50,100 \
    --binary \
    --binary_k 2048 \
    --answer_match_type dpr_string \
    --include_title_in_passage \
    --device_ids 0,1,2,3
# The result should be logged as follows:
# Recall at 1: 0.4058 (1465/3610)
# Recall at 2: 0.5191 (1874/3610)
# Recall at 5: 0.6443 (2326/3610)
# Recall at 10: 0.7144 (2579/3610)
# Recall at 20: 0.7687 (2775/3610)
# Recall at 50: 0.8233 (2972/3610)
# Recall at 100: 0.8504 (3070/3610)
```

**5. Train a reader**

```sh
$ python train_reader.py \
    --train_file <WORK_DIR>/reader_data/nq_train.jsonl \
    --dev_file <WORK_DIR>/reader_data/nq_dev.jsonl  \
    --output_dir <WORK_DIR>/reader \
    --train_num_passages 24 \
    --eval_num_passages 100 \
    --max_input_length 256 \
    --include_title_in_passage \
    --shuffle_positive_passage \
    --shuffle_negative_passage \
    --num_dataloader_workers 1 \
    --base_pretrained_model bert-base-uncased \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --learning_rate 1e-5 \
    --warmup_proportion 0.1 \
    --accumulate_grad_batches 4 \
    --gradient_clip_val 2.0 \
    --max_epochs 20 \
    --gpus 4 \
    --precision 16 \
    --accelerator ddp \
    --answer_normalization_type dpr
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
    --accelerator ddp
# The result should be printed as follows:
# --------------------------------------------------------------------------------
# DATALOADER:0 TEST RESULTS
# {'test_answer_accuracy': 0.3944272994995117,
#  'test_classifier_precision': 0.5893570780754089}
# --------------------------------------------------------------------------------
```

```sh
$ python evaluate_reader.py \
    --reader_file <WORK_DIR>/reader/lightning_logs/version_0/checkpoints/best.ckpt \
    --test_file <WORK_DIR>/reader_data/nq_test.jsonl \
    --test_num_passages 100 \
    --test_max_load_passages 100 \
    --test_batch_size 4 \
    --gpus 4 \
    --accelerator ddp
# The result should be printed as follows:
# --------------------------------------------------------------------------------
# DATALOADER:0 TEST RESULTS
# {'test_answer_accuracy': 0.3889196813106537,
#  'test_classifier_precision': 0.5728532075881958}
# --------------------------------------------------------------------------------
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
$ docker build -t soseki .
$ docker run --rm -v <WORK_DIR>:/app/models -p 8501:8501 -it soseki \
    streamlit run demo.py --browser.serverAddress localhost --browser.serverPort 8501 -- \
        --onnx_model_dir models/onnx \
        --passage_db_file models/passages.db \
        --passage_embeddings_file models/passage_embeddings.idx
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
