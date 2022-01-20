import ast
import csv
import dataclasses
import logging
import re
import unicodedata
from argparse import ArgumentParser, Namespace

import torch
import regex
from torch.nn.parallel import DataParallel
from tqdm import tqdm

from soseki.biencoder.modeling import BiencoderLightningModule
from soseki.retriever.binary_retriever import BinaryRetriever
from soseki.retriever.dense_retriever import DenseRetriever
from soseki.utils.data_utils import batch_iter, writeitem_json


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()

# Regular expressions used in the original DPR's `SimpleTokenizer`
# https://github.com/facebookresearch/DPR/blob/6b7e36d31850498c253e26f4b2628eccf1a8924e/dpr/utils/tokenizers.py#L157
ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
NON_WS = r"[^\p{Z}\p{C}]"
tokenizer_regexp = regex.compile(
    "(%s)|(%s)" % (ALPHA_NUM, NON_WS), flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
)


def passage_has_answer(answer: str, passage: str, match_type: str = "dpr_string") -> bool:
    if match_type == "dpr_string":
        # The "string" setting in the original DPR
        # https://github.com/facebookresearch/DPR/blob/6b7e36d31850498c253e26f4b2628eccf1a8924e/dpr/data/qa_validation.py#L110
        answer = unicodedata.normalize("NFD", answer)
        passage = unicodedata.normalize("NFD", passage)

        answer_tokens = [m.group().lower() for m in tokenizer_regexp.finditer(answer)]
        passage_tokens = [m.group().lower() for m in tokenizer_regexp.finditer(passage)]

        for i in range(0, len(passage_tokens) - len(answer_tokens) + 1):
            if answer_tokens == passage_tokens[i : i + len(answer_tokens)]:
                return True

    elif match_type == "dpr_regex":
        # The "regex" setting in the original DPR
        # https://github.com/facebookresearch/DPR/blob/6b7e36d31850498c253e26f4b2628eccf1a8924e/dpr/data/qa_validation.py#L123
        answer = unicodedata.normalize("NFD", answer)
        passage = unicodedata.normalize("NFD", passage)
        try:
            pattern = re.compile(answer, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
            if pattern.search(passage) is not None:
                return True

        except BaseException:
            return False

    elif match_type == "simple_nfd":
        # Simple matching of texts after unicode normalization to the NFD form
        answer = unicodedata.normalize("NFD", answer).lower()
        passage = unicodedata.normalize("NFD", passage).lower()

        answer = " ".join(answer.split())
        passage = " ".join(passage.split())

        return answer in passage

    elif match_type == "simple_nfkc":
        # Simple matching of texts after unicode normalization to the NFKC form
        # Most suitable for Japanese texts
        answer = unicodedata.normalize("NFKC", answer).lower()
        passage = unicodedata.normalize("NFKC", passage).lower()

        answer = " ".join(answer.split())
        passage = " ".join(passage.split())

        return answer in passage

    elif match_type == "none":
        # No normalization
        return answer in passage

    else:
        ValueError("Invalid match_type is specified.")

    return False


def main(args: Namespace):
    # initialize question encoder and its tokenization
    biencoder = BiencoderLightningModule.load_from_checkpoint(args.biencoder_file, map_location="cpu")
    question_encoder = biencoder.question_encoder.eval()
    encoder_tokenization = biencoder.tokenization

    if args.device_ids is not None:
        device_ids = [int(i) for i in args.device_ids.split(",")]
        question_encoder.to(device_ids[0])
        if len(device_ids) > 1:
            question_encoder = DataParallel(question_encoder, device_ids=device_ids)
    else:
        device_ids = []

    # initialize passage retriever
    if args.binary:
        logger.info("Using BinaryRetriever.")
        retriever = BinaryRetriever(args.passage_embeddings_file, args.passage_db_file)
    else:
        logger.info("Using DenseRetriever.")
        retriever = DenseRetriever(args.passage_embeddings_file, args.passage_db_file)

    num_passages = len(retriever.passage_db)
    top_ks = [int(k) for k in args.top_k.split(",")]
    retrieval_k = max(top_ks)

    positive_ranks = []
    output_items = []

    with open(args.qa_file) as f, tqdm() as pbar:
        for rows in batch_iter(csv.reader(f, delimiter="\t"), batch_size=args.batch_size):
            questions = [row[0] for row in rows]
            answer_lists = [ast.literal_eval(row[1]) for row in rows]

            with torch.no_grad():
                encoder_inputs = dict(encoder_tokenization.tokenize_questions(
                    questions,
                    padding=True,
                    truncation=True,
                    max_length=args.max_question_length,
                    return_tensors="pt",
                ))
                if device_ids:
                    encoder_inputs = {key: tensor.to(device_ids[0]) for key, tensor in encoder_inputs.items()}

                encoded_questions = question_encoder(**encoder_inputs).cpu().numpy()

            if args.binary:
                retrieved_passage_lists = retriever.retrieve_top_k_passages(
                    encoded_questions, k=retrieval_k, binary_k=args.binary_k
                )
            else:
                retrieved_passage_lists = retriever.retrieve_top_k_passages(encoded_questions, k=retrieval_k)

            for question, answers, retrieved_passages in zip(questions, answer_lists, retrieved_passage_lists):
                positive_rank = None
                output_ctxs = []
                for rank, retrieved_passage in enumerate(retrieved_passages, start=1):
                    if args.include_title_in_passage:
                        passage = " ".join([retrieved_passage.title, retrieved_passage.text])
                    else:
                        passage = retrieved_passage.text

                    has_answer = any(
                        passage_has_answer(answer, passage, match_type=args.answer_match_type)
                        for answer in answers
                    )
                    retrieved_passage.has_answer = has_answer
                    output_ctxs.append(dataclasses.asdict(retrieved_passage))

                    if has_answer and positive_rank is None:
                        positive_rank = rank

                output_items.append({"question": question, "answers": answers, "ctxs": output_ctxs})
                positive_ranks.append(positive_rank if positive_rank is not None else num_passages)

            pbar.update(len(rows))

    for k in sorted(top_ks):
        num_correct_at_k = sum(int(rank <= k) for rank in positive_ranks)
        recall_at_k = num_correct_at_k / len(positive_ranks)
        logger.info(f"Recall at {k}: {recall_at_k:.4f} ({num_correct_at_k}/{len(positive_ranks)})")

    if args.output_file is not None:
        writeitem_json(output_items, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--biencoder_file", type=str, required=True)
    parser.add_argument("--passage_db_file", type=str, required=True)
    parser.add_argument("--passage_embeddings_file", type=str, required=True)
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_question_length", type=int, default=256)
    parser.add_argument("--top_k", type=str, default="1,5,20,50,100")
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--binary_k", type=int, default=2048)
    parser.add_argument("--answer_match_type", type=str, default="dpr_string")
    parser.add_argument("--include_title_in_passage", action="store_true")
    parser.add_argument("--device_ids", type=str)
    args = parser.parse_args()

    main(args)
