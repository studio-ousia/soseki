from argparse import ArgumentParser, Namespace

import faiss
import numpy as np
from torch.nn.parallel import DataParallel
from tqdm import tqdm

from soseki.biencoder.modeling import BiencoderLightningModule
from soseki.passage_db.lmdb_passage_db import LMDBPassageDB
from soseki.utils.data_utils import batch_iter


def main(args: Namespace):
    # load the biencoder
    biencoder = BiencoderLightningModule.load_from_checkpoint(args.biencoder_file, map_location="cpu")
    biencoder.freeze()
    biencoder.question_encoder = None  # to free up memory
    biencoder.passage_encoder.eval()

    if args.device_ids is not None:
        device_ids = [int(i) for i in args.device_ids.split(",")]
        biencoder.passage_encoder.to(device_ids[0])
        if len(device_ids) > 1:
            biencoder.passage_encoder = DataParallel(biencoder.passage_encoder, device_ids=device_ids)
    else:
        device_ids = []

    # load the passage db
    passage_db = LMDBPassageDB(args.passage_db_file)

    index = None
    is_binary = biencoder.hparams["binary"]

    # iterate over passages in the passage db
    with tqdm(total=len(passage_db)) as pbar:
        for passages in batch_iter(passage_db, batch_size=args.batch_size):
            # get embeddings of the passages
            titles = [passage.title for passage in passages]
            texts = [passage.text for passage in passages]
            encoder_inputs = dict(biencoder.tokenization.tokenize_passages(
                titles,
                texts,
                padding=True,
                truncation="only_second",
                max_length=args.max_passage_length,
                return_tensors="pt",
            ))
            if device_ids:
                encoder_inputs = {key: tensor.to(device_ids[0]) for key, tensor in encoder_inputs.items()}

            embeddings = biencoder.passage_encoder(**encoder_inputs).cpu().numpy()

            if index is None:
                # initialize the faiss index with the dimensionality of the embeddings
                dim_size = embeddings.shape[1]
                if is_binary:
                    index = faiss.IndexBinaryIDMap2(faiss.IndexBinaryFlat(dim_size))
                else:
                    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim_size))

            # format the embeddings for indexing
            if is_binary:
                embeddings = np.where(embeddings < 0, 0, 1)
                embeddings = np.packbits(embeddings).reshape(embeddings.shape[0], -1)
            else:
                embeddings = embeddings.astype(np.float32)

            # add the embeddings and the corresponding passage ids to the faiss index
            passage_ids = np.array([passage.id for passage in passages], dtype=np.int64)
            index.add_with_ids(embeddings, passage_ids)

            pbar.update(len(passages))

    # write the faiss index to file
    if is_binary:
        faiss.write_index_binary(index, args.output_file)
    else:
        faiss.write_index(index, args.output_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--biencoder_file", type=str, required=True)
    parser.add_argument("--passage_db_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_passage_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device_ids", type=str)
    args = parser.parse_args()

    main(args)
