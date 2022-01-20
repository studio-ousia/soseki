import logging
import os
from argparse import ArgumentParser, Namespace

from soseki.passage_db.lmdb_passage_db import LMDBPassageDB


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def main(args: Namespace) -> None:
    if os.path.exists(args.db_file):
        logger.error("A database file already exists at the --db_file path. Exiting.")
        exit(1)

    logger.info("Building LMDBPassageDB")
    passage_db = LMDBPassageDB.from_passage_file(
        args.passage_file,
        args.db_file,
        db_map_size=args.db_map_size,
        skip_header=args.skip_header,
        chunk_size=args.chunk_size,
    )
    logger.info("Finished building LMDBPassageDB")
    logger.info("The number of indexed passages: %d", len(passage_db))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--passage_file", type=str, required=True)
    parser.add_argument("--db_file", type=str, required=True)
    parser.add_argument("--db_map_size", type=int, default=2147483648)
    parser.add_argument("--skip_header", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=1024)
    args = parser.parse_args()

    main(args)
