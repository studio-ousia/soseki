import logging
from argparse import ArgumentParser

from pytorch_lightning.trainer import Trainer

from soseki.reader.modeling import ReaderLightningModule


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()
logging.getLogger("lightning").setLevel(logging.ERROR)


def main(args: ArgumentParser):
    overriding_hparams = {}

    overriding_hparams["test_file"] = args.test_file
    overriding_hparams["test_gold_passages_file"] = args.test_gold_passages_file

    if args.test_num_passages is not None:
        overriding_hparams["eval_num_passages"] = args.test_num_passages
    if args.test_max_load_passages is not None:
        overriding_hparams["eval_max_load_passages"] = args.test_max_load_passages
    if args.test_batch_size is not None:
        overriding_hparams["eval_batch_size"] = args.test_batch_size

    model = ReaderLightningModule.load_from_checkpoint(args.reader_file, **overriding_hparams)

    trainer = Trainer.from_argparse_args(args, logger=False)
    trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--reader_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--test_gold_passages_file", type=str)
    parser.add_argument("--test_num_passages", type=int)
    parser.add_argument("--test_max_load_passages", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
