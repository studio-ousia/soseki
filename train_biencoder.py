from argparse import ArgumentParser, Namespace

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything

from soseki.biencoder.modeling import BiencoderLightningModule


def main(args: Namespace) -> None:
    seed_everything(args.random_seed, workers=True)

    model = BiencoderLightningModule(args)
    checkpoint_callback = ModelCheckpoint(
        filename="best", monitor="val_avg_rank", save_last=True, save_top_k=1, mode="min"
    )
    trainer = Trainer.from_argparse_args(args, default_root_dir=args.output_dir, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--random_seed", type=int, default=1)
    parser = BiencoderLightningModule.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
