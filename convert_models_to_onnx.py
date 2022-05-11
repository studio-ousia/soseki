import json
import logging
import os
from argparse import ArgumentParser, Namespace

import torch

from soseki.biencoder.modeling import BiencoderLightningModule
from soseki.reader.modeling import ReaderLightningModule


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger()


def convert_biencoder_to_onnx(biencoder_ckpt_file: str, output_dir: str, verbose: bool = False) -> None:
    biencoder = BiencoderLightningModule.load_from_checkpoint(biencoder_ckpt_file, map_location="cpu")
    biencoder.freeze()

    with torch.no_grad():
        question_encoder_module = biencoder.question_encoder
        passage_encoder_module = biencoder.passage_encoder
        question_encoder_module.eval()
        passage_encoder_module.eval()

        biencoder_tokenizer = biencoder.tokenization.tokenizer
        sample_text = "This is a sample text." + biencoder_tokenizer.sep_token + "This is another sample text."
        sample_inputs = dict(biencoder_tokenizer(sample_text, return_tensors="pt"))  # BatchEncoding -> dict

        input_names = list(sample_inputs.keys())
        output_names = ["pooled_output"]

        dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
        dynamic_axes["pooled_output"] = {0: "batch"}

        question_encoder_output_file = os.path.join(output_dir, "question_encoder.onnx")
        torch.onnx.export(
            model=question_encoder_module,
            args=(sample_inputs, {}),
            f=question_encoder_output_file,
            input_names=input_names,
            output_names=output_names,
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

        passage_encoder_output_file = os.path.join(output_dir, "passage_encoder.onnx")
        torch.onnx.export(
            model=passage_encoder_module,
            args=(sample_inputs, {}),
            f=passage_encoder_output_file,
            input_names=input_names,
            output_names=output_names,
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

        biencoder_hparams_file = os.path.join(output_dir, "biencoder_hparams.json")
        json.dump(biencoder.hparams, open(biencoder_hparams_file, "w"), indent=4)


def convert_reader_to_onnx(reader_ckpt_file: str, output_dir: str, verbose: bool = False) -> None:
    reader = ReaderLightningModule.load_from_checkpoint(reader_ckpt_file, map_location="cpu")
    reader.freeze()

    with torch.no_grad():
        reader_module = reader.reader
        reader_module.eval()

        reader_tokenizer = reader.tokenization.tokenizer
        sample_text = "This is a sample text." + reader_tokenizer.sep_token + "This is another sample text."
        sample_inputs = dict(reader_tokenizer(sample_text, return_tensors="pt"))  # BatchEncoding -> dict

        input_names = list(sample_inputs.keys())
        output_names = ["classifier_logits", "start_logits", "end_logits"]

        dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
        dynamic_axes["classifier_logits"] = {0: "batch"}
        dynamic_axes["start_logits"] = {0: "batch", 1: "sequence"}
        dynamic_axes["end_logits"] = {0: "batch", 1: "sequence"}

        output_file = os.path.join(output_dir, "reader.onnx")
        torch.onnx.export(
            model=reader_module,
            args=(sample_inputs, {}),
            f=output_file,
            input_names=input_names,
            output_names=output_names,
            opset_version=12,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
        )

        reader_hparams_file = os.path.join(output_dir, "reader_hparams.json")
        json.dump(reader.hparams, open(reader_hparams_file, "w"), indent=4)


def main(args: Namespace) -> None:
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.biencoder_ckpt_file is not None:
        logger.info("Converting the biencoder model to onnx files")
        convert_biencoder_to_onnx(args.biencoder_ckpt_file, args.output_dir, verbose=args.verbose)

    if args.reader_ckpt_file is not None:
        logger.info("Converting the reader model to onnx files")
        convert_reader_to_onnx(args.reader_ckpt_file, args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--biencoder_ckpt_file", type=str)
    parser.add_argument("--reader_ckpt_file", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args)
