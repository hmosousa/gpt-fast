# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import time
from pathlib import Path

import fire
import torch
from sentencepiece import SentencePieceProcessor

from src.model import Transformer
from src.quantize import (
    WeightOnlyInt4GPTQQuantHandler,
    WeightOnlyInt4QuantHandler,
    WeightOnlyInt8QuantHandler,
)

logging.basicConfig(level=logging.INFO)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main(
    checkpoint_path: str = "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth",
    mode: str = "int8",
    groupsize: int = 128,
    calibration_tasks: list = ["hellaswag"],
    calibration_limit: int = 1000,
    calibration_seq_length: int = 100,
    pad_calibration_inputs: bool = False,
    percdamp: float = 0.01,
    blocksize: int = 128,
) -> None:
    """
    Quantize a model.

        checkpoint_path: Path to the model checkpoint - in the .pth format - to be quantized.
        mode: ['int8', 'int4', 'int4-gptq'] type of quantization to perform
        groupsize: Group size for int4 quantization.
        calibration_tasks: tasks to do gptq calibration on, if doing gptq
        calibration_limit: number of samples to use for gptq calibration
        calibration_seq_length: length of sequences to use for gptq calibration
        pad_calibration_inputs: pads sequences shorter than calibration_seq_length to that length, yielding more calibration inputs but running much slower
        percdamp: gptq percentage dampening
        blocksize: blocksize for gptq
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file(), checkpoint_path

    dir_name = checkpoint_path.parent
    model_name = checkpoint_path.name

    logging.info("Loading model...")
    t0 = time.time()

    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=torch.bfloat16, device="cpu")

    if mode == "int8":
        logging.info(
            "Quantizing model weights for int8 weight-only symmetric per-channel quantization"
        )
        quant_handler = WeightOnlyInt8QuantHandler(model)
        quantized_state_dict = quant_handler.create_quantized_state_dict()

        new_model_name = model_name.replace(".pth", "_int8.pth")

    elif mode == "int4":
        logging.info(
            "Quantizing model weights for int4 weight-only affine per-channel groupwise quantization"
        )
        quant_handler = WeightOnlyInt4QuantHandler(model, groupsize)
        quantized_state_dict = quant_handler.create_quantized_state_dict()

        new_model_name = model_name.replace(".pth", f"_int4.g{groupsize}.pth")

    elif mode == "int4-gptq":
        logging.info(
            "Quantizing model weights for int4 weight-only affine per-channel groupwise quantization using GPTQ..."
        )
        quant_handler = WeightOnlyInt4GPTQQuantHandler(model, groupsize)

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path
        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

        quantized_state_dict = quant_handler.create_quantized_state_dict(
            tokenizer,
            blocksize,
            percdamp,
            groupsize,
            calibration_tasks,
            calibration_limit,
            calibration_seq_length,
            pad_calibration_inputs,
        )

        new_model_name = model_name.replace(".pth", f"_int4-gptq.g{groupsize}.pth")

    else:
        raise ValueError(
            f"Invalid quantization mode {mode} needs to be one of [int8, int4, int4-gpptq]"
        )

    quantize_path = dir_name / new_model_name
    logging.info(f"Writing quantized weights to {quantize_path}")
    quantize_path.unlink(missing_ok=True)
    torch.save(quantized_state_dict, quantize_path)
    logging.info(f"Quantization complete took {time.time() - t0:.02f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
