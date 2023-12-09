# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from pathlib import Path
from typing import Optional

import lm_eval
from lm_eval.base import BaseLM
import torch
import torch._dynamo.config
import torch._inductor.config
from sentencepiece import SentencePieceProcessor

from src.generate import _load_model, model_forward
from src.utils import setup_cache_padded_seq_input_pos_max_seq_length_for_prefill, encode_tokens

torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.epilogue_fusion = False
torch._dynamo.config.cache_size_limit = 100000



class SimpleGPTEvalWrapper(BaseLM):
    """
    A wrapper class for SimpleGPT, providing integration with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_seq_length: Optional[int] = None,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device("mps")
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        encoded = encode_tokens(
            self._tokenizer, string, bos=True, eos=False, device=self._device
        )
        # encoded is a pytorch tensor, but some internal logic in the
        # eval harness expects it to be a list instead
        # TODO: verify this for multi-batch as well
        encoded = encoded.tolist()
        return encoded

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        # TODO: make batches work
        inps = inps.squeeze(0)

        max_new_tokens = 1
        (
            seq,
            input_pos,
            max_seq_length,
        ) = setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
            self._model,
            inps,
            max_new_tokens,
            self.max_length,
        )
        x = seq.index_select(0, input_pos).view(1, -1)
        logits = model_forward(self._model, x, input_pos)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception("unimplemented")


@torch.no_grad()
def eval(
    model,
    tokenizer,
    tasks: list = ["hellaswag"],
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        model: The pre-trained language model to evaluate.
        tokenizer: The tokenizer to use for encoding/decoding text.
        task (str): The name of the evaluation task to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    Returns:
        eval_results (dict): A dictionary of evaluation results for the specified task(s).
    """
    model_eval_wrapper = SimpleGPTEvalWrapper(
        model,
        tokenizer,
        max_seq_length,
    )

    if "hendrycks_test" in tasks:
        tasks.remove("hendrycks_test")
        tasks += [x for x in lm_eval.tasks.hendrycks_test.create_all_tasks().keys()]
    task_dict = lm_eval.tasks.get_task_dict(tasks)

    eval_results = lm_eval.evaluator.evaluate(
        model_eval_wrapper,
        task_dict,
        limit=limit,
    )
    return eval_results


def main(
    checkpoint_path: Path = Path(
        "checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"
    ),
    compile: bool = False,
    tasks: list = ["hellaswag"],
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        checkpoint_path (Path): The path to the model checkpoint file to load.
        compile (bool): Whether or not to compile the model for optimization.
        task (Optional[str]): The name of the evaluation task or a list of tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    """

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    device = "mps"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, False)

    torch.mps.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds.")

    model.eval()

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    torch.manual_seed(1234)

    if compile:
        global model_forward
        model_forward = torch.compile(
            model_forward, mode="reduce-overhead", dynamic=True, fullgraph=True
        )
        torch._inductor.config.coordinate_descent_tuning = True

    t1 = time.time()
    result = eval(
        model,
        tokenizer,
        tasks,
        limit,
        max_seq_length,
    )
    print(f"Time to run eval: {time.time() - t1:.02f} seconds.")
    print(f"For model {checkpoint_path}")
    for task, res in result["results"].items():
        print(f"{task}: {res}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["hellaswag"],
        help="list of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="number of samples to evalulate"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="maximum length sequence to evaluate",
    )

    args = parser.parse_args()
    main(
        Path(args.checkpoint_path),
        args.compile,
        args.tasks,
        args.limit,
        args.max_seq_length,
    )
