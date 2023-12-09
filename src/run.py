# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import itertools
import time
from pathlib import Path
from typing import Optional

import fire
import torch
import torch._dynamo.config
import torch._inductor.config
from sentencepiece import SentencePieceProcessor

from src.generate import generate, load_model, decode_one_token, prefill, model_forward, logits_to_prob
from src.utils import encode_tokens

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = (
    True  # Experimental feature to reduce compilation times, will be on by default in future
)


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: str = "checkpoints/meta-llama/Llama-2-7b-chat-hf/model_int8.pth",
    compile: bool = True,
    profile: Optional[Path] = None,
    compile_prefill: bool = False,
    draft_checkpoint_path: Optional[str] = None,
    speculate_k: int = 5,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.

    prompt: str, Input prompt
    interactive: bool Whether to launch in interactive mode
    num_samples: int, Number of samples.
    max_new_tokens: int, Maximum number of new tokens
    top_k: int, Top-k for sampling.
    temperature: float, Temperature for sampling
    checkpoint_path: Path, Model checkpoint path.
    compile: bool Whether to compile the model
    compile_prefill: bool Whether to compile the prefill (improves prefill perf, but higher compile times)
    profile: Path, Profile path.
    speculate_k: int, Speculative execution depth
    draft_checkpoint_path: Path, Draft checkpoint path.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    rank = maybe_init_dist()

    device = "mps"
    precision = torch.float32
    is_speculative = False
    if draft_checkpoint_path is not None:
        draft_checkpoint_path = Path(draft_checkpoint_path)
        is_speculative = True
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision)

    if is_speculative:
        draft_model = load_model(draft_checkpoint_path, device, precision)
    else:
        draft_model = None

    torch.mps.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    prompt_length = encoded.size(0)

    torch.manual_seed(1234)
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    if compile:
        if is_speculative:
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    aggregate_metrics = {
        "tokens_per_sec": [],
        "accept_counts": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        torch.mps.synchronize()
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x: x
        t0 = time.perf_counter()

        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            prof = torch.profiler.profile()

        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics["accept_counts"].append(metrics["accept_counts"])

        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")

        torch.mps.synchronize()
        t = time.perf_counter() - t0

        if not interactive:
            print(tokenizer.decode(y.tolist()))
        else:
            print()

        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")

    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")


if __name__ == "__main__":
    fire.Fire(main)
