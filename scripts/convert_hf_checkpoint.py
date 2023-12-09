# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import re
from pathlib import Path

import fire
import torch

from src.model import ModelArgs

logging.basicConfig(level=logging.INFO)


def _permute(w, config):
    dim = config.dim
    return (
        w.view(config.n_head, 2, config.head_dim // 2, dim)
        .transpose(1, 2)
        .reshape(config.head_dim * config.n_head, dim)
    )


@torch.inference_mode()
def convert_hf_checkpoint(checkpoint_dir: Path) -> None:
    """Convert HF checkpoint to .pth format."""
    model_name = checkpoint_dir.name

    config = ModelArgs.from_name(model_name)
    logging.info(f"Model config {config.__dict__}")

    model_map_json = checkpoint_dir / "pytorch_model.bin.index.json"
    assert model_map_json.is_file()
    bin_index = json.load(model_map_json.open())

    weight_map = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
    }
    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}

    merged_result = {}
    for file in sorted(bin_files):
        state_dict = torch.load(
            str(file), 
            map_location="cpu", 
            mmap=True, 
            weights_only=True
        )
        merged_result.update(state_dict)

    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r"(\d+)", "{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = weight_map[abstract_key]
            if new_key is None:
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]
        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            q = _permute(q, config)
            k = _permute(k, config)
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    logging.info(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")


def main(checkpoint_dir: str = "checkpoints/meta-llama/llama-2-7b-chat-hf"):
    """Convert HuggingFace checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    convert_hf_checkpoint(checkpoint_dir)


if __name__ == "__main__":
    fire.Fire(main)
