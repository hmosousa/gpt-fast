# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Optional

import fire
from huggingface_hub import snapshot_download
from requests.exceptions import HTTPError

ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"


def hf_download(repo_id: str, hf_token: Optional[str] = None) -> None:
    repo_dir = CHECKPOINT_DIR / repo_id
    repo_dir.mkdir(exist_ok=True, parents=True)

    try:
        ignore_patterns = ["*.safetensors*", "*.md", "*.txt", ".gitattributes"]
        snapshot_download(
            repo_id,
            local_dir=str(repo_dir),
            local_dir_use_symlinks=False,
            token=hf_token,
            ignore_patterns=ignore_patterns,
        )
    except HTTPError as e:
        if e.response.status_code == 401:
            print(
                "You need to pass a valid `--hf_token=...` to download private checkpoints."
            )
        else:
            raise e


def main(
    repo_id: str = "meta-llama/llama-2-7b-chat-hf", 
    hf_token: str = None
):
    """
    Download model weights form HuggingFace.
        repo_id: Repository ID to download from.
        hf_token: HuggingFace API token.
    """
    hf_download(repo_id, hf_token)


if __name__ == "__main__":
    fire.Fire(main)
