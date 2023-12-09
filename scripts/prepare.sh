python scripts/download.py --repo_id 'meta-llama/Llama-2-7b-chat-hf' --hf_token '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'
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
python scripts/quantize.py --checkpoint_path checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth --mode int4
