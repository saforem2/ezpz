"""
cria.py

From Wikipedia:

> A cria (pronounced /kriː.ə/) is a juvenile llama, alpaca, vicuña, or
> guanaco.[1]
"""
import shlex
import subprocess
import time

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

import ezpz

mname_from = "meta-llama/Llama-2-7b-hf"
mname_tiny = "tiny-random-llama-2"
vocab_keep_items = 3000


@ezpz.timeitlogit(rank=ezpz.get_rank())
def main() -> None:
    t0 = time.perf_counter()
    config = LlamaConfig.from_pretrained(mname_from)
    config.update(
        dict(
            hidden_size=16,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=256,
            num_key_value_heads=4,
            vocab_size=vocab_keep_items,
        )
    )
    print("new config", config)

    tiny_model = LlamaForCausalLM(config)
    print(f"num of params {tiny_model.num_parameters()}")

    tiny_model.bfloat16()
    tiny_model.save_pretrained(mname_tiny)

    tokenizer_fast = LlamaTokenizerFast.from_pretrained(mname_from)
    tmp_dir = f"/tmp/{mname_from}"
    tokenizer_fast.save_pretrained(tmp_dir)
    closing_pat = '},"merges": []}}'
    cmd = (
        "perl -0777 -pi -e "
        f"'s|({vocab_keep_items-1}).*|$1{closing_pat}|msg' "
        f"{tmp_dir}/tokenizer.json"
    )
    _ = subprocess.run(shlex.split(cmd), capture_output=True, text=True)

    tokenizer_fast_tiny = LlamaTokenizerFast.from_pretrained(tmp_dir)
    tokenizer_fast_tiny.save_pretrained(mname_tiny)

    model_inputs = tokenizer_fast_tiny(
        "Making tiny model", return_tensors="pt"
    )
    gen_tokens = tiny_model.generate(**model_inputs, max_new_tokens=100)
    print(tokenizer_fast_tiny.batch_decode(gen_tokens, skip_special_tokens=True))
    print("Random output should be expected, but no crashing")

    print(f"Model+Tokenizer saved in {mname_tiny}")
    end_time = time.perf_counter()
    timings = {
        "main/total": end_time - t0,
        "timings/training_start": 0.0,
        "timings/train_duration": end_time - t0,
        "timings/end-to-end": end_time - t0,
    }
    print(f"Timings: {timings}")


if __name__ == "__main__":
    main()
