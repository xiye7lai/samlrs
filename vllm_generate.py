import json
import time
import os

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from rec_utils import load_config


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def resolve_long_input_path(dataset_name: str) -> str:
    mapping = {
        "movie": "/root/samlrs/data/ml-1m-processed/test/head.json",
        "steam": "/root/samlrs/data/steam-processed/test/head.json",
        "adm":   "/root/samlrs/data/adm-processed/test/head.json",
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset_name for long path: {dataset_name}")
    return mapping[dataset_name]


def resolve_tail_input_path(dataset_name: str) -> str:
    mapping = {
        "movie": "/root/samlrs/data/ml-1m-processed/test/tail.json",
        "steam": "/root/samlrs/data/steam-processed/test/tail.json",
        "adm":   "/root/samlrs/data/adm-processed/test/tail.json",
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unknown dataset_name for tail path: {dataset_name}")
    return mapping[dataset_name]


def run_generation_for_split(llm, lora_path, input_path, output_path):
    ds = load_dataset("json", data_files=input_path)["train"]
    print(f"--------------- split input: {input_path} -----------------")
    print(ds[0])

    prompts_all = [generate_prompt(ex["instruction"], ex.get("input")) for ex in ds]
    corrects_all = [ex["output"] for ex in ds]

    print('--------------- sample prompt -----------------')
    print(prompts_all[0])
    print('--------------- sample correct -----------------')
    print(corrects_all[0])

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.9,
        top_k=40,
        max_tokens=128,
    )

    start = time.time()
    results_gathered = list(
        map(
            lambda x: x.outputs[0].text,
            llm.generate(
                prompts=prompts_all,
                sampling_params=sampling_params,
                lora_request=LoRARequest("rec_adapter", 1, lora_path),
            ),
        )
    )
    results = [r.replace("</s>", "").lstrip() for r in results_gathered]
    print(f"time elapsed for split `{input_path}`: {time.time() - start:.2f}s")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "a", encoding="utf-8") as f:
        for idx in range(len(prompts_all)):
            d = {
                "prompt": prompts_all[idx],
                "correct": corrects_all[idx],
                "generated": results[idx],
            }
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved generated JSONL to: {output_path} (size={len(prompts_all)})")


def main():
    start = time.time()
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    config = load_config()

    base_model = config["base_model"]
    lora_weights = config["lora_path"]
    dataset_name = config["dataset_name"]

    long_input_path = resolve_long_input_path(dataset_name)
    tail_input_path = resolve_tail_input_path(dataset_name)

    long_generated_path = config["head_generated_path"]
    tail_generated_path = config["tail_generated_path"]

    world_size = 1
    print(f"model path: {base_model}")

    llm = LLM(
        model=base_model,
        tensor_parallel_size=world_size,
        dtype="bfloat16",
        enable_lora=True,
    )

    run_generation_for_split(
        llm=llm,
        lora_path=lora_weights,
        input_path=long_input_path,
        output_path=long_generated_path,
    )

    run_generation_for_split(
        llm=llm,
        lora_path=lora_weights,
        input_path=tail_input_path,
        output_path=tail_generated_path,
    )

    print(f"both splits finished generating in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
