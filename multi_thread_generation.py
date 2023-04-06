import os
import threading
import json
from copy import deepcopy

from typing import List, Dict

import torch

from generate_text import generate_text
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def multi_thread_generation(
    gpu_ids: List[str],
    model_name: str,
    input_list: List[str],
    batch_size: int = 16,
    max_length: int = 256,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_thread = len(gpu_ids)
    num_input_per_thread = len(input_list) // num_thread + 1

    lock = threading.Lock()

    results = []

    def process_thread(gpu_id: str, input_list: List[str], show_progress: bool = False):
        thread_name = threading.current_thread().name

        device = torch.device(f"cuda:{gpu_id}")
        model_copy = deepcopy(model).to(device)
        tokenizer_copy = deepcopy(tokenizer)
        print(
            f"{thread_name} is using GPU {gpu_id} to generate {len(input_list)} inputs."
        )
        generated = generate_text(
            input_list=input_list,
            model=model_copy,
            tokenizer=tokenizer_copy,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            show_progress=show_progress,
        )
        with lock:
            print(f"{thread_name} complete {len(generated)} generations.")
            results.extend(generated)

    threads = []
    for i in range(num_thread):
        start = i * num_input_per_thread
        end = (i + 1) * num_input_per_thread
        t = threading.Thread(
            target=process_thread, args=(gpu_ids[i], input_list[start:end], i == 0)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("/mnt/datasets/safe-rlhf", split="train")

    results = multi_thread_generation(
        gpu_ids=["4", "5"],
        model_name="google/flan-t5-large",
        input_list=ds["input"][:20],
    )

    for result in results:
        print(result.prompt)
        print(result.outputs)
        print()
