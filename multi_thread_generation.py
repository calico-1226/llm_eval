import os
import threading
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["https_proxy"] = "http://127.0.0.1:7899"
os.environ["http_proxy"] = "http://127.0.0.1:7899"

from typing import List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader

from generate_text import generate_text


def multi_thread_generation(
    gpu_ids: List[str],
    model_name: str,
    input_list: List[str],
    batch_size: int = 16,
    max_length: int = 256,
):
    num_thread = len(gpu_ids)
    num_input_per_thread = len(input_list) // num_thread + 1

    lock = threading.Lock()

    results = []

    def process_thread(gpu_id: str, input_list: List[str], show_progress: bool = False):
        generated = generate_text(
            input_list,
            model_name,
            batch_size=batch_size,
            max_length=max_length,
            gpu_id=gpu_id,
            show_progress=show_progress,
        )
        with lock:
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
    data_dir = "red-team-attempts"
    with open(f"/mnt/datasets/{data_dir}-single.json", "r") as f:
        custom_dataset = json.load(f)

    results = multi_thread_generation(
        gpu_ids=["0", "1"],
        model_name="google/flan-t5-xl",
        input_list=custom_dataset["train"][:25],
    )

    for result in results:
        print(result.prompt)
        print(result.outputs)
        print()
