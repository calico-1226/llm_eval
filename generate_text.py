import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["https_proxy"] = "http://127.0.0.1:7899"
os.environ["http_proxy"] = "http://127.0.0.1:7899"

from typing import List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator


def generate_text(
    input_list: List[str],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 256,
    do_sample: bool = True,
    num_return_sequences: int = 5,
    show_progress: bool = True,
):
    """Generate text from a list of inputs using a pretrained model.

    Args:
        input_list (List[str]): List of inputs to generate text from.
        model_name (str): Name of the pretrained model to use.
        max_length (int, optional): Maximum length of the generated text. Defaults to 256.
        do_sample (bool, optional): Whether to sample from the model's output. Defaults to True.
        num_return_sequences (int, optional): Number of sequences to return. Defaults to 5.
        show_progress (bool, optional): Whether to show a progress bar. Defaults to True.
    """
    accelerator = Accelerator()
    device = accelerator.device

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_inputs = tokenizer(
        input_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()

    generated = {}
    for input_ids, attention_mask in dataloader:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
            )

        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated = {
            input_list[i]: output_texts[
                i * num_return_sequences : (i + 1) * num_return_sequences
            ]
            for i in range(len(input_list))
        }

    return generated

def main():
    data_dir = "red-team-attempts"
    with open(f"/mnt/datasets/{data_dir}-single.json", "r") as f:
        custom_dataset = json.load(f)

    generated = generate_text(
        input_list=custom_dataset["train"],
        model_name="google/flan-t5-xl",
        batch_size=16,
        max_length=100,
    )

    for k, v in generated.items():
        print("input: ", k)
        print("output: ")
        for i, reply in enumerate(v):
            print(f"{i+1}. {reply}")



if __name__ == "__main__":
    main()
