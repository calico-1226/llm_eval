import os
import json
from dataclasses import dataclass

from typing import List, Dict
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, DataLoader


from transformers import PreTrainedTokenizer, PreTrainedModel


@dataclass
class Query:
    prompt: str
    outputs: List[str]
    scores: List[int] = None


def generate_text(
    input_list: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 16,
    max_length: int = 256,
    do_sample: bool = True,
    num_return_sequences: int = 5,
    device: torch.device = torch.device("cuda:0"),
    show_progress: bool = False,
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
    model.to(device)

    tokenized_inputs = tokenizer(
        input_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()

    if show_progress:
        dataloader = tqdm(dataloader)

    generated = []
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
        # generated = {
        #     input_list[i]: output_texts[
        #         i * num_return_sequences : (i + 1) * num_return_sequences
        #     ]
        #     for i in range(len(input_list))
        # }
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        assert len(input_texts) * num_return_sequences == len(output_texts)
        for i in range(len(input_texts)):
            query = Query(
                prompt=input_texts[i],
                outputs=output_texts[
                    i * num_return_sequences : (i + 1) * num_return_sequences
                ],
            )
            generated.append(query)

    return generated


def main():
    data_dir = "red-team-attempts"
    with open(f"/root/data/datasets/hh_rlhf/{data_dir}-single.json", "r") as f:
        custom_dataset = json.load(f)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

    generated = generate_text(
        input_list=custom_dataset["train"][:10],
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        max_length=100,
    )

    for query in generated:
        print(f"Prompt: {query.prompt}")
        print(f"Outputs: {query.outputs}")
        print()


if __name__ == "__main__":
    main()
