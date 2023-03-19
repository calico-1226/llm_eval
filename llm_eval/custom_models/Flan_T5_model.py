from typing import Optional, List, Union, Tuple, Dict
import bigbench.api.model as model
import bigbench.api.results as results
import bigbench.models.model_utils as model_utils
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np
import random
import scipy

MODEL_INFO = {
    "google/flan-t5-small": model.ModelData(
        model_family="FLAN-T5",
        model_name="flan-t5-small",
        non_embedding_params=100,
        flop_matched_non_embedding_params=100,
        total_params=100,
        training_batch_size=64,
        training_steps=100 * 32 * 1024,
        description="see",
        decoding_params={},
    ),
    "google/flan-t5-base": model.ModelData(
        model_family="FLAN-T5",
        model_name="flan-t5-base",
        non_embedding_params=100,
        flop_matched_non_embedding_params=100,
        total_params=100,
        training_batch_size=64,
        training_steps=100 * 32 * 1024,
        description="see",
        decoding_params={},
    ),
    "google/flan-t5-large": model.ModelData(
        model_family="FLAN-T5",
        model_name="flan-t5-large",
        non_embedding_params=100,
        flop_matched_non_embedding_params=100,
        total_params=100,
        training_batch_size=64,
        training_steps=100 * 32 * 1024,
        description="see",
        decoding_params={},
    ),
}


def compute_loss(self, labels, logits, shape):
    shape = logits.shape
    logits = logits.reshape(-1, shape[-1])

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    # When scoring step N, the target token is the next input token at step N+1, so we
    # shift all labels one step to the left before giving the labels to the loss function.
    shifted_labels = torch.roll(labels, -1)
    # always mask the last shifted token (== first token before the shift)
    shifted_labels[:, -1] = -100
    shifted_labels = shifted_labels.reshape(-1)
    # Clip negative/masked labels to zero - those will get masked later anyway
    unmasked_loss = loss_fn(logits, torch.nn.functional.relu(shifted_labels)).reshape(
        shape[:-1]
    )
    # make sure only labels that are not equal to -100 affect the loss
    loss_mask = (shifted_labels != -100).type_as(unmasked_loss).reshape(shape[:-1])
    masked_loss = unmasked_loss * loss_mask
    reduced_masked_loss = torch.sum(masked_loss, axis=1)
    return (-reduced_masked_loss).cpu().detach().numpy().tolist()


def set_seed(seed: int):
    """sets random number generator seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BIGBenchHF(model.Model):
    """A BIGBench api-compatible Huggingface model

    Args:
    model_name: name of model to load, must be in MODEL_NAMES
    """

    def __init__(self, model_name="google/flan-t5-base", max_length=40) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_name = model_name
        self._max_length = max_length
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model_data(self) -> model.ModelData:
        return MODEL_INFO[self._model_name]

    def generate_text(
        self,
        inputs: Union[str, List[str]],
        max_length: int = 0,
        stop_string: Optional[str] = None,
        output_regex: Optional[str] = None,
    ) -> Union[str, List[str]]:
        """Generates text for given inputs.

        Args:
          inputs: String or list of strings as inputs for model.
          max_length: Maximum string length of output, if 0 uses max_length passed
            to constructor
          stop_string: If specified, model output will be truncated to the shortest
            string which includes stop_string.
          output_regex: If specified, the first match to the python regular
            expression output_regex in the model output will be returned. If there is
            no match, an empty string will be returned.

        Returns:
          String or list of strings generated by model.

        Raises:
          ValueError if max_length is invalid
        """
        max_length = max_length or self._max_length

        if isinstance(inputs, str):
            input_list = [inputs]
        else:
            input_list = inputs

        generated = []
        for idx, prompt in enumerate(input_list):
            input = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            response = self._model.generate(**input)
            response = self._tokenizer.batch_decode(response, skip_special_tokens=True)
            generated.append(response)

        if isinstance(inputs, str):
            generated = generated[0]

        generated = model_utils.postprocess_output(
            generated, max_length, stop_string, output_regex
        )

        return generated

    def cond_log_prob(
        self,
        inputs: Union[str, List[str]],
        targets: Union[List[str], List[List[str]]],
        batch_size: int = 64,
        absolute_normalization: Optional[bool] = False,
    ) -> Union[List[float], List[List[float]]]:
        """Computes conditional log probabilities of targets given inputs.

        Args:
          `inputs`: A single string input or a list of string inputs.

          `targets`: Possible string outputs for each input. If input is a
             string, this is a list `[t_1, t_2, ..., t_n]` of possible string
             outputs. If input is a list of strings, then this is a nested
             list `[[t_1, t_2, ..., t_n], ...]` with length equal to `len(inputs)`.

           `absolute_normalization`: When True, the function returns the log
             probability of unconstrained generation or the target sequence. When
             False (default), log probabilities are normalized so that the probabilities
             of generating `targets` sum to 1. Note that setting `absolute_normalization`
             to True restricts the class of models that can be evaluated to those that
             can assign absolute probabilities to sequences.

           Returns:
             If a single string input is provided, returns a list of
             log-probabilities `[lp_1, lp_2, ..., lp_n]` predicted by the model,
             where  `lp_i = log(prob(t_i | input)`  is the conditional log-prob
             to generate target `t_i` given input. If a list of string inputs
             was provided, returns a list of such elements of the form
             `[[lp_1, lp_2, ..., lp_n], ...]`, where each element contains the
             log-probabilities for the corresponding input and targets.
             In this case, the length of the returned list is `len(input)`.
        """
        sep = ""
        if isinstance(inputs, str):
            target_list = targets
            input_list = [inputs] * len(target_list)
            # input_list = [inputs for target in target_list]
            shape = None

        else:
            target_list = targets
            input_list = sum(
                [
                    [
                        inpt,
                    ]
                    * len(target)
                    for inpt, target in zip(inputs, targets)
                ],
                [],
            )
            target_list = sum(
                [
                    [inpt + sep + tgt for tgt in target]
                    for inpt, target in zip(inputs, targets)
                ],
                [],
            )
            # To be fixed: Maybe #answer of each question is different!
            shape = (len(inputs), -1)

        encoder_max_len = 64
        decoder_max_len = 64
        encoder_inputs = self._tokenizer(
            input_list,
            truncation=True,
            return_tensors="pt",
            max_length=encoder_max_len,
            padding=True,
        )
        decoder_inputs = self._tokenizer(
            target_list,
            truncation=True,
            return_tensors="pt",
            max_length=decoder_max_len,
            padding=True,
        )
        labels = decoder_inputs["input_ids"]
        labels[labels == self._tokenizer.pad_token_id] = -100

        num_examples = len(input_list)
        loss = []
        for idx in range(0, num_examples, batch_size):
            batch_encoder_ids = encoder_inputs["input_ids"][
                idx : min(idx + batch_size, num_examples), :
            ].to(self._device)
            batch_encoder_mask = encoder_inputs["attention_mask"][
                idx : min(idx + batch_size, num_examples), :
            ].to(self._device)
            batch_label = labels[idx : min(idx + batch_size, num_examples), :].to(
                self._device
            )

            batch_logits = self._model(
                input_ids=batch_encoder_ids,
                attention_mask=batch_encoder_mask,
                labels=batch_label,
            ).logits
            batch_loss = compute_loss(self, batch_label, batch_logits, shape)
            loss += batch_loss

        start, scores = 0, []
        if isinstance(inputs, str):
            scores.append(loss)
        else:
            for target in targets:
                scores.append(loss[start : start + len(target)])
                start += len(target)

        if not absolute_normalization:
            scores = [
                list(score_row - scipy.special.logsumexp(score_row))
                for score_row in scores
            ]

        if isinstance(inputs, str):
            scores = scores[0]

        return scores


if __name__ == "__main__":
    """Use this to run a simple test of the HF model types."""

    # test a few gpt models
    for model_name in ["google/flan-t5-base"]:
        print("-" * 80)
        print(f"model: {model_name}")
        set_seed(42)

        model = BIGBenchHF(model_name=model_name)
        prompt = "It was the best of times, it was"
        response = model.generate_text(
            inputs=prompt,
            max_length=32,
            stop_string=".",
        )

        print(f"prompt: {prompt}")
        print(f"response: {response}")

        prompts = ["These are the times that", "Stately, plump Buck Mulligan"]
        responses = model.generate_text(inputs=prompts, max_length=32, stop_string=".")

        for p, r in zip(prompts, responses):
            print(f"prompt: {p}")
            print(f"response: {r}")

        # for testing, the prompt here has no trailing space, while the
        # next scoring example has a trailing space
        prompt = (
            f"What color is the sky? Answer: blue\n" f"What color is grass? Answer:"
        )
        choices = ("red", "blue", "green")

        scores = model.cond_log_prob(inputs=prompt, targets=choices)

        print("\n")
        print(f"prompt:\n{prompt}")
        print(f"scores:")
        for c, s in zip(choices, scores):
            print(f"  {c:>8}: {s:0.2f}")

        prompts = [
            f"What color is the sky? Answer: blue\n" f"What color is grass? Answer: ",
            f"What is 1+1? Answer: 2\n" f"What is 2+2? Answer: ",
        ]
        choices = [("red", "blue", "green"), ("1", "2", "3", "4")]

        scores = model.cond_log_prob(inputs=prompts, targets=choices)

        for p, c, s in zip(prompts, choices, scores):
            print("\n")
            print(f"prompt:\n{p}")
            print(f"scores:")
            for ci, si in zip(c, s):
                print(f"  {ci:>8}: {si:0.2f}")
