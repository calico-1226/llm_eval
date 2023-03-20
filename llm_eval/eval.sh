#!/bin/bash
# alignment
python eval_task.py --task convinceme
python eval_task.py --task hhh_alignment
# social bias
python eval_task.py --task bbq_lite_json
python eval_task.py --task diverse_social_bias
# racial bias
python eval_task.py --task bias_from_probabilities
python eval_task.py --task unqover
# gender bi
python eval_task.py --task gender_sensitivity_english
python eval_task.py --task linguistic_mappings
# religious bias
python eval_task.py --task muslim_violence_bias
# toxicity
python eval_task.py --task hinglish_toxicity
# inclusion
python eval_task.py --task gender_inclusive_sentences_german
# truthfulness
python eval_task.py --task fact_checker
python eval_task.py --task truthful_qa
# human-like behavior
python eval_task.py --task causal_judgment
# emotional intelligence
python eval_task.py --task dark_humor_detection
