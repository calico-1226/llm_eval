#!/bin/bash
model_name="google/flan-t5-large"
GPU="0,1,2,3"
batch_size=64

# alignment
echo "$GPU: $model_name - convinceme"
python eval.py --task convinceme --model_name $model_name --gpu $GPU --batch_size $batch_size
echo "$GPU: $model_name - hhh_alignment"
python eval.py --task hhh_alignment --model_name $model_name --gpu $GPU --batch_size $batch_size

# social bias
echo "$GPU: $model_name - bbq_lite_json"
python eval.py --task bbq_lite_json --model_name $model_name --gpu $GPU --batch_size $batch_size
echo "$GPU: $model_name - diverse_social_bias"
python eval.py --task diverse_social_bias --model_name $model_name --gpu $GPU --batch_size $batch_size

# racial bias
echo "$GPU: $model_name - bias_from_probabilities"
python eval.py --task bias_from_probabilities --model_name $model_name --gpu $GPU --batch_size $batch_size
echo "$GPU: $model_name - unqover"
python eval.py --task unqover --model_name $model_name --gpu $GPU --batch_size $batch_size

# gender bias
echo "$GPU: $model_name - gender_sensitivity_english"
python eval.py --task gender_sensitivity_english --model_name $model_name --gpu $GPU --batch_size $batch_size
echo "$GPU: $model_name - linguistic_mappings"
python eval.py --task linguistic_mappings --model_name $model_name --gpu $GPU --batch_size $batch_size

# religious bias
echo "$GPU: $model_name - muslim_violence_bias"
python eval.py --task muslim_violence_bias --model_name $model_name --gpu $GPU --batch_size $batch_size

# toxicity
echo "$GPU: $model_name - hinglish_toxicity"
python eval.py --task hinglish_toxicity --model_name $model_name --gpu $GPU --batch_size $batch_size

# inclusion
echo "$GPU: $model_name - gender_inclusive_sentences_german"
python eval.py --task gender_inclusive_sentences_german --model_name $model_name --gpu $GPU --batch_size $batch_size

# truthfulness
echo "$GPU: $model_name - fact_checker"
python eval.py --task fact_checker --model_name $model_name --gpu $GPU --batch_size $batch_size
echo "$GPU: $model_name - truthful_qa"
python eval.py --task truthful_qa --model_name $model_name --gpu $GPU --batch_size $batch_size

# human-like behavior
echo "$GPU: $model_name - causal_judgment"
python eval.py --task causal_judgment --model_name $model_name --gpu $GPU --batch_size $batch_size

# emotional intelligence
echo "$GPU: $model_name - dark_humor_detection"
python eval.py --task dark_humor_detection --model_name $model_name --gpu $GPU --batch_size $batch_size
