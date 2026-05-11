import argparse
import os
import pathlib
import pickle
import random
import evaluate
import numpy as np
import torch
from tqdm import tqdm
import math

import sys
sys.path.append(".")

from modules.uncertainty_utils import (
    compute_entropy,
    compute_sequence_entropy,
    compute_layer_wise_variance,
    compute_uncertainty_score,
    EarlyWarningMonitor,
    AdaptiveThreshold,
    compute_generation_confidence,
    compute_max_prob,
    extract_intermediate_states,
    extract_layer_signal,
    get_mean_pooling,
    get_last_layer
)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model_path', type=str, default='facebook/opt-350m')
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument("--beam_search", action='store_true')

parser.add_argument("--image_folder", type=str, default="")
parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
parser.add_argument("--outdir", type=str, default="/output/")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--num_chunks", type=int, default=1)
parser.add_argument("--chunk_idx", type=int, default=0)
parser.add_argument("--reason", type=str, choices=['cot', 'none'], default=None)

parser.add_argument("--enable_early_warning", action='store_true', default=True,
                   help='Enable early warning mechanism based on uncertainty')
parser.add_argument("--entropy_threshold", type=float, default=0.7,
                   help='Threshold for entropy-based early warning')
parser.add_argument("--variance_threshold", type=float, default=0.5,
                   help='Threshold for variance-based early warning')
parser.add_argument("--early_stop_consecutive", type=int, default=2,
                   help='Number of consecutive high uncertainty steps before early stop')
parser.add_argument("--record_uncertainty", action='store_true', default=True,
                   help='Record uncertainty metrics for each generation')

parser.add_argument("--layer_strategy", type=str, default='last_layer',
                   choices=['25%', '50%', '75%', 'last_layer', 'eos', 'mean_pooling'],
                   help='Strategy for extracting hidden states from layers')
parser.add_argument("--eval_all_layers", action='store_true', default=False,
                   help='Evaluate all layer extraction strategies simultaneously')

args = parser.parse_args()

device = 'cuda'

seed_value = 10
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if 'cogvlm' in args.model_path.lower():
    from modules.models.cogvlm_models import CogVLMModel
    model = CogVLMModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif 'contactdoctor' in args.model_path.lower():
    from modules.models.biomedllama_models import BioMedLlamaModel
    model = BioMedLlamaModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif 'qwen' in args.model_path.lower():
    from modules.models.qwen_models import QwenModel
    model = QwenModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif 'liuhaotian' in args.model_path.lower():
    from modules.models.llava_models import HuggingfaceModel as LlavaModel
    model = LlavaModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
else:
    from modules.models.vision_models import VisionModel
    model = VisionModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)

import json
questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

if args.reason == 'none' or args.reason is None:
    prefix_prompt = 'Answer this question in only a word or a phrase. '
elif args.reason == 'cot':
    prefix_prompt = "Your task is to answer the question provided to you to the best of your abilities.\n" \
                    "### Format of the answer:\n" \
                    "<Steps for reasoning out what the answer would be>. ANSWER: <answer>\n" \
                    "Always state the answer by the end of your reasoning process." \
                    "End your response with the word 'ANSWER:' followed by the final answer." \
                    "Now you have understood the format and guidelines. Please answer according to the guidelines:\n"
else:
    raise ValueError(f"Unknown reasoning type: {args.reason}")

rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

def predict_with_uncertainty(model, input_data, temperature, top_p, enable_early_warning=False,
                               entropy_threshold=0.7, variance_threshold=0.5,
                               consecutive_threshold=2, early_stop=False):
    """
    Extended predict function that also computes uncertainty metrics.
    Returns: (answer, log_likelihoods, embeddings, uncertainty_info)
    """
    input_ids = input_data['input_ids'].to(device=device, non_blocking=True)
    input_text = model.tokenizer.batch_decode(input_ids)[0]
    pad_token_id = model.tokenizer.eos_token_id

    if model.stop_sequences is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList
        stopping_criteria = StoppingCriteriaList([
            __import__('modules.models.vision_models', fromlist=['StoppingCriteriaSub']).StoppingCriteriaSub(
                stops=model.stop_sequences,
                initial_length=len(input_ids[0]),
                tokenizer=model.tokenizer
            )
        ])
    else:
        stopping_criteria = None

    generation_input_data = {k: v for k, v in input_data.items() if k != 'input_text'}

    with torch.no_grad():
        outputs = model.model.generate(
            **generation_input_data,
            max_new_tokens=model.max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
        )

    if len(outputs.sequences[0]) > model.token_limit:
        raise ValueError(
            'Generation exceeding token limit %d > %d',
            len(outputs.sequences[0]), model.token_limit)

    full_answer_list = model.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)

    if full_answer_list[0].startswith(input_text):
        input_data_offset = len(input_text)
        n_input_token = model.tokenizer(input_text, return_tensors="pt")['input_ids'].shape[1]
    else:
        input_data_offset = 0
        n_input_token = 1

    answer_list = [full_answer[input_data_offset:] for full_answer in full_answer_list]
    sliced_answer_list = []
    last_token_embedding_list = []
    log_likelihoods_list = []

    transition_scores = model.model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    uncertainty_info = {
        'token_entropies': [],
        'max_probabilities': [],
        'layer_variances': [],
        'early_stop_triggered': False,
        'early_stop_reason': None,
        'confidence_scores': [],
        'generation_steps': 0,
        'layer_strategy': args.layer_strategy
    }
    
    layer_strategies_to_eval = []
    if args.eval_all_layers:
        layer_strategies_to_eval = ['25%', '50%', '75%', 'last_layer', 'eos', 'mean_pooling']
    else:
        layer_strategies_to_eval = [args.layer_strategy]
    
    uncertainty_info['layer_variances_by_strategy'] = {}
    uncertainty_info['layer_rep_norm_by_strategy'] = {}
    uncertainty_info['layer_eigen_score_by_strategy'] = {}

    if args.enable_early_warning:
        monitor = EarlyWarningMonitor(
            entropy_threshold=entropy_threshold,
            variance_threshold=variance_threshold,
            consecutive_threshold=consecutive_threshold
        )

    for ans_id, answer in enumerate(answer_list):
        stop_at = len(answer)
        sliced_answer = answer
        if model.stop_sequences is not None:
            for stop in model.stop_sequences:
                if stop in answer:
                    stop_at = answer.find(stop)
                    sliced_answer = answer[:stop_at]
                    break

        sliced_answer = sliced_answer.strip()
        sliced_answer_list.append(sliced_answer)

        token_stop_index = model.tokenizer(
            full_answer_list[ans_id][:input_data_offset + stop_at],
            return_tensors="pt"
        )['input_ids'].shape[1]
        n_generated = token_stop_index - n_input_token

        if n_generated == 0:
            n_generated = 1

        if 'decoder_hidden_states' in outputs.keys():
            hidden = outputs.decoder_hidden_states
        else:
            hidden = outputs.hidden_states

        if len(hidden) == 1:
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            last_input = hidden[-1]
        else:
            if len(hidden) > n_generated:
                last_input = hidden[n_generated]
            else:
                last_input = hidden[n_generated - 1]

        last_token_embedding_list.append(last_input[-1][ans_id][-1, :].cpu())

        log_likelihoods = [score.item() for score in transition_scores[ans_id]]
        if len(log_likelihoods) == 1:
            log_likelihoods = log_likelihoods
        else:
            if len(log_likelihoods) > n_generated:
                log_likelihoods = log_likelihoods[:n_generated+1]
            else:
                log_likelihoods = log_likelihoods[:n_generated]
        if len(log_likelihoods) == 0:
            pass

        log_likelihoods_list.append(log_likelihoods)

        if args.record_uncertainty and outputs.scores is not None:
            token_entropies = compute_sequence_entropy(outputs.scores)
            uncertainty_info['token_entropies'] = token_entropies
            uncertainty_info['generation_steps'] = len(token_entropies)

            for logits in outputs.scores[:len(token_entropies)]:
                max_prob = compute_max_prob(logits)
                uncertainty_info['max_probabilities'].append(max_prob)

            if hidden is not None:
                layer_variances = compute_layer_wise_variance(hidden)
                uncertainty_info['layer_variances'] = layer_variances
                
                for strategy in layer_strategies_to_eval:
                    extracted_state = extract_layer_signal(hidden, strategy, n_generated)
                    if extracted_state is not None and isinstance(extracted_state, torch.Tensor):
                        var = extracted_state.var(dim=0).mean().item()
                        rep_norm = extracted_state.norm(dim=-1).mean().item()
                        
                        try:
                            mat = extracted_state @ extracted_state.T
                            eigen_score = torch.linalg.eigvals(mat).abs().mean().item()
                        except:
                            eigen_score = 0.0
                        
                        uncertainty_info['layer_variances_by_strategy'][strategy] = var
                        uncertainty_info['layer_rep_norm_by_strategy'][strategy] = rep_norm
                        uncertainty_info['layer_eigen_score_by_strategy'][strategy] = eigen_score
                    else:
                        uncertainty_info['layer_variances_by_strategy'][strategy] = 0.0
                        uncertainty_info['layer_rep_norm_by_strategy'][strategy] = 0.0
                        uncertainty_info['layer_eigen_score_by_strategy'][strategy] = 0.0

            for i, entropy_val in enumerate(token_entropies):
                max_prob_val = uncertainty_info['max_probabilities'][i] if i < len(uncertainty_info['max_probabilities']) else 0.5
                confidence = compute_generation_confidence(entropy_val, max_prob_val)
                uncertainty_info['confidence_scores'].append(confidence)

                if args.enable_early_warning:
                    variance_val = layer_variances[i] if i < len(layer_variances) else 0.0
                    monitor.update(entropy_val, variance_val)
                    should_stop, reason = monitor.check_early_stop()
                    if should_stop:
                        uncertainty_info['early_stop_triggered'] = True
                        uncertainty_info['early_stop_reason'] = reason
                        break

    if len(sliced_answer_list) == 1:
        return sliced_answer_list[0], log_likelihoods_list[0], last_token_embedding_list[0], uncertainty_info
    else:
        last_token_embedding_list = torch.stack(last_token_embedding_list).permute(1, 0, 2)
        return sliced_answer_list, log_likelihoods_list, last_token_embedding_list, uncertainty_info

sequences = []
number_of_generations = args.num_generations_per_prompt

for line in tqdm(questions, total=len(questions)):
    idx = line["question_id"]
    cur_prompt = prefix_prompt + line["text"]
    image_name = line['image']
    image_path = os.path.join(args.image_folder, image_name)

    most_likely_output_text, most_likely_log_likelihood, most_likely_embedding, most_likely_uncertainty = \
        predict_with_uncertainty(
            model, model.process_input(cur_prompt, image_path),
            temperature=0.1, top_p=0.9,
            enable_early_warning=False
        )

    generation_list = []
    generation_log_likelihood_list = []
    embedding_list = []
    all_uncertainty_info = []

    for i in range(number_of_generations):
        gen_output, gen_log_likelihood, gen_embedding, uncertainty_info = \
            predict_with_uncertainty(
                model, model.process_input(cur_prompt, image_path),
                temperature=args.temperature, top_p=args.top_p,
                enable_early_warning=args.enable_early_warning,
                entropy_threshold=args.entropy_threshold,
                variance_threshold=args.variance_threshold,
                consecutive_threshold=args.early_stop_consecutive
            )
        generation_list.append(gen_output)
        generation_log_likelihood_list.append(gen_log_likelihood)
        embedding_list.append(gen_embedding)
        all_uncertainty_info.append(uncertainty_info)

    embedding_array = np.array(torch.stack(embedding_list).tolist())

    sequence_dict = {
        'question_id': idx,
        'question_text': cur_prompt,
        'image': line['image'],
    }

    sequence_dict['generations_text'] = generation_list
    sequence_dict['generations_log_likelihood'] = generation_log_likelihood_list
    sequence_dict['most_likely_generation_text'] = most_likely_output_text
    sequence_dict['most_likely_generation_log_likelihood'] = most_likely_log_likelihood

    reference_answers = line['answers']

    rouge_types = ['rouge1', 'rouge2', 'rougeL']
    for rouge_type in rouge_types:
        if rouge_type in line:
            sequence_dict[rouge_type + '_reference_answers'] = line[rouge_type]
        else:
            sequence_dict[rouge_type + '_reference_answers'] = None
        sequence_dict[rouge_type + '_to_target'] = 0.0

    sequence_dict['exact_match'] = 0.0
    sequence_dict['answers'] = reference_answers
    for answer in reference_answers:
        predictions = [sequence_dict['most_likely_generation_text'].lstrip()]
        references = [answer]
        results = exact_match_metric.compute(
            predictions=predictions,
            references=references,
            ignore_case=True,
            ignore_punctuation=True
        )
        sequence_dict['exact_match'] = max(results['exact_match'], sequence_dict['exact_match'])
        rouge_results = rouge.compute(predictions=predictions, references=references)
        for rouge_type in rouge_types:
            sequence_dict[rouge_type + '_to_target'] = max(
                rouge_results[rouge_type],
                sequence_dict[rouge_type + '_to_target']
            )

    sequence_dict['internal_embedding'] = embedding_array
    sequence_dict['uncertainty_info'] = all_uncertainty_info

    early_stop_count = sum(1 for u in all_uncertainty_info if u.get('early_stop_triggered', False))
    sequence_dict['early_stop_count'] = early_stop_count
    sequence_dict['early_stop_rate'] = early_stop_count / number_of_generations if number_of_generations > 0 else 0

    sequences.append(sequence_dict)

pathlib.Path(f'{args.outdir}').mkdir(parents=True, exist_ok=True)

with open(f'{args.outdir}/generations_with_uncertainty.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)

print("Done!")
print(f"Generated {len(sequences)} sequences with uncertainty metrics")
