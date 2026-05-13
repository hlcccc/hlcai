import argparse
import copy
import math
import os
import pathlib
import pickle
import random

import evaluate
import numpy as np
import torch
from tqdm import tqdm

import sys

sys.path.append(".")

from modules.uncertainty_utils import (
    AdaptiveThreshold,
    EarlyWarningMonitor,
    compute_generation_confidence,
    compute_layer_spread,
    compute_layer_statistics,
    compute_max_prob,
    compute_temporal_drift,
    compute_token_entropy_from_logits,
    compute_uncertainty_score,
    extract_layer_signal,
    reduce_to_token_vector,
    summarize_generation_signals,
)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

parser = argparse.ArgumentParser()
parser.add_argument("--type_of_question", type=str)
parser.add_argument("--num_generations_per_prompt", type=int, default=5)
parser.add_argument("--fraction_of_data_to_use", type=float, default=1.0)
parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--dataset", type=str, default="coqa")
parser.add_argument("--beam_search", action="store_true")

parser.add_argument("--image_folder", type=str, default="")
parser.add_argument("--question_file", type=str, default="tables/question.jsonl")
parser.add_argument("--outdir", type=str, default="/output/")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--num_chunks", type=int, default=1)
parser.add_argument("--chunk_idx", type=int, default=0)
parser.add_argument("--reason", type=str, choices=["cot", "none"], default=None)

parser.add_argument("--enable_early_warning", action="store_true", default=True)
parser.add_argument("--entropy_threshold", type=float, default=0.7)
parser.add_argument("--variance_threshold", type=float, default=0.5)
parser.add_argument("--early_stop_consecutive", type=int, default=2)
parser.add_argument("--record_uncertainty", action="store_true", default=True)
parser.add_argument(
    "--layer_strategy",
    type=str,
    default="last_layer",
    choices=["25%", "50%", "75%", "last_layer", "eos", "mean_pooling"],
)
parser.add_argument("--eval_all_layers", action="store_true", default=True)
parser.add_argument("--warning_confidence_threshold", type=float, default=0.35)
parser.add_argument("--warning_drift_threshold", type=float, default=0.25)
parser.add_argument("--warning_risk_threshold", type=float, default=0.62)
parser.add_argument("--warning_warmup_steps", type=int, default=2)
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--smoke_test", action="store_true", default=False)

args = parser.parse_args()

device = "cuda"

seed_value = 10
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

if "cogvlm" in args.model_path.lower():
    from modules.models.cogvlm_models import CogVLMModel
    model = CogVLMModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif "contactdoctor" in args.model_path.lower():
    from modules.models.biomedllama_models import BioMedLlamaModel
    model = BioMedLlamaModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif "qwen" in args.model_path.lower():
    from modules.models.qwen_models import QwenModel
    model = QwenModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
elif "liuhaotian" in args.model_path.lower():
    from modules.models.llava_models import HuggingfaceModel as LlavaModel
    model = LlavaModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)
else:
    from modules.models.vision_models import VisionModel
    model = VisionModel(model_name=args.model_path, stop_sequences=[], max_new_tokens=args.max_new_tokens)

import json
questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
if args.max_samples is not None:
    questions = questions[: args.max_samples]

if args.reason == "none" or args.reason is None:
    prefix_prompt = "Answer this question in only a word or a phrase. "
elif args.reason == "cot":
    prefix_prompt = (
        "Your task is to answer the question provided to you to the best of your abilities.\n"
        "### Format of the answer:\n"
        "<Steps for reasoning out what the answer would be>. ANSWER: <answer>\n"
        "Always state the answer by the end of your reasoning process."
        "End your response with the word 'ANSWER:' followed by the final answer."
        "Now you have understood the format and guidelines. Please answer according to the guidelines:\n"
    )
else:
    raise ValueError(f"Unknown reasoning type: {args.reason}")

rouge = evaluate.load('rouge')
exact_match_metric = evaluate.load("exact_match")

def get_layer_strategies_to_eval(hidden_states):
    if not args.eval_all_layers:
        return [args.layer_strategy]
    base = ["last_layer", "mean_pooling", "25%", "50%", "75%"]
    available = [f"layer_{idx}" for idx in range(len(hidden_states or []))]
    return available + [strategy for strategy in base if strategy not in available]


def unwrap_generation_inputs(input_data):
    cleaned = copy.deepcopy(input_data)
    cleaned.pop("input_text", None)
    if hasattr(cleaned, "to") and hasattr(cleaned, "keys"):
        return cleaned.to(device)
    for key, value in list(cleaned.items()):
        if isinstance(value, torch.Tensor):
            cleaned[key] = value.to(device)
    return cleaned


def build_forward_kwargs(model_inputs, sequence_ids):
    kwargs = {}
    for key, value in model_inputs.items():
        if key == "input_ids" or value is None:
            continue
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == sequence_ids.shape[0]:
            if key == "attention_mask":
                if value.shape[1] != sequence_ids.shape[1]:
                    pad_len = sequence_ids.shape[1] - value.shape[1]
                    if pad_len > 0:
                        pad = torch.ones(
                            value.shape[0],
                            pad_len,
                            dtype=value.dtype,
                            device=sequence_ids.device,
                        )
                        value = torch.cat([value.to(sequence_ids.device), pad], dim=1)
                    else:
                        value = value[:, : sequence_ids.shape[1]].to(sequence_ids.device)
                else:
                    value = value.to(sequence_ids.device)
            elif key == "token_type_ids":
                if value.shape[1] != sequence_ids.shape[1]:
                    pad_len = sequence_ids.shape[1] - value.shape[1]
                    if pad_len > 0:
                        tail_value = value[:, -1:].to(sequence_ids.device)
                        pad = tail_value.expand(value.shape[0], pad_len)
                        value = torch.cat([value.to(sequence_ids.device), pad], dim=1)
                    else:
                        value = value[:, : sequence_ids.shape[1]].to(sequence_ids.device)
                else:
                    value = value.to(sequence_ids.device)
            else:
                value = value.to(sequence_ids.device)
        kwargs[key] = value
    kwargs["input_ids"] = sequence_ids
    if "position_ids" in model_inputs:
        kwargs["position_ids"] = torch.arange(
            sequence_ids.shape[1],
            device=sequence_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(sequence_ids.shape[0], -1)
    kwargs["use_cache"] = False
    kwargs["return_dict"] = True
    kwargs["output_hidden_states"] = True
    return kwargs


def sample_next_token(logits, temperature, top_p):
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)

    scaled_logits = logits / max(float(temperature), 1e-5)
    probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_idx)
        filtered_probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
        log_probs = torch.log(filtered_probs.clamp(min=1e-12))
        return next_token.squeeze(-1), log_probs

    next_token = torch.multinomial(probs, num_samples=1)
    log_probs = torch.log(probs.clamp(min=1e-12))
    return next_token.squeeze(-1), log_probs


def strip_stop_sequences(text, stop_sequences):
    stop_at = len(text)
    for stop in stop_sequences or []:
        if stop and stop in text:
            stop_at = min(stop_at, text.find(stop))
    return text[:stop_at].strip()


def ensure_finite_scalar(name, value, question_id=None, generation_idx=None, step_idx=None):
    if value is None:
        return
    if np.isfinite(float(value)):
        return
    context = []
    if question_id is not None:
        context.append(f"question_id={question_id}")
    if generation_idx is not None:
        context.append(f"generation_idx={generation_idx}")
    if step_idx is not None:
        context.append(f"step={step_idx}")
    context_str = ", ".join(context)
    raise ValueError(f"Non-finite value detected for {name}" + (f" ({context_str})" if context_str else ""))


def validate_uncertainty_info(uncertainty_info, question_id=None, generation_idx=None):
    numeric_lists = [
        "token_entropies",
        "max_probabilities",
        "layer_variances",
        "layer_spreads",
        "temporal_drifts",
        "warning_scores",
        "confidence_scores",
    ]
    for key in numeric_lists:
        for step_idx, value in enumerate(uncertainty_info.get(key, [])):
            ensure_finite_scalar(key, value, question_id=question_id, generation_idx=generation_idx, step_idx=step_idx)

    summary = uncertainty_info.get("signal_summary", {}) or {}
    for key, value in summary.items():
        ensure_finite_scalar(f"signal_summary.{key}", value, question_id=question_id, generation_idx=generation_idx)

    ensure_finite_scalar(
        "combined_online_uncertainty",
        uncertainty_info.get("combined_online_uncertainty"),
        question_id=question_id,
        generation_idx=generation_idx,
    )


def predict_with_uncertainty(
    model,
    input_data,
    temperature,
    top_p,
    enable_early_warning=False,
    entropy_threshold=0.7,
    variance_threshold=0.5,
    consecutive_threshold=2,
    question_id=None,
    generation_idx=None,
):
    model_inputs = unwrap_generation_inputs(input_data)
    input_ids = model_inputs["input_ids"]
    sequence_ids = input_ids.clone()
    prompt_length = input_ids.shape[1]
    eos_token_id = model.tokenizer.eos_token_id

    uncertainty_info = {
        "token_entropies": [],
        "max_probabilities": [],
        "layer_variances": [],
        "layer_spreads": [],
        "temporal_drifts": [],
        "warning_scores": [],
        "early_stop_triggered": False,
        "early_stop_reason": None,
        "confidence_scores": [],
        "generation_steps": 0,
        "layer_strategy": args.layer_strategy,
        "layer_features_by_strategy": {},
        "signal_summary": {},
    }

    adaptive_threshold = AdaptiveThreshold(initial_alpha=1.0, learning_rate=0.01)
    monitor = None
    if enable_early_warning:
        monitor = EarlyWarningMonitor(
            entropy_threshold=entropy_threshold,
            variance_threshold=variance_threshold,
            consecutive_threshold=consecutive_threshold,
            confidence_threshold=args.warning_confidence_threshold,
            drift_threshold=args.warning_drift_threshold,
            risk_threshold=args.warning_risk_threshold,
            warmup_steps=args.warning_warmup_steps,
        )

    layer_trajectories = {}
    log_likelihoods = []
    most_recent_hidden_states = None

    with torch.no_grad():
        for step_idx in range(model.max_new_tokens):
            outputs = model.model(**build_forward_kwargs(model_inputs, sequence_ids))
            logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states
            most_recent_hidden_states = hidden_states

            next_token, token_log_probs = sample_next_token(logits, temperature=temperature, top_p=top_p)
            next_token = next_token.reshape(1)
            sampled_log_prob = token_log_probs[0, next_token.item()].item()
            log_likelihoods.append(float(sampled_log_prob))
            sequence_ids = torch.cat([sequence_ids, next_token.view(1, 1).to(sequence_ids.device)], dim=1)

            token_entropy = compute_token_entropy_from_logits(logits)
            max_prob = compute_max_prob(logits)
            confidence = compute_generation_confidence(token_entropy, max_prob)
            if args.smoke_test:
                ensure_finite_scalar(
                    "token_entropy",
                    token_entropy,
                    question_id=question_id,
                    generation_idx=generation_idx,
                    step_idx=step_idx,
                )
                ensure_finite_scalar(
                    "max_prob",
                    max_prob,
                    question_id=question_id,
                    generation_idx=generation_idx,
                    step_idx=step_idx,
                )
                ensure_finite_scalar(
                    "confidence",
                    confidence,
                    question_id=question_id,
                    generation_idx=generation_idx,
                    step_idx=step_idx,
                )
            uncertainty_info["token_entropies"].append(float(token_entropy))
            uncertainty_info["max_probabilities"].append(float(max_prob))
            uncertainty_info["confidence_scores"].append(float(confidence))

            available_strategies = get_layer_strategies_to_eval(hidden_states)
            step_feature_snapshot = {}
            step_layer_vectors = []
            step_variances = []

            for strategy in available_strategies:
                layer_state = extract_layer_signal(hidden_states, strategy, sequence_length=sequence_ids.shape[1])
                layer_vector = reduce_to_token_vector(layer_state)
                if layer_vector is None:
                    continue
                step_layer_vectors.append(layer_vector)
                step_variances.append(float(layer_vector.var(unbiased=False).item()))
                layer_trajectories.setdefault(strategy, []).append(layer_vector)
                stacked_history = torch.stack(layer_trajectories[strategy], dim=0)
                step_feature_snapshot[strategy] = compute_layer_statistics(stacked_history)

            uncertainty_info["layer_features_by_strategy"] = step_feature_snapshot

            layer_spread = compute_layer_spread(step_layer_vectors)
            drift_values = []
            for trajectory in layer_trajectories.values():
                if len(trajectory) >= 2:
                    drift_values.append(compute_temporal_drift(torch.stack(trajectory, dim=0)))
            mean_drift = float(np.mean(drift_values)) if drift_values else 0.0
            mean_variance = float(np.mean(step_variances)) if step_variances else 0.0

            uncertainty_info["layer_variances"].append(mean_variance)
            uncertainty_info["layer_spreads"].append(layer_spread)
            uncertainty_info["temporal_drifts"].append(mean_drift)

            if monitor is not None:
                should_stop, reason, snapshot = monitor.update(
                    entropy=token_entropy,
                    variance=mean_variance,
                    confidence_gap=1.0 - confidence,
                    layer_spread=layer_spread,
                    drift=mean_drift,
                )
                if args.smoke_test:
                    ensure_finite_scalar(
                        "warning_score",
                        snapshot["risk_score"],
                        question_id=question_id,
                        generation_idx=generation_idx,
                        step_idx=step_idx,
                    )
                uncertainty_info["warning_scores"].append(snapshot["risk_score"])
                adaptive_threshold.update(snapshot["risk_score"] - 0.5)
                if should_stop:
                    uncertainty_info["early_stop_triggered"] = True
                    uncertainty_info["early_stop_reason"] = reason
                    break
            else:
                uncertainty_info["warning_scores"].append(0.0)

            if next_token.item() == eos_token_id:
                break

            decoded_partial = model.tokenizer.decode(sequence_ids[0, prompt_length:], skip_special_tokens=False)
            if any(stop in decoded_partial for stop in (model.stop_sequences or []) if stop):
                break

    prompt_text = model.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    decoded_full = model.tokenizer.decode(sequence_ids[0], skip_special_tokens=False)
    if decoded_full.startswith(prompt_text):
        decoded_text = decoded_full[len(prompt_text) :]
    else:
        decoded_text = model.tokenizer.decode(sequence_ids[0, prompt_length:], skip_special_tokens=False)
    decoded_text = strip_stop_sequences(decoded_text, model.stop_sequences)

    if most_recent_hidden_states is not None:
        final_state = extract_layer_signal(
            most_recent_hidden_states,
            args.layer_strategy,
            sequence_length=sequence_ids.shape[1],
        )
        final_vector = reduce_to_token_vector(final_state)
    else:
        final_vector = None
    if final_vector is None:
        final_vector = torch.zeros(1)

    uncertainty_info["generation_steps"] = len(uncertainty_info["token_entropies"])
    uncertainty_info["signal_summary"] = summarize_generation_signals(
        uncertainty_info["token_entropies"],
        uncertainty_info["confidence_scores"],
        uncertainty_info["layer_spreads"],
        uncertainty_info["temporal_drifts"],
        uncertainty_info["warning_scores"],
    )
    uncertainty_info["combined_online_uncertainty"] = compute_uncertainty_score(
        uncertainty_info["signal_summary"]["entropy_signal"],
        1.0 - uncertainty_info["signal_summary"]["confidence_signal"],
        float(uncertainty_info["early_stop_triggered"]),
        layer_instability=uncertainty_info["signal_summary"]["layer_instability"],
        disagreement=uncertainty_info["signal_summary"]["warning_signal"],
    )
    if args.smoke_test:
        validate_uncertainty_info(
            uncertainty_info,
            question_id=question_id,
            generation_idx=generation_idx,
        )

    return decoded_text, log_likelihoods, final_vector.cpu(), uncertainty_info

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
            enable_early_warning=False,
            question_id=idx,
            generation_idx=-1,
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
                consecutive_threshold=args.early_stop_consecutive,
                question_id=idx,
                generation_idx=i,
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
    sequence_dict['online_uncertainty'] = float(
        np.mean([u.get('combined_online_uncertainty', 0.0) for u in all_uncertainty_info])
    )

    sequences.append(sequence_dict)

pathlib.Path(f'{args.outdir}').mkdir(parents=True, exist_ok=True)

with open(f'{args.outdir}/generations_with_uncertainty.pkl', 'wb') as outfile:
    pickle.dump(sequences, outfile)

summary = {
    "total_samples": len(sequences),
    "num_generations_per_prompt": number_of_generations,
    "smoke_test": bool(args.smoke_test),
    "samples_with_early_stop": int(sum(1 for seq in sequences if seq.get("early_stop_count", 0) > 0)),
    "avg_early_stop_rate": float(np.mean([seq.get("early_stop_rate", 0.0) for seq in sequences])) if sequences else 0.0,
    "avg_online_uncertainty": float(np.mean([seq.get("online_uncertainty", 0.0) for seq in sequences])) if sequences else 0.0,
}
with open(f"{args.outdir}/smoke_test_summary.json", "w") as outfile:
    json.dump(summary, outfile, indent=2)

print("Done!")
print(f"Generated {len(sequences)} sequences with uncertainty metrics")
