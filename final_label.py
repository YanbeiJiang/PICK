import json
import os
import math
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
)
from scipy.stats import ttest_rel


def load_json_data(file_paths):
    data_from_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_from_files.append(data)
        else:
            print(f"Warning: File {file_path} does not exist")
    return data_from_files


def calculate_distribution_entropy(distribution):
    entropy = 0
    for prob in distribution.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


def combine_data(object_data, relation_data, global_data):
    combined_results = {}
    true_labels = {}
    predictions = {}

    for item in global_data["global_results"]:
        filename = item["filename"]
        prefix = filename.split('/')[-1].split('.jpg')[0]
        true_labels[prefix] = 0 if item["label"] == "positive" else 1

        confidence = item["confidence"]
        global_distribution = {
            "pos": confidence if item["prediction"] == "positive" else 1 - confidence,
            "neg": 1 - confidence if item["prediction"] == "positive" else confidence
        }
        global_entropy = calculate_distribution_entropy(global_distribution)
        combined_results[prefix] = {
            "global": {
                "distribution": global_distribution,
                "weight": 1,
                "entropy": global_entropy
            },
            "object": None,
            "relation": None
        }

    for prefix, data in object_data["prefix_level"].items():
        if prefix in combined_results:
            object_entropy = calculate_distribution_entropy(data["final_distribution"])
            combined_results[prefix]["object"] = {
                "distribution": data["final_distribution"],
                "weight": data["count"] * 4,
                "entropy": object_entropy
            }

    for prefix, data in relation_data["prefix_level"].items():
        if prefix in combined_results:
            relation_entropy = calculate_distribution_entropy(data["final_distribution"])
            combined_results[prefix]["relation"] = {
                "distribution": data["final_distribution"],
                "weight": data["count"],
                "entropy": relation_entropy
            }

    for prefix, data in combined_results.items():
        total_weight = 0
        pos_weighted_sum = 0
        neg_weighted_sum = 0

        for level_data in data.values():
            if level_data is not None:
                effective_weight = level_data["weight"] * (1 - level_data["entropy"])
                total_weight += effective_weight
                pos_weighted_sum += effective_weight * level_data["distribution"]["pos"]
                neg_weighted_sum += effective_weight * level_data["distribution"]["neg"]

        if total_weight > 0:
            final_pos = pos_weighted_sum / total_weight
            final_neg = neg_weighted_sum / total_weight
            norm_factor = final_pos + final_neg
            if norm_factor > 0:
                final_pos /= norm_factor
                final_neg /= norm_factor
            final_prediction = 0 if final_pos > final_neg else 1
            predictions[prefix] = final_prediction

    prefixes = sorted([prefix for prefix in true_labels if prefix in predictions])
    y_true_list = [true_labels[prefix] for prefix in prefixes]
    y_pred_list = [predictions[prefix] for prefix in prefixes]
    return y_true_list, y_pred_list


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {item["filename"]: (item["label"], item["prediction"])
            for item in data["results"]}


def main(args):
    with open(args.object_level, "r") as f:
        object_data = json.load(f)
    with open(args.relation_level, "r") as f:
        relation_data = json.load(f)
    with open(args.global_level, "r") as f:
        global_data = json.load(f)

    valid_true, valid_pred = combine_data(object_data, relation_data, global_data)

    baseline_dict = load_results(args.baseline_path)
    y_pred_base = []
    y_true = []
    for fname in baseline_dict:
        label_b, pred_b = baseline_dict[fname]
        y_pred_base.append(pred_b)
        y_true.append(label_b)

    y_true = np.array(y_true)
    y_pred_base = np.array(y_pred_base)
    y_true_ours = np.asarray(valid_true)
    y_pred_ours = np.asarray(valid_pred)

    labels = [0, 1]
    n_iter = 10
    rng = np.random.default_rng(42)

    metrics = {cls: {m: {"base": [], "ours": []}
                     for m in ["accuracy", "precision", "recall", "f1"]}
               for cls in labels}

    for _ in range(n_iter):
        idx = rng.choice(len(y_true_ours), size=len(y_true_ours), replace=True)
        y_t = y_true[idx]
        y_t_ours = y_true_ours[idx]
        y_b = y_pred_base[idx]
        y_o = y_pred_ours[idx]

        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(
            y_t, y_b, labels=labels, zero_division=0
        )
        prec_o, rec_o, f1_o, _ = precision_recall_fscore_support(
            y_t_ours, y_o, labels=labels, zero_division=0
        )

        acc_b = rec_b
        acc_o = rec_o

        for i, cls in enumerate(labels):
            metrics[cls]["precision"]["base"].append(prec_b[i])
            metrics[cls]["precision"]["ours"].append(prec_o[i])
            metrics[cls]["recall"]["base"].append(rec_b[i])
            metrics[cls]["recall"]["ours"].append(rec_o[i])
            metrics[cls]["f1"]["base"].append(f1_b[i])
            metrics[cls]["f1"]["ours"].append(f1_o[i])
            metrics[cls]["accuracy"]["base"].append(acc_b[i])
            metrics[cls]["accuracy"]["ours"].append(acc_o[i])

    for cls in labels:
        print(f"\n=== Class {cls} ===")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            base_samples = np.array(metrics[cls][metric]["base"])
            ours_samples = np.array(metrics[cls][metric]["ours"])
            t, p = ttest_rel(ours_samples, base_samples, alternative="greater")
            print(
                f"{metric:9s}  "
                f"mean(base)={base_samples.mean():.4f}  "
                f"mean(ours)={ours_samples.mean():.4f}  "
                f"p‑value={p:.10f}"
            )

    accuracy_0 = accuracy_score(valid_true, valid_pred)
    precision_0 = precision_score(valid_true, valid_pred, zero_division=0, pos_label=0)
    recall_0 = recall_score(valid_true, valid_pred, zero_division=0, pos_label=0)
    f1_0 = f1_score(valid_true, valid_pred, pos_label=0)
    accuracy_1 = accuracy_score(valid_true, valid_pred)
    precision_1 = precision_score(valid_true, valid_pred, zero_division=0, pos_label=1)
    recall_1 = recall_score(valid_true, valid_pred, zero_division=0, pos_label=1)
    f1_1 = f1_score(valid_true, valid_pred, pos_label=1)

    summary = {
        "accuracy_0": accuracy_0,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "accuracy_1": accuracy_1,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
    }

    with open(args.output_json, "w") as f:
        json.dump({"valid_true": valid_true, "valid_pred": valid_pred, "summary": summary}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object+relation+global emotional reasoning")
    parser.add_argument("--object_level", type=str, default="emotional_classification_results_object.json",
                        help="Path to the object-level JSON file")
    parser.add_argument("--relation_level", type=str, default="emotional_classification_results_relation.json",
                        help="Path to the relation-level JSON file")
    parser.add_argument("--global_level", type=str, default="emotional_classification_results_global.json",
                        help="Path to the global-level JSON file")
    parser.add_argument("--baseline_path", type=str, default="。/outputs/gemini.json",
                        help="Path to baseline result JSON")
    parser.add_argument("--output_json", type=str, default="./outputs/ours_object_relation_global.json",
                        help="Path to save the final evaluation result")
    args = parser.parse_args()
    main(args)
