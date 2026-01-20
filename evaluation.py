import torch
from torch.utils.data import DataLoader, Subset
from train import MILModel
from data import BagDataset, process_labels
from sklearn.metrics import accuracy_score, f1_score
import os
import pandas as pd
import numpy as np
import krippendorff as kd
import argparse
import matplotlib.pyplot as plt

from visualization import setup_plots

def remove_nans():
    nas = []  
    for experiment in sorted(os.listdir("experiments/final_eval/")):
        df = pd.read_csv(f"experiments/final_eval/{experiment}")
        nas.append(df[df["panel_label_selected"].isna()]["block_id"].values.tolist())
    flat_nas = [item for sublist in nas for item in sublist]
    remove = pd.Series(flat_nas).unique()

    for experiment in sorted(os.listdir("experiments/final_eval/")):
            df = pd.read_csv(f"experiments/final_eval/{experiment}")
            df = df[~df["block_id"].isin(remove)]
            print(f"Removed {len(remove)} instances from {experiment} due to NaN values in panel_label_selected.")
            df.to_csv(f"experiments/final_eval/{experiment}", index=False)



def bootstrap_all_metrics(df, num_iterations=2000, alpha=0.05):
    n = len(df)
    metrics = {
        "acc_cons": [], "acc_v5": [], "acc_v20": [],
        "f1_cons": [], "f1_v5": [], "f1_v20": [],
        "alpha_cons": [], "alpha_v5": [], "alpha_v20": []
    }

    for _ in range(num_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        df_sample = df.iloc[idx]

        # Accuracy
        metrics["acc_cons"].append(accuracy_score(df_sample["cons_label"], df_sample["pred_class"]))
        metrics["acc_v5"].append(accuracy_score(df_sample["panel_label_selected"], df_sample["pred_class"]))
        metrics["acc_v20"].append(accuracy_score(df_sample["panel_label_all"], df_sample["pred_class"]))

        # F1 macro
        metrics["f1_cons"].append(f1_score(df_sample["cons_label"], df_sample["pred_class"], average="macro", zero_division=0))
        metrics["f1_v5"].append(f1_score(df_sample["panel_label_selected"], df_sample["pred_class"], average="macro",zero_division=0))
        metrics["f1_v20"].append(f1_score(df_sample["panel_label_all"], df_sample["pred_class"], average="macro",zero_division=0))

        # Krippendorff's alpha
        metrics["alpha_cons"].append(kd.alpha(df_sample[["cons_label", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0]))
        metrics["alpha_v5"].append(kd.alpha(df_sample[["panel_label_selected", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0]))
        metrics["alpha_v20"].append(kd.alpha(df_sample[["panel_label_all", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0]))

    summary = {}
    for k, values in metrics.items():
        mean = np.mean(values)
        ci_low = np.percentile(values, 100 * alpha / 2)
        ci_high = np.percentile(values, 100 * (1 - alpha / 2))
        summary[k] = mean
        summary[k + "_dev"] = (ci_high - ci_low) / 2
    return summary



def calculate_agreement():
    results = []

    for experiment in sorted(os.listdir("experiments/final_eval/")):
        print(f"Processing {experiment}...")
        df = pd.read_csv(f"experiments/final_eval/{experiment}")
        experiment = experiment.replace("virtual_", "virtual")

        summary = bootstrap_all_metrics(df)

        results.append(
            {
                "experiment": experiment.replace(".csv", ""),
                "model_strat": experiment.split("_")[0],
                "panel_strat": experiment.split("_")[1],

                "acc_consensus": summary["acc_cons"],
                "acc_consensus_dev": summary["acc_cons_dev"],
                "acc_virtual_5": summary["acc_v5"],
                "acc_virtual_5_dev": summary["acc_v5_dev"],
                "acc_virtual_20": summary["acc_v20"],
                "acc_virtual_20_dev": summary["acc_v20_dev"],

                "f1_consensus": summary["f1_cons"],
                "f1_consensus_dev": summary["f1_cons_dev"],
                "f1_virtual_5": summary["f1_v5"],
                "f1_virtual_5_dev": summary["f1_v5_dev"],
                "f1_virtual_20": summary["f1_v20"],
                "f1_virtual_20_dev": summary["f1_v20_dev"],
                "avg_f1": (summary["f1_cons"] + summary["f1_v5"] + summary["f1_v20"]) / 3,

                "alpha_consensus": summary["alpha_cons"],
                "alpha_consensus_dev": summary["alpha_cons_dev"],
                "alpha_virtual_5": summary["alpha_v5"],
                "alpha_virtual_5_dev": summary["alpha_v5_dev"],
                "alpha_virtual_20": summary["alpha_v20"],
                "alpha_virtual_20_dev": summary["alpha_v20_dev"],

                "avg_acc": (summary["acc_cons"] + summary["acc_v5"] + summary["acc_v20"]) / 3,
                "avg_alpha": (summary["alpha_cons"] + summary["alpha_v5"] + summary["alpha_v20"]) / 3,

                "mean_entropy": np.mean(df["entropy"]),
                "mean_frequency_entropy": np.mean(df["frequency_entropy"])
            }
        )

    pd.DataFrame(results).round(4).to_csv("experiments/acc_alpha_summary.csv", index=False)


# def calculate_agreement_certain(entropy_threshold):
#     results = []
#     for experiment in sorted(os.listdir("experiments/final_eval/")):
#             df = pd.read_csv(f"experiments/final_eval/{experiment}")
#             experiment = experiment.replace("virtual_", "virtual")
#             df = df.dropna(subset=["panel_label_selected"]) # just in case there actually are instances not having been rated by a single pathologist in the panel
#             removed_instances = df[df["entropy"] > entropy_threshold]
#             df = df[df["entropy"] <= entropy_threshold]  
#             acc_cons = accuracy_score(df["cons_label"], df["pred_class"])
#             acc_virtual20 = accuracy_score(df["panel_label_all"], df["pred_class"])
#             alpha_cons = kd.alpha(df[["cons_label", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
#             alpha_virtual5 = kd.alpha(df[["panel_label_selected", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
#             acc_virtual5 = accuracy_score(df["panel_label_selected"], df["pred_class"])
#             alpha_virtual20 = kd.alpha(df[["panel_label_all", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
#             avg_acc = (acc_cons + acc_virtual5 + acc_virtual20) / 3
#             avg_alpha = (alpha_cons + alpha_virtual5 + alpha_virtual20) / 3

#             results.append(
#                 {
#                     "experiment": experiment.replace(".csv", ""),
#                     "model_strat": experiment.split("_")[0],
#                     "panel_strat": experiment.split("_")[1],
#                     "acc_consensus": acc_cons,
#                     "acc_virtual_5": acc_virtual5,
#                     "acc_virtual_20": acc_virtual20,
#                     "alpha_consensus": alpha_cons,
#                     "alpha_virtual_5": alpha_virtual5,
#                     "alpha_virtual_20": alpha_virtual20,
#                     "avg_acc": avg_acc,
#                     "avg_alpha": avg_alpha,
#                     "removed instances": len(removed_instances),
#                     "rm_class_0": len(removed_instances[removed_instances["cons_label"] == 0]),
#                     "rm_class_1": len(removed_instances[removed_instances["cons_label"] == 1]),
#                     "rm_class_2": len(removed_instances[removed_instances["cons_label"] == 2]),
#                     "remainining_class_0": len(df[df["cons_label"] == 0]),
#                     "remaining_class_1": len(df[df["cons_label"] == 1]),
#                     "remaining_class_2": len(df[df["cons_label"] == 2]),
#                     "mean_entropy": np.mean(df["entropy"])
#                 }
#             )

#     pd.DataFrame(results).to_csv(f"experiments/certain_{round(entropy_threshold, 2)}_acc_alpha_summary.csv", index=False)

# def calculate_agreement_uncertain(entropy_threshold):
#     results = []
#     for experiment in sorted(os.listdir("experiments/final_eval/")):
#             df = pd.read_csv(f"experiments/final_eval/{experiment}")
#             experiment = experiment.replace("virtual_", "virtual")
#             df = df.dropna(subset=["panel_label_selected"]) # just in case there actually are instances not having been rated by a single pathologist in the panel
#             removed_instances = df[df["entropy"] < entropy_threshold]
#             df = df[df["entropy"] >= entropy_threshold]  
#             acc_cons = accuracy_score(df["cons_label"], df["pred_class"])
#             acc_virtual20 = accuracy_score(df["panel_label_all"], df["pred_class"])
#             alpha_cons = kd.alpha(df[["cons_label", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
#             alpha_virtual5 = kd.alpha(df[["panel_label_selected", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
#             acc_virtual5 = accuracy_score(df["panel_label_selected"], df["pred_class"])
#             alpha_virtual20 = kd.alpha(df[["panel_label_all", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
#             avg_acc = (acc_cons + acc_virtual5 + acc_virtual20) / 3
#             avg_alpha = (alpha_cons + alpha_virtual5 + alpha_virtual20) / 3

#             results.append(
#                 {
#                     "experiment": experiment.replace(".csv", ""),
#                     "model_strat": experiment.split("_")[0],
#                     "panel_strat": experiment.split("_")[1],
#                     "acc_consensus": acc_cons,
#                     "acc_virtual_5": acc_virtual5,
#                     "acc_virtual_20": acc_virtual20,
#                     "alpha_consensus": alpha_cons,
#                     "alpha_virtual_5": alpha_virtual5,
#                     "alpha_virtual_20": alpha_virtual20,
#                     "avg_acc": avg_acc,
#                     "avg_alpha": avg_alpha,
#                     "removed instances": len(removed_instances),
#                     "rm_class_0": len(removed_instances[removed_instances["cons_label"] == 0]),
#                     "rm_class_1": len(removed_instances[removed_instances["cons_label"] == 1]),
#                     "rm_class_2": len(removed_instances[removed_instances["cons_label"] == 2]),
#                     "remaining_class_0": len(df[df["cons_label"] == 0]),
#                     "remaining_class_1": len(df[df["cons_label"] == 1]),
#                     "remaining_class_2": len(df[df["cons_label"] == 2]),
#                     "mean_entropy": np.mean(df["entropy"]),

#                 }
#             )

#     pd.DataFrame(results).to_csv(f"experiments/uncertain_{round(entropy_threshold, 2)}_acc_alpha_summary.csv", index=False)


def plot_acc_all_thresholds():
    setup_plots()
    result_per_threshold = []
    for entropy_threshold in np.arange(0, 1.05, 0.05):
        entropy_threshold_condition = entropy_threshold + 0.06  # as small value added to softmax to prevent log(0)
        results_threshold_per_experiment = []
        for experiment in sorted(os.listdir("experiments/final_eval/")):
            df = pd.read_csv(f"experiments/final_eval/{experiment}")
            experiment = experiment.replace("virtual_", "virtual")
            df = df.dropna(subset=["panel_label_selected"])  # just in case there actually are instances not having been rated by a single pathologist in the panel
            overall_length = len(df)
            removed_instances = df[df["entropy"] > entropy_threshold_condition]
            df = df[df["entropy"] <= entropy_threshold_condition]  
            f1_cons = f1_score(df["cons_label"], df["pred_class"], average='macro', zero_division=0)
            f1_virtual20 = f1_score(df["panel_label_all"], df["pred_class"], average='macro', zero_division=0)
            f1_virtual5 = f1_score(df["panel_label_selected"], df["pred_class"], average='macro', zero_division=0)
            avg_f1 = (f1_cons + f1_virtual5 + f1_virtual20) / 3

            results_threshold_per_experiment.append(
                {
                    "experiment": experiment.replace(".csv", ""),
                    "model_strat": experiment.split("_")[0],
                    "panel_strat": experiment.split("_")[1],
                    "f1_consensus": f1_cons,
                    "f1_virtual_5": f1_virtual5,
                    "f1_virtual_20": f1_virtual20,
                    "avg_f1": avg_f1,
                    "removed_instances": len(removed_instances),
                    "rm_class_0": len(removed_instances[removed_instances["cons_label"] == 0]),
                    "rm_class_1": len(removed_instances[removed_instances["cons_label"] == 1]),
                    "rm_class_2": len(removed_instances[removed_instances["cons_label"] == 2]),
                    "remaining_class_0": len(df[df["cons_label"] == 0]),
                    "remaining_class_1": len(df[df["cons_label"] == 1]),
                    "remaining_class_2": len(df[df["cons_label"] == 2]),
                    "mean_entropy": np.mean(df["entropy"]),
                    "mean_frequency_entropy": np.mean(df["frequency_entropy"])
                }
            )
        results_threshold_per_experiment = pd.DataFrame(results_threshold_per_experiment)
        results_threshold_per_experiment.to_csv(f"experiments/certain_{round(entropy_threshold, 2)}_f1_summary.csv", index=False)

        for plot_cond in ["virtual5", "virtual20", "consensus"]:
            avg_f1 = np.mean(results_threshold_per_experiment.loc[results_threshold_per_experiment["experiment"].str.contains(plot_cond), "avg_f1"])
            result_per_threshold.append(
                {
                    "plot_cond": plot_cond,
                    "entropy_threshold": entropy_threshold,
                    "confidence_threshold": 1 - entropy_threshold,
                    "avg_f1": avg_f1,
                    "mean_removed_instances": np.mean(results_threshold_per_experiment.loc[results_threshold_per_experiment["experiment"].str.contains(plot_cond), "removed_instances"]) / overall_length,
                }
            )
    result_per_threshold = pd.DataFrame(result_per_threshold)

    fig, ax1 = plt.subplots()

    for plot_cond in ["virtual20", "virtual5", "consensus"]:
        if plot_cond == "virtual5":
            text = "Golden Panel"
        elif plot_cond == "virtual20":
            text = "Non-selective Panel"
        elif plot_cond == "consensus":
            text = "Consensus"

        subset = result_per_threshold[result_per_threshold["plot_cond"] == plot_cond]
        ax1.plot(subset["confidence_threshold"], subset["avg_f1"], marker='o', label=text)
    ax1.set_title("Average F1 Score per Entropy Threshold")
    ax1.set_ylim(0.7, 1.01)
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Average F1 Score")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  
    bar_width = 0.01 
    for i, plot_cond in enumerate(["virtual5", "virtual20", "consensus"]):
        if plot_cond == "virtual5":
            text = "Golden Panel"
        elif plot_cond == "virtual20":
            text = "Non-selective Panel"
        elif plot_cond == "consensus":
            text = "Consensus"

        subset = result_per_threshold[result_per_threshold["plot_cond"] == plot_cond]
        ax2.bar(subset["confidence_threshold"] + (i * bar_width) - (bar_width), subset["mean_removed_instances"], 
            alpha=0.4, label=f"Removed Instances: {text}", width=bar_width, align='center')

    ax2.set_ylabel("Mean Removed Instances")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y')

    fig.tight_layout()  
    lines, labels = ax1.get_legend_handles_labels() 
    lines2, labels2 = ax2.get_legend_handles_labels() 
    ax2.legend(lines + lines2, labels + labels2, loc='upper left') 
    
    plt.savefig("experiments/figs/avg_f1_all_thresholds.png")
    result_per_threshold.to_csv(f"experiments/avg_f1_all_thresholds.csv", index=False)

def plot_difference_increase_to_cons():
    setup_plots()
    results = []
    baseline_df = pd.read_csv("experiments/acc_alpha_summary.csv")
    baseline_f1 = baseline_df.groupby("model_strat")["avg_f1"].mean()

    for entropy_threshold in np.arange(0.15, 1.05, 0.05):
        entropy_threshold = round(entropy_threshold, 2)
        entropy_df = pd.read_csv(f"experiments/certain_{entropy_threshold}_f1_summary.csv")
        entropy_f1 = entropy_df.groupby("model_strat")["avg_f1"].mean()

        diff = entropy_f1 - baseline_f1
        
        diff_to_cons = diff - diff["consensus"]
        diff_to_cons.drop("consensus", inplace=True)

        print(diff_to_cons)

        for model_strat, f1_diff in diff_to_cons.items():
            if model_strat == "virtual5":
                model_strat = "Golden Panel"
            elif model_strat == "virtual20":
                model_strat = "Non-selective Panel"
            results.append({
                "entropy_threshold": entropy_threshold,
                "confidence_threshold": round(1 - entropy_threshold, 2),             
                "model_strat": model_strat,
                "f1_diff": f1_diff
            })

    results_df = pd.DataFrame(results)
    pivot_df = results_df.pivot(index="confidence_threshold", columns="model_strat", values="f1_diff")

    # Plot
    pivot_df.plot(kind="bar", edgecolor='black')
    plt.ylim(-0.09, 0.09)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("F1 Score Difference")
    plt.title("F1 Score Difference by Confidence Threshold and Model Strategy")
    plt.legend(title="Model Strategy")
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig("experiments/figs/f1_diff_to_cons.png")


def plot_multiclass_ece(bin_width=0.1, label_col="cons_label", conf_base="softmax"):
    setup_plots()
    best_experiments = ["virtual_20_cluster_2_13_14_15_16", "virtual_5_cluster_2_13_14_15_16", "consensus_cluster_2_13_14_15_16"]
    all_results = []

    for experiment in best_experiments:
        df = pd.read_csv(f'experiments/final_eval/{experiment}.csv')
        preds = df[["softmax_scores_0", "softmax_scores_1", "softmax_scores_2"]].values
        
        if conf_base == "softmax":    
            confidences = preds.max(axis=1)
        elif conf_base == "percentage":
            confidences = df["percentage_agreement_mode"]
        
        predictions = preds.argmax(axis=1)
        true_labels = df[label_col].values
        correct = (predictions == true_labels).astype(int)

        bin_edges = np.arange(0, 1.0001, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        acc_per_bin = []
        conf_per_bin = []
        counts = []

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > low) & (confidences <= high)
            if np.sum(mask) > 0:
                acc = np.mean(correct[mask])
                conf = np.mean(confidences[mask])
                acc_per_bin.append(acc)
                conf_per_bin.append(conf)
                counts.append(np.sum(mask))
            else:
                acc_per_bin.append(0)
                conf_per_bin.append(0)
                counts.append(0)

        results_df = pd.DataFrame({
            "experiment": experiment,
            "bin_centers": bin_centers,
            "accuracy": acc_per_bin,
            "confidence": conf_per_bin,
            "counts": counts
        })
        all_results.append(results_df)

    combined_results = pd.concat(all_results)

    plt.figure()
    for experiment in best_experiments:
        exp_data = combined_results[combined_results['experiment'] == experiment]
        if "consensus" in experiment:
            label = "Consensus"
        elif "virtual_5" in experiment:
            label = "Golden Panel"
        elif "virtual_20" in experiment:
            label = "Non-selective Panel"

        plt.plot(exp_data["bin_centers"], exp_data["accuracy"], 'o-', alpha = 0.7)
        plt.bar(exp_data['bin_centers'], exp_data['accuracy'], width=bin_width * 0.9, alpha=0.4, edgecolor='black', label=label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel("Confidence")
    plt.xlim(0.3,1)
    plt.ylabel("Accuracy")
    plt.title(f"Confidence Calibration (on Consensus Labels)")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiments/figs/multiclass_ece.png")
    plt.close()

        # plt.figure()
        # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        # plt.bar(bin_centers, acc_per_bin, width=bin_width * 0.9, alpha=0.7, edgecolor='black', label='Empirical Accuracy')
        # # plt.plot(bin_centers, conf_per_bin, 'o-', color='black', label='Avg Confidence')
        # plt.xlabel("Confidence")
        # plt.ylabel("Accuracy")
        # plt.title(f"Top-label Calibration (label: {label_col})")
        # plt.ylim(0, 1)
        # plt.grid(True, linestyle="--", linewidth=0.5)
        # plt.legend()
        # plt.savefig("experiments/figs/multiclass_ece.png")
        # plt.close()

def plot_multiclass_ece_separate(bin_width=0.1, label_col="cons_label", conf_base="softmax"):
    best_experiments = [
        "virtual_20_cluster_2_13_14_15_16",
        "virtual_5_cluster_2_13_14_15_16",
        "consensus_cluster_2_13_14_15_16"
    ]

    colors = {
        "virtual_20": "tab:blue",
        "virtual_5": "tab:orange",
        "consensus": "tab:red"
    }

    for experiment in best_experiments:
        df = pd.read_csv(f'experiments/final_eval/{experiment}.csv')
        preds = df[["softmax_scores_0", "softmax_scores_1", "softmax_scores_2"]].values

        if conf_base == "softmax":
            confidences = preds.max(axis=1)
        elif conf_base == "percentage":
            confidences = df["percentage_agreement_mode"]

        predictions = preds.argmax(axis=1)
        true_labels = df[label_col].values
        correct = (predictions == true_labels).astype(int)

        bin_edges = np.arange(0, 1.0001, bin_width)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        acc_per_bin = []
        conf_per_bin = []
        counts = []

        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > low) & (confidences <= high)
            if np.sum(mask) > 0:
                acc_per_bin.append(np.mean(correct[mask]))
                conf_per_bin.append(np.mean(confidences[mask]))
                counts.append(np.sum(mask))
            else:
                acc_per_bin.append(0)
                conf_per_bin.append(0)
                counts.append(0)

        results_df = pd.DataFrame({
            "bin_centers": bin_centers,
            "accuracy": acc_per_bin,
            "confidence": conf_per_bin,
            "counts": counts
        })

        if "consensus" in experiment:
            label = "Consensus"
            color = colors["consensus"]
        elif "virtual_5" in experiment:
            label = "Golden Panel"
            color = colors["virtual_5"]
        elif "virtual_20" in experiment:
            label = "Non selective Panel"
            color = colors["virtual_20"]

        total = np.sum(counts)
        if total > 0:
            ece = np.sum([
                (counts[i] / total) * abs(acc_per_bin[i] - conf_per_bin[i])
                for i in range(len(counts))
            ])
        else:
            ece = float("nan")

        print(f"{label} ECE: {ece:.4f}")



        plt.figure()
        plt.plot(results_df["bin_centers"], results_df["accuracy"], marker="o", alpha=0.7, color=color)
        plt.bar(results_df["bin_centers"], results_df["accuracy"],
                width=bin_width * 0.9, alpha=0.4, edgecolor="black", color=color, label=label)

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
        plt.xlabel("Confidence")
        plt.xlim(0.3, 1)
        plt.ylabel("Accuracy")
        plt.title(f"Confidence Calibration {label}")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"experiments/figs/multiclass_ece_{experiment}.png")
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate agreement metrics for model predictions.")
    parser.add_argument("--entropy_threshold", type=float, default=1, help="Threshold for filtering uncertain predictions.")
    args = parser.parse_args()

    # remove_nans()
    # calculate_agreement()
    # plot_acc_all_thresholds()
    # plot_difference_increase_to_cons()
    plot_multiclass_ece_separate(conf_base="percentage")