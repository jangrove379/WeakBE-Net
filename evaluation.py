import torch
from torch.utils.data import DataLoader, Subset
from train import MILModel
from data import BagDataset, process_labels
from sklearn.metrics import accuracy_score
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
            df.to_csv(f"experiments/final_eval/{experiment}", index=False)


def calculate_agreement():
    results = []
    for experiment in sorted(os.listdir("experiments/final_eval/")):
            df = pd.read_csv(f"experiments/final_eval/{experiment}")
            experiment = experiment.replace("virtual_", "virtual")
            df = df.dropna(subset=["panel_label_selected"]) # just in case there actually are instances not having been rated by a single pathologist in the panel
            acc_cons = accuracy_score(df["cons_label"], df["pred_class"])
            acc_virtual20 = accuracy_score(df["panel_label_all"], df["pred_class"])
            alpha_cons = kd.alpha(df[["cons_label", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
            alpha_virtual5 = kd.alpha(df[["panel_label_selected", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
            acc_virtual5 = accuracy_score(df["panel_label_selected"], df["pred_class"])
            alpha_virtual20 = kd.alpha(df[["panel_label_all", "pred_class"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
            avg_acc = (acc_cons + acc_virtual5 + acc_virtual20) / 3
            avg_alpha = (alpha_cons + alpha_virtual5 + alpha_virtual20) / 3

            results.append(
                {
                    "experiment": experiment.replace(".csv", ""),
                    "model_strat": experiment.split("_")[0],
                    "panel_strat": experiment.split("_")[1],
                    "acc_consensus": acc_cons,
                    "acc_virtual_5": acc_virtual5,
                    "acc_virtual_20": acc_virtual20,
                    "alpha_consensus": alpha_cons,
                    "alpha_virtual_5": alpha_virtual5,
                    "alpha_virtual_20": alpha_virtual20,
                    "avg_acc": avg_acc,
                    "avg_alpha": avg_alpha,
                    "mean_entropy": np.mean(df["entropy"]),
                    "mean_frequency_entropy": np.mean(df["frequency_entropy"])
                }
            )

    pd.DataFrame(results).to_csv(f"experiments/acc_alpha_summary.csv", index=False)


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
    for entropy_threshold in np.arange(0.15, 1.05, 0.05):
        entropy_threshold_condition = entropy_threshold + 0.06 # as small value added to softmax to prevent log(0)
        results_threshold_per_experiment = []
        for experiment in sorted(os.listdir("experiments/final_eval/")):
                df = pd.read_csv(f"experiments/final_eval/{experiment}")
                experiment = experiment.replace("virtual_", "virtual")
                df = df.dropna(subset=["panel_label_selected"]) # just in case there actually are instances not having been rated by a single pathologist in the panel
                overall_length = len(df)
                removed_instances = df[df["entropy"] > entropy_threshold_condition]
                df = df[df["entropy"] <= entropy_threshold_condition]  
                acc_cons = accuracy_score(df["cons_label"], df["pred_class"])
                acc_virtual20 = accuracy_score(df["panel_label_all"], df["pred_class"])
                acc_virtual5 = accuracy_score(df["panel_label_selected"], df["pred_class"])
                avg_acc = (acc_cons + acc_virtual5 + acc_virtual20) / 3

                results_threshold_per_experiment.append(
                    {
                        "experiment": experiment.replace(".csv", ""),
                        "model_strat": experiment.split("_")[0],
                        "panel_strat": experiment.split("_")[1],
                        "acc_consensus": acc_cons,
                        "acc_virtual_5": acc_virtual5,
                        "acc_virtual_20": acc_virtual20,
                        "avg_acc": avg_acc,
                        "removed_instances": len(removed_instances),
                        "rm_class_0": len(removed_instances[removed_instances["cons_label"] == 0]),
                        "rm_class_1": len(removed_instances[removed_instances["cons_label"] == 1]),
                        "rm_class_2": len(removed_instances[removed_instances["cons_label"] == 2]),
                        "remainining_class_0": len(df[df["cons_label"] == 0]),
                        "remaining_class_1": len(df[df["cons_label"] == 1]),
                        "remaining_class_2": len(df[df["cons_label"] == 2]),
                        "mean_entropy": np.mean(df["entropy"]),
                        "mean_frequency_entropy": np.mean(df["frequency_entropy"])
                    }
                )
        results_threshold_per_experiment = pd.DataFrame(results_threshold_per_experiment)
        results_threshold_per_experiment.to_csv(f"experiments/certain_{round(entropy_threshold, 2)}_acc_alpha_summary.csv", index=False)

        for plot_cond in ["virtual5", "virtual20", "consensus"]:
            avg_acc = np.mean(results_threshold_per_experiment.loc[results_threshold_per_experiment["experiment"].str.contains(plot_cond), "avg_acc"])
            result_per_threshold.append(
                {
                    "plot_cond": plot_cond,
                    "entropy_threshold": entropy_threshold,
                    "avg_acc": avg_acc,
                    "mean_removed_instances": np.mean(results_threshold_per_experiment.loc[results_threshold_per_experiment["experiment"].str.contains(plot_cond), "removed_instances"]) / overall_length,
                }
            )
    result_per_threshold = pd.DataFrame(result_per_threshold)

    fig, ax1 = plt.subplots()

    # Plotting average accuracy
    for plot_cond in ["virtual5", "virtual20", "consensus"]:
        if plot_cond == "virtual5":
            text = "Golden Panel"
        elif plot_cond == "virtual20":
            text = "Non-selective Panel"
        elif plot_cond == "consensus":
            text = "Consensus"

        subset = result_per_threshold[result_per_threshold["plot_cond"] == plot_cond]
        ax1.plot(subset["entropy_threshold"], subset["avg_acc"], marker='o', label=text)
    ax1.set_title("Average Accuracy vs. Entropy Threshold")
    ax1.set_ylim(0.7, 1.01)
    ax1.set_xlabel("Entropy Threshold")
    ax1.set_ylabel("Average Accuracy")
    ax1.tick_params(axis='y')

    # Creating a second y-axis for mean_removed_instances
    ax2 = ax1.twinx()  # Create the second y-axis first
    bar_width = 0.01  # Width of each bar
    for i, plot_cond in enumerate(["virtual5", "virtual20", "consensus"]):
        if plot_cond == "virtual5":
            text = "Golden Panel"
        elif plot_cond == "virtual20":
            text = "Non-selective Panel"
        elif plot_cond == "consensus":
            text = "Consensus"

        subset = result_per_threshold[result_per_threshold["plot_cond"] == plot_cond]
        ax2.bar(subset["entropy_threshold"] + (i * bar_width) - (bar_width), subset["mean_removed_instances"], 
            alpha=0.4, label=f"Removed Instances: {text}", width=bar_width, align='center')

    ax2.set_ylabel("Mean Removed Instances")
    ax2.set_ylim(0, 1.05)  # Set y-axis limits for removed instances
    ax2.tick_params(axis='y')

    fig.tight_layout()  # To ensure the layout is clean
    lines, labels = ax1.get_legend_handles_labels()  # Get handles and labels from ax1
    lines2, labels2 = ax2.get_legend_handles_labels()  # Get handles and labels from ax2
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')  # Combine legends into one
    
    plt.savefig("experiments/figs/avg_acc_all_thresholds.png")
    result_per_threshold.to_csv(f"experiments/avg_acc_all_thresholds.csv", index=False)

def plot_difference_increase_to_cons():
    setup_plots()
    results = []
    baseline_df = pd.read_csv("experiments/acc_alpha_summary.csv")
    baseline_acc = baseline_df.groupby("model_strat")["avg_acc"].mean()

    for entropy_threshold in np.arange(0.15, 1.05, 0.05):
        entropy_threshold = round(entropy_threshold, 2)
        entropy_df = pd.read_csv(f"experiments/certain_{entropy_threshold}_acc_alpha_summary.csv")
        entropy_acc = entropy_df.groupby("model_strat")["avg_acc"].mean()

        diff = entropy_acc - baseline_acc
        
        diff_to_cons = diff - diff["consensus"]
        diff_to_cons.drop("consensus", inplace=True)

        print(diff_to_cons)

        for model_strat, acc_diff in diff_to_cons.items():
            if model_strat == "virtual5":
                model_strat = "Golden Panel"
            elif model_strat == "virtual20":
                model_strat = "Non-selective Panel"
            results.append({
                "entropy_threshold": entropy_threshold,
                "model_strat": model_strat,
                "acc_diff": acc_diff
            })

    results_df = pd.DataFrame(results)
    pivot_df = results_df.pivot(index="entropy_threshold", columns="model_strat", values="acc_diff")

    # Plot
    pivot_df.plot(kind="bar", edgecolor='black')
    plt.xlabel("Entropy Threshold")
    plt.ylabel("Accuracy Difference")
    plt.title("Accuracy Difference by Entropy Threshold and Model Strategy")
    plt.legend(title="Model Strategy")
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.savefig("experiments/figs/acc_diff_to_cons.png")


def plot_ece():
    df = pd.read_csv("experiments/final_eval/virtual_20_cluster_2_13_14_15_16.csv")
    bins = []
    bin_width = 0.05
    for i_class in [0, 1, 2]:
        for softmax_score in np.arange(0, 1.01, bin_width):
            df_class = df[
                (df[f"softmax_scores_{i_class}"] > softmax_score - bin_width) &
                (df[f"softmax_scores_{i_class}"] <= softmax_score)  
                # (df["cons_label"] == i_class)
            ]
            acc_cons = accuracy_score(df_class["cons_label"], df_class["pred_class"])
            acc_virtual20 = accuracy_score(df_class["panel_label_all"], df_class["pred_class"])
            acc_virtual5 = accuracy_score(df_class["panel_label_selected"], df_class["pred_class"])
            acc_avg = (acc_cons + acc_virtual5 + acc_virtual20) / 3
            bins.append(
                {
                    "confidence": softmax_score,
                    "acc_consensus": acc_cons,
                    "acc_virtual_5": acc_virtual5,
                    "acc_virtual_20": acc_virtual20,
                    "avg_acc": acc_avg,
                    "class": i_class,
                    "length": len(df_class)
                }
            )
    bins = pd.DataFrame(bins)
    print(bins)
    bins.to_csv("experiments/acc_bins.csv", index=False)

    for class_id in [0, 1, 2]:
        class_df = bins[bins["class"] == class_id]

        plt.figure()
        plt.bar(
            class_df["confidence"],
            class_df["avg_acc"],
            width=0.04,
            edgecolor="black",
            color="lightblue",
            alpha=0.7,
            label="Average Accuracy\n(all three test sets)"
        )
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")

        plt.xlabel("Confidence (softmax score)")
        plt.ylabel("Average Accuracy\n(all three test sets)")
        plt.title(f"Calibration for Class {class_id}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"experiments/figs/plot_ece_class_{class_id}.png")
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate agreement metrics for model predictions.")
    parser.add_argument("--entropy_threshold", type=float, default=1, help="Threshold for filtering uncertain predictions.")
    args = parser.parse_args()

    remove_nans()
    calculate_agreement()
    plot_acc_all_thresholds()
    plot_difference_increase_to_cons()
    plot_ece()
       
