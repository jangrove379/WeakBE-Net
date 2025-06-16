import torch
from torch.utils.data import DataLoader, Subset
from train import MILModel
from data import BagDataset, process_labels
import os
import pandas as pd
import numpy as np
import krippendorff as kd
import argparse


def load_models(checkpoint_paths, device):
    models = []
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"Checkpoint file {path} does not exist. Skipping.") # <- for path 8 fold 4 and 5
            continue 
        model = MILModel.load_from_checkpoint(path,
            feature_dim=2560,
            hidden_dim=16,
            num_classes=3,
            output_dim=3,
            lr=1e-5,
            wd=1e-5,
            drop_out=0.0,
            class_weights=None, 
            diff_weights=None,
            run_dir=None,
            strict=False  
        )
        model.to(device)
        model.eval()
        models.append(model)
    return models

def get_panel_labels():
    path_list = args.panel_pathologists
    path_list_all = list(range(1, 21))

    labels = pd.read_csv(args.label_file)
    cons = labels[["block_id", "dx"]]
    cons = cons[["block_id", "dx"]].replace({1: 0, 2: np.nan, 3: 1, 4: 2})
    cons = cons.dropna(subset=["dx"])
    rater_labels = labels.loc[:,"p53":]
    rater_labels = rater_labels.drop(["p53"], axis=1).replace({1: 0, 2: np.nan, 3: 1, 4: 2, 0: np.nan})
    rater_labels = pd.concat([labels["block_id"], rater_labels], axis=1)

    avg_alpha_path = []
    for path in range (1,21):
        alpha_per_fold = []
        for fold in range(1,6):
            try:
                df = pd.read_csv(f"{args.intra_results_dir}/{args.intra_results_name}_path_{path}_fold_{fold}.csv")
            except FileNotFoundError:
                raise FileNotFoundError(f"{args.intra_results_dir}/{args.intra_results_name}_{path}_fold_{fold}.csv not available. If you haven't, run intra_evaluation.py first!")
            alpha = kd.alpha(df[["pred_class", "label"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
            alpha_per_fold.append(alpha)
        avg_alpha_path.append(np.mean(alpha_per_fold))
    avg_alpha = pd.DataFrame({"path_id": list(range(1, 21)), "alpha": avg_alpha_path})
    avg_alpha.to_csv("experiments/intra/aa_krippendorff_alpha_per_path.csv", index=False)

    panel_labels = []
    for panel in [path_list, path_list_all]:
        path_scores = avg_alpha[avg_alpha["path_id"].isin(panel)][["path_id", "alpha"]].sort_values(by="alpha", ascending=False)
        path_list_priority = path_scores["path_id"].tolist()
        col_list = [f"path_{i}" for i in panel]

        modes = rater_labels[col_list].mode(axis=1, numeric_only=True)
        num_different_modes = modes.notna().sum(axis=1)

        decided_cases_indices = num_different_modes[num_different_modes == 1].index
        decided_df = modes[modes.index.isin(decided_cases_indices)][[0]].rename(columns={0: "labels"})
        decided_df["block_id"] = rater_labels.loc[decided_cases_indices, "block_id"].values
        
        
        undecided_cases_indices = num_different_modes[num_different_modes > 1].index
        undecided_cases_basis = rater_labels[rater_labels.index.isin(undecided_cases_indices)]
        undecided_df = pd.DataFrame({"block_id": undecided_cases_basis["block_id"]})

        undecided_labels = []
        for _, row in undecided_cases_basis.iterrows():
            for path in path_list_priority:
                col = f"path_{path}"
                cell_value = row[f"path_{path}"]
                if pd.notna(cell_value):
                    undecided_labels.append(cell_value)
                    break
            else:
                undecided_labels.append(np.nan) 

        undecided_df["labels"] = undecided_labels
        panel_labels.append(pd.concat([undecided_df, decided_df], axis=0).sort_index())
    return panel_labels[0], panel_labels[1] # <- note: panel_labels[0] corresponds to selected panel, while second return corresponds to all 20 pathologists 



def run_ensemble_evaluation(device):
    if (args.experiment_name_base == "agg" and args.train_pathologists is None) or (args.experiment_name_base != "agg" and args.train_pathologists is not None):
        raise ValueError("Either both 'experiment_name_base' and 'pathologists' must be provided, or neither.")

    if args.experiment_name_base == "agg":
        checkpoint_paths = [f"/data/archief/AMC-data/Barrett/experiments/jans_experiments/{args.experiment_name_base}_Pathologist_{j}_fold_{i + 1}/best_model.ckpt" for i in range(5) for j in args.train_pathologists]
    else:
        checkpoint_paths = [f"/data/archief/AMC-data/Barrett/experiments/jans_experiments/{args.experiment_name_base}_fold_{i+1}/best_model.ckpt" for i in range(5)]


    os.makedirs(args.output_dir, exist_ok=True)
    models = load_models(checkpoint_paths, device)
    dataloader = get_dataloader()
    results = []
    panel_labels_selected, panel_labels_all = get_panel_labels()

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            block_id = batch["block_id"][0]
            cons_labels = batch["cons_label"]

            panel_label_selected = panel_labels_selected[panel_labels_selected["block_id"] == block_id]["labels"].values
            panel_label_all = panel_labels_all[panel_labels_all["block_id"] == block_id]["labels"].values

            logits = []
            for model in models:
                logit = model(features)
                logits.append(logit)  # Change here to get the predicted class index
            avg_score = sum(logits) / len(logits)
            pred_class = avg_score.argmax(dim=1)
            softmax_scores = torch.softmax(avg_score, dim=1)

            results.append({
                "block_id": block_id,
                "pred_score_0": logit[0][0].item(),
                "pred_score_1": logit[0][1].item(),
                "pred_score_2": logit[0][2].item(),
                "softmax_scores_0": softmax_scores[0][0].item(),
                "softmax_scores_1": softmax_scores[0][1].item(),
                "softmax_scores_2": softmax_scores[0][2].item(),
                "pred_class": pred_class.item(),
                "cons_label": cons_labels.item(),
                "panel_label_selected": panel_label_selected[0] if len(panel_label_selected) > 0 else np.nan,
                "panel_label_all": panel_label_all[0] if len(panel_label_all) > 0 else np.nan
            })

    df = pd.DataFrame(results)
    print(df)
    file_name = os.path.join(args.output_dir, f"{args.output_name}.csv")
    df.to_csv(file_name, index=False)
    print(f"Saved predictions to {file_name}")


def get_dataloader():
    dataset = BagDataset(args.features_dir, use_p53=True, label_file=args.label_file, experiment_mode="final_cons", path_id=None) # <- final_cons so that no filtering is done -> we need the whole set
    test_idx = [
        i for i, block_id in enumerate(dataset.block_ids)
        if 1000 < int(block_id.split("-")[1]) <= 1100
    ]
    dataset = Subset(dataset, test_idx)

    return DataLoader(dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Evaluation for MIL Models")
    parser.add_argument('--experiment_name_base', type=str, required=True, choices=['agg', 'agg_cons'], help='Base name for the experiment') # <- agg for panel of paths, final_cons for consensus
    parser.add_argument('--features_dir', type=str, default="/data/archief/AMC-data/Barrett/LANS_features/Virchow_HE_P53_1mpp_v2", help='Directory containing features')
    parser.add_argument('--intra_results_dir', type=str, default='/home/jmgrove/experiments/intra', help='Path to intra results directory')
    parser.add_argument('--intra_results_name', type=str, default='evaluation_results_final_intra', help='Start to name of the intra results file')
    parser.add_argument("--label_file", type=str, default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv')
    parser.add_argument('--output_dir', type=str, default='/home/jmgrove/experiments/final_eval/', help='Directory to save output results')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file')
    parser.add_argument('--panel_pathologists', type=str, nargs='+', required=True, help='Pathologist ids of panel')
    parser.add_argument('--train_pathologists', type=str, nargs='+', default=None, help='Pathologist indices for training (only used if experiment_name_base is agg)')
    args = parser.parse_args() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    run_ensemble_evaluation(device)