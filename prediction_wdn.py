import torch
from torch.utils.data import DataLoader, Subset
from data import BagDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import pandas as pd
import numpy as np
import krippendorff as kd
import argparse
from train_wdn import DoctorNetModel, WeightedDoctorNetModel


def load_doctor_net_model(checkpoint_path, num_raters, device):
    """Load a single Doctor Net model."""
    # Import the DoctorNetModel class from your training script    
    model = DoctorNetModel.load_from_checkpoint(
        checkpoint_path,
        feature_dim=2560,
        hidden_dim=16,
        num_classes=3,
        num_raters=num_raters,
        run_dir=None,
        class_weights=None,
        strict=False  
    )
    model.to(device)
    model.eval()
    return model


def load_weighted_doctor_net_model(checkpoint_path, doctor_net_checkpoint, num_raters, device):
    """Load a single Weighted Doctor Net model."""
    
    model = WeightedDoctorNetModel.load_from_checkpoint(
        checkpoint_path,
        feature_dim=2560,
        hidden_dim=16,
        num_classes=3,
        num_raters=num_raters,
        run_dir=None,
        doctor_net_checkpoint=doctor_net_checkpoint,
        strict=False  
    )
    model.to(device)
    model.eval()
    return model


def get_panel_labels(args):
    """Get panel labels based on pathologist reliability."""
    path_list = [int(p) for p in args.panel_pathologists]
    path_list_all = list(range(1, 21))

    labels = pd.read_csv(args.label_file)
    cons = labels[["block_id", "dx"]]
    cons = cons[["block_id", "dx"]].replace({1: 0, 2: np.nan, 3: 1, 4: 2})
    cons = cons.dropna(subset=["dx"])
    
    rater_labels = labels.loc[:, "p53":]
    rater_labels = rater_labels.drop(["p53"], axis=1).replace({1: 0, 2: np.nan, 3: 1, 4: 2, 0: np.nan})
    rater_labels = pd.concat([labels["block_id"], rater_labels], axis=1)

    # Get reliability scores if intra results are available
    if args.intra_results_dir and args.intra_results_name:
        avg_alpha = get_alpha_scores(args.intra_results_dir, args.intra_results_name, args.label_file)
    else:
        # Use simple inter-rater agreement if intra results not available
        avg_alpha = pd.DataFrame({
            "path_id": list(range(1, 21)),
            "overall": get_mean_inter_rater_agreement(args.label_file)
        })

    panel_labels = []
    for panel in [path_list, path_list_all]:
        path_scores = avg_alpha[avg_alpha["path_id"].isin(panel)][["path_id", "overall"]].sort_values(by="overall", ascending=False)
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
                cell_value = row[f"path_{path}"]
                if pd.notna(cell_value):
                    undecided_labels.append(cell_value)
                    break
            else:
                undecided_labels.append(np.nan) 

        undecided_df["labels"] = undecided_labels
        panel_labels.append(pd.concat([undecided_df, decided_df], axis=0).sort_index())
    
    return panel_labels[0], panel_labels[1]


def get_alpha_scores(intra_results_dir, intra_results_name, label_file):
    avg_intra_alpha_path = []
    avg_intra_f1_path = []
    for path in range (1,21):
        alpha_per_fold = []
        f1_per_fold = []
        for fold in range(1,6):
            try:
                df = pd.read_csv(f"{intra_results_dir}/{intra_results_name}_path_{path}_fold_{fold}.csv")
            except FileNotFoundError:
                raise FileNotFoundError(f"{intra_results_dir}/{intra_results_name}_{path}_fold_{fold}.csv not available. If you haven't, run intra_evaluation.py first!")
            alpha = kd.alpha(df[["pred_class", "label"]].T.values, level_of_measurement="ordinal", value_domain=[0, 1.0, 2.0])
            f1 = f1_score(df["label"], df["pred_class"], average="macro")
            alpha_per_fold.append(alpha)
            f1_per_fold.append(f1)
        avg_intra_alpha_path.append(np.mean(alpha_per_fold))
        avg_intra_f1_path.append(np.mean(f1_per_fold))
    avg_alpha = pd.DataFrame({"path_id": list(range(1, 21)), "intra": avg_intra_alpha_path})
    avg_inter_alpha = get_mean_inter_rater_agreement(label_file)
    avg_alpha["inter"] = avg_inter_alpha
    avg_alpha["overall"] = 0.5*(avg_alpha["inter"] + avg_alpha["intra"])
    avg_alpha.to_csv("experiments/reliability_scores.csv", index=False)
    avg_f1 = pd.DataFrame({"path_id": list(range(1,21)), "intra_f1": avg_intra_f1_path})
    avg_f1.to_csv("experiments/intra_f1.csv", index=False)
    
    return avg_alpha


def get_mean_inter_rater_agreement(label_file, common_samples=False):
    data =  pd.read_csv(label_file)
    data = data.loc[:,"p53":]
    data = data.drop(["p53"], axis=1).replace({1: 0, 2: np.nan, 3: 1, 4: 2, 0 : np.nan})
    
    # note: use of lists might be more efficient -> artifact from EDA
    pairwise_corr = pd.DataFrame(np.zeros((20, 20)), index=range(1,21), columns=range(1,21))
    number_of_common_rows = pd.DataFrame(np.zeros((20, 20)), index=range(1,21), columns=range(1,21))

    for i in range(1, 21):
        for j in range(1, 21):
            pairwise_df = data[[f"path_{i}", f"path_{j}"]]
            number_of_common_rows.loc[i,j] = len(pairwise_df.dropna())
            if ((len(pairwise_df[pairwise_df.isna().any(axis=1)]) == len(pairwise_df)) or len(pairwise_df) < 100):
                pairwise_corr.loc[i,j] = np.nan
            else:
                pairwise_corr.loc[i,j] = kd.alpha(pairwise_df.T.values, level_of_measurement="ordinal", value_domain=[0, 1, 2])

    if common_samples:
        return number_of_common_rows.apply(lambda row: row.drop(row.name).mean(), axis=1).reset_index(drop=True)

    return pairwise_corr.apply(lambda row: row.drop(row.name).mean(), axis=1).reset_index(drop=True)


def get_dataloader(args):
    """Create dataloader for test set."""
    dataset = BagDataset(
        args.features_dir, 
        use_p53=True, 
        label_file=args.label_file, 
        experiment_mode="final_cons", 
        path_id=None
    )
    
    # Get test indices (block IDs > 1000)
    test_idx = [
        i for i, block_id in enumerate(dataset.block_ids)
        if 1000 < int(block_id.split("-")[1])
    ]
    
    dataset = Subset(dataset, test_idx)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def run_wdn_prediction(args, device):
    """Run prediction for WDN models."""
    
    print(f"\nRunning WDN prediction: {args.model_type}")
    
    # Load model based on type
    
    # Get dataloader
    dataloader = get_dataloader(args)
    
    # Get panel labels if needed
    if args.use_panel_labels:
        panel_labels_selected, panel_labels_all = get_panel_labels(args)
    
    # Run predictions
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            models = []
            if args.model_type == 'doctor_net':
                checkpoint_paths = [f"/data/archief/AMC-data/Barrett/experiments/jans_experiments/Guan_CV_phase1_fold_{i+1}/best_model.ckpt" for i in range(5)] 
                for path in checkpoint_paths:
                    model = load_doctor_net_model(path, args.num_raters, device)
                    models.append(model)
            elif args.model_type == 'weighted_doctor_net':
                checkpoint_paths = [f"/data/archief/AMC-data/Barrett/experiments/jans_experiments/Guan_CV_phase2_fold_{i+1}/best_model.ckpt" for i in range(5)] 
                doctornet_checkpoints = [f"/data/archief/AMC-data/Barrett/experiments/jans_experiments/Guan_CV_phase1_fold_{i+1}/best_model.ckpt" for i in range(5)] 
                for path, dn_path in zip(checkpoint_paths, doctornet_checkpoints):
                    model = load_weighted_doctor_net_model(path, dn_path, args.num_raters, device)
                    models.append(model)
            else:
                raise ValueError(f"Unknown model type: {args.model_type}")

            features = batch["features"].to(device)
            block_id = batch["block_id"][0]
            cons_labels = batch["cons_label"]
            
            # Get panel labels if available
            if args.use_panel_labels:
                panel_label_selected = panel_labels_selected[panel_labels_selected["block_id"] == block_id]["labels"].values
                panel_label_all = panel_labels_all[panel_labels_all["block_id"] == block_id]["labels"].values
            else:
                panel_label_selected = [np.nan]
                panel_label_all = [np.nan]
            

            logits = []
            frequency_classes_prediction = torch.zeros((3,), device=device)
            if args.model_type == 'weighted_doctor_net':
                weights_list = []
            for model in models:
                if args.model_type == 'doctor_net':
                    rater_logits_list = model(features)
                    rater_logits = torch.cat(rater_logits_list, dim=0)
                    curr_logits = rater_logits.mean(dim=0, keepdim=True)
                    logits.append(curr_logits)
                    rater_predicted_classes = torch.argmax(rater_logits, dim=1)
                    frequency_classes_prediction.scatter_add_(
                        dim=0,
                        index=rater_predicted_classes,
                        src=torch.ones_like(rater_predicted_classes, dtype=torch.float)
                    )

                elif args.model_type == 'weighted_doctor_net':
                    rater_logits, curr_logits, weights = model(features)
                    weights_list.append(weights)
                    logits.append(curr_logits)
                    rater_predicted_classes = torch.argmax(rater_logits, dim=1)
                    frequency_classes_prediction.scatter_add_(
                        dim=0,
                        index=rater_predicted_classes,
                        src=weights.squeeze()
                    )

            mean_logit_over_folds = sum(logits) / len(logits)
            softmax_probs = torch.softmax(mean_logit_over_folds, dim=1)
            pred_class = mean_logit_over_folds.argmax(dim=1)

            # Calculate entropy
            entropy = -torch.sum(softmax_probs * torch.log(softmax_probs + 1e-10), dim=1)

            result = {
                "block_id": block_id,
                "pred_score_0": mean_logit_over_folds[0][0].item(),
                "pred_score_1": mean_logit_over_folds[0][1].item(),
                "pred_score_2": mean_logit_over_folds[0][2].item(),
                "softmax_scores_0": softmax_probs[0][0].item(),
                "softmax_scores_1": softmax_probs[0][1].item(),
                "softmax_scores_2": softmax_probs[0][2].item(),
                "pred_class": pred_class.item(),
                "cons_label": cons_labels.item(),
                "entropy": entropy.item(),
                "frequency_classes_prediction_0": frequency_classes_prediction[0].item(),
                "frequency_classes_prediction_1": frequency_classes_prediction[1].item(),
                "frequency_classes_prediction_2": frequency_classes_prediction[2].item(),
            }

            # Add panel labels if available
            if args.use_panel_labels:
                result["panel_label_selected"] = panel_label_selected[0] if len(panel_label_selected) > 0 else np.nan
                result["panel_label_all"] = panel_label_all[0] if len(panel_label_all) > 0 else np.nan
                        
            results.append(result)
    
    # Convert to dataframe
    df = pd.DataFrame(results)
    df['percentage_agreement_mode'] = df.apply(calculate_percentage_agreement_mode, axis=1)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.output_name}.csv")
    df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to: {output_file}")
    

    # Save learned weights if WDN
    if args.model_type == 'weighted_doctor_net':
        avg_weights = torch.stack(weights_list).mean(dim=0).squeeze().cpu().numpy()
        weights_file = os.path.join(args.output_dir, f"{args.output_name}_weights.csv")
        pd.DataFrame(avg_weights).to_csv(weights_file, index=False)
        print(f"Saved learned weights to: {weights_file}")

def calculate_percentage_agreement_mode(row):
    freq_preds = [row['frequency_classes_prediction_0'], row['frequency_classes_prediction_1'], row['frequency_classes_prediction_2']]
    mode = np.argmax(freq_preds) 
    total_freq = sum(freq_preds)
    percentage_agreement_mode = freq_preds[mode] / total_freq if total_freq != 0 else 0
    
    return percentage_agreement_mode



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WDN Evaluation (Single Split)")
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['baseline', 'doctor_net', 'weighted_doctor_net'],
                       help='Type of model to evaluate')
    parser.add_argument('--num_raters', type=int, default=20,
                       help='Number of raters/pathologists')
    
    parser.add_argument('--features_dir', type=str, 
                       default="/data/archief/AMC-data/Barrett/LANS_features/Virchow_HE_P53_1mpp_v2",
                       help='Directory containing features')
    parser.add_argument("--label_file", type=str, 
                       default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv',
                       help='Path to label file')
    
    parser.add_argument('--use_panel_labels',
                       help='Whether to compute metrics against panel labels')
    parser.add_argument('--panel_pathologists', type=str, nargs='+', default=None,
                       help='Pathologist IDs for panel (required if use_panel_labels=True)')
    parser.add_argument('--intra_results_dir', type=str, default='/home/jmgrove/experiments/intra', help='Path to intra results directory')
    parser.add_argument('--intra_results_name', type=str, default='evaluation_results_final_intra', help='Start to name of the intra results file')
        
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save output results')
    parser.add_argument('--output_name', type=str, required=True,
                       help='Name of the output file (without .csv)')
    
    args = parser.parse_args()
    
    # Validation
    
    if args.use_panel_labels and args.panel_pathologists is None:
        raise ValueError("--panel_pathologists is required when --use_panel_labels is set")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_wdn_prediction(args, device)