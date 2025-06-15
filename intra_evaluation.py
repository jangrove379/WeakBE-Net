import torch
from torch.utils.data import DataLoader, Subset
from train import MILModel
from data import BagDataset, process_labels, get_dataloaders
import os
import pandas as pd
import argparse


def load_model(checkpoint_path, device):
    model = MILModel.load_from_checkpoint(checkpoint_path,
        feature_dim=2560,
        hidden_dim=16,
        num_classes=3,
        output_dim=3,
        lr=1e-5,
        wd=1e-5,
        drop_out=0.0,
        class_weights=None,  # or load from checkpoint if needed
        diff_weights=None,
        run_dir=None,
        strict=False  # Allow loading with unexpected keys
    )
    model.to(device)
    model.eval()
    return model


def run_evaluation(device, output_dir, experiment_name):
    dataset = BagDataset(args.features_dir, use_p53=True, path_id=args.path_id, label_file=args.label_file, experiment_mode="intra")
    _, _, _, dataloader, _, _ = next(get_dataloaders(dataset, path=args.path_id, k_folds=5, batch_size=1, experiment_mode="intra"))

    for i in range(5):
        checkpoint_path = f"/data/archief/AMC-data/Barrett/experiments/jans_experiments/{args.experiment_name_base}_Pathologist_{args.path_id}_fold_{i + 1}/best_model.ckpt"
        os.makedirs(output_dir, exist_ok=True)
        model = load_model(checkpoint_path, device)
        results = []
        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(device)
                block_id = batch["block_id"][0]
                cons_labels = batch["cons_label"]
                raters_labels = batch["rater_labels"]
                target = process_labels(cons_labels, raters_labels, method='path', add_consensus=False, path_id=args.path_id)

                logit = model(features)
                pred_class = logit.argmax(dim=1)
                softmax_scores = torch.softmax(logit, dim=1)  
                softmax_scores = torch.softmax(logit, dim=1)  
                entropy = -torch.sum(softmax_scores * torch.log(softmax_scores + 1e-10), dim=1) 
                results.append({
                    "block_id": block_id,
                    "pred_score_0": logit[0][0].item(),
                    "pred_score_1": logit[0][1].item(),
                    "pred_score_2": logit[0][2].item(),
                    "pred_class": pred_class.item(),
                    "label": target.item(),
                    "softmax_scores_0": softmax_scores[0][0].item(),
                    "softmax_scores_1": softmax_scores[0][1].item(),
                    "softmax_scores_2": softmax_scores[0][2].item(),
                    "entropy": entropy.item(),
                    "max_softmax_score": softmax_scores.max().item(),
                    "max_pred_score": logit.max().item(),
                    "predictive_variance": torch.var(softmax_scores, dim=1).item()
                })

        df = pd.DataFrame(results)
        print(df)
        file_name = os.path.join(output_dir, f"evaluation_results_{experiment_name}_path_{args.path_id}_fold_{i+1}.csv")
        df.to_csv(file_name, index=False)
        print(f"Saved predictions to {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Evaluation for MIL Models")
    parser.add_argument('--experiment_name_base', type=str, default='final_intra', help='Base name for the experiment')
    parser.add_argument('--features_dir', type=str, default="/data/archief/AMC-data/Barrett/LANS_features/Virchow_HE_P53_1mpp_v2", help='Directory containing features')
    parser.add_argument('--output_dir', type=str, default='/home/jmgrove/experiments/intra/', help='Directory to save output results')
    parser.add_argument('--path_id', type=int, default=None, help='Pathologist index')
    parser.add_argument("--label_file", type=str, default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv')
    args = parser.parse_args()
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    run_evaluation(device, args.output_dir, args.experiment_name_base)