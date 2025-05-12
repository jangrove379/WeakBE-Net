import torch
from torch.utils.data import DataLoader
from train import MILModel
from data import EvalDataset
import os
import pandas as pd


def load_models(checkpoint_paths, device):
    models = []
    for path in checkpoint_paths:
        model = MILModel.load_from_checkpoint(path)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def run_ensemble_evaluation(models, dataloader, device, output_dir, experiment_name):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            block_id = batch["block_id"][0]

            scores = []
            for model in models:
                pred_score = model(features)
                scores.append(pred_score.item())
#TODO: change pred_class for classification instead of regression
            avg_score = sum(scores) / len(scores)
            pred_class = 0 if avg_score < 0.5 else (1 if avg_score < 1.5 else 2)

            results.append({
                "block_id": block_id,
                "pred_score": avg_score,
                "pred_class": pred_class
            })

    df = pd.DataFrame(results)
    print(df)
    file_name = os.path.join(output_dir, f"evaluation_results_{experiment_name}.csv")
    df.to_csv(file_name, index=False)
    print(f"Saved predictions to {file_name}")

#TODO change paths
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # experiment_name_base = 'regression_supervised_by_cons_0.5mpp'
    # features_dir = "/data/archief/AMC-data/Barrett/Bolero_features/Virchow_HE_P53_0.5mpp"
    output_dir = "/home/jmgrove/experiments/ensemble/"

    # paths to all 5 folds
    checkpoint_paths = [f"/home/mbotros/experiments/lans_weaklysupervised/{experiment_name_base}_fold_{i + 1}/best_model.ckpt" for i in range(5)]

    dataset = EvalDataset(features_dir, use_p53=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    models = load_models(checkpoint_paths, device)
    run_ensemble_evaluation(models, dataloader, device, output_dir, f"ensemble_{experiment_name_base}")