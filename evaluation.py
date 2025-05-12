import torch
from torch.utils.data import DataLoader
from train import MILModel
from data import EvalDataset
import os
import pandas as pd


def load_model(checkpoint_path, device):
    model = MILModel.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def run_evaluation(model, dataloader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            block_id = batch["block_id"][0]
#TODO: change pred_class for classification instead of regression
            pred_score = model(features)
            pred_class = torch.where(pred_score < 0.5, 0,
                                     torch.where(pred_score < 1.5, 1, 2))

            results.append({
                "block_id": block_id,
                "pred_score": pred_score.item(),
                "pred_class": pred_class.item()
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    print(f"Saved predictions to {os.path.join(output_dir, 'evaluation_results.csv')}")

#TODO change paths

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # features_dir = "/data/archief/AMC-data/Barrett/Bolero_features/Virchow_HE_P53_1mpp"
    # checkpoint_path = "/home/mbotros/experiments/lans_weaklysupervised/test_evaluation_fold_4/best_model.ckpt"
    # output_dir = "/home/mbotros/experiments/bolero_results/test/"

    dataset = EvalDataset(features_dir, use_p53=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = load_model(checkpoint_path, device)

    run_evaluation(model, dataloader, device, output_dir)