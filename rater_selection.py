import os
import pandas as pd
import numpy as np
import argparse

from wandb import Api

from sklearn.cluster import KMeans  
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import rcParams


from prediction import get_alpha_scores 




def setup_plotting():
    rcParams['font.family'] = 'STIXGeneral'
    rcParams['font.size'] = 12  # Optional: adjust to match Overleaf's text size

    rcParams.update({
        "font.size": 12,       
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300
        }
    )
    
    FIG_WIDTH = 7
    FIG_HEIGHT = FIG_WIDTH * 0.66  
    FIGSIZE_HORIZONTAL = (FIG_WIDTH, FIG_HEIGHT)

    rcParams['figure.figsize'] = FIGSIZE_HORIZONTAL  


def get_rater_selection():
    setup_plotting()

    preds_cluster = pd.DataFrame()
    pathologists = list(range(1, 21))
    illegible_pathologists = {8, 12, 18}
    viable_pathologists = [p for p in pathologists if p not in illegible_pathologists]  # manual removal of pathologists that did not converge -> ran in notebooks/convergence.ipynb TODO: automate
    for path in viable_pathologists:
        df = pd.read_csv(f"experiments/intra/wo1000_predictions_path_{path}.csv")
        preds_cluster[f"path_{path}"] = df["pred_class"]

    alpha_scores = get_alpha_scores(args.intra_results_dir, args.intra_results_name, args.label_file)
    alpha_scores = alpha_scores[alpha_scores["path_id"].isin(viable_pathologists)]  # filter out illegible pathologists
    alpha_scores["cluster"] = get_clusters(preds_cluster, viable_pathologists)
    alpha_scores.to_csv("experiments/selection.csv", index=False)
    print("Alpha scores with clusters:\n" , alpha_scores)

    selection = alpha_scores.loc[alpha_scores.groupby('cluster')['overall'].idxmax(), ['path_id', "overall", "cluster"]]
    print("The following pathologists are selected using maximum scores per cluster: ", selection["path_id"].values.tolist())
    print("The following pathologists are selected using maximum scores over all clusters: ", alpha_scores.nlargest(5, 'overall')["path_id"].values.tolist())


def get_clusters(preds, viable_pathologists):
    preds_transposed = preds.T  

    kmeans = KMeans(n_clusters=5, random_state=42)  
    labels = kmeans.fit_predict(preds_transposed)  

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(preds_transposed)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap='tab10', s=100)
    plt.title('PCA of Predictions Colored by Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    for i, paths in enumerate(viable_pathologists):
        plt.text(pca_coords[i, 0], pca_coords[i, 1], str(paths), fontsize=9)
    os.makedirs("experiments/figs", exist_ok=True)
    save_path = 'experiments/figs/predictions_clusters_pca.png'
    plt.savefig(save_path)

    return labels





# def assess_convergence_per_model(df, metric='val_loss', window=10, abs_threshold=0.0001):
#     series = df[metric].values
#     converged = False
#     slope = None
#     std_dev = None
#     starting_point_i = None

#     for i in range(len(series) - window + 1):
#         window_vals = series[i:i+window]
#         x = np.arange(window)
#         slope = abs(np.polyfit(x, window_vals, 1)[0])
#         print(abs(slope), "threshold", abs_threshold)


#         if abs(slope) <= abs_threshold:
#             converged = True
#             std_dev = np.std(window_vals)
#             starting_point_i = i
#             break  
#         else: 
#             slope = None
#             std_dev = None
#             starting_point_i = None

#     return {
#         f'{metric}_slope': slope,
#         f'{metric}_std': std_dev,
#         f'{metric}_converged': converged,
#         f'{metric}_starting_point_i': starting_point_i
#     }


# def assess_all_models(model_dfs, window=10):
#     results = []

#     for (model, fold), df in model_dfs.items():
#         res_loss = assess_convergence_per_model(df, metric='val_loss', window=window)
#         res_acc = assess_convergence_per_model(df, metric='val_accuracy', window=window) 

#         results.append({
#             'model': model,
#             'fold': fold,
#             **res_loss,
#             **res_acc
#         })

#     return pd.DataFrame(results)

# def get_runs_from_wandb():
#     api = Api()
#     runs = api.runs(args.wandb_experiment)

#     histories = {}
#     for run in runs:
#         run_id = run.id
            
#         if run_id.startswith("agg"):
#             history = run.history(keys=['val_loss', 'val_accuracy'])
#             path_id = int(run_id.split('_')[2])
#             fold_id = int(run_id.split('_')[4])
#             histories[path_id, fold_id] = history
#             print(f"Stored run ID: {run_id}")


# def get_convergences():
#     histories = pd.DataFrame(get_runs_from_wandb())
#     print(histories)

#     convergence = assess_all_models()
#     fully_converged = convergence[(convergence["val_loss_converged"] == True) & (convergence["val_accuracy_converged"] == True)]
#     acc_converged = convergence[(convergence["val_loss_converged"] == False) & (convergence["val_accuracy_converged"] == True)]
#     loss_converged = convergence[(convergence["val_loss_converged"] == True) & (convergence["val_accuracy_converged"] == False)]




#     fig, axes = plt.subplots(20, 5, figsize=(20, 70))
#     model_dfs = {} 
#     for path_id in range(1, 21):
#         for fold_id in range(1, 6):
#             df = pd.read_csv(f"../../../experiments/histories/agg_Pathologist_{path_id}_fold_{fold_id}_history_final.csv")
#             loss_df = df["val_loss"]
#             acc_df = df["val_accuracy"]

#             model_dfs[(path_id, fold_id)] = df  

#             if (path_id, fold_id) in zip(fully_converged["model"], fully_converged["fold"]):
#                 row = fully_converged[(fully_converged["model"] == path_id) & (fully_converged["fold"] == fold_id)]
#                 ax = axes[path_id-1, fold_id-1]
#                 ax.set_facecolor("lightgrey")
#                 ax.plot(acc_df, label=f"Pathologist {path_id}, Fold {fold_id} (Acc)")
#                 ax.plot(loss_df, label=f"Pathologist {path_id}, Fold {fold_id} (Loss)")
#                 ax.set_title(f"Pathologist {path_id}, Fold {fold_id}")
#                 ax.set_ylim(0, 1)
#                 starting_point_loss = row["val_loss_starting_point_i"].values[0]
#                 starting_point_acc = row["val_accuracy_starting_point_i"].values[0]

#                 ax.axvspan(starting_point_loss, starting_point_loss + 10, alpha=0.5, color='orange')
#                 ax.axvspan(starting_point_acc, starting_point_acc + 10, alpha=0.5, color='blue')

#             elif (path_id, fold_id) in zip(acc_converged["model"], acc_converged["fold"]):
#                 row = acc_converged[(acc_converged["model"] == path_id) & (acc_converged["fold"] == fold_id)]
#                 ax = axes[path_id-1, fold_id-1]
#                 ax.set_facecolor("aliceblue")
#                 ax.plot(acc_df, label=f"Pathologist {path_id}, Fold {fold_id} (Acc)")
#                 ax.plot(loss_df, label=f"Pathologist {path_id}, Fold {fold_id} (Loss)")
#                 ax.set_title(f"Pathologist {path_id}, Fold {fold_id}")
#                 ax.set_ylim(0, 1)

#                 starting_point_acc = row["val_accuracy_starting_point_i"].values[0]
#                 ax.axvspan(starting_point_acc, starting_point_acc + 10, alpha = 0.5, color='blue')

#             elif (path_id, fold_id) in zip(loss_converged["model"], loss_converged["fold"]):
#                 row = loss_converged[(loss_converged["model"] == path_id) & (loss_converged["fold"] == fold_id)]
#                 ax = axes[path_id-1, fold_id-1]
#                 ax.set_facecolor("bisque")
#                 ax.plot(acc_df, label=f"Pathologist {path_id}, Fold {fold_id} (Acc)")
#                 ax.plot(loss_df, label=f"Pathologist {path_id}, Fold {fold_id} (Loss)")
#                 ax.set_title(f"Pathologist {path_id}, Fold {fold_id}")
#                 ax.set_ylim(0, 1)

#                 starting_point_loss = row["val_loss_starting_point_i"].values[0]
#                 ax.axvspan(starting_point_loss, starting_point_loss + 10, alpha=0.5, color='orange')

#             else:
#                 ax = axes[path_id-1, fold_id-1]
#                 ax.set_facecolor("white")
#                 ax.plot(acc_df, label=f"Pathologist {path_id}, Fold {fold_id} (Acc)")
#                 ax.plot(loss_df, label=f"Pathologist {path_id}, Fold {fold_id} (Loss)")
#                 ax.set_title(f"Pathologist {path_id}, Fold {fold_id}")
#                 ax.set_ylim(0, 1)

#             loss_points = merged_df[(merged_df['path'] == path_id) & (merged_df['fold'] == fold_id)][['index_loss', 'value_loss']]
#             acc_points = merged_df[(merged_df['path'] == path_id) & (merged_df['fold'] == fold_id)][['index_acc', 'value_acc']]

#             for _, row_point in loss_points.iterrows():
#                 ax.plot(row_point['index_loss'], row_point['value_loss'], 'o', color='orange')
#                 ax.text(row_point['index_loss'], row_point['value_loss'] + 0.02, f"{row_point['value_loss']:.2f}", 
#                         color='orange', fontsize=8)

#             for _, row_point in acc_points.iterrows():
#                 ax.plot(row_point['index_acc'], row_point['value_acc'], 'o', color='blue')
#                 ax.text(row_point['index_acc'], row_point['value_acc'] + 0.02, f"{row_point['value_acc']:.2f}", 
#                         color='blue', fontsize=8)

# plt.tight_layout()
# plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rater selection based on alpha scores and prediction clustering.")
    parser.add_argument("--label_file", type=str, default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv')
    parser.add_argument('--intra_results_dir', type=str, default='/home/jmgrove/experiments/intra', help='Path to intra results directory')
    parser.add_argument('--intra_results_name', type=str, default='evaluation_results_final_intra', help='Start to name of the intra results file')
    parser.add_argument('--wandb_experiment', type=str, default='jangrove-jg-university-of-amsterdam/WeakBE-Net_no_ind', help='WandB experiment name')

    args = parser.parse_args()

    get_rater_selection()
