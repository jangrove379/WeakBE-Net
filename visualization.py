import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rcParams
from prediction import get_mean_inter_rater_agreement

def plot_acc():
    df = pd.read_csv('experiments/acc_alpha_summary.csv')
    df["experiment"] = df["experiment"].str.replace("virtual_", "virtual")
    df["label_strategy"] = df["experiment"].str.split("_").str[0]
    df["panel_strategy"] = df["experiment"].str.split("_").str[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = {'virtual20': 'r', 'virtual5': 'y', 'consensus': 'b'}  
    scatter = ax.scatter(df['acc_consensus'], df['acc_virtual_5'], df['acc_virtual_20'], 
                         c=df['label_strategy'].map(colors), marker='o')

    for i in range(len(df)):
        ax.text(df['acc_consensus'].iloc[i], df['acc_virtual_5'].iloc[i], df['acc_virtual_20'].iloc[i], 
                df['panel_strategy'].iloc[i], size=10, zorder=1)

    ax.set_xlabel('Acc Consensus')
    ax.set_ylabel('Acc Virtual 5')
    ax.set_zlabel('Acc Virtual 20')

    ax.set_xlim(0.74, 0.83)
    ax.set_ylim(0.74, 0.83)
    ax.set_zlim(0.74, 0.83)

    plt.title('3D Scatter Plot of Accuracies')
    plt.savefig('experiments/figs/acc_3d_scatter_plot.png')

def plot_alpha():
    df = pd.read_csv('experiments/acc_alpha_summary.csv')
    df["experiment"] = df["experiment"].str.replace("virtual_", "virtual")
    df["label_strategy"] = df["experiment"].str.split("_").str[0]
    df["panel_strategy"] = df["experiment"].str.split("_").str[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    colors = {'virtual20': 'r', 'virtual5': 'y', 'consensus': 'b'} 
    scatter = ax.scatter(df['alpha_consensus'], df['alpha_virtual_5'], df['alpha_virtual_20'], 
                         c=df['label_strategy'].map(colors), marker='o')

    for i in range(len(df)):
        ax.text(df['alpha_consensus'].iloc[i], df['acc_virtual_5'].iloc[i], df['alpha_virtual_20'].iloc[i], 
                df['panel_strategy'].iloc[i], size=10, zorder=1)

    ax.set_xlabel('Alpha Consensus')
    ax.set_ylabel('Alpha Virtual 5')
    ax.set_zlabel('Alpha Virtual 20')


    plt.title('3D Scatter Plot of alpha Scores')
    plt.savefig('experiments/figs/alpha_3d_scatter_plot.png')


def plot_avg():
    df = pd.read_csv('experiments/acc_alpha_summary.csv')
    df["experiment"] = df["experiment"].str.replace("virtual_", "virtual")
    df["label_strategy"] = df["experiment"].str.split("_").str[0]
    df["panel_strategy"] = df["experiment"].str.split("_").str[1]

    plt.figure()
    colors = {'virtual20': 'r', 'virtual5': 'y', 'consensus': 'b'}
    plt.scatter(df['avg_alpha'], df['avg_acc'], c=df['label_strategy'].map(colors), marker='o')

    for i in range(len(df)):
        plt.text(df['avg_alpha'].iloc[i], df['avg_acc'].iloc[i], df['panel_strategy'].iloc[i], 
                size=10, zorder=1)

    plt.xlabel('Average Alpha')
    plt.ylabel('Average Accuracy')
    plt.title('2D Scatter Plot of Average Alpha vs Average Accuracy')
    plt.savefig('experiments/figs/avg_alpha_accuracy_scatter_plot.png')

def plot_avg_acc_panel():
    df = pd.read_csv('experiments/acc_alpha_summary.csv')
    df["experiment"] = df["experiment"].str.replace("virtual_", "virtual")
    df["label_strategy"] = df["experiment"].str.split("_").str[0]
    df["panel_strategy"] = df["experiment"].str.split("_").str[1]

    plt.figure()
    plt.bar(df['panel_strategy'], df['avg_acc'], color='b')
    plt.xlabel('Panel Strategy')
    plt.ylabel('Average Accuracy')
    plt.title('Bar Plot of Average Accuracy by Selection of Panel Strategy')
    plt.xticks()
    plt.tight_layout()
    plt.savefig('experiments/figs/avg_acc_by_panel_strategy.png')


def plot_avg_acc_label():
    df = pd.read_csv('experiments/acc_alpha_summary.csv')
    df["experiment"] = df["experiment"].str.replace("virtual_", "virtual")
    df["label_strategy"] = df["experiment"].str.split("_").str[0]
    df["panel_strategy"] = df["experiment"].str.split("_").str[1]

    plt.figure()
    plt.bar(df['label_strategy'], df['avg_acc'], color='b')
    plt.xlabel('Label Strategy')
    plt.ylabel('Average Accuracy')
    plt.title('Bar Plot of Average Accuracy by Aggregation Strategy')
    plt.xticks()
    plt.tight_layout()
    plt.savefig('experiments/figs/avg_acc_by_agg_strategy.png')


def plot_reliability(label_file):
    df = pd.read_csv('experiments/reliability_scores.csv')
    df.rename(columns={'intra': 'Intra-Rater Reliability', 'inter': 'Inter-Rater Reliability', 'overall': 'Overall Reliability'}, inplace=True)
    
    plt.figure()
    total_samples_list = []
    for i in range(1,21):
        test_samples = pd.read_csv(f"experiments/intra/evaluation_results_final_intra_path_{i}_fold_1.csv")
        total_samples = len(test_samples) * 1 / 0.2
        total_samples_list.append(total_samples)
    total_samples_array = np.array(total_samples_list)
    print(total_samples_array)
    colors = plt.cm.Greys(total_samples_array / np.max(total_samples_array)) 
    plt.bar(df.index, df['Intra-Rater Reliability'], color=colors, edgecolor='black')  
    plt.title('Intra-Rater Reliability')
    plt.ylabel("Krippendorff's alpha")
    plt.ylim(0, 1)
    plt.xlabel("Pathologists")
    plt.xticks(ticks=np.arange(0, len(df)), labels=[f'{i}' for i in range(1, len(df) + 1)], rotation=0)
    plt.grid(axis='y')  # Added horizontal grid
    plt.tight_layout()
    
    # Color bar for training samples
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=0, vmax=np.max(total_samples_array)))
    sm.set_array([])  
    cbar = plt.colorbar(sm)
    cbar.set_label('Number of Total Samples')
    
    plt.savefig('experiments/figs/intra.png')


    plt.figure()
    mean_common_samples = get_mean_inter_rater_agreement(label_file, common_samples=True)
    colors = plt.cm.Greys(mean_common_samples / np.max(mean_common_samples))  
    bars = plt.bar(df.index, df['Inter-Rater Reliability'], color=colors, edgecolor='black')
    plt.title('Mean Pairwise Inter-Rater Reliability')
    plt.ylabel("Krippendorff's alpha")
    plt.ylim(0, 1)
    plt.xlabel("Pathologists")
    plt.xticks(ticks=np.arange(0, len(df)), labels=[f'{i}' for i in range(1, len(df) + 1)], rotation=0)
    plt.grid(axis='y')  # Added horizontal grid
    plt.tight_layout()
    
    sm = plt.cm.ScalarMappable(cmap='Greys', norm=plt.Normalize(vmin=0, vmax=np.max(mean_common_samples)))
    sm.set_array([])  
    cbar = plt.colorbar(sm)
    cbar.set_label('Mean Common Samples')
    
    plt.savefig('experiments/figs/inter.png')


    plt.figure()
    plt.bar(df.index, df['Overall Reliability'], color='grey', edgecolor='black')
    plt.title('Overall Reliability')
    plt.ylabel("Mean Krippendorff's alpha")
    plt.ylim(0, 1)
    plt.xlabel("Pathologists")
    plt.xticks(ticks=np.arange(0, len(df)), labels=[f'{i}' for i in range(1, len(df) + 1)], rotation=0)
    plt.grid(axis='y')  # Added horizontal grid
    plt.tight_layout()
    plt.savefig('experiments/figs/overall.png')


def setup_plots():
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



if __name__ == "__main__":
    label_file = 'code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv'


    setup_plots()
    plot_acc()
    plot_alpha()
    plot_avg()
    plot_avg_acc_panel()
    plot_avg_acc_label()
    plot_reliability(label_file)