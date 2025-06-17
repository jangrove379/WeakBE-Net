import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rcParams


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


def plot_reliability():
    df = pd.read_csv('experiments/reliability_scores.csv')
    df.rename(columns={'intra': 'Inter-Rater Reliability', 'inter': 'Inter-Rater Reliability', 'overall': 'Average Reliability'}, inplace=True)
    fig, ax = plt.subplots()
    bars = df.iloc[:,1:].plot(kind='bar', width=0.4, edgecolor='black', ax=ax)

    # plt.title('Mean Pairwise Agreement for max-aggregated Diagnoses')
    # plt.ylabel("Mean Pairwise Agreement (Krippendorff's Alpha)")
    plt.xlabel("Pathologists")
    plt.xticks(ticks=np.arange(0,20), labels=[f'{i}' for i in range(1,21)], rotation=0)
    plt.ylim(0, 1)
    plt.savefig('experiments/figs/reliability.png')

# def plot_self_vs_consensus():




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
    setup_plots()
    plot_acc()
    plot_alpha()
    plot_avg()
    plot_avg_acc_panel()
    plot_avg_acc_label()
    plot_reliability()