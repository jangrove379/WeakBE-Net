import pandas as pd

from prediction import get_alpha_scores 


def get_rater_selection():
    avg_alpha = get_alpha_scores()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rater selection based on alpha scores and prediction clustering.")
    parser.add_argument("--label_file", type=str, default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv')
    args = parser.parse_args()

    get_rater_selection()
