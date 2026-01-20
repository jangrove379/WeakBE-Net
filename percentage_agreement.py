import numpy as np
import pandas as pd
import os


def calculate_percentage_agreement_mode(row):
    freq_preds = [row['frequency_classes_prediction_0'], row['frequency_classes_prediction_1'], row['frequency_classes_prediction_2']]
    print("freq_preds", freq_preds)
    mode = np.argmax(freq_preds) 
    print("mode", str(mode))
    total_freq = sum(freq_preds)
    percentage_agreement_mode = freq_preds[mode] / total_freq if total_freq != 0 else 0
    
    return percentage_agreement_mode


def add_percentage_agreement():
    for experiment in sorted(os.listdir("experiments/final_eval/")):
        df = pd.read_csv(f'experiments/final_eval/{experiment}')
        df['percentage_agreement_mode'] = df.apply(calculate_percentage_agreement_mode, axis=1)
        df.to_csv(f'experiments/final_eval/{experiment}', index=False)


if __name__ == "__main__":
    add_percentage_agreement()