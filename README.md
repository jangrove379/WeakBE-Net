# Weakly Supervised Barrett's Dysplasia Grading

This repository provides code for predicting dysplasia grade from whole-slide images (WSIs) of Barrett's esophagus biopsies using weakly supervised learning. This approach is based on Multiple Instance Learning (MIL) to perform slide-level classification based on patch-level features extracted from WSIs.

## Pipeline
## Feature Extraction
Model Features are extracted from WSIs using feature_extraction.py.
This step takes several hours, possibly up to two days.

### Training Individual Models
Each pathologist model is trained independently using train.py.
Depending on the sparsity of individual rater sets, this step takes 5-10 minutes per fold, combining to 25-50 minutes per rater.
Usage of loops as part of bash-scripts is advised.
Example command:
```bash
python train.py --experiment_mode intra1000 --path_id <PATHOLOGIST_ID>
```


### Intra-Rater Evaluation
After training, intra_evaluation.py is run for each pathologist to compute intra-rater agreement and derive scores for rater selection.
This script is run in a matter of seconds to a minute per pathologist. 
Usage of loops as part of bash-scripts is advised.
Example command:
```bash
python intra_evaluation.py --path_id <PATHOLOGIST_ID>
```


### Rater Selection
The script rater_selection.py identifies the best-performing pathologists based on intra-rater and inter-rater reliability, using both cluster-based and overall-best strategies.
Execution is expected to take not more than a few seconds.

### Predictions
Predictions are generated using prediction.py:
Runtime per execution is limited to a few seconds. 

#### Consensus model:
```bash
python prediction.py --experiment_name "agg_cons" --output_name <OUTPUT> --panel_pathologists <PATH_IDS>
```


#### Ensemble models:
```bash
python prediction.py --experiment_name "agg" --output_name <OUTPUT> --panel_pathologists <PATH_IDS> --train_pathologists <PATH_IDS>
```

Here, --panel_pathologists specifies the evaluation panel, while --train_pathologists determines the training panel (a selected subset or all 20 pathologists).

### Performance Evaluation
Metrics for accuracy and uncertainty calibration are computed with evaluation.py.
Given the employment of bootstrapping, each evaluation run takes around a minute.

### Visualization
Generate plots and figures from model outputs using visualization.py.
Runtime estimate: ~20 seconds