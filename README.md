# Weakly Supervised Barrett's Dysplasia Grading

This repository provides code for predicting dysplasia grade from whole-slide images (WSIs) of Barrett's esophagus biopsies using weakly supervised learning. This approach is based on Multiple Instance Learning (MIL) to perform slide-level classification based on patch-level features extracted from WSIs.

## Pipeline
### Training Individual Models
Each pathologist model is trained independently using train.py.
Example command:

python train.py --experiment_mode intra1000 --path_id <PATHOLOGIST_ID>


### Intra-Rater Evaluation
After training, intra_evaluation.py is run for each pathologist to compute intra-rater agreement and derive scores for rater selection.
Example command:

python intra_evaluation.py --path_id <PATHOLOGIST_ID>


### Rater Selection
The script rater_selection.py identifies the best-performing pathologists based on intra-rater and inter-rater reliability, using both cluster-based and overall-best strategies.

Predictions
Predictions are generated using prediction.py:

Consensus model:

python prediction.py --experiment_name "agg_cons" --output_name <OUTPUT> --panel_pathologists <PATH_IDS>


Ensemble models:

python prediction.py --experiment_name "agg" --output_name <OUTPUT> --panel_pathologists <PATH_IDS> --train_pathologists <PATH_IDS>


Here, --panel_pathologists specifies the evaluation panel, while --train_pathologists determines the training panel (from 1 to 20 pathologists or a selected subset).

### Performance Evaluation
Metrics for accuracy and uncertainty calibration are computed with evaluation.py.

### Visualization
Generate plots and figures from model outputs using visualization.py.
