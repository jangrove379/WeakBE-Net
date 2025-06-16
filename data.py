import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import re
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from torch.utils.data import DataLoader, Subset



def filter_image_files(image_files, stain='HE'):
    """  Filters and selects representative image files for each unique block identifier.
    Ensures that only one file per block is returned, currently just the first available match

     Args:
        image_files (list of Path): List of file paths to image files.
        stain (str): the staining type (HE or P53)

    Returns:
        Tuple[list, list]:
            - block_identifiers (list of str): List of unique block identifiers.
            - files (list of Path): List of selected image file paths, one per block.


    """
    # get all blocks identifiers
    block_ids = sorted(list(set([str(f).split('/')[-1].split(stain)[0] for f in image_files])))
    block_identifiers = []
    files = []

    for block_id in block_ids:
        # find all files with this block id
        file = [f for f in image_files if block_id in str(f)][
            0]  # select the first file (1) HE.tiff or (2) HE_1.tiff if not (1) not there
        block_identifiers.append((block_id[:-1]))
        files.append(file)

    return block_identifiers, files


class LANSFileDataset:
    """
    A file dataset class used to keep track of the associated files in the LANS dataset.
    """

    def __init__(self, data_dir, stain="HE"):

        # location of data
        self.data_dir = data_dir
        self.stain = stain

        # load the images
        self.image_files = [f for f in sorted(data_dir.rglob("*.tiff")) if self.stain in str(f) and 'tm' not in str(f)]

        # only take the first file for each block identifier (otherwise we process so much)
        self.block_identifiers, self.image_files = filter_image_files(self.image_files, stain=stain)

        # get the corresponding annotation and mask files
        self.annotation_files = [str(f).split('.')[0] + '.xml' for f in self.image_files]
        self.mask_files = [str(f).split('.')[0] + '_tm.tiff' for f in self.image_files]

        print('Selected {} {} image files!'.format(len(self.image_files), self.stain))
        # check if all annotations and masks are present before continue
        for f in self.annotation_files:
            assert (Path(f).exists()), 'Annotation missing: {}'.format(f)
        for f in self.mask_files:
            assert (Path(f).exists()), 'Mask missing: {}'.format(f)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        return self.image_files[item], self.annotation_files[item], self.mask_files[item], self.block_identifiers[item]


class BagDataset:
    """
    A PyTorch Dataset class for loading pre-extracted feature representations of WSIs
    along with their corresponding labels and spatial coordinates. Each WSI (bag) consists of multiple
    instances (patch-level features).
    """

    def __init__(self, features_dir, label_file, use_p53, path_id, include_ind=False, binary=False, experiment_mode="final_cons"):
        """
        Initializes the BagDataset by loading features, coordinates, and labels
        from the specified directory and label file.

        Args:
            features_dir (str): Path to the directory containing `.pt` files with WSI features and `.npy` files
                with patch coordinates.
            label_file (str): Path to the CSV file containing the labels for each WSI or block.
            use_p53 (bool): Whether to use features derived from p53 (if available for sample). Default is 'True'.
            binary (bool, optional): Whether to use a binary classification setting (e.g., non-dysplastic vs dysplastic).
                If `True`, labels will be binarized. Default is `False`.
        """

        # all files and all labels
        self.HE_feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f and 'HE' in f])
        self.P53_feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f and 'P53' in f])
        self.coord_files = sorted([f for f in os.listdir(features_dir) if '.npy' in f and 'HE' in f])
        self.use_p53 = use_p53

        # load and update labels df
        self.labels = pd.read_csv(label_file, index_col=0)
        self.labels = self.update_consensus_labels(include_ind=include_ind)
        self.labels = self.update_p53_labels()
        self.labels = self.update_individual_labels()

        # filter out where we don't have both a file and a label
        self.block_id_files = [f.split('-HE')[0] for f in self.HE_feature_files]
        self.block_ids = [x for x in self.labels['block_id'] if x in self.block_id_files and (int(x.split("-")[1]) <= 1100)]

        # for intra-rater agreement, filter out instances without a label
        if (path_id is not None) & ((experiment_mode == "intra") |  (experiment_mode == "final_path") | (experiment_mode == "intra1000")):
            path_id = path_id + 2 # +2 because block_id, dx, and p53 are included but path_id is supposed to NOT be 0-indexed
            mask = ~self.labels.iloc[:, path_id].isin([3, 4])
            valid = self.labels.loc[mask]
            valid = valid[valid['block_id'].isin(self.block_ids)].reset_index(drop=True)
            self.block_ids = valid['block_id'].tolist()

            self.coordinates = [
                np.load(os.path.join(features_dir, f"{bid}-HE-coords.npy"))
                for bid in self.block_ids
            ]

            self.cons_labels = [
                torch.tensor(dx, dtype=torch.long)
                for dx in valid["dx"].tolist()
            ]

            self.p53_labels = [
                torch.tensor(p, dtype=torch.long)
                for p in valid["p53"].tolist()
            ]

            rater_start = valid.columns.get_loc("p53") + 1
            self.rater_labels = [
                torch.tensor(row[rater_start:].astype(int).values, dtype=torch.long)
                for _, row in valid.iterrows()
            ]

        elif experiment_mode == 'final_cons':
            # get coordinates and labels
            self.coordinates = [np.load(os.path.join(features_dir, b + '-HE-coords.npy')) for b in self.block_ids]
            self.cons_labels = [
                torch.tensor(self.labels.loc[self.labels['block_id'] == b, 'dx'].iat[0], dtype=torch.long)
                for b in self.block_ids]
            self.p53_labels = [torch.tensor(self.labels.loc[self.labels['block_id'] == b, 'p53'].iat[0], dtype=torch.long)
                            for b in self.block_ids]
            self.rater_labels = [torch.tensor(self.labels.loc[self.labels['block_id'] == b, self.labels.columns[
                                                                                            self.labels.columns.get_loc(
                                                                                                'p53') + 1:]].values.flatten(),
                                            dtype=torch.long)
                                for b in self.block_ids]

        if binary:
            self.cons_labels = [torch.tensor(0 if label < 1 else 1, dtype=torch.float64).unsqueeze(0) for label in
                                self.cons_labels]
            
        #case difficulty
        temp_diff = self.calculate_case_difficulty()
        self.difficulty = [
                torch.tensor(diff, dtype=torch.float64)
                    for diff in temp_diff.values.to_numpy().flatten()
                ]
        self.filtered_difficulty = [
            torch.tensor(temp_diff.loc[temp_diff.index == b].item(), dtype=torch.float64)
            for b in self.block_ids
        ]
           
        self.features = []
        self.p53_file_available = []

        for block_id in self.block_ids:

            he_features = torch.load(os.path.join(features_dir, block_id + '-HE-features.pt'))

            # check if p53 features are available for this block
            matching_p53 = next((item for item in self.P53_feature_files if block_id + '-' in item), None)

            if matching_p53 and self.use_p53:
                p53_features = torch.load(os.path.join(features_dir, matching_p53))
                stack = torch.cat([he_features, p53_features], dim=0)
                self.features.append(stack)
                self.p53_file_available.append(1)
            else:
                self.features.append(he_features)
                self.p53_file_available.append(0)


        print('{} of which {} have p53 features available'.format(len(self.block_ids), sum(self.p53_file_available)))
        print('Using p53 features: {}'.format(self.use_p53))

    def update_consensus_labels(self, include_ind=False):
        """
                        include_ind=False   include_ind=True
        1 = NDBE        => 0                => 0
        2 = IND         => Nan              => 1
        3 = LGD         => 1                => 2
        4 = HGD         => 2                => 3
        """
        labels = self.labels.copy()

        if include_ind:
            labels["dx"] = labels["dx"].replace({1: 0, 2: 1, 3: 2, 4: 3})
        else:
            labels["dx"] = labels["dx"].replace({1: 0, 2: np.nan, 3: 1, 4: 2})
            labels = labels.dropna(subset=["dx"])

        return labels

    def update_individual_labels(self):
        """
        0 = not rated   => 3
        1 = NDBE        => 0
        2 = IND         => 4
        3 = LGD         => 1
        4 = HGD         => 2
        """
        labels = self.labels.copy()
        columns_after_dx = labels.columns[labels.columns.get_loc("p53") + 1:]
        labels[columns_after_dx] = labels[columns_after_dx].replace({0: 3, 1: 0, 2: 4, 3: 1, 4: 2})
        return labels

    def calculate_case_difficulty(self):
        labels = self.labels.copy()

        basis_agreement = labels.drop(["p53"], axis=1).reset_index(drop=True)
        basis_ranges = labels.drop(["dx", "p53"], axis=1)

        percentage_agreement = pd.DataFrame(columns=["agreement"])
        diagnoses_ranges = pd.DataFrame()

        for case_id in labels["block_id"]:

            patient_data = basis_agreement.loc[basis_agreement["block_id"] == case_id].drop(["block_id"], axis=1).dropna(axis=1, how="all")

            consensus_dx = patient_data["dx"].values[0]
            value_counts = patient_data.T.value_counts()
            consensus_rating_count = value_counts[consensus_dx] - 1 # note: subtract 1 to not count the consensus dx itself

            count = patient_data.shape[1] - 1

            corr = pd.DataFrame(data=(consensus_rating_count / count), columns=["agreement"], index=[case_id])
            percentage_agreement = pd.concat([percentage_agreement, corr])

            case_data = basis_ranges[basis_ranges["block_id"] == case_id].drop(["block_id"], axis=1).dropna(axis=1, how="all")
            diagnoses_ranges = pd.concat([diagnoses_ranges, pd.DataFrame(case_data.T.value_counts()).T])

        diagnoses_ranges.index = basis_ranges["block_id"].unique()
        diagnoses_ranges.rename(columns={1.0: "1", 2.0: "2", 3.0: "3", 4.0: "4"}, inplace=True)
        num_of_distinct_diagnoses = diagnoses_ranges.notna().sum(axis=1)

        case_diff = pd.DataFrame(percentage_agreement)
        case_diff["distinct_diagnoses"] = num_of_distinct_diagnoses

        scaler = MinMaxScaler()
        case_diff[["distinct_diagnoses_scaled"]] = (1 - scaler.fit_transform(case_diff[["distinct_diagnoses"]]))

        case_diff["comb_cont"] = (case_diff["agreement"] + case_diff["distinct_diagnoses_scaled"]) / 2
        case_diff["comb_final"] = pd.qcut(case_diff["comb_cont"], q=3, labels=[3, 2, 1])

        return case_diff["comb_final"]



    def update_p53_labels(self):
        """ New mapping:
        0: OE
        1: WT
        2: NM
        3: DC
        4: IND
        5: not present
        """
        labels = self.labels.copy()
        labels["p53"] = self.labels["p53"].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5})

        return labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        return {
            "features": self.features[idx],  # (num_patches, feature_dim)
            "cons_label": self.cons_labels[idx],  # (1,) Consensus label
            "coordinates": self.coordinates[idx],  # (num_patches, 2) Patch (x, y) coordinates
            "block_id": self.block_ids[idx],  # Block identifier
            "p53_file_available": self.p53_file_available[idx],  # Boolean indicating p53 file availability
            "p53_label": self.p53_labels[idx],  # p53 mutation status label
            "rater_labels": self.rater_labels[idx],  # (20) Individual rater labels
            "difficulty": self.difficulty[idx]  # Difficulty of the case
        }


class EvalDataset(Dataset):
    """
    A lightweight Dataset for evaluation/inference.
    Only loads features (+ coordinates) and optionally block_ids.
    """

    def __init__(self, features_dir, use_p53=True):
        self.HE_feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f and 'HE' in f])
        self.P53_feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f and 'P53' in f])
        self.coord_files = sorted([f for f in os.listdir(features_dir) if '.npy' in f and 'HE' in f])
        self.use_p53 = use_p53
        self.features_dir = features_dir
        self.block_ids = [f.split('-HE')[0] for f in self.HE_feature_files]

        self.samples = []
        for block_id in self.block_ids:
            sample = {"block_id": block_id}

            coords_path = os.path.join(features_dir, block_id + '-HE-coords.npy')
            sample["coordinates"] = np.load(coords_path)

            he_features_path = os.path.join(features_dir, block_id + '-HE-features.pt')
            he_features = torch.load(he_features_path)

            matching_p53 = next((item for item in self.P53_feature_files if block_id + '-' in item), None)

            if matching_p53 and self.use_p53:
                p53_features_path = os.path.join(features_dir, matching_p53)
                p53_features = torch.load(p53_features_path)
                features = torch.cat([he_features, p53_features], dim=0)
            else:
                features = he_features

            sample["features"] = features

            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def get_class_weights(dataset):
    """
    Computes class weights for handling imbalanced datasets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset from which to compute class weights.

    Returns:
        class_weights (torch.Tensor): A tensor containing the class weights, where each weight corresponds to a class label.
    """

    labels = [sample["cons_label"].item() for sample in dataset]
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(labels), y=labels),
                                 dtype=torch.float)

    return class_weights


def process_labels(cons_labels, rater_labels, method="random", add_consensus=False, path_id=None):
    """
    Processes a single sample's labels by selecting randomly, averaging, or returning all valid labels.
    """
    rater_labels = rater_labels.squeeze(0)
    valid_rater_labels = rater_labels[(rater_labels != 3) & (rater_labels != 4)]

    if add_consensus:
        valid_labels = torch.cat([cons_labels, valid_rater_labels])
    else:
        valid_labels = valid_rater_labels

    if method == 'random':
        random_idx = torch.randint(0, len(valid_labels), (1,))
        return valid_labels[random_idx].unsqueeze(0).long()
    elif method == 'average':
        return valid_labels.float().mean().unsqueeze(0)
    elif method == 'all':
        return valid_labels.float()     
    elif method == 'path':
        if path_id is None:
            raise ValueError("path_id must be provided when method is 'path'")
        return rater_labels[path_id-1].unsqueeze(0).long()



def get_dataloaders(dataset, k_folds=5, batch_size=32, seed=42, test_size=0.2, path=None, experiment_mode="final_cons"):
    """
    Balances the dataset by undersampling based on filtered_difficulty,
    then performs stratified test split and stratified K-Fold CV.

    Yields per fold:
        fold               : int
        train_loader       : DataLoader
        val_loader         : DataLoader
        test_loader        : DataLoader
        class_weights      : torch.Tensor
        difficulty_weights : torch.Tensor
    """

    if (path is not None) & ((experiment_mode == 'intra') or (experiment_mode == 'intra1000')):

        if experiment_mode == 'intra1000':
            filtered_indices = [
                i for i, block_id in enumerate(dataset.block_ids)
                if int(block_id.split("-")[1]) <= 1000
            ]
        else:
            filtered_indices = list(range(len(dataset)))  # no filtering

        filtered_difficulties = np.array([dataset.filtered_difficulty[i].item() for i in filtered_indices])
        dataset = Subset(dataset, filtered_indices)

        balanced_subset, balanced_difficulties = undersample_dataset(dataset, filtered_difficulties, seed=seed)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_idx, test_idx = next(sss.split(np.zeros(len(balanced_difficulties)), balanced_difficulties))

        test_subset = Subset(balanced_subset, test_idx)
        train_val_difficulties = balanced_difficulties[train_val_idx]

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for fold, (tr_idx_local, val_idx_local) in enumerate(skf.split(train_val_idx, train_val_difficulties), start=1):
            tr_idx_global = [train_val_idx[i] for i in tr_idx_local]
            val_idx_global = [train_val_idx[i] for i in val_idx_local]

            train_subset = Subset(balanced_subset, tr_idx_global)
            val_subset   = Subset(balanced_subset, val_idx_global)

            class_weights = get_class_weights(train_subset)

            tr_difficulties = balanced_difficulties[tr_idx_global]
            counts = Counter(tr_difficulties)
            total = len(tr_difficulties)
            inv_freq = {d: total / counts[d] for d in counts}
            difficulty_weights = torch.tensor([inv_freq[d] for d in tr_difficulties], dtype=torch.float32)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
            
            yield fold, train_loader, val_loader, test_loader, class_weights, difficulty_weights
    else:
        test_idx = [
            i for i, block_id in enumerate(dataset.block_ids)
            if 1000 < int(block_id.split("-")[1]) <= 1100
        ]
        
        # test_subset = Subset(dataset, test_idx) 
        train_val_idx = [i for i in range(len(dataset)) if i not in test_idx]
        train_val_subset = Subset(dataset, train_val_idx)
 
        kfold = KFold(n_splits=k_folds, shuffle=False)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_subset)):
            fold = fold + 1
            train_subset = Subset(train_val_subset, train_idx)
            val_subset = Subset(train_val_subset, val_idx)

            class_weights = get_class_weights(train_subset)

            print("Size test_subset: {}".format(len(test_idx)))
            print("Size train_subset: {}".format(len(train_subset)))
            print("Class weights train_subset: {}".format(class_weights))
            print("Size val_subset: {}".format(len(val_subset)))

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

            yield fold, train_loader, val_loader, None, class_weights, None
           


def undersample_dataset(dataset, filtered_difficulties, seed=42):
    """
    Undersamples dataset to balance it by difficulty.

    Args:
        dataset: original dataset
        filtered_difficulties: np.ndarray of difficulty labels
        seed: random seed for reproducibility

    Returns:
        balanced_dataset: Subset of original dataset
        balanced_difficulties: np.ndarray of difficulty labels for the subset
    """
    rng = np.random.default_rng(seed)
    unique_difficulties, counts = np.unique(filtered_difficulties, return_counts=True)
    min_count = counts.min()

    indices = []
    for difficulty in unique_difficulties:
        difficulty_indices = np.where(filtered_difficulties == difficulty)[0]
        rng.shuffle(difficulty_indices)
        indices.extend(difficulty_indices[:min_count])

    indices = sorted(indices) 
    balanced_dataset = Subset(dataset, indices)
    balanced_difficulties = filtered_difficulties[indices]

    return balanced_dataset, balanced_difficulties

