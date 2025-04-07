import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight


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

    def __init__(self, features_dir, label_file, use_p53, include_ind=False, binary=False):
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
        self.block_id_files = [f.split('HE')[0] for f in self.HE_feature_files]
        self.block_ids = [x for x in self.labels['block_id'] if x in self.block_id_files]

        # get coordinates and labels
        self.coordinates = [np.load(os.path.join(features_dir, b + 'HE-coords.npy')) for b in self.block_ids]
        self.cons_labels = [
            torch.tensor(self.labels.loc[self.labels['block_id'] == b, 'dx'].iat[0], dtype=torch.float32)
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
        self.features = []
        self.p53_file_available = []

        for block_id in self.block_ids:

            he_features = torch.load(os.path.join(features_dir, block_id + 'HE-features.pt'))

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
            "rater_labels": self.rater_labels[idx]  # (20) Individual rater labels
        }


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


def process_labels(cons_labels, rater_labels, method="random", add_consensus=False):
    """
    Processes a single sample's labels by selecting randomly, averaging, or returning all valid labels.
    """
    rater_labels = rater_labels.squeeze(0)
    valid_rater_labels = rater_labels[(rater_labels != 3) & (rater_labels != 4)]  # exclude not rated (3) and IND (4)

    if add_consensus:
        valid_labels = torch.cat([cons_labels, valid_rater_labels])
    else:
        valid_labels = valid_rater_labels

    if method == 'random':
        random_idx = torch.randint(0, len(valid_labels), (1,))
        return valid_labels[random_idx].unsqueeze(0).float()
    elif method == 'average':
        return valid_labels.float().mean().unsqueeze(0)
    elif method == 'all':
        return valid_labels.float()


def get_dataloaders(dataset, k_folds=5, batch_size=32, seed=42):
    """ Splits the dataset into training and validation sets using K-Fold cross-validation
    and returns corresponding DataLoaders for each fold.

    Args:
        dataset (torch.utils.data.Dataset): The full dataset to be split.
        k_folds (int, optional): Number of folds for cross-validation. Default is 5.
        batch_size (int, optional): Batch size for the DataLoaders. Default is 32.
        seed (int, optional): Seed for drawing samples.

    Returns:
         Generator[Tuple[int, DataLoader, DataLoader, torch.Tensor]]: A generator yielding:
            - fold (int): The current fold number (starting at 1).
            - train_loader (DataLoader): DataLoader for the training subset.
            - val_loader (DataLoader): DataLoader for the validation subset.
            - class_weights (torch.Tensor): Class weights computed from the training subset.

    """
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        fold = fold + 1
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        class_weights = get_class_weights(train_subset)

        print("Size train_subset: {}".format(len(train_subset)))
        print("Class weights train_subset: {}".format(class_weights))
        print("Size val_subset: {}".format(len(val_subset)))

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        yield fold, train_loader, val_loader, class_weights
