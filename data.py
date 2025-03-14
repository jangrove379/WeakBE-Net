from pathlib import Path
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn.functional as F
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
        file = [f for f in image_files if block_id in str(f)][0] # select the first file (1) HE.tiff or (2) HE_1.tiff if not (1) not there
        block_identifiers.append((block_id[:-1]))
        files.append(file)

    return block_identifiers, files


class LANSFileDataset:
    """
    A file dataset class used to keep track of the associated files in the LANS dataset.
    """

    def __init__(self, data_dir, stain='HE'):

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
    def __init__(self, features_dir, label_file, binary=False):
        """
        Initializes the BagDataset object by loading features, coordinates, and labels
        from the specified directory and label file.

        Args:
            features_dir (str): Path to the directory containing `.pt` files with WSI features and `.npy` files
                with patch coordinates.
            label_file (str): Path to the CSV file containing the labels for each WSI or block.
            binary (bool, optional): Whether to use a binary classification setting (e.g., non-dysplastic vs dysplastic).
                If `True`, labels will be binarized. Default is `False`.
        """

        # all files and all labels
        self.HE_feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f and 'HE' in f])
        self.P53_feature_files = sorted([f for f in os.listdir(features_dir) if '.pt' in f and 'P53' in f])
        self.coord_files = sorted([f for f in os.listdir(features_dir) if '.npy' in f and 'HE' in f])
        self.labels = pd.read_csv(label_file)

        # filter out where we don't have both a file and a label
        block_id_files = [f.split('HE')[0] for f in self.HE_feature_files]
        self.block_ids = [x for x in self.labels['block id'] if x in block_id_files]

        self.features = []
        self.p53_available = []

        for block_id in self.block_ids:

            he_features = torch.load(os.path.join(features_dir, block_id + 'HE-features.pt'))

            # check if p53 features are available for this block
            matching_p53 = next((item for item in self.P53_feature_files if block_id + '-' in item), None)

            if matching_p53:
                p53_features = torch.load(os.path.join(features_dir, matching_p53))
                stack = torch.cat([he_features, p53_features], dim=0)
                self.features.append(stack)
                self.p53_available.append(block_id)
            else:
                self.features.append(he_features)

        print('{} of which {} have p53 features'.format(len(self.block_ids), len(self.p53_available)))
        # convert grades to tensors
        self.labels = [torch.tensor(self.labels[self.labels['block id'] == b]['dx'].values[0], dtype=torch.long) for b in self.block_ids]
        self.coordinates = [np.load(os.path.join(features_dir, b + 'HE-coords.npy')) for b in self.block_ids]

        if binary:
            self.labels = [torch.tensor(0 if label < 1 else 1, dtype=torch.float64).unsqueeze(0) for label in self.labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray, str]:
                - features (torch.Tensor): Tensor of shape (num_patches, feature_dim)
                - label (torch.Tensor): Tensor of shape (num_patches, 1) containing the label for the corresponding WSI.
                - coordinates (np.ndarray): Array of shape (num_patches, 2) representing the (x, y) coordinates of each patch.
                - block_id (str): Identifier for the block, useful for tracking the source of each bag.
        """
        features = self.features[idx]
        label = self.labels[idx]
        coordinates = self.coordinates[idx]
        block_id = self.block_ids[idx]

        return features, label, coordinates, block_id


def collate_fn(batch):
    """ Makes sure batches are the same length by padding. A mask is used to keep track of padded instances.

    Args:
        batch: (n_feat, n_labels, n_coords, n_block_ids)
    """
    features, labels, coords, block_ids = zip(*batch)
    max_patches = max(f.shape[0] for f in features)

    padded_features = []
    masks = []
    for f in features:
        pad_size = max_patches - f.shape[0]
        padded = F.pad(f, (0, 0, 0, pad_size))
        mask = torch.cat([torch.ones(f.shape[0]), torch.zeros(pad_size)])
        padded_features.append(padded)
        masks.append(mask)

    return torch.stack(padded_features), torch.stack(masks), torch.tensor(labels), coords, block_ids


def get_class_weights(dataset):
    """
    Computes class weights for handling imbalanced datasets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset from which to compute class weights.

    Returns:
        class_weights (torch.Tensor): A tensor containing the class weights, where each weight corresponds to a class label.
    """
    labels = [label.item() for _, label, _, _ in dataset]
    class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(labels), y=labels), dtype=torch.float)
    return class_weights


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

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        yield fold, train_loader, val_loader, class_weights

