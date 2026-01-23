#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import yaml

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAUROC, BinaryPrecision,
    BinaryRecall, ConfusionMatrix, MulticlassAccuracy
)

from aggregators.abmil import AttentionMIL
from data import BagDataset, get_dataloaders, process_labels


def train(args):
    pl.seed_everything(args.seed)

    dataset = BagDataset(
        features_dir=args.features_dir,
        label_file=args.label_file,
        use_p53=args.use_p53,
        binary=args.binary,
        include_ind=args.include_ind,
        path_id=args.path_id,
        experiment_mode=args.experiment_mode)

    print('Total length dataset: {}'.format(len(dataset)))
    labels = np.array(dataset.cons_labels)
    num_classes = len(np.unique(labels))
    
    # Get number of raters from the dataset
    num_raters = dataset.get_num_raters()
    print('Number of raters: {}'.format(num_raters))
    print('Label counts: {}'.format(np.unique(labels, return_counts=True)))
    print('Using class weights: {}'.format(args.use_class_weights))
    print('Using features from: {}'.format(args.features_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.features_dir, 'extract_config.yaml')) as file:
        feat_extraction_config = yaml.safe_load(file)

    best_auc_scores = []
    best_acc_scores = []

    print('Running {} folds total'.format(args.k_folds))
    
    fold_iterator = get_dataloaders(
        dataset,
        k_folds=args.k_folds,
        batch_size=args.batch_size,
        seed=args.seed,
        path=args.path_id, 
        experiment_mode=args.experiment_mode)
    
    for fold, train_loader, val_loader, _, class_weights, difficulty_weights in fold_iterator:
        # Skip to specific fold if specified (for Phase 2 per-fold training)
        if args.specific_fold is not None and fold != args.specific_fold:
            continue
        
        fold_dir = os.path.join(args.exp_dir, '{}_fold_{}'.format(args.run_name, fold))
        os.makedirs(fold_dir, exist_ok=True)

        config = {
            'feature_extraction_config': feat_extraction_config,
            'hidden_dim': args.hidden_dim,
            'nr_epochs': args.nr_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'wd': args.wd,
            'k_folds': args.k_folds,
            'drop_out': args.drop_out,
            'class_weights': class_weights.numpy() if class_weights is not None else None,
            'difficulty_weights': difficulty_weights.numpy() if difficulty_weights is not None else None,
            'num_raters': num_raters,
            'wdn_phase': args.wdn_phase
        }

        print('Starting fold {}'.format(fold))
        run = wandb.init(
            project=args.project_name,
            id='{}_fold_{}'.format(args.run_name, fold),
            name='{}_fold_{}'.format(args.run_name, fold),
            config=config,
            dir=fold_dir)

        wandb_logger = WandbLogger(log_model=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=fold_dir,
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            filename='best_model')

        class_weights = class_weights if args.use_class_weights else None
        
        # Choose model based on WDN phase
        if args.wdn_phase == 'doctor_net':
            # Phase 1: Train individual doctor models
            model = DoctorNetModel(
                feature_dim=feat_extraction_config['model']['feature_dim'],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_raters=num_raters,
                lr=args.lr,
                wd=args.wd,
                drop_out=args.drop_out,
                run_dir=fold_dir,
                class_weights=class_weights)
        
        elif args.wdn_phase == 'weighted_doctor_net':
            # Phase 2: Load frozen doctor models and train averaging weights
            if args.doctor_net_checkpoint is None:
                raise ValueError("Must provide --doctor_net_checkpoint for weighted_doctor_net phase")
            
            model = WeightedDoctorNetModel(
                feature_dim=feat_extraction_config['model']['feature_dim'],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_raters=num_raters,
                lr=args.wdn_lr,
                wd=args.wd,
                run_dir=fold_dir,
                doctor_net_checkpoint=args.doctor_net_checkpoint
                )
        
        else:  # baseline
            model = MILModel(
                feature_dim=feat_extraction_config['model']['feature_dim'],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                output_dim=num_classes,
                lr=args.lr,
                wd=args.wd,
                drop_out=args.drop_out,
                run_dir=fold_dir,
                class_weights=class_weights,
                diff_weights=difficulty_weights)

        trainer = pl.Trainer(
            max_epochs=args.nr_epochs,
            devices=[0],
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback])

        trainer.fit(model, train_loader, val_loader)

        # Load best model and validate
        best_model_path = checkpoint_callback.best_model_path
        print('Best model for fold {} saved at: {}'.format(fold, best_model_path))
        
        if args.wdn_phase == 'doctor_net':
            best_model = DoctorNetModel.load_from_checkpoint(
                best_model_path,
                feature_dim=feat_extraction_config['model']['feature_dim'],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_raters=num_raters,
                run_dir=fold_dir,
                class_weights=class_weights)
        
        elif args.wdn_phase == 'weighted_doctor_net':
            best_model = WeightedDoctorNetModel.load_from_checkpoint(
                best_model_path,
                feature_dim=feat_extraction_config['model']['feature_dim'],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_raters=num_raters,
                run_dir=fold_dir,
                doctor_net_checkpoint=args.doctor_net_checkpoint
                )
        
        else:
            best_model = MILModel.load_from_checkpoint(
                best_model_path,
                feature_dim=feat_extraction_config['model']['feature_dim'],
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                output_dim=num_classes,
                run_dir=fold_dir,
                class_weights=class_weights,
                diff_weights=difficulty_weights)

        best_model.final_validation = True
        results = trainer.validate(best_model, dataloaders=val_loader)

        best_acc_scores.append(results[0]['final_val_accuracy'])
        run.finish()

    print(f"AUC: {np.mean(best_auc_scores):.2f} ± {np.std(best_auc_scores):.2f}")
    print(f"Accuracy: {np.mean(best_acc_scores):.2f} ± {np.std(best_acc_scores):.2f}")


class DoctorNetModel(pl.LightningModule):
    """Phase 1: Train individual doctor models with shared backbone"""
    
    def __init__(self, feature_dim=512, hidden_dim=512, num_classes=3, 
                 num_raters=20, lr=1e-5, wd=1e-4, drop_out=0.2,
                 class_weights=None, run_dir=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.num_raters = num_raters
        self.lr = lr
        self.wd = wd
        self.class_weights = class_weights
        self.run_dir = run_dir
        
        # Shared backbone (Inception equivalent)
        self.backbone = AttentionMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output features, not classes
            drop_out=drop_out)
        
        # Individual output heads for each rater
        self.rater_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_raters)
        ])
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics
        self.multi_class_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.conf_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass")
        self.class_labels = ['NDBE', 'LGD', 'HGD'] if num_classes == 3 else ['NDBE', 'IND', 'LGD', 'HGD']
        
        # Validation tracking
        self.val_preds = []
        self.val_labels = []
        self.val_block_ids = []
        self.final_validation = False
    
    def forward(self, bag_features):
        # Get shared representation
        features = self.backbone(bag_features.to(torch.float32))
        
        # Get predictions from all rater heads
        rater_logits = [head(features) for head in self.rater_heads]
        return rater_logits
    
    def training_step(self, batch):
        bag_features = batch["features"]
        rater_labels = batch["rater_labels"].squeeze(0)  # (num_raters,)
        
        rater_logits = self(bag_features)  # List of (1, num_classes) tensors
        
        # Compute loss only for raters who labeled this sample
        total_loss = 0
        num_valid_raters = 0
        
        for rater_idx, logits in enumerate(rater_logits):
            label = rater_labels[rater_idx]
            # Skip if not rated (label == 3) or indefinite (label == 4)
            if label not in [3, 4]:
                loss = self.criterion(logits, label.unsqueeze(0))
                total_loss += loss
                num_valid_raters += 1
        
        if num_valid_raters > 0:
            total_loss = total_loss / num_valid_raters
        
        self.log('train_loss', total_loss, on_step=True, on_epoch=True)
        return total_loss
    
    def validation_step(self, batch):
        bag_features = batch["features"]
        cons_labels = batch["cons_label"]
        block_ids = batch["block_id"]
        
        rater_logits = self(bag_features)
        
        # Average predictions across all rater models
        stacked_logits = torch.stack(rater_logits, dim=0)  # (num_raters, 1, num_classes)
        avg_logits = stacked_logits.mean(dim=0)  # (1, num_classes)
        
        loss = self.criterion(avg_logits, cons_labels)
        
        probs = torch.softmax(avg_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        if self.final_validation:
            self.val_preds.append(preds)
            self.val_labels.append(cons_labels)
            self.val_block_ids.append(block_ids)
        else:
            self.log('val_loss', loss, prog_bar=True)
            acc = self.multi_class_accuracy(preds, cons_labels)
            self.log('val_accuracy', acc, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if self.final_validation:
            preds = torch.cat(self.val_preds)
            labels = torch.cat(self.val_labels)
            
            # Compute metrics
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            self.log('final_val_accuracy', acc)
            
            # Confusion matrix
            conf_mat = self.conf_matrix(preds, labels).cpu().numpy()
            plt.figure(figsize=(7, 7))
            sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                       xticklabels=self.class_labels, yticklabels=self.class_labels)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Doctor Net Confusion Matrix")
            wandb.log({"Confusion Matrix": wandb.Image(plt)})
            plt.close()
            
            # Per-class metrics
            auc_per_class, precision, recall, f1 = self.compute_per_class_metrics(
                preds.cpu().numpy(), labels.cpu().numpy())
            
            for i, class_n in enumerate(self.class_labels):
                self.log(f'final_val_{class_n}_auc', auc_per_class[i])
                self.log(f'final_val_{class_n}_precision', precision[i])
                self.log(f'final_val_{class_n}_recall', recall[i])
    
    def compute_per_class_metrics(self, preds, labels):
        # Compute one-vs-rest AUC for each class
        from sklearn.preprocessing import label_binarize
        labels_bin = label_binarize(labels, classes=range(self.num_classes))
        preds_bin = label_binarize(preds, classes=range(self.num_classes))
        
        auc_per_class = []
        for i in range(self.num_classes):
            if len(np.unique(labels_bin[:, i])) > 1:
                auc_score = roc_auc_score(labels_bin[:, i], preds_bin[:, i])
                auc_per_class.append(auc_score)
            else:
                auc_per_class.append(0.0)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        return auc_per_class, precision, recall, f1
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


class WeightedDoctorNetModel(pl.LightningModule):
    """Phase 2: Freeze doctor models and learn averaging weights"""
    
    def __init__(self, feature_dim=512, hidden_dim=512, num_classes=3,
                 num_raters=20, lr=1e-3, wd=1e-4, run_dir=None,
                 doctor_net_checkpoint=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.num_raters = num_raters
        self.lr = lr
        self.wd = wd
        self.run_dir = run_dir
        
        # Load frozen doctor net
        self.doctor_net = DoctorNetModel.load_from_checkpoint(
            doctor_net_checkpoint,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_raters=num_raters,
            run_dir=run_dir)
        
        # Freeze doctor net
        for param in self.doctor_net.parameters():
            param.requires_grad = False
        self.doctor_net.eval()
        
        # Averaging logits (to be passed through softmax for weights)
        self.averaging_logits = nn.Parameter(torch.zeros(num_raters))
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics (same as DoctorNet)
        self.multi_class_accuracy = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.conf_matrix = ConfusionMatrix(num_classes=num_classes, task="multiclass")
        self.class_labels = ['NDBE', 'LGD', 'HGD'] if num_classes == 3 else ['NDBE', 'IND', 'LGD', 'HGD']
        
        self.val_preds = []
        self.val_labels = []
        self.val_block_ids = []
        self.final_validation = False
    
    def forward(self, bag_features):
        with torch.no_grad():
            rater_logits = self.doctor_net(bag_features)
        
        # Get averaging weights
        weights = torch.softmax(self.averaging_logits, dim=0).unsqueeze(0)  # (1, num_raters)
        
        # Weighted average of rater predictions
        stacked_logits = torch.stack(rater_logits, dim=0)  # (num_raters, 1, num_classes)
        stacked_logits = stacked_logits.squeeze(1)  # (num_raters, num_classes)
        stacked_logits = stacked_logits * weights.T  # (num_raters, num_classes)

        weighted_logits = (stacked_logits.T * weights).T.sum(dim=0, keepdim=True)  # (1, num_classes)
        
        return stacked_logits, weighted_logits, weights
    
    def training_step(self, batch):
        bag_features = batch["features"]
        rater_labels = batch["rater_labels"].squeeze(0)
        
        # Create target from raters who labeled this sample
        valid_labels = rater_labels[(rater_labels != 3) & (rater_labels != 4)]
        if len(valid_labels) == 0:
            return None
        
        # Target is distribution over valid labels
        target_dist = torch.zeros(self.num_classes, device=self.device)
        for label in valid_labels:
            target_dist[label] += 1
        target_dist = target_dist / target_dist.sum()
        
        stacked_logits, weighted_logits, weights = self(bag_features)
        
        # KL divergence loss between prediction and target distribution
        pred_probs = torch.softmax(weighted_logits, dim=1).squeeze(0)
        loss = nn.functional.kl_div(
            torch.log(pred_probs + 1e-10),
            target_dist,
            reduction='batchmean')
        
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch):
        bag_features = batch["features"]
        cons_labels = batch["cons_label"]
        block_ids = batch["block_id"]
        
        stacked_logits, weighted_logits, weights = self(bag_features)
        
        loss = self.criterion(weighted_logits, cons_labels)
        
        probs = torch.softmax(weighted_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        if self.final_validation:
            self.val_preds.append(preds)
            self.val_labels.append(cons_labels)
            self.val_block_ids.append(block_ids)
        else:
            self.log('val_loss', loss, prog_bar=True)
            acc = self.multi_class_accuracy(preds, cons_labels)
            self.log('val_accuracy', acc, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if self.final_validation:
            preds = torch.cat(self.val_preds)
            labels = torch.cat(self.val_labels)
            
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            self.log('final_val_accuracy', acc)
            
            # Save learned weights
            weights = torch.softmax(self.averaging_logits, dim=0).cpu().numpy()
            weights_df = pd.DataFrame({
                'rater_id': range(self.num_raters),
                'weight': weights
            })
            weights_df.to_csv(os.path.join(self.run_dir, 'learned_weights.csv'), index=False)
            
            # Confusion matrix and metrics (same as DoctorNet)
            conf_mat = self.conf_matrix(preds, labels).cpu().numpy()
            plt.figure(figsize=(7, 7))
            sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                       xticklabels=self.class_labels, yticklabels=self.class_labels)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Weighted Doctor Net Confusion Matrix")
            wandb.log({"Confusion Matrix": wandb.Image(plt)})
            plt.close()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


class MILModel(pl.LightningModule):
    """Baseline: Standard MIL model trained on consensus labels"""
    # ... (keep your existing MILModel implementation)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default='WeakBE-Net_WDN')
    parser.add_argument("--binary", type=bool, default=False)
    parser.add_argument("--include_ind", type=bool, default=False)
    parser.add_argument("--use_p53", type=bool, default=True)
    parser.add_argument("--use_class_weights", type=bool, default=True)
    parser.add_argument("--nr_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--exp_dir", type=str, default='/data/archief/AMC-data/Barrett/experiments/jans_experiments')
    parser.add_argument("--features_dir", type=str, default='/data/archief/AMC-data/Barrett/LANS_features/old_stuff/Virchow_HE_P53_1mpp_v2')
    parser.add_argument("--label_file", type=str, default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv')
    parser.add_argument("--wandb_key", type=str)
    parser.add_argument("--test", type=bool, default=True)
    parser.add_argument("--path_id", type=int, default=None)
    parser.add_argument("--experiment_mode", type=str, default="final_cons", 
                       choices=["intra", "intra1000", "final_cons", "final_path"])
    
    # WDN-specific arguments
    parser.add_argument("--wdn_phase", type=str, default="baseline",
                       choices=["baseline", "doctor_net", "weighted_doctor_net"],
                       help="Which phase: baseline (standard MIL), doctor_net (phase 1), or weighted_doctor_net (phase 2)")
    parser.add_argument("--doctor_net_checkpoint", type=str, default=None,
                       help="Path to trained doctor net checkpoint (required for phase 2)")
    parser.add_argument("--wdn_lr", type=float, default=0.03,
                       help="Learning rate for phase 2 (averaging weights)")
    parser.add_argument("--specific_fold", type=int, default=None,
                       help="Run only a specific fold (used for Phase 2 per-fold training)")
    
    args = parser.parse_args()
    train(args)