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
        include_ind=args.include_ind)

    print('Total length dataset: {}'.format(len(dataset)))
    labels = np.array(dataset.cons_labels)
    num_classes = len(np.unique(labels))
    print('Label counts: {}'.format(np.unique(labels, return_counts=True)))
    print('Using class weights: {}'.format(args.use_class_weights))
    print('Using features from: {}'.format(args.features_dir))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config for extraction
    with open(os.path.join(args.features_dir, 'extract_config.yaml')) as file:
        feat_extraction_config = yaml.safe_load(file)

    # log best
    best_auc_scores = []
    best_acc_scores = []

    print('Running {} folds total'.format(args.k_folds))
    for fold, train_loader, val_loader, class_weights in get_dataloaders(dataset,
                                                                         k_folds=args.k_folds,
                                                                         batch_size=args.batch_size,
                                                                         seed=args.seed):
        fold_dir = os.path.join(args.exp_dir, '{}_fold_{}'.format(args.run_name, fold))
        os.makedirs(fold_dir, exist_ok=True)

        # load the feature extraction config that was used
        config = {'feature_extraction_config': feat_extraction_config,
                  'hidden_dim': args.hidden_dim,
                  'nr_epochs': args.nr_epochs,
                  'batch_size': args.batch_size,
                  'lr': args.lr,
                  'wd': args.wd,
                  'k_folds': args.k_folds,
                  'drop out': args.drop_out,
                  'class weights:': class_weights.numpy()}

        print('Starting fold {}'.format(fold))
        run = wandb.init(project=args.project_name,
                         id='{}_fold_{}'.format(args.run_name, fold),
                         name='{}_fold_{}'.format(args.run_name, fold),
                         config=config,
                         dir=fold_dir)

        wandb_logger = WandbLogger(log_model=True)
        checkpoint_callback = ModelCheckpoint(dirpath=fold_dir,
                                              save_top_k=1,
                                              monitor='val_loss',
                                              mode='min',
                                              filename='best_model')

        class_weights = class_weights if args.use_class_weights else None
        model = MILModel(feature_dim=feat_extraction_config['model']['feature_dim'],
                         hidden_dim=args.hidden_dim,
                         num_classes=num_classes,
                         output_dim=1,
                         lr=args.lr,
                         wd=args.wd,
                         drop_out=args.drop_out,
                         run_dir=fold_dir,
                         class_weights=class_weights)

        trainer = pl.Trainer(max_epochs=args.nr_epochs,
                             devices=[0],
                             logger=wandb_logger,
                             log_every_n_steps=1,
                             callbacks=[checkpoint_callback])

        trainer.fit(model, train_loader, val_loader)

        # Load best model and validate
        best_model_path = checkpoint_callback.best_model_path
        print('Best model for fold {} saved at: {}'.format(fold, best_model_path))
        best_model = MILModel.load_from_checkpoint(best_model_path,
                                                   feature_dim=feat_extraction_config['model']['feature_dim'],
                                                   hidden_dim=args.hidden_dim,
                                                   num_classes=num_classes,
                                                   output_dim=1,
                                                   run_dir=fold_dir,
                                                   binary=args.binary,
                                                   class_weights=class_weights)

        best_model.final_validation = True
        results = trainer.validate(best_model, dataloaders=val_loader)

        auc_metric = 'final_val_auc' if args.binary else 'final_val_NDBE_auc'
        best_auc_scores.append(results[0][auc_metric])
        best_acc_scores.append(results[0]['final_val_accuracy'])
        run.finish()

    # print average over folds
    print(f"AUC: {np.mean(best_auc_scores):.2f} ± {np.std(best_auc_scores):.2f}")
    print(f"Accuracy: {np.mean(best_acc_scores):.2f} ± {np.std(best_acc_scores):.2f}")


class MILModel(pl.LightningModule):
    """ Implements a standard MIL model for regression.
    """

    def __init__(self,
                 feature_dim=512,
                 hidden_dim=512,
                 output_dim=1,
                 num_classes=3,
                 lr=1e-5,
                 wd=1e-4,
                 drop_out=0.2,
                 class_weights=None,
                 run_dir=None):

        super(MILModel, self).__init__()
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.model = AttentionMIL(feature_dim=feature_dim,
                                  hidden_dim=hidden_dim,
                                  output_dim=output_dim,
                                  drop_out=drop_out)
        self.lr = lr
        self.wd = wd
        self.criterion = nn.MSELoss()

        # Metrics + Confusion Matrix
        if num_classes == 2:
            self.binary_accuracy = BinaryAccuracy()
            self.binary_auroc = BinaryAUROC()
            self.binary_precision = BinaryPrecision()
            self.binary_recall = BinaryRecall()
            self.metrics = {'accuracy': self.binary_accuracy,
                            'auc': self.binary_auroc,
                            'precision': self.binary_precision,
                            'recall': self.binary_recall}
            self.conf_matrix = ConfusionMatrix(num_classes=2, task="binary")
            self.class_labels = ['Non-Dysplastic', 'Dysplastic']
        elif num_classes == 3:
            self.multi_class_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='micro')
            self.metrics = {'accuracy': self.multi_class_accuracy}
            self.conf_matrix = ConfusionMatrix(num_classes=self.num_classes, task="multiclass")
            self.class_labels = ['NDBE', 'LGD', 'HGD']
        elif num_classes == 4:
            self.multi_class_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='micro')
            self.metrics = {'accuracy': self.multi_class_accuracy}
            self.conf_matrix = ConfusionMatrix(num_classes=self.num_classes, task="multiclass")
            self.class_labels = ['NDBE', 'IND', 'LGD', 'HGD']

        # for final validation round: store predictions
        self.val_pred_score = []
        self.val_pred_class = []
        self.val_labels = []
        self.val_block_ids = []
        self.val_p53_available = []
        self.val_p53_labels = []
        self.final_validation = False
        self.run_dir = run_dir

    def forward(self, bag_features):
        logits = self.model(bag_features.to(torch.float32))
        return logits.squeeze(0)

    def training_step(self, batch, batch_idx):
        bag_features = batch["features"]
        cons_labels = batch["cons_label"]
        raters_labels = batch["rater_labels"]
        target = process_labels(cons_labels, raters_labels, method='all', add_consensus=False)

        pred_score = self(bag_features)

        loss = self.criterion(pred_score.expand_as(target), target)
        pred_class = torch.where(pred_score < 0.5, 0,
                                 torch.where(pred_score < 1.5, 1, 2))

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        for name, metric in self.metrics.items():
            self.log('train_{}'.format(name), metric(pred_class, cons_labels), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        bag_features = batch["features"]
        cons_labels = batch["cons_label"]
        block_ids = batch["block_id"]
        p53_file_available = batch["p53_file_available"]
        p53_labels = batch["p53_label"]
        raters_labels = batch["rater_labels"]
        target = process_labels(cons_labels, raters_labels, method='all', add_consensus=False)

        pred_score = self(bag_features)

        loss = self.criterion(pred_score.expand_as(target), target)
        pred_class = torch.where(pred_score < 0.5, 0,
                                 torch.where(pred_score < 1.5, 1, 2))

        # Store predictions if final validation
        if self.final_validation:
            self.val_pred_score.append(pred_score)
            self.val_pred_class.append(pred_class)
            self.val_labels.append(cons_labels)
            self.val_block_ids.append(block_ids)
            self.val_p53_available.append(p53_file_available)
            self.val_p53_labels.append(p53_labels)
        else:
            self.log('val_loss', loss, prog_bar=True)
            for name, metric in self.metrics.items():
                self.log('val_{}'.format(name), metric(pred_class, cons_labels), prog_bar=True)

        return loss

    def compute_confusion_matrix(self, preds, labels):
        """Generate and log confusion matrix"""
        conf_mat = self.conf_matrix(preds, labels).cpu().numpy()
        plt.figure(figsize=(7, 7))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_labels,
                    yticklabels=self.class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Validation Confusion Matrix")
        wandb.log({"Confusion Matrix": wandb.Image(plt)})
        plt.savefig(os.path.join(self.run_dir, 'confusion_matrix.png'))
        plt.close()

    def compute_per_class_metrics(self, preds, score, labels):
        # compute auc per class
        auc_per_class = []
        for i in range(len(self.class_labels)):
            binary_true = (labels == i).astype(int)

            if i == 0:
                auc_score = roc_auc_score(binary_true, -score)  # if NDBE is seen as the positive class
            else:
                auc_score = roc_auc_score(binary_true, score)

            auc_per_class.append(auc_score)

        # compute precision recall and f1 per class
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
        return auc_per_class, precision, recall, f1

    def compute_roc_curve(self, labels, score):
        fpr, tpr, _ = roc_curve(labels, score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        wandb.log({"ROC Curve": wandb.Image(plt)})
        plt.savefig(os.path.join(self.run_dir, 'roc_curve.png'))
        plt.close()

    def on_validation_epoch_end(self):
        """Compute confusion matrix only at final validation round"""
        if self.final_validation:
            preds = torch.cat(self.val_pred_class)
            score = torch.cat(self.val_pred_score)
            labels = torch.cat(self.val_labels)
            p53_labels = torch.cat(self.val_p53_labels)
            p53_available = [item.cpu().numpy() for tup in self.val_p53_available for item in tup]
            block_ids = [item for tup in self.val_block_ids for item in tup]

            print('Final evaluation on: {} samples'.format(len(preds)))
            self.compute_confusion_matrix(preds, labels)

            # store prediction results as csv: to-do add probabilities
            results_df = pd.DataFrame({'block_id': block_ids,
                                       'label': labels.cpu().numpy(),
                                       'p53_label': p53_labels.cpu().numpy(),
                                       'p53_available': p53_available,
                                       'pred_score': score.cpu().numpy(),
                                       'pred_class': preds.cpu().numpy()})

            if self.num_classes == 1:  # binary
                results_df['pred_score'] = score.cpu().numpy()
                self.compute_roc_curve(labels.cpu().numpy().astype(int), score.cpu().numpy())
                self.log('final_val_accuracy', accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy()))
            else:  # multi-class
                self.log('final_val_accuracy', accuracy_score(y_true=labels.cpu().numpy(), y_pred=preds.cpu().numpy()))
                auc_per_class, precision, recall, f1 = self.compute_per_class_metrics(preds=preds.cpu().numpy(),
                                                                                      score=score.cpu().numpy(),
                                                                                      labels=labels.cpu().numpy())
                for i, class_n in enumerate(self.class_labels):
                    self.log('final_val_{}_auc'.format(class_n), auc_per_class[i])

            print(results_df)
            results_save_path = os.path.join(self.run_dir, 'results.csv')
            print('Saving results to: {}'.format(results_save_path))
            results_df.to_csv(results_save_path, index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='ABMIL_regression_please_all_raters-T3', help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--project_name", type=str, default='WeakBE-Net_no_ind', help="the name of this project")
    parser.add_argument("--binary", type=bool, default=False, help="whether to run in binary setup")
    parser.add_argument("--include_ind", type=bool, default=False, help="whether to include IND cases")
    parser.add_argument("--use_p53", type=bool, default=True, help="whether use p53 features if available")
    parser.add_argument("--use_class_weights", type=bool, default=True, help="whether to use frequency based weights")
    parser.add_argument("--nr_epochs", type=int, default=150, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="the size of mini batches")
    parser.add_argument("--hidden_dim", type=int, default=16, help="hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-5, help="initial the learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay (L2)")
    parser.add_argument("--drop_out", type=float, default=0.0, help="drop out rate")
    parser.add_argument("--k_folds", type=int, default=5, help="number of folds")
    parser.add_argument("--exp_dir", type=str,
                        default='/home/mbotros/experiments/lans_weaklysupervised/')
    parser.add_argument("--features_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS_features/Virchow_HE_P53')
    parser.add_argument("--label_file", type=str,
                        default='/data/archief/AMC-data/Barrett/LANS/lans_train_labels_with_ind.csv')
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    parser.add_argument("--test", type=bool, help="whether to also test", default=True)
    args = parser.parse_args()

    train(args)
