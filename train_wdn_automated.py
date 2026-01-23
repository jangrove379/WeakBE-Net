#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated script for two-phase WDN training.
Properly handles Phase 1 → Phase 2 transition for each fold.
"""

import argparse
import os
import subprocess
import glob


def train_phase1(args):
    """Train Doctor Net (Phase 1) for all folds."""
    print("\n" + "="*80)
    print("PHASE 1: Training Doctor Net for all {} folds".format(args.k_folds))
    print("="*80 + "\n")
    
    cmd = [
        "python", "code/WeakBE-Net/train_wdn.py",
        "--run_name", f"{args.run_name}_phase1",
        "--wdn_phase", "doctor_net",
        "--nr_epochs", str(args.phase1_epochs),
        "--lr", str(args.lr),
        "--k_folds", str(args.k_folds),
        "--batch_size", str(args.batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--drop_out", str(args.drop_out),
        "--wd", str(args.wd),
        "--exp_dir", args.exp_dir,
        "--features_dir", args.features_dir,
        "--label_file", args.label_file,
        "--project_name", args.project_name,
        "--experiment_mode", args.experiment_mode,
        "--seed", str(args.seed)
    ]
    
    if args.use_p53:
        cmd.extend(["--use_p53", "True"])
    if args.use_class_weights:
        cmd.extend(["--use_class_weights", "True"])
    if args.path_id is not None:
        cmd.extend(["--path_id", str(args.path_id)])
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n[ERROR] Phase 1 training failed!")
        return False
    
    print("\n[SUCCESS] Phase 1 completed for all folds")
    return True


def find_phase1_checkpoints(args):
    """Find all Phase 1 checkpoint files."""
    base_pattern = os.path.join(
        args.exp_dir, 
        f"{args.run_name}_phase1_fold_*",
        "best_model.ckpt"
    )
    
    checkpoints = sorted(glob.glob(base_pattern))
    
    print(f"\nFound {len(checkpoints)} Phase 1 checkpoints:")
    for ckpt in checkpoints:
        print(f"  - {ckpt}")
    
    return checkpoints


def train_phase2(args, phase1_checkpoints):
    """Train Weighted Doctor Net (Phase 2) for each fold."""
    print("\n" + "="*80)
    print("PHASE 2: Training Weighted Doctor Net for each fold")
    print("="*80 + "\n")
    
    if len(phase1_checkpoints) != args.k_folds:
        print(f"[ERROR] Expected {args.k_folds} checkpoints but found {len(phase1_checkpoints)}")
        return False
    
    results = []
    
    for fold_idx, checkpoint_path in enumerate(phase1_checkpoints, start=1):
        print("\n" + "-"*80)
        print(f"Training Phase 2 for Fold {fold_idx}/{args.k_folds}")
        print(f"Using checkpoint: {checkpoint_path}")
        print("-"*80 + "\n")
        
        cmd = [
            "python", "code/WeakBE-Net/train_wdn.py",
            "--run_name", f"{args.run_name}_phase2",
            "--wdn_phase", "weighted_doctor_net",
            "--doctor_net_checkpoint", checkpoint_path,
            "--nr_epochs", str(args.phase2_epochs),
            "--wdn_lr", str(args.wdn_lr),
            "--k_folds", "5",  # Only run single fold
            "--batch_size", str(args.batch_size),
            "--hidden_dim", str(args.hidden_dim),
            "--wd", str(args.wd),
            "--exp_dir", args.exp_dir,
            "--features_dir", args.features_dir,
            "--label_file", args.label_file,
            "--project_name", args.project_name,
            "--experiment_mode", args.experiment_mode,
            "--seed", str(args.seed),
            "--specific_fold", str(fold_idx)   # But only train this specific fold

        ]
        
        if args.use_p53:
            cmd.extend(["--use_p53", "True"])
        if args.path_id is not None:
            cmd.extend(["--path_id", str(args.path_id)])
        
        print("Running command:")
        print(" ".join(cmd))
        print()
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\n[ERROR] Phase 2 training failed for fold {fold_idx}")
            results.append(False)
        else:
            print(f"\n[SUCCESS] Phase 2 completed for fold {fold_idx}")
            results.append(True)
    
    success_count = sum(results)
    print("\n" + "="*80)
    print(f"PHASE 2 SUMMARY: {success_count}/{args.k_folds} folds completed successfully")
    print("="*80 + "\n")
    
    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description="Automated two-phase WDN training script"
    )
    
    # Experiment settings
    parser.add_argument("--run_name", type=str, required=True,
                       help="Base name for experiment (phase1/phase2 will be appended)")
    parser.add_argument("--project_name", type=str, default='WeakBE-Net_WDN')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_folds", type=int, default=5)
    
    # Phase control
    parser.add_argument("--phases", type=str, default="both", 
                       choices=["phase1", "phase2", "both"],
                       help="Which phases to run: phase1 only, phase2 only, or both")
    
    # Data settings
    parser.add_argument("--exp_dir", type=str, default='/data/archief/AMC-data/Barrett/experiments/jans_experiments')
    parser.add_argument("--features_dir", type=str, default='/data/archief/AMC-data/Barrett/LANS_features/old_stuff/Virchow_HE_P53_1mpp_v2')
    parser.add_argument("--label_file", type=str, default='code/WeakBE-Net/notebooks/EDA/data/lans_all_labels.csv')
    parser.add_argument("--use_p53", type=bool, default=True)
    parser.add_argument("--experiment_mode", type=str, default="final_cons",
                       choices=["intra", "intra1000", "final_cons", "final_path"])
    parser.add_argument("--path_id", type=int, default=None)
    
    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_class_weights", type=bool, default=True)
    
    # Phase 1 hyperparameters
    parser.add_argument("--phase1_epochs", type=int, default=150,
                       help="Number of epochs for Phase 1 (Doctor Net)")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate for Phase 1")
    parser.add_argument("--wd", type=float, default=1e-5,
                       help="Weight decay")
    
    # Phase 2 hyperparameters
    parser.add_argument("--phase2_epochs", type=int, default=50,
                       help="Number of epochs for Phase 2 (WDN)")
    parser.add_argument("--wdn_lr", type=float, default=0.03,
                       help="Learning rate for Phase 2 (paper suggests 0.03)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("WEIGHTED DOCTOR NET - AUTOMATED TWO-PHASE TRAINING")
    print("="*80)
    print(f"Run name: {args.run_name}")
    print(f"K-folds: {args.k_folds}")
    print(f"Phases to run: {args.phases}")
    print(f"Phase 1 epochs: {args.phase1_epochs}")
    print(f"Phase 2 epochs: {args.phase2_epochs}")
    print(f"Phase 2 LR: {args.wdn_lr}")
    print("="*80 + "\n")
    
    # Phase 1: Train Doctor Net
    if args.phases in ["phase1", "both"]:
        success = train_phase1(args)
        if not success:
            print("\n[FATAL] Phase 1 failed. Aborting.")
            return
    
    # Phase 2: Train Weighted Doctor Net
    if args.phases in ["phase2", "both"]:
        # Find Phase 1 checkpoints
        phase1_checkpoints = find_phase1_checkpoints(args)
        
        if len(phase1_checkpoints) == 0:
            print("\n[ERROR] No Phase 1 checkpoints found!")
            print(f"Expected to find checkpoints matching:")
            print(f"  {args.exp_dir}/{args.run_name}_phase1_fold_*/best_model.ckpt")
            print("\nPlease run Phase 1 first or check the run_name.")
            return
        
        if len(phase1_checkpoints) != args.k_folds:
            print(f"\n[WARNING] Found {len(phase1_checkpoints)} checkpoints but expected {args.k_folds}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        success = train_phase2(args, phase1_checkpoints)
        if not success:
            print("\n[WARNING] Some Phase 2 folds failed. Check logs above.")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")
    
    # Print summary of where to find results
    print("Results locations:")
    print(f"  Phase 1: {args.exp_dir}/{args.run_name}_phase1_fold_*/")
    print(f"  Phase 2: {args.exp_dir}/{args.run_name}_phase2_fold*/")
    print()


if __name__ == '__main__':
    main()