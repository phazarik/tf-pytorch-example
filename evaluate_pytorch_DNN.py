#!/usr/bin/env python3
# -------------------------------------------------------------------------
# This is a simple PyTorch-based DNN evaluation script that
# - Reads into text files containing physics-processes.
# - Evaluates a .pt model for two classes of physics-processes: WZ and ZZ
# - input dataset: rows = events, columns: properties of each event
#                                                       - Prachurjya
# -------------------------------------------------------------------------

import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn

# ANSI colors for decoration
RESET  = "\033[0m"
RED    = "\033[31m"
CYAN   = "\033[1;34m"
YELLOW = "\033[33m"
YELLOW_BOLD = "\033[1;33m"

def main():

    modelname = "pytorch-DNN"
    modelfile = f"trained_models/{modelname}/model_{modelname}.pt"
    minfile = f"trained_models/{modelname}/scaling_parameters_min.txt"
    maxfile = f"trained_models/{modelname}/scaling_parameters_max.txt"

    time_start = time.time()
    time_buffer = time.time()

    ## Input files
    file_wz = "input_datasets/input_WZ.txt"
    file_zz = "input_datasets/input_ZZ.txt"

    ## Read datasets and give labels
    df_wz = read_txt_into_df(file_wz, truth=0)
    df_zz = read_txt_into_df(file_zz, truth=1)
    df = pd.concat([df_wz, df_zz], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"\n{YELLOW}Dataframe ready!{RESET}")
    print(df.head())
    print(f"Time Taken to read dataframe = {CYAN}{timedelta(seconds=int(time.time()-time_buffer))}{RESET}")
    time_buffer = time.time()

    ## Input feature names
    train_var = ["var0", "var1", "var2", "var3", "var4", "var5",
                 "var6", "var7", "var8", "var9", "var10", "var11"]

    ## Check model and scaling files
    if not os.path.exists(modelfile):
        print(f"{RED}[Error]{RESET} Model file not found: {modelfile}")
        sys.exit(1)
    if not os.path.exists(minfile) or not os.path.exists(maxfile):
        print(f"{RED}[Error]{RESET} Scaling parameter files not found in trained_models/{modelname}")
        sys.exit(1)

    ## Prepare numpy arrays and apply min-max scaling
    X = df[train_var].values.astype(np.float32)
    y = df["truth"].values.astype(np.float32)
    print(f"\n{YELLOW}Applying min-max scaling using files from trained_models/{modelname}{RESET}")
    X = ApplyMinMax(X, minfile, maxfile)
    print("Numpy arrays ready.")
    print(f"Time Taken to prepare arrays = {CYAN}{timedelta(seconds=int(time.time()-time_buffer))}{RESET}")
    time_buffer = time.time()

    ## Define DNN architecture (must match trainer!)
    class DNNModel(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32,1), nn.Sigmoid()
            )
        def forward(self, x): return self.layers(x)

    ## Load model
    n_features = X.shape[1]
    model = DNNModel(n_features)
    model.load_state_dict(torch.load(modelfile, map_location=torch.device("cpu")))
    model.eval()
    print(f"\n{YELLOW}Model loaded: {modelfile}{RESET}")

    ## Predict
    print(f"\n{YELLOW}Predicting scores for evaluation dataset...{RESET}")
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X)).numpy().flatten()
    df["score"] = y_pred

    ## ROC info
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_score = auc(fpr, tpr)
    tpr_percent = tpr * 100
    fnr_percent = (1 - fpr) * 100

    ## Prepare histograms and ROC plots
    mybins = np.arange(0, 1.02, 0.02)
    density_ = False

    eval_scores_sig, bins_sig_eval, weights_sig_eval, counts_sig_eval, errors_sig_eval = extract_plot(df, 1, mybins, density_)
    eval_scores_bkg, bins_bkg_eval, weights_bkg_eval, counts_bkg_eval, errors_bkg_eval = extract_plot(df, 0, mybins, density_)

    print("ROC and score histograms computed.")

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    if density_:
        ax[0].hist(eval_scores_sig, label=f'Eval Sig [{len(eval_scores_sig)}]', alpha=0.3, bins=mybins, density=density_)
        ax[0].hist(eval_scores_bkg, label=f'Eval Bkg [{len(eval_scores_bkg)}]', alpha=0.3, bins=mybins, density=density_)
    else:
        ax[0].errorbar(bins_sig_eval[:-1] + np.diff(bins_sig_eval) / 2,
                       counts_sig_eval,
                       yerr=errors_sig_eval,
                       color='xkcd:green', label=f'Eval Sig [{len(eval_scores_sig)}]',
                       fmt='o', markersize=3)

        ax[0].errorbar(bins_bkg_eval[:-1] + np.diff(bins_bkg_eval) / 2,
                       counts_bkg_eval,
                       yerr=errors_bkg_eval,
                       color='xkcd:blue', label=f'Eval Bkg [{len(eval_scores_bkg)}]',
                       fmt='o', markersize=3)

    ax[0].set_xlabel('Score')
    if density_:
        ax[0].set_ylabel('Counts (normalized)')
    else:
        ax[0].set_ylabel('Counts')
    ax[0].legend(loc='best')

    ax[1].plot(tpr_percent, fnr_percent, label='Eval ROC (AUC = %0.4f)' % auc_score)
    ax[1].set_xlabel('Signal efficiency (%)')
    ax[1].set_ylabel('Background rejection (%)')
    ax[1].legend(loc='best', fontsize=8)

    fig.suptitle(f"{modelname} evaluation", fontsize=12)
    plt.tight_layout()
    outdir = f"trained_models/{modelname}"
    os.makedirs(outdir, exist_ok=True)
    perffile = f"{outdir}/performance_eval.png"
    plt.savefig(perffile)
    print(f"File created: {YELLOW_BOLD}{perffile}{RESET}")

    print(f"Time Taken to produce evaluation plots = {CYAN}{timedelta(seconds=int(time.time()-time_buffer))}{RESET}")
    time_buffer = time.time()

    time_end = time.time()
    print("\nDone!")
    print(f"Total runtime = {CYAN}{timedelta(seconds=int(time_end - time_start))}{RESET}\n")

# -----------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------

def read_txt_into_df(txtfile, truth):
    df = pd.DataFrame()
    if not os.path.exists(txtfile):
        print(f"{RED}[Error]: File not found: {txtfile}{RESET}")
        return df
    df = pd.read_csv(txtfile, sep="\s+", header=0)
    df['truth'] = truth
    return df

def ApplyMinMax(X, min_filename, max_filename):
    minval = np.loadtxt(min_filename)
    maxval = np.loadtxt(max_filename)
    diff = maxval - minval
    normed_X = X.copy()
    nonconst = np.where(diff != 0)[0]
    normed_X[:, nonconst] = 2 * ((X[:, nonconst] - minval[nonconst]) / diff[nonconst]) - 1.0
    return normed_X

def extract_plot(df_, truth_, mybins, density_):
    scores_ = df_[df_['truth'] == truth_]['score'].values
    counts_, bins_, _ = plt.hist(scores_, bins=mybins, density=density_)
    errors_ = np.sqrt(counts_)
    return scores_, bins_, np.ones_like(scores_), counts_, errors_

if __name__ == "__main__": main()
