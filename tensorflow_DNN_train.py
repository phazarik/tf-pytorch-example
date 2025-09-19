#--------------------------------------------------------------------
# This is a simple tensorflow based DNN trainer script that
# - Reads into text file containing physics-processes.
# - classifies between two classes of physics-processes: WZ and ZZ
# - input dataset: rows = events, columns: properites of each event
#                                                       - Prachurjya
#--------------------------------------------------------------------

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from datetime import timedelta

## tensorflow modules
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras import layers

## sklearn for handling ROC etc.
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc, accuracy_score
from sklearn.inspection import permutation_importance

## ANSI colors for decoration
RESET  = "\033[0m"
RED    = "\033[31m"
CYAN   = "\033[1;34m"
YELLOW = "\033[33m"
YELLOW_BOLD = "\033[1;33m"

def main():

    modelname = "tensorflow-DNN"
    os.makedirs(f"trained_models/{modelname}", exist_ok=True)
    
    ## ----------------------------------------------------------------------------
    ##                          HANDLING INPUT FILES
    ## ----------------------------------------------------------------------------
    time_start = time.time()
    time_buffer = time.time()
    
    ## Read the files and give them truth information.
    df_wz = read_txt_into_df("input_datasets/input_WZ.txt", truth=0) ## signal
    df_zz = read_txt_into_df("input_datasets/input_ZZ.txt", truth=1) ## background

    ## Combine the dataframes and randomize the rows.
    df = pd.concat([df_wz, df_zz], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"\n{YELLOW}Dataframe ready!{RESET}")
    print(df)

    print(f"Time Taken to read dataframe = {CYAN}{timedelta(seconds=int(time.time()-time_buffer))}{RESET}")
    time_buffer = time.time() ## Reseting the buffer
    
    ## Plot the input variables beforehand to see which ones are good.
    ## Then pick the input variables.
    train_var = ["var0", "var1", "var2", "var3", "var4", "var5", "var6", "var7", "var8", "var9", "var10", "var11"]

    ## ----------------------------------------------------------------------------
    ##                 Preparing numpy arrays and min-max scaling
    ## ----------------------------------------------------------------------------
    
    ## Split the df into two parts
    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df['truth']) ## from sklearn
    nsig_train = len(df_train.query('truth == 1'))
    nsig_test  = len(df_test.query('truth == 1'))
    nbkg_train = len(df_train.query('truth == 0'))
    nbkg_test  = len(df_test.query('truth == 0'))
    print(f'\n{YELLOW}Training statistics:{RESET}')
    print(f'nSig split into train and test: {nsig_train}, {nsig_test}')
    print(f'nBkg split into train and test: {nbkg_train}, {nbkg_test}')

    ## Convert these dfs into numpy arrays that go into the training and testing.
    ## X = input features, y = truth labels
    X_train= df_train[train_var].values
    y_train= df_train['truth'].values
    X_test = df_test[train_var].values
    y_test = df_test['truth'].values
    
    ## Find min-max of the train array and keep them as text files.
    ## Saving into text files is important for future-use.
    print(f"\n{YELLOW}min-max scaling of input features:{RESET}")
    FindMinMax(X_train, modelname)
    
    ## Scale the train and test arrays using these min-max values from the text files.
    X_train = ApplyMinMax(X_train, f'trained_models/{modelname}/scaling_parameters_min.txt', f'trained_models/{modelname}/scaling_parameters_max.txt')
    X_test  = ApplyMinMax(X_test,  f'trained_models/{modelname}/scaling_parameters_min.txt', f'trained_models/{modelname}/scaling_parameters_max.txt')
    print(f"Numpy arrays ready.")

    ## ----------------------------------------------------------------------------
    ##                        Defining and training the DNN
    ## ----------------------------------------------------------------------------

    n_features = X_train.shape[1]
    epochs_, batch_ = 30, 512

    ## Rule of thumb:
    ## - Larger batch size:
    ##     * Pros: smoother loss curve per epoch (better statistics)
    ##     * Cons: slower convergence, may get stuck in sharp minima
    ## - Smaller batch size:
    ##     * Pros: faster convergence, better generalization
    ##     * Cons: noisier loss curve per epoch
    ## - More epochs:
    ##     * Pros: network can learn more complex patterns
    ##     * Cons: may overfit if too large
    ##
    ## Suggested settings for ~100,000 events and 12 input features:
    ## - Batch size: 256–1024 (512 is a reasonable middle ground)
    ## - Epochs: 20–50 (monitor validation loss for early stopping)
    ## - Use EarlyStopping to avoid over-training
    
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    ## Training (might take some time)
    print(f'\n{YELLOW}Starting the training!{RESET}')
    time_buffer = time.time() ## Reseting the buffer
    history = model.fit(
        X_train, y_train,
        epochs=epochs_,
        batch_size=batch_,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[] ## Put EarlyStopping here if needed
    )

    modelfile = f'trained_models/{modelname}/model_{modelname}.keras'
    model.save(modelfile)
    print(f'Success!\nFile created: {YELLOW_BOLD}{modelfile}{RESET}')
    
    ## ----------------------------------------------------------------------------
    ##                        Plotting loss and accuracy 
    ## ----------------------------------------------------------------------------

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    epochs_run = len(history.history['loss'])

    # Subplot 1: accuracy vs epoch
    ax[0].plot(range(1, epochs_run + 1), history.history['accuracy'],     label='Train Accuracy')
    ax[0].plot(range(1, epochs_run + 1), history.history['val_accuracy'], label='Val Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_xlim(0, epochs_run)
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Accuracy-vs-Epoch', fontsize=10)
    ax[0].legend(loc='best')
    
    # Subplot 2: loss vs epoch
    ax[1].plot(range(1, epochs_run + 1), history.history['loss'],     label='Train Loss')
    ax[1].plot(range(1, epochs_run + 1), history.history['val_loss'], label='Val Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_xlim(0, epochs_run)
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    ax[1].legend(loc='best')
    ax[1].set_title('Loss-vs-Epoch', fontsize=10)
    
    fig.suptitle(modelname, fontsize=12)

    plt.tight_layout()
    figname_loss = f'trained_models/{modelname}/loss-and-accuracy.png'
    plt.savefig(figname_loss)
    print(f"File created: {YELLOW_BOLD}{figname_loss}{RESET}")

    print(f"Time Taken to train the network = {CYAN}{timedelta(seconds=int(time.time()-time_buffer))}{RESET}")
    time_buffer = time.time() ## Reseting the buffer
    
    ## ----------------------------------------------------------------------------
    ##                     Use the trained model to predict
    ## ----------------------------------------------------------------------------

    print(f"\n{YELLOW}Trying to predict the truth using the trained model:{RESET}")
    y_predicted_train = model.predict(X_train)
    y_predicted_test  = model.predict(X_test)

    ## Add the predicted labels as "score" in the dfs to compare against "truth"
    df_train['score'] = y_predicted_train
    df_test['score']  = y_predicted_test

    ## ROC information for the test dataset
    fpr, tpr, _ = roc_curve(y_test, y_predicted_test)
    auc_score = auc(tpr,1-fpr)
    tpr=tpr*100
    fnr=(1-fpr)*100
    
    ## ROC information for training data
    fpr1, tpr1, _ = roc_curve(y_train, y_predicted_train)
    auc_score1 = auc(tpr1,1-fpr1)
    tpr1=tpr1*100
    fnr1=(1-fpr1)*100

    ## ----------------------------------------------------------------------------
    ##                        Plot the DNN-score and ROC
    ## ----------------------------------------------------------------------------

    mybins = np.arange(0, 1.02, 0.02)
    density_ = False
    train_scores_sig, bins_sig_train, weights_sig_train, counts_sig_train, errors_sig_train = extract_plot(df_train, 1, mybins, density_)
    train_scores_bkg, bins_bkg_train, weights_bkg_train, counts_bkg_train, errors_bkg_train = extract_plot(df_train, 0, mybins, density_)
    test_scores_sig, bins_sig_test, weights_sig_test, counts_sig_test, errors_sig_test =      extract_plot(df_test,  1, mybins, density_)
    test_scores_bkg, bins_bkg_test, weights_bkg_test, counts_bkg_test, errors_bkg_test =      extract_plot(df_test,  0, mybins, density_)
    print('ROC values loaded.')

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    decorate_hist = {'bins':mybins, 'histtype':'step', 'linewidth':1.5, 'density': density_, 'log': False, 'alpha':1.0}
    decorate_point ={'marker':'o',  'markersize':3,    'linestyle':''}

    if density_:
        ax[0].hist(train_scores_sig, color='xkcd:greenish', label=f'Train Sig [{len(train_scores_sig)}]', alpha=0.3, bins=mybins, density=density_)
        ax[0].hist(train_scores_bkg, color='xkcd:sky blue', label=f'Train Bkg [{len(train_scores_bkg)}]', alpha=0.3, bins=mybins, density=density_)
        ax[0].hist(test_scores_sig,  color='xkcd:green', label=f'Test Sig [{len(test_scores_sig)}]', **decorate_hist)
        ax[0].hist(test_scores_bkg,  color='xkcd:blue',  label=f'Test Bkg [{len(test_scores_bkg)}]',**decorate_hist)
        
    else:
        ## Calculate the scaling factors to normalize train histograms to match test integrals
        scale_factor_sig = np.sum(counts_sig_test) / np.sum(counts_sig_train) if np.sum(counts_sig_train) > 0 else 1
        scale_factor_bkg = np.sum(counts_bkg_test) / np.sum(counts_bkg_train) if np.sum(counts_bkg_train) > 0 else 1

        ## Make train plots, normalized to match test histogram integrals
        ax[0].hist(train_scores_sig, color='xkcd:greenish', label=f'Train Sig [{len(train_scores_sig)}]', 
                   alpha=0.3, bins=mybins, density=False, weights=np.ones_like(train_scores_sig) * scale_factor_sig)
        ax[0].hist(train_scores_bkg, color='xkcd:sky blue', label=f'Train Bkg [{len(train_scores_bkg)}]', 
                   alpha=0.3, bins=mybins, density=False, weights=np.ones_like(train_scores_bkg) * scale_factor_bkg)
    
        ## Make test plots with error bars
        ax[0].errorbar(bins_sig_test[:-1] + np.diff(bins_sig_test) / 2, 
                       counts_sig_test, 
                       yerr=errors_sig_test, 
                       color='xkcd:green', label=f'Test Sig [{len(test_scores_sig)}]', 
                       fmt='o', markersize=3)
    
        ax[0].errorbar(bins_bkg_test[:-1] + np.diff(bins_bkg_test) / 2, 
                       counts_bkg_test, 
                       yerr=errors_bkg_test, 
                       color='xkcd:blue', label=f'Test Bkg [{len(test_scores_bkg)}]', 
                       fmt='o', markersize=3)

    ax[0].set_xlabel('Score')
    if density_:     ax[0].set_ylabel('Counts (normalized)')
    if not density_: ax[0].set_ylabel('Counts (train normalised to test)')
    #ax[0].set_yscale('log')
    ax[0].legend(loc='best')
    
    ax[1].plot(tpr, fnr, color='xkcd:denim blue', label='Training ROC (AUC = %0.4f)' % auc_score)
    ax[1].plot(tpr1,fnr1,color='xkcd:sky blue',   label='Testing ROC  (AUC = %0.4f)'  % auc_score1)
    ax[1].set_xlabel('Signal efficiency (%)')
    ax[1].set_ylabel('Background rejection (%)')
    ax[1].legend(loc='best', fontsize=8)
    
    fig.suptitle(modelname, fontsize=12)
    
    plt.tight_layout()
    figname_nnscore = f"trained_models/{modelname}/performance.png"
    plt.savefig(figname_nnscore)
    print(f"File created: {YELLOW_BOLD}{figname_nnscore}{RESET}")

    ## Optional, takes longer time:
    time_buffer = time.time() ## Reseting the buffer
    result = get_feature_importance(model, X_test, y_test, working_point=0.5, iterations=30)
    plot_feature_importance(result, modelname, train_var)

    print(f"Time Taken to calculate feature importance = {CYAN}{timedelta(seconds=int(time.time()-time_buffer))}{RESET}")
    print("\nDone!")
    time_end = time.time()
    print(f"Toral runtime = {CYAN}{timedelta(seconds=int(time_end-time_start))}{RESET}\n")

################################################################################################
##                                   UTILITY FUNCTIONS                                        ##
################################################################################################
    
# ----- Read space-separated txt file into pandas df -----
def read_txt_into_df(txtfile, truth):
    df = pd.DataFrame()
    if not os.path.exists(txtfile): print(f"{RED}[Error]: File not found: {txtfile}{RESET}")
    df = pd.read_csv(txtfile, sep="\s+", header=0) ## space separated, header = first row
    df['truth'] = truth
    return df

# ----- min-max-scaling -----
def FindMinMax(X, modelname):
    maxval = X.max(axis=0)
    minval = X.min(axis=0)
    print("Min Values found: ", minval)
    print("Max Values found: ", maxval)

    ## Warning for columns carrying identical values
    identical_cols = np.where(minval == maxval)[0]
    if len(identical_cols)>0: print(f"{YELLOW}[Warning]: Columns with identical values: {identical_cols}{RESET}")
    
    ## Save min and max values to separate text files without headers
    maxfile = f'trained_models/{modelname}/scaling_parameters_max.txt'
    minfile = f'trained_models/{modelname}/scaling_parameters_min.txt'
    np.savetxt(maxfile, maxval, fmt='%.6f')
    np.savetxt(minfile, minval, fmt='%.6f')
    print(f"File created: {YELLOW_BOLD}{maxfile}{RESET}")
    print(f"File created: {YELLOW_BOLD}{minfile}{RESET}")

def ApplyMinMax(X, min_filename, max_filename):
    minval = np.loadtxt(min_filename)
    maxval = np.loadtxt(max_filename)
    diff = maxval - minval
    normed_X = X.copy()    
    nonconst = np.where(diff != 0)[0] ## Scale the data only for non-constant columns
    normed_X[:, nonconst] = 2 * ((X[:, nonconst] - minval[nonconst]) / diff[nonconst]) - 1.0
    return normed_X

# ------ extract values to plot from a given df with truth and nnscore -----
def extract_plot(df_, truth_, mybins, density_):
    scores_ = df_[df_['truth'] == truth_]['score']
    hist_ = plt.hist(scores_, bins=mybins, density=density_)
    counts_, bins_, _ = hist_
    errors_ = np.sqrt(counts_)
    integral_ = np.sum(counts_)
    scale_ = integral_ / len(scores_) if len(scores_) > 0 else 1
    weights_ = np.ones_like(scores_)
    if density_ == True: weights_ = np.ones_like(scores_) * (scale_ / len(scores_)) #Normalise integral to 1    
    return scores_, bins_, weights_, counts_, errors_

# ----- Optional: Checking feature importance -----
def get_feature_importance(model, X_test, y_test, working_point=0.5, iterations=10):

    from tqdm import tqdm
    import io
    from functools import partial
    from contextlib import redirect_stdout, redirect_stderr
    import logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    tf.get_logger().setLevel('ERROR')

    N = min(5000, len(X_test))
    X_eval = X_test[:N]
    y_eval = y_test[:N]

    print(f"\n{YELLOW}Calculating input feature importance.{RESET}")
    total_permutations = iterations * X_eval.shape[1]
    pbar = tqdm(total=total_permutations, desc="Processing", unit="iter",
                colour="green", ncols=100, leave=True,
                bar_format="{l_bar}{bar}| [{elapsed} < {remaining}, {n_fmt}/{total_fmt}]")
    keras_accuracy = partial(keras_accuracy_base, nncut=working_point, pbar=pbar)

    ## Suppress internal prints from sklearn/joblib/TensorFlow
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        result = permutation_importance(
            model, X_eval, y_eval,
            n_repeats=iterations, scoring=keras_accuracy,
            random_state=42, n_jobs=1
        )
    pbar.close()
    return result

def keras_accuracy_base(model, X, y, nncut, pbar):
    y_pred = model.predict(X, verbose=0)
    y_pred_classes = (y_pred > nncut).astype(int).flatten()
    acc = accuracy_score(y, y_pred_classes)
    pbar.update(1)
    return acc

def plot_feature_importance(result, modelname, train_var):
    importance_scores = result.importances_mean
    feature_names = train_var

    ## Sort importance scores and corresponding feature names
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_importance_scores = importance_scores[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]
    print("Sorted Features and Importance Scores:")
    for name, score in zip(sorted_feature_names, sorted_importance_scores):  print(f"{name:<20} {score:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(sorted_feature_names, sorted_importance_scores, color='xkcd:greenish')
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance from Neural Network')
    ax.grid(axis='x', linestyle='--')
    ax.text(0.98, 0.05, modelname, transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
            bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', pad=2))
    plt.tight_layout()
    figname_importance = f'trained_models/{modelname}/feature_importance.png'
    plt.savefig(figname_importance, bbox_inches='tight')  ## Save with tight bounding box
    print(f"File created: {YELLOW_BOLD}{figname_importance}{RESET}")
    
# ----- EXECUTION -----
if __name__=="__main__": main()
