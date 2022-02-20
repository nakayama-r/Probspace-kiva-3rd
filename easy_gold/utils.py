# coding:utf-8

import os
from pathlib import Path
import sys
import argparse
import pdb

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import pickle

import time
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
from functools import partial
import scipy as sp
import matplotlib.pyplot as plt
#from matplotlib_venn import venn2
import lightgbm as lgb
from sklearn import preprocessing
import seaborn as sns
import gc
import psutil
import os
from IPython.display import FileLink
import statistics
import json
import ast
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import collections
import random
import functools
from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix,f1_score,classification_report
from sklearn.metrics import mean_squared_error
# The metric in question
from sklearn.metrics import cohen_kappa_score
import copy
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from distutils.util import strtobool
import math
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Union
from sklearn.decomposition import PCA
#import dask.dataframe as dd
import re
from sklearn.cluster import KMeans

from contextlib import contextmanager
from collections import deque

#import eli5
#from eli5.sklearn import PermutationImportance

import shutil
import array
#import sqlite3



#from tsfresh.utilities.dataframe_functions import roll_time_series
#from tsfresh import extract_features
SEED_NUMBER=2020
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED_NUMBER)

pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 1000)

EMPTY_NUM=-999

# https://github.com/lopuhin/kaggle-imet-2019/blob/master/imet/utils.py#L17
ON_KAGGLE = False#'KAGGLE_URL_BASE'in os.environ
#print(" os.environ :",  os.environ)

print("ON_KAGGLE:", ON_KAGGLE)

if not ON_KAGGLE:
    #import slackweb

    try:
        import wandb
        from wandb.lightgbm import wandb_callback
    except:
        print(f"error : cannot import wandb")

else:
    import warnings
    warnings.simplefilter('ignore')

PROJECT_NAME = "probspace_kiva"
INPUT_DIR = Path("../data/raw")
PROC_DIR = Path("../data/proc")
LOG_DIR = Path("../data/log")
OUTPUT_DIR = Path("../data/submission")
PATH_TO_GRAPH_DIR=Path("../data/graph")
PATH_TO_MODEL_DIR=Path("../data/model")
PATH_TO_UPLOAD_MODEL_PARENT_DIR=Path("../data/model")
PATH_TO_FEATURES_DIR=Path("../data/features")


class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    blue = "\033[94m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer2(label: str) -> None:
    """compute the time the code block takes to run.
    """
    p = psutil.Process(os.getpid())
    start = time.time()  # Setup - __enter__
    m0 = p.memory_info()[0] / 2. ** 30
    print(color(f"{label}: Start at {start}; RAM USAGE AT START {m0}"))
    try:
        yield  # yield to body of `with` statement
    finally:  # Teardown - __exit__
        m1 = p.memory_info()[0] / 2. ** 30
        delta = m1 - m0
        sign = '+' if delta >= 0 else '-'
        delta = math.fabs(delta)
        end = time.time()
        print(color(f"{label}: End at {end} ({end - start}[s] elapsed); RAM USAGE AT END {m1:.2f}GB ({sign}{delta:.2f}GB)", color=Colors.red))

@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)

def cpu_dict(my_dictionary, text=None):

    size = sys.getsizeof(json.dumps(my_dictionary))
    #size += sum(map(sys.getsizeof, my_dictionary.values())) + sum(map(sys.getsizeof, my_dictionary.keys()))
    print(f"{text} size :  {size}")

def cpu_stats(text=None):
    
    #if not ON_KAGGLE:
        pid = os.getpid()
        py = psutil.Process(pid)
        memory_use = py.memory_info()[0] / 2. ** 30
        
        print('{} memory GB:'.format(text) + str(memory_use))#str(np.round(memory_use, 2)))
    
def reduce_mem_Series(se, verbose=True, categories=False):
    
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    col_type = se.dtype
    
    best_type = None
    if (categories==True) & (col_type == "object"):
        se = se.astype("category")
        best_type = "category"
    elif col_type in numeric2reduce:
        downcast = "integer" if "int" in str(col_type) else "float"
        se = pd.to_numeric(se, downcast=downcast)
        best_type = se.dtype.name
        
    if verbose and best_type is not None and best_type != str(col_type):
            print(f"Series '{se.index}' converted from {col_type} to {best_type}")
            
    return se
    

def reduce_mem_usage(df, verbose=True, categories=False):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2    
        #start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if (categories==True) & (col_type == "object"):
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        #end_mem = memory_usage_mb(df, deep=deep)
        end_mem = df.memory_usage().sum() / 1024**2
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
        
    return df


@contextmanager
def timer(name):
    
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.6f} s')
    

def normal_sampling(mean, label_k, std=2, under_limit=1e-15):

    val =  math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)
    if val < under_limit:
        val = under_limit
    
    return val
    
    
def compHist(np_oof, np_y_pred, np_y_true, title_str):

    np_list = [np_oof, np_y_true, np_y_pred]
    label_list = ["oof", "true", "pred"]
    color_list = ['red', 'blue', 'green']

    for np_data, label, color in zip(np_list, label_list, color_list):
        
        sns.distplot(
            np_data,
            #bins=sturges(len(data)),
            color=color,
            kde=True,
            label=label
        )


    plt.savefig(str(PATH_TO_GRAPH_DIR / f"{title_str}_compHist.png"))
    plt.close()


def compPredTarget(y_pred, y_true, index_list, title_str, lm_flag=False):


    
        df_total = pd.DataFrame({"Prediction" : y_pred.flatten(),
                                 "Target" : y_true.flatten(),
                                 "Difference" : y_true.flatten() -y_pred.flatten()
                                 #"type" : np.full(len(y_pred), "oof")
                                 }, index=index_list)
        
        print(df_total)
    

        print("Difference > 0.1 : ", df_total[np.abs(df_total["Difference"]) > 0.1].Difference.count())
        #print(df_total[df_total["type"]=="valid_train"].Difference)
        
        fig = plt.figure()
        sns.displot(df_total.Difference,bins=10)
        plt.savefig(str(PATH_TO_GRAPH_DIR / f"{title_str}_oof_diff_distplot.png"))
        plt.close()
        
        #pdb.set_trace()

        if lm_flag:
            plt.figure()
            fig2 = sns.lmplot(x="Target", y="Prediction", data=df_total, palette="Set1")
            #fig.set_axis_labels('target', 'pred')
            plt.title(title_str)
            plt.tight_layout()
            plt.savefig(str(PATH_TO_GRAPH_DIR / f"{title_str}_oof_true_lm.png"))
                
            plt.close()

def dimensionReductionPCA(df, _n_components, prefix="PCA_"):



    pca = PCA(n_components=_n_components)
    pca.fit(df)

    reduced_feature = pca.transform(df)
    df_reduced = pd.DataFrame(reduced_feature, columns=[f"{prefix}{x + 1}" for x in range(_n_components)], index=df.index)
    print(f"df_reduced:{df_reduced}")

    df_tmp = pd.DataFrame(pca.explained_variance_ratio_, index=[f"{prefix}{x + 1}" for x in range(_n_components)])
    print(df_tmp)

    import matplotlib.ticker as ticker
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()

    

    path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_PCA.png")
    #print("save: ", path_to_save)
    plt.savefig(path_to_save)
    plt.show(block=False)
    plt.close()


    # df_comp = pd.DataFrame(pca.components_, columns=df.columns, index=[f"{prefix}{x + 1}" for x in range(_n_components)])
    # print(df_comp)

    # plt.figure(figsize=(6, 6))
    # for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns):
    #     plt.text(x, y, name)
    # plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
    # plt.grid()
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    
    # path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_PCA_scatter.png")
    # #print("save: ", path_to_save)
    # plt.savefig(path_to_save)
    # plt.show(block=False)
    # plt.close()

    return df_reduced



def addNanPos(df, cols_list:list, suffix="nan_pos"):
    
    for col in cols_list:
        if df[col].isnull().any():
            df["{}_{}".format(col, suffix)] = df[col].map(lambda x: 1 if pd.isna(x) else 0)
        
    return df
        

def get_feature_importances(X, y, shuffle=False):
    # 必要ならば目的変数をシャッフル
    if shuffle:
        y = np.random.permutation(y)

    # モデルの学習
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)

    # 特徴量の重要度を含むデータフレームを作成
    imp_df = pd.DataFrame()
    imp_df["feature"] = X.columns
    imp_df["importance"] = clf.feature_importances_
    return imp_df.sort_values("importance", ascending=False)


def nullImporcance(df_train_X, df_train_y, th=80, n_runs=100):
    
    
    # 実際の目的変数でモデルを学習し、特徴量の重要度を含むデータフレームを作成
    actual_imp_df = get_feature_importances(df_train_X, df_train_y, shuffle=False)
    
    # 目的変数をシャッフルした状態でモデルを学習し、特徴量の重要度を含むデータフレームを作成
    N_RUNS = n_runs
    null_imp_df = pd.DataFrame()
    for i in range(N_RUNS):
        print("run : {}".format(i))
        imp_df = get_feature_importances(df_train_X, df_train_y, shuffle=True)
        imp_df["run"] = i + 1
        null_imp_df = pd.concat([null_imp_df, imp_df])
        
        
    def display_distributions(actual_imp_df, null_imp_df, feature, path_to_save_dir):
        # ある特徴量に対する重要度を取得
        actual_imp = actual_imp_df.query("feature == '{}'".format(feature))["importance"].mean()
        null_imp = null_imp_df.query("feature == '{}'".format(feature))["importance"]
    
        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        a = ax.hist(null_imp, label="Null importances")
        ax.vlines(x=actual_imp, ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
        ax.legend(loc="upper right")
        ax.set_title("Importance of {}".format(feature), fontweight='bold')
        plt.xlabel("Null Importance Distribution for {}".format(feature))
        plt.ylabel("Importance")
        plt.show()
        
        path_to_save = os.path.join(str(path_to_save_dir), "null_imp_{}".format(feature))
        plt.savefig(path_to_save)

    # 実データにおいて特徴量の重要度が高かった上位5位を表示
    for feature in actual_imp_df["feature"]:
        display_distributions(actual_imp_df, null_imp_df, feature, PATH_TO_GRAPH_DIR)
        
    # 閾値を設定
    THRESHOLD = th
    
    # 閾値を超える特徴量を取得
    null_features = []
    for feature in actual_imp_df["feature"]:
        print("Null :: {}".format(feature))
        actual_value = actual_imp_df.query("feature=='{}'".format(feature))["importance"].values
        null_value = null_imp_df.query("feature=='{}'".format(feature))["importance"].values
        percentage = (null_value < actual_value).sum() / null_value.size * 100
        print("actual_value: {}, null_value : {}, percentage : {}".format(actual_value, null_value, percentage))
        if percentage < THRESHOLD and (100-THRESHOLD) < percentage:
            null_features.append(feature)

    return null_features
    
    

def makeFourArithmeticOperations(df, col1, col2):
    
    
    
    new_col = "auto__{}_add_{}".format(col1, col2)
    df[new_col] = df[col1] + df[col2]
    
    new_col = "auto__{}_diff_{}".format(col1, col2)
    df[new_col] = df[col1] - df[col2]
    
    new_col = "auto__{}_multiply_{}".format(col1, col2)
    df[new_col] = df[col1] * df[col2]
    
    new_col = "auto__{}_devide_{}".format(col1, col2)
    df[new_col] = df[col1] / df[col2]
    
    return df

def procAgg(df:pd.DataFrame, base_group_col:str, agg_col:str, agg_list:list):
    
    
    for agg_func in agg_list:
        new_col = "auto__{}_{}_agg_by_{}".format(agg_col, agg_func, base_group_col)
        map_dict = df.groupby(base_group_col)[agg_col].agg(agg_func)
        print(new_col)
        print(map_dict)
        df[new_col] = df[base_group_col].map(map_dict)
        df[new_col] = reduce_mem_Series(df[new_col])
    
        #df = makeFourArithmeticOperations(df, new_col, agg_col)
    
    return df

def aggregationFE(df:pd.DataFrame, base_group_cols:list, agg_cols:list, agg_func_list:list=['count', 'max', 'min', 'sum', 'mean', "nunique", "std", "median", "skew"]):
    
    for b in base_group_cols:
        for a in agg_cols:
            df = procAgg(df, b, a, agg_func_list)
    
    return df

def makeInteractionColumn(df:pd.DataFrame, inter_cols:list):
    print(inter_cols)
    for c in inter_cols:
            
        col_name = "inter_" + "_".join(c)
        print(col_name)
        #df[col_name] = "_"
        
        for i, col in enumerate(c):
            print(col)
            if i == 0:
                df[col_name] = df[col]
            else:
                #
                #print(df[col])
                df[col_name] = df[col_name].map(lambda x : str(x)) + "_" + df[col].map(lambda x : str(x))
        
        #print(df[col_name].unique())
        
    print("****")
    return df

def interactionFE(df:pd.DataFrame, cols:list=[], inter_nums:list=[]):
    
    for inter_num in inter_nums:
        
        inter_cols = itertools.combinations(cols, inter_num)
        df = makeInteractionColumn(df, inter_cols)
        
#        for c in itertools.combinations(cols, inter_num):
#            
#            col_name = "inter_" + "_".join(c)
#            print(col_name)
#            #df[col_name] = "_"
#            
#            for i, col in enumerate(c):
#                print(col)
#                if i == 0:
#                    df[col_name] = df[col]
#                else:
#                    #
#                    #print(df[col])
#                    df[col_name] = df[col_name].map(lambda x : str(x)) + "_" + df[col].map(lambda x : str(x))
#            
#            print(df[col_name].unique())
            
            
    return df

def interactionFEbyOne(df:pd.DataFrame, inter_col:str, target_cols:list, inter_nums:list=[1]):
    
    for inter_num in inter_nums:
        
        comb = itertools.combinations(target_cols, inter_num)
        for c in comb:
            
            if not inter_col in c:
            
                inter_cols = (inter_col,) + c
                print(inter_cols)
                df = makeInteractionColumn(df, [inter_cols])
    
    return df
    
def calcSmoothingParam(num_of_data, k=100, f=100):
    
    param = 1 / (1 + np.exp(-(num_of_data - k)/f))
    return param
    
def calcSmoothingTargetMean(df:pd.DataFrame, col_name, target_name):
    
    #print(df[target_name])
    
    all_mean = df[target_name].mean()
    #print(all_mean)
    #sys.exit()
    
    df_vc = df[col_name].value_counts()
    gp_mean_dict = df.groupby(col_name)[target_name].mean()
    
    smooth_target_mean = df_vc.copy()
    
    for key, val in gp_mean_dict.items():
        
        n=df_vc[key]
        
        param = calcSmoothingParam(num_of_data=n)
        smooth = param * val + (1-param)*all_mean

        smooth_target_mean[key] = smooth

        print("label : {}, n = {}, val={}, all = {}, param = {}, final={}".format(key, n, val, all_mean, param, smooth))
   
    del smooth_target_mean, df_vc
    gc.collect()
    
    return smooth_target_mean

def targetEncoding(df_train_X, df_train_y, encoding_cols:list, _n_splits=4, smooth_flag=0):
    
    
    dict_all_train_target_mean = {}
    for c in encoding_cols:
        # print("Target Encoding : {}".format(c))
        # print(f"df_train_X[c] : {df_train_X[c].shape}")
        # print(f"df_train_y : {df_train_y.shape}")
        
        
        
        #df_data_tmp = pd.DataFrame({c: df_train_X[c], "target":df_train_y})
        df_data_tmp = pd.DataFrame(df_train_X[c])
        df_data_tmp["target"] = df_train_y#.loc[:,0]
        
        
        
        #nan_mean= -999#df_data_tmp["target"].mean()
        nan_mean=df_data_tmp["target"].mean()
        
        if smooth_flag:
            all_train_target_mean=calcSmoothingTargetMean(df_data_tmp, c, "target")
        else:
            all_train_target_mean = df_data_tmp.groupby(c)["target"].mean()
        
        dict_all_train_target_mean[c] = all_train_target_mean
        #print(all_train_target_mean)
        
        #df_test_X[c] = df_test_X[c].map(all_train_target_mean)
        
        tmp = np.repeat(np.nan, df_train_X.shape[0])
        
        kf = KFold(n_splits=_n_splits, shuffle=True, random_state=0)
        for idx_1, idx_2 in kf.split(df_train_X):
            
            if smooth_flag:
                target_mean=calcSmoothingTargetMean(df_data_tmp.iloc[idx_1], c, "target")
            else:
                target_mean = df_data_tmp.iloc[idx_1].groupby(c)["target"].mean()
            
            tmp[idx_2] = df_train_X[c].iloc[idx_2].map(target_mean)
            
            idx_1_unique = df_data_tmp.iloc[idx_1][c].unique()
            idx_2_unique = df_data_tmp.iloc[idx_2][c].unique()
            for c2 in idx_2_unique:
                if not c2 in idx_1_unique:
                    pass
                    #print("TARGET ENCORDING ERROR {}: {} replace to {}".format(c, c2, nan_mean))
                    
        
        df_train_X[c] = tmp
        df_train_X[c].fillna(value=nan_mean, inplace=True)
        #print(df_train_X.loc[df_train_X[c].isnull(), c])
        #showNAN(df_train_X)
        
    
    return df_train_X, dict_all_train_target_mean
            
         
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def calc_smoothing(se_gp_count, se_gp_mean, prior, min_samples_leaf=1, smoothing=1):
    

    smoothing = 1 / (1 + np.exp(-(se_gp_count - min_samples_leaf) / smoothing))
    se_smoothing_mean = prior * (1 - smoothing) + se_gp_mean * smoothing
    
    return se_smoothing_mean #, smoothing

def TEST__calc_smoothing():
    
    se_count = pd.Series(np.arange(2000000))
    
    cpu_stats("before calc_smoothing")

    se_count, smoothing = calc_smoothing(se_count, se_count, prior=50, min_samples_leaf=100, smoothing=300)

    cpu_stats("after calc_smoothing")

    #fig = plt.Figure()
    plt.plot(se_count, smoothing, label="smoothing")
    plt.show()

def target_encode_with_smoothing(trn_series=None, 
                  #tst_series=None, 
                  target_se=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  #noise_level=0,
                  agg_val="mean",
                  ):
    """
    from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features/notebook
    
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    """ 
    assert len(trn_series) == len(target_se)
    #assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target_se], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target_se.name].agg([agg_val, "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    if agg_val == "mean":
        prior = target_se.mean()
    elif agg_val == "std":
        prior = target_se.std()
    # The bigger the count the less full_avg is taken into account
    averages[target_se.name] = prior * (1 - smoothing) + averages[agg_val] * smoothing
    averages.drop([agg_val, "count"], axis=1, inplace=True)
    
    return averages
    
    
    # # Apply averages to trn and tst series
    # ft_trn_series = pd.merge(
    #     trn_series.to_frame(trn_series.name),
    #     averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
    #     on=trn_series.name,
    #     how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    
    # # pd.merge does not keep the index so restore it
    # ft_trn_series.index = trn_series.index 
    # ft_tst_series = pd.merge(
    #     tst_series.to_frame(tst_series.name),
    #     averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
    #     on=tst_series.name,
    #     how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    
    
    # pd.merge does not keep the index so restore it
    #ft_tst_series.index = tst_series.index
    #return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def showNAN(df):
    # print(df.isnull())
    # df.isnull().to_csv(PROC_DIR/'isnull.csv')
    # print(df.isnull().sum())
    # df.isnull().sum().to_csv(PROC_DIR/'isnull_sum.csv')
    
    # total = df.isnull().sum().sort_values(ascending=False)
    # print(total)
    # print(f"count : {df.isnull().count()}")
    # percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    # missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    #df=df.replace([np.inf, -np.inf], np.nan)
    
    nan_dict = {}
    for col in df.columns:
        total = df[col].isnull().sum()
        percent = total / df[col].isnull().count()
        nan_dict[col] = [total, percent]
    
    missing_data = pd.DataFrame(nan_dict, index=['Total', 'Percent']).T
    missing_data = missing_data.sort_values('Percent', ascending=False)
    
    
    nan_data = missing_data.loc[missing_data["Percent"] > 0, :]
    
    if not ON_KAGGLE:
        print("****show nan*****")
        print(nan_data)
        print("****show nan end*****\n")
    
    nan_list = list(nan_data.index)
    
    #del missing_data, nan_data
    #gc.collect()
    
    return nan_list


def accumAdd(accum_dict, dict_key_name, add_val, _empty_val=EMPTY_NUM):
    if accum_dict[dict_key_name] == _empty_val:
        accum_dict[dict_key_name] = add_val
    else:
        accum_dict[dict_key_name] += add_val
        
        
    return accum_dict

def getColumnsFromParts(colums_parts, base_columns):
    
    new_cols=[]
    for col_p in colums_parts:
        for col in base_columns:
            if col_p in col:
                if not col in new_cols:
                    new_cols.append(col)
                    #print("find from the part : {}".format(col))
    return new_cols.copy()

def checkCorreatedFeatures(df, exclude_columns=[], th=0.995, use_cols=[]):
    counter = 0
    to_remove = []
    
    if len(use_cols)==0:
        use_cols = df.columns

    for feat_a in use_cols:
        if feat_a in exclude_columns:
            continue
        
        for feat_b in df.columns:
            if feat_b in exclude_columns:
                continue
            
            if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:
                #print('{}: FEAT_A: {}, FEAT_B: {}'.format(counter, feat_a, feat_b))
                c = np.corrcoef(df[feat_a], df[feat_b])[0][1]
                if c > th:
                    counter += 1
                    to_remove.append(feat_b)
                    print('{}: FEAT_A: {}, FEAT_B (removed): {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
    
    
    return to_remove.copy()


def addStrToLastWithoutContinuous(chain_string, added_str, splitter="_"):
    string_list = chain_string.split(splitter)
    if string_list[-1] != added_str:
        return chain_string + splitter + added_str
    else:
        return chain_string
     
def adv2(_df_train, _df_test, drop_cols):
        
        df_train = _df_train.copy()
        df_test = _df_test.copy()
        
        print(len(df_train))
        print(len(df_test))
        
        df_train["isTest"] = 0
        df_test["isTest"] = 1
        drop_cols.append("isTest")
        
        df = pd.concat([df_train, df_test])
        
        
        #train 0, test 1
        
        df_X = df.drop(columns= drop_cols)
        df_y = df["isTest"]
        columns=df_X.columns.to_list()

        train, test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42, shuffle=True)
        del df, df_y, df_X
        gc.collect()

        train = lgb.Dataset(train, label=y_train)
        test = lgb.Dataset(test, label=y_test)

        param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.05,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1,
         'importance_type':'gain',
        }

        
        num_round = 1000
        clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 500)

        feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),columns)), columns=['Value','Feature'])

        plt.figure(figsize=(20, 20))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(100))
        plt.title('LightGBM Features')
        plt.tight_layout()
        plt.show()
        plt.savefig(str(PATH_TO_GRAPH_DIR / 'lgbm_importances-01.png'))



def adversarialValidation(_df_train, _df_test, drop_cols, sample_flag=False):
        
        df_train = _df_train.copy()
        df_test = _df_test.copy()
        
        if sample_flag:
            num_test = len(df_test)
            df_train = df_train.sample(n=num_test)
        
        print(f"df_train : {len(df_train)}")
        print(f"df_test : {len(df_test)}")
        
        df_train["isTest"] = 0
        df_test["isTest"] = 1
        drop_cols.append("isTest")
        
        df = pd.concat([df_train, df_test])
        
        
        #train 0, test 1
        
         
        
        df_X = df.drop(columns= drop_cols)
        df_y = df["isTest"]
        
        adv_params = {
            'learning_rate': 0.05, 
            'n_jobs': -1,
            'seed': 50,
            'objective':'binary',
            'boosting_type':'gbdt',
            'is_unbalance': False,
            'importance_type':'gain',
            'metric': 'auc',
            'verbose': 1,
        }
        
        model = lgb.LGBMClassifier(n_estimators=100)
        model.set_params(**adv_params)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_validate(model, df_X, df_y, cv=skf, return_estimator=True, scoring="roc_auc")
        
        adv_acc = score['test_score'].mean()
        print('Adv AUC:', score['test_score'].mean())
        
        feature_imp = pd.DataFrame(sorted(zip(score['estimator'][0].feature_importances_,df_X.columns), reverse=True), columns=['Value','Feature'])
        print(feature_imp)
        #graphImportance(feature_imp, 50)
        #base_name = '../data/features/adv_feature_imp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #feature_imp.to_csv(base_name + '.csv')
#        f = open(base_name + '.pkl', 'wb')
#        pickle.dump(feature_imp, f)
#        f.close()

        for i in range(len(feature_imp["Feature"])):
            if feature_imp["Value"].values[i] > 0:
                str_col = "\'" + feature_imp["Feature"].values[i] + "\',"
                print(str_col)
        
        return adv_acc, feature_imp
                


def get_too_many_null_attr(data, rate=0.9):
    # many_null_cols = []
    # for col in data.columns:
    #     print(col)
    #     if data[col].isnull().sum() / data.shape[0] > 0.9:
    #         many_null_cols.append(col) 
    # print("DONE!!!!!")
    many_null_cols = [col for col in data.columns if data[col].isnull().sum() / data.shape[0] > rate]
    return many_null_cols

def get_too_many_repeated_val(data, rate=0.95):
    big_top_value_cols = [col for col in data.columns if data[col].value_counts(dropna=False, normalize=True).values[0] > rate]
    return big_top_value_cols




def get_useless_columns(data, null_rate=0.95, repeat_rate=0.95):
    too_many_null = get_too_many_null_attr(data, null_rate)
    print("More than {}% null: ".format(null_rate) + str(len(too_many_null)))
    print(too_many_null)
    too_many_repeated = get_too_many_repeated_val(data, repeat_rate)
    print("More than {}% repeated value: ".format(repeat_rate) + str(len(too_many_repeated)))
    print(too_many_repeated)
    cols_to_drop = list(set(too_many_null + too_many_repeated))

    return cols_to_drop

def get_useless_columnsTrainTest(df_train, df_test, null_rate=0.95, repeat_rate=0.95):
    
    
    drop_train = set(get_useless_columns(df_train, null_rate=null_rate, repeat_rate=repeat_rate))
    drop_test = set(get_useless_columns(df_test, null_rate=null_rate, repeat_rate=repeat_rate)) if not df_test.empty else set([])
    
    
    s_symmetric_difference = drop_train ^ drop_test
    if s_symmetric_difference:
        print("{} are not included in each set".format(s_symmetric_difference))

    cols_to_drop = list((drop_train) & (drop_test))
    print("intersection cols_to_drop")
    print(cols_to_drop)
    
    return cols_to_drop

def transformCosCircle(df, time_col_str):
    
    
    val = [float(x) for x in df[time_col_str].unique()]
    val.sort()
    #print(val)
    num = len(val)
    unit = 180.0 / num
    #print(unit)
    trans_val = [x * unit for x in val]
    #print(trans_val)
    df[time_col_str + "_angle_rad"] = np.deg2rad(df[time_col_str].replace(val, trans_val))
    df[time_col_str + "_cos"] =  np.cos(df[time_col_str + "_angle_rad"])
    df[time_col_str + "_sin"] =  np.sin(df[time_col_str + "_angle_rad"])
    
    df = df.drop(columns=[time_col_str + "_angle_rad"])
    
    #print(df[time_col_str])
    
    return df


def extract_time_features(df, date_col):

    df[date_col] = pd.to_datetime(df[date_col])
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    #df['hour'] = df[date_col].dt.hour
    df['year'] = df[date_col].dt.year
    #df["seconds"] = df[date_col].dt.second
    df['dayofweek'] = df[date_col].dt.dayofweek #0:monday to 6: sunday
    #df['week'] = df[date_col].dt.week # the week ordinal of the year
    df['weekofyear'] = df[date_col].dt.weekofyear # the week ordinal of the year
    
    
    df['dayofyear'] = df[date_col].dt.dayofyear #1-366 
    df['quarter'] = df[date_col].dt.quarter
    df['is_month_start'] = df[date_col].dt.is_month_start
    df['is_month_end'] = df[date_col].dt.is_month_end
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end
    df['is_year_start'] = df[date_col].dt.is_year_start
    df['is_year_end'] = df[date_col].dt.is_year_end
    df['is_leap_year'] = df[date_col].dt.is_leap_year

    df['days_in_month'] = df['date'].dt.daysinmonth
    df["days_from_end_of_month"] = df['days_in_month'] - df["day"]
    df["days_rate_in_month"] = (df["day"] -1) / (df['days_in_month'] - 1)

    df["s_m_e_in_month"] = df["day"].map(lambda x: 0 if x <= 10 else (1 if x <= 20 else 2))
    
    # df = transformCosCircle(df, "day")
    # df = transformCosCircle(df, "month")
    # df = transformCosCircle(df, "dayofweek")
    # df = transformCosCircle(df, "weekofyear")
    # df = transformCosCircle(df, "dayofyear")
    
    
    return df

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    data = None
    with open(path, mode='rb') as f:
        data = pickle.load(f)
    return data

def procLabelEncToColumns(df_train, df_test, col_list):
    
    df_train["train_test_judge"] = "train"
    df_test["train_test_judge"] = "test"
    
    
    df = pd.concat([df_train, df_test])
    
    for c in col_list:
        df[c] = procLabelEncToSeries(df[c])
        
    df_train = df.loc[df["train_test_judge"]=="train"].drop(columns=["train_test_judge"])
    df_test = df.loc[df["train_test_judge"]=="test"].drop(columns=["train_test_judge"])
    
    return df_train, df_test

def procLabelEncToSeries(se):

    val_list = list(se.dropna().unique())
    val_list.sort()
    #print(df[f].unique())
    
    replace_map = dict(zip(val_list, np.arange(len(val_list))))
    se = se.map(replace_map)
    #pdb.set_trace()
    return se
    
def proclabelEncodings(df, not_proc_list=[]):
    #lbl = preprocessing.LabelEncoder()
    
    if not ON_KAGGLE:
        print("**label encoding**")
    decode_dict = {}
    for f in df.columns:
        if df[f].dtype.name =='object':
            
            if f in not_proc_list:
                continue
            
            if not ON_KAGGLE:
                print(f)
            
            val_list = list(df[f].dropna().unique())
            val_list.sort()
            #print(df[f].unique())
            
            replace_map = dict(zip(val_list, np.arange(len(val_list))))
            df[f] = df[f].map(replace_map)
            #print(df[f].unique())
            inverse_dict =  get_swap_dict(replace_map)
            decode_dict[f] = inverse_dict
            
            
            #lbl.fit(list(df[f].dropna().unique()))
            #print(list(lbl.classes_))
            #df[f] = lbl.transform(list(df[f].values))
                
    if not ON_KAGGLE:
        print("**label encoding end **\n")
        print("**for dicode**")
        #print(f"{decode_dict}")
            
    return df, decode_dict


def qwk(act,pred,n=4,hist_range=(0,3), weights=None):
    O = confusion_matrix(act,pred, labels=[0, 1, 2, 3],sample_weight = weights)
    O = np.divide(O,np.sum(O))
    
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            W[i][j] = ((i-j)**2)/((n-1)**2)
    
            
    act_hist = np.histogram(act,bins=n,range=hist_range, weights=weights)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range, weights=weights)[0]
    
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E))
    
    num = np.sum(np.multiply(W,O))
    den = np.sum(np.multiply(W,E))
        
    return 1-np.divide(num,den)


def calcClass(X, coef):
    
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            X_p[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            X_p[i] = 2
        else:
            X_p[i] = 3
            
    return X_p


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        
        #print(coef)
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            #elif pred >= coef[2] and pred < coef[3]:
            #    X_p[i] = 3
            else:
                X_p[i] = 3

        #ll = cohen_kappa_score(y, X_p, weights='quadratic')
        
        ll = qwk(y, X_p)
        #print(ll)
        return -ll

    def fit(self, X, y, initial_coef):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        #initial_coef = th_list
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        return self.coefficients()

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            #elif pred >= coef[2] and pred < coef[3]:
            #    X_p[i] = 3
            else:
                X_p[i] = 3
        return X_p

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']

def calcDropColsFromPermuationImportance(path_to_dir):
    
    ppath_to_dir = Path(path_to_dir)
    df_total = pd.DataFrame()
    
    for i, f in enumerate(ppath_to_dir.glob("permutation_feature_imp_*.csv")):
        
        
        df_imp = pd.read_csv(f, index_col=1).rename(columns={'weight': 'weight{}'.format(i)})
        
        if i == 0:
            df_total = df_imp['weight{}'.format(i)]
        else:
            df_total = pd.concat([df_total, df_imp['weight{}'.format(i)]], axis=1)
    
    df_total["mean"] = df_total.mean(axis=1)
    df_total.to_csv(ppath_to_dir/("total_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"))
    
    drop_list = list(df_total.loc[df_total["mean"] <= 0].index.values)
    for col in drop_list:
        print('"{}",'.format(col))
    
    return drop_list

def stract_hists(feature, train, test, adjust=False, plot=False):
    n_bins = 10
    train_data = train[feature]
    test_data = test[feature]
    if adjust:
        test_data *= train_data.mean() / test_data.mean()
    perc_90 = np.percentile(train_data, 95)
    train_data = np.clip(train_data, 0, perc_90)
    test_data = np.clip(test_data, 0, perc_90)
    train_hist = np.histogram(train_data, bins=n_bins)[0] / len(train_data)
    test_hist = np.histogram(test_data, bins=n_bins)[0] / len(test_data)
    msre = mean_squared_error(train_hist, test_hist)
    #print(msre)
    if plot:
        print(msre)
        plt.bar(range(n_bins), train_hist, color='blue', alpha=0.5)
        plt.bar(range(n_bins), test_hist, color='red', alpha=0.5)
        plt.show()
    return msre

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

def testComp():
    
    y_pred = np.arange(5).flatten()
    y_true = np.arange(5).flatten()* 2
    compPredTarget(y_pred, y_true, np.arange(5).flatten(), title_str="oof_diff")
    
    
def transformMultiOneHot(df, col_name, splitter=",", drop_original_col=True):
    
    original_elements_list = df[col_name].dropna().unique()

    print(original_elements_list)
    duplicate_set = set([col  for cols in original_elements_list for col in cols.split(splitter) ])
    print(duplicate_set)
    
    for c in duplicate_set:
        c_name = f"OH_{col_name}_{c}"
        #df[c_name] = 0
        df[c_name] = df[col_name].map(lambda x:(1 if c in x.split(splitter) else 0) if pd.isnull(x) == False else 0)
        print(df[c_name].value_counts())
     
    if drop_original_col:
        df.drop(col_name, axis=1, inplace=True)
    return df    

def showROC(test_y, pred_y, path_to_save):
    # FPR, TPR(, しきい値) を算出
    fpr, tpr, thresholds = roc_curve(test_y, pred_y)

    # ついでにAUCも
    auc_val = auc(fpr, tpr)

    # ROC曲線をプロット
    plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc_val)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)

    plt.savefig(path_to_save)
    plt.close()

def visualizeComp(df, y_true, y_pred, category_cols=[]):

    print("vis")
    print(f"df : {df.shape}")
    print(f"y_true : {y_true}")
    print(f"y_pred : {y_pred}")


    
    target_col = "target"
    df["pred"] = y_pred.astype(float)
    df[target_col] = y_true.astype(float)
    
    if len(category_cols) == 0:
        category_cols = df.select_dtypes(include=['object']).columns
    
    for col in category_cols:
        
        vals = df[col].unique()
        for val in vals:
            
            
            str_title = f"{col}_{val}"
            
            try:

                tmp_df = df.loc[df[col]==val, [target_col, "pred"]]
                print(tmp_df)
                
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), "{}_result_comp_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"), str_title))
                showROC(test_y=tmp_df[target_col].values, pred_y=tmp_df["pred"].values, path_to_save=path_to_save)

                # plt.figure()
                
                
                # fig2 = sns.lmplot(x=target_col, y="pred", data=tmp_df, palette="Set1")
                # #fig.set_axis_labels('target', 'pred')
                # plt.title(str_title)
                # plt.tight_layout()
                
    
                # plt.savefig(os.path.join(str(PATH_TO_GRAPH_DIR), "{}_result_comp_{}.png".format(datetime.now().strftime("%Y%m%d_%H%M%S"), str_title)))
                # plt.close()
            except Exception as e:
                print(e)
                print("******[error col_val : {}]******".format(str_title))
                
    
def test_inter():
    
    cols = ["area", "station", "baba"]
    inter_cols = itertools.combinations(cols, 2)
    for col in inter_cols:
        print(col)
    
    inter_cols = itertools.combinations(cols, 3)
    for col in inter_cols:
        print(col)
    
def calcPermutationWeightMean(df_list):
    
    # e_df = df_list[0]["weight"]
    
    # if len(df_list) > 1:
    
    #     for df in df_list[1:]:
    #         e_df += df["weight"]
    
    # mean = e_df/len(df_list)
    
    # #print(e_df)
    # print(mean[mean<0])
    
    total_df = pd.DataFrame()
    for i, df in enumerate(df_list):
        total_df[f"weight_{i}"] = df["weight"]
        
    #print(total_df)
    
    total_df["weight_mean"] = total_df.mean(axis=1)
    total_df = total_df.sort_values("weight_mean", ascending=False)
    #print(total_df.loc[total_df["weight_mean"]<0])
    
    return total_df

def testpermu():
    
    df0 = pd.read_csv(PATH_TO_FEATURES_DIR/"permutation_feature_imp_fold020200728_003522.csv", index_col=0)
    
    df0 = df0.set_index("feature")

    
    df1 = pd.read_csv(PATH_TO_FEATURES_DIR/"permutation_feature_imp_fold120200728_003957.csv", index_col=0)  
    df1 = df1.set_index("feature")
    
    idx = "madori"
    print(df0.loc[idx])
    print(df1.loc[idx])
    
    df_total = df0["weight"] + df1["weight"]
    print(df_total.loc[idx])
    
    df2 = pd.read_csv(PATH_TO_FEATURES_DIR/"permutation_feature_imp_fold220200728_004355.csv", index_col=0)  
    df2 = df2.set_index("feature")
    
    df3 = pd.read_csv(PATH_TO_FEATURES_DIR/"permutation_feature_imp_fold320200728_004850.csv", index_col=0)  
    df3 = df3.set_index("feature")
    
    df4 = pd.read_csv(PATH_TO_FEATURES_DIR/"permutation_feature_imp_fold420200728_005234.csv", index_col=0)  
    df4 = df4.set_index("feature")
    
    df_list = [df0, df1, df2, df3, df4]
    total_df = calcPermutationWeightMean(df_list)
    print("permutation feature importance weight under 0")
    print(total_df.loc[total_df["weight_mean"]<=0].index)
    
    total_df.to_csv(PATH_TO_FEATURES_DIR/"test_perm.csv", index=True)


def comparisonSub():

    mode = "nn"
    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train_{mode}.pkl')
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test_{mode}.pkl')
    print(df_test["seq_scored"].unique())
    test_pub_mol_ids = df_test.loc[df_test["seq_scored"]==68, "id"].unique()
    test_pra_mol_ids = df_test.loc[df_test["seq_scored"]==91, "id"].unique()


    test_pub_ids = [f"{mol_id}_{i}" for mol_id in test_pub_mol_ids for i in range(68) ]
    test_pra_ids = [f"{mol_id}_{i}" for mol_id in test_pra_mol_ids for i in range(91) ]

    path_to_dir=OUTPUT_DIR/ "fix_pair_value"
    pp_dir = Path(path_to_dir)
    
    target_cols = ['reactivity', 'deg_Mg_50C','deg_Mg_pH10']
    y_pred_list=[]
    oof_list = []
    name_lsit = []
    for i, f in enumerate(pp_dir.glob('*--_submission.csv')):
        #oof_f_name = f.name
        #print(oof_f_name)
        
        
        # df_oof = pd.read_csv(str(f.parent/oof_f_name), index_col=0)[target_cols]
        # print(f"df_oof : {df_oof}")
        # df_oof.sort_values('id_seqpos',inplace=True, ascending=True)
        # df_oof['sub_no'] = i
        # showNAN(df_oof)

        # oof_list.append(df_oof)

        #pred_f_name = oof_f_name.replace("oof", "submission")
        pred_f_name  = f.name
        print(pred_f_name)
        name_lsit.append(pred_f_name)
        
        df_pred = pd.read_csv(str(f.parent/pred_f_name), index_col=0)[target_cols]
        print(f"df_pred : {df_pred}")
        df_pred.sort_values('id_seqpos',inplace=True, ascending=True)
        df_pred['sub_no'] = i
        showNAN(df_pred)

        y_pred_list.append(df_pred)

    from sklearn.metrics import mean_absolute_error as mae
    for t in target_cols:
        print(f'\nMean abs difference in {t}:\n')
        for i in range(len(y_pred_list)):
            for j in range(i+1, len(y_pred_list)):
                # df_i, df_j = oof_list[i], oof_list[j]
                # abs_diff= mae(oof_list[i][t], oof_list[j][t])
                

                df_sub_i, ddf_sub_j = y_pred_list[i], y_pred_list[j]
                pub_abs_pred_diff= mae(df_sub_i.loc[test_pub_ids, t], ddf_sub_j.loc[test_pub_ids, t])
                pra_abs_pred_diff= mae(df_sub_i.loc[test_pra_ids, t], ddf_sub_j.loc[test_pra_ids, t])
                print(f'{name_lsit[i]} and {name_lsit[j]}:  pub {pub_abs_pred_diff:.5f}, pra {pra_abs_pred_diff:.5f}')

                pub_corr = df_sub_i.loc[test_pub_ids, t].corr(ddf_sub_j.loc[test_pub_ids, t])
                pri_corr = df_sub_i.loc[test_pra_ids, t].corr(ddf_sub_j.loc[test_pra_ids, t])
                #print(f'{name_lsit[i]} and {name_lsit[j]}:  pub {pub_corr:.5f}, pra {pri_corr:.5f}')


    idx = np.random.randint(0,len(y_pred_list[0]), 500)
    subs_vis = []
    for i in range(len(y_pred_list)):
        df_vis = y_pred_list[i].iloc[idx]
        df_vis.loc[:,target_cols] = df_vis[target_cols].values
        subs_vis.append(df_vis)
    df_vis = pd.concat(subs_vis) 

    import seaborn as sns
    sns.set_style(style="ticks")
    sns.pairplot(df_vis, hue="sub_no")
    plt.savefig(PATH_TO_GRAPH_DIR/"seaborntest.png") 


def rolling_window(a, w):
    s0, s1 = a.strides
    m, n = a.shape
    return np.lib.stride_tricks.as_strided(
        a, 
        shape=(m-w+1, w, n), 
        strides=(s0, s0, s1)
    )


def make_time_series(x, windows_size, pad_num):
  x = np.pad(x, [[ windows_size-1, 0], [0, 0]], constant_values=pad_num)
  x = rolling_window(x, windows_size)
  return x






def calcEvalScoreDict(y_true, y_pred, eval_metric_func_dict):
    #print("y_true")
    #print(np.unique(y_true, return_counts=True))
    #print("y_pred")
    #print(np.unique(y_pred, return_counts=True))

    eval_score_dict={}
    for name, f in eval_metric_func_dict.items():
        score  = f(y_pred=y_pred, y_true=y_true)
        eval_score_dict[name] = score
    
    return eval_score_dict

def TestReduceALL():

    x = np.array([[[0, 0, 0], [1, 0, 3], [2, 2, 2]], [[0, 0, 0], [0, 0, 0], [1, 0, 3]]])

    pdb.set_trace()


def searchCheckptFile(ppath_to_save_dir, ppath_to_model, prefix):

    model_name = ppath_to_model.stem.split("__")[-1]

    fold_num = int(prefix.replace("fold_", ""))
    ckpt_name = f"{model_name}_train_model.ckpt"
    if fold_num > 0:
        ckpt_name = ckpt_name.replace(".ckpt", f"-v{fold_num}.ckpt")
    #multiLabelNet_train_model-v4.ckpt

    return ppath_to_save_dir/ckpt_name

def numpy_normalize(v):

    return v/np.linalg.norm(v)
    
if __name__ == '__main__':
    TestReduceALL()