# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:14:29 2020

@author: p000526841
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime
import inspect
#from matplotlib_venn import venn2

from utils import *



plt.rcParams['font.family'] = 'IPAexGothic'


@contextmanager
def save_fig(path_to_save=PATH_TO_GRAPH_DIR/f"tmp.png"):
    
    plt.figure()
    yield
    plt.savefig(path_to_save)

def showCorr(df, str_value_name, show_percentage=0.6):
        corrmat = df.corr()
    
        num_of_col = len(corrmat.columns)
        cols = corrmat.nlargest(num_of_col, str_value_name)[str_value_name]
        tmp = cols[(cols >= show_percentage) | (cols <= -show_percentage)]

        print("*****[ corr : " + str_value_name + " ]*****")
        print(tmp)
        print("*****[" + str_value_name + "]*****")
        print("\n")
        
        #print(tmp[0])

def showBoxPlot(df, str_val1, str_va2):
    plt.figure(figsize=(15, 8))
    plt.xticks(rotation=90, size='small')
    
    #neigh_median = df.groupby([str_val1],as_index=False)[str_va2].median().sort_values(str_va2)
    #print(neigh_median)
    #col_order = neigh_median[str_val1].values
    #sns.boxplot(x=df[str_val1], y =df[str_va2], order=col_order)

    sns.boxplot(x=df[str_val1], y =df[str_va2])
    plt.tight_layout()
    path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_box_plot_{}.png".format(str_val1))
    #print("save: ", path_to_save)
    plt.savefig(path_to_save)
    plt.show(block=False) 
    plt.close()

def createVenn(train_set, test_set, title_str, path_to_save, train_label="train", test_label="test"):
    plt.figure()
    #venn2(subsets=[train_set,test_set],set_labels=(train_label,test_label))
    plt.title(f'{title_str}',fontsize=20)
    #path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_venn_{}.png".format(title_str))
    #print("save: ", path_to_save)
    plt.savefig(path_to_save)
    plt.show(block=False) 
    plt.close()

def showValueCount(df_train, df_test, str_value, str_target, debug=True, regression_flag=1, _fig_size=(20,10)):

        if str_value == str_target:
            df_test[str_value] = np.nan
        
        df = pd.concat([df_train, df_test])
        if not str_value in df.columns:
            print(str_value, " is not inside columns")
            return 
        
        
        se_all = df[str_value]
        se_train = df_train[str_value]
        se_test = df_test[str_value]
        
        all_desc = se_all.describe()
        train_desc = se_train.describe()
        test_desc = se_test.describe()
        df_concat_desc = pd.concat([train_desc, test_desc, all_desc], axis=1, keys=['train', 'test', "all"])
        
        if debug:
            print("***[" + str_value + "]***")
            print("describe :")
            print(df_concat_desc)
        
        num_nan_all = se_all.isna().sum()
        num_nan_train = se_train.isna().sum()
        num_nan_test = se_test.isna().sum()
        df_concat_num_nan = pd.DataFrame([num_nan_train, num_nan_test, num_nan_all], columns=["num_of_nan"], index=['train', 'test', "all"]).transpose()
        
        if debug:
            print("Num of Nan : ")
            print(df_concat_num_nan)
        
        df_value = se_all.value_counts(dropna=False)
        df_value_percentage = (df_value / df_value.sum()) * 100
        
        
        df_value_train = se_train.value_counts(dropna=False)
        df_value_train_percentage = (df_value_train / df_value_train.sum()) * 100
        
        df_value_test = se_test.value_counts(dropna=False)
        df_value_test_percentage = (df_value_test / df_value_test.sum()) * 100
        
        df_concat = pd.concat([df_value_train, df_value_train_percentage, df_value_test, df_value_test_percentage, df_value, df_value_percentage], axis=1, keys=['train', "train rate", 'test', "test rate", "all", "all rate"], sort=True)
        
        train_values = set(se_train.unique())
        test_values = set(se_test.unique())

        xor_values = test_values - train_values 
        if xor_values:
            #print(f'Replace {len(xor_values)} in {col} column')
            print(f'{xor_values} is only found in test, not train!!!')
            
            #full_data.loc[full_data[col].isin(xor_values), col] = 'xor'
            
        xor_values_train = train_values - test_values
        if xor_values_train:
            #print(f'Replace {len(xor_values)} in {col} column')
            print(f'{xor_values_train} is only found in train, not test!!!' )
            
            #full_data.loc[full_data[col].isin(xor_values), col] = 'xor'
        
        
        if debug:
            
            # plt.figure()
            # venn2(subsets=[train_values,test_values],set_labels=('train','test'))
            # plt.title(f'{str_value}',fontsize=20)
            # path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_venn_{}.png".format(str_value))
            # #print("save: ", path_to_save)
            # plt.savefig(path_to_save)
            # plt.show(block=False) 
            # plt.close()
            path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_venn_{}.png".format(str_value))
            createVenn(train_set=train_values, test_set=test_values, title_str=str_value, path_to_save=path_to_save, train_label="train", test_label="test")
            
            print("value_counts :")
            print(df_concat)
            
            plt.figure(figsize=_fig_size)
            df_graph = df_concat[['train', 'test', "all"]].reset_index()
            df_graph = pd.melt(df_graph, id_vars=["index"], value_vars=['train', 'test', "all"])
            sns.barplot(x='index', y='value', hue='variable', data=df_graph)
            #sns.despine(fig)

            #df_concat[['train', 'test', "all"]].dropna().plot.bar(figsize=_fig_size)

            
            plt.ylabel('Number of each element', fontsize=12)
            plt.xlabel(str_value, fontsize=12)
            plt.xticks(rotation=90, size='small')
            plt.tight_layout()
            path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_num_each_elments_{}.png".format(str_value))
            #print("save: ", path_to_save)
            plt.savefig(path_to_save)
            plt.show(block=False) 
            plt.close()
        
            plt.figure(figsize=_fig_size)
            df_graph = df_concat[['train rate', 'test rate', "all rate"]].reset_index()
            df_graph = pd.melt(df_graph, id_vars=["index"], value_vars=['train rate', 'test rate', "all rate"])
            sns.barplot(x='index', y='value', hue='variable', data=df_graph)
            
            #df_concat[['train rate', 'test rate', "all rate"]].plot.bar(figsize=_fig_size)
            plt.ylabel('rate of each element', fontsize=12)
            plt.xlabel(str_value, fontsize=12)
            plt.xticks(rotation=90, size='small')
            plt.tight_layout()
            path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_rate_each_elments_{}.png".format(str_value))
            #print("save: ", path_to_save)
            plt.savefig(path_to_save)
            plt.show(block=False) 
            plt.close()

        
        
        
        if str_value != str_target and str_target in df.columns:
            
            if regression_flag == 1:
                if debug:
                    showBoxPlot(df_train, str_value, str_target)
            
            else:
                
                df_train_small = df.loc[df[str_target].isnull() == False, [str_value, str_target]]
                df_stack = df_train_small.groupby(str_value)[str_target].value_counts().unstack()
                
                if debug:
                    print("---")
                
                col_list = []
                df_list = []
                
                if debug:
                    plt.figure(figsize=_fig_size)
                    g = sns.countplot(x=str_value, hue = str_target, data=df, order=df_stack.index)
                    plt.xticks(rotation=90, size='small')
                    ax1 = g.axes
                    ax2 = ax1.twinx()
                    
                for col in df_stack.columns:
                    col_list += [str(col), str(col)+"_percent"]
                    df_percent = (df_stack.loc[:, col] / df_stack.sum(axis=1))
                    
                    df_list += [df_stack.loc[:, col], df_percent]
                    
                    if debug:
                        #print(df_percent.index)
                        xn = range(len(df_percent.index))
                        sns.lineplot(x=xn, y=df_percent.values, ax=ax2)
                        #sns.lineplot(data=df_percent, ax=ax2)
                        #sns.lineplot(data=df_percent, y=(str(col)+"_percent"), x=df_percent.index)
                    
                df_conc = pd.concat(df_list, axis=1, keys=col_list)
                
                if debug:
                    print(df_conc.T)
                    #print(df_stack.columns)
                    #print(df_stack.index)

                    #plt.tight_layout()
                    path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_count_line_{}.png".format(str_value))
                    #print("save: ", path_to_save)
                    plt.savefig(path_to_save)
                    plt.show(block=False) 
                    plt.close()
        
                
        if debug:
            print("******\n")
        
        del df
        gc.collect()
        
        return df_concat



def showJointPlot(_df_train, _df_test, str_value, str_target, debug=True, regression_flag=1, corr_flag=False, empty_nums=[], log_flag=1, _fig_size=(20, 10)):
        print("now in function : ", inspect.getframeinfo(inspect.currentframe())[2])
        
        df_train = _df_train.copy()
        df_test = _df_test.copy()
        
        if str_value == str_target:
            df_test[str_target] = np.nan
        

        if len(empty_nums) >0:
            for e in empty_nums:
                df_train[str_value] = df_train[str_value].replace(e, np.nan)
                df_test[str_value] = df_test[str_value].replace(e, np.nan)
        
        if log_flag==1:
            df_train[str_value] = np.log1p(df_train[str_value])
            df_test[str_value] = np.log1p(df_test[str_value])
        
        df = pd.concat([df_train, df_test])

        if not str_value in df.columns:
            print(str_value + " is not inside columns")
            return 
        

        se_all = df[str_value]
        se_train = df_train[str_value]
        se_test = df_test[str_value]
        
        all_desc = se_all.describe()
        train_desc = se_train.describe()
        test_desc = se_test.describe()
        df_concat_desc = pd.concat([train_desc, test_desc, all_desc], axis=1, keys=['train', 'test', "all"])
        
        print("***[" + str_value + "]***")
        print("describe :")
        print(df_concat_desc)
        
        num_nan_all = se_all.isna().sum()
        num_nan_train = se_train.isna().sum()
        num_nan_test = se_test.isna().sum()
        df_concat_num_nan = pd.DataFrame([num_nan_train, num_nan_test, num_nan_all], columns=["num_of_nan"], index=['train', 'test', "all"]).transpose()
        
        print("Num of Nan : ")
        print(df_concat_num_nan)
        
        skew_all = se_all.skew()
        skew_train = se_train.skew()
        skew_test = se_test.skew()
        df_concat_skew = pd.DataFrame([skew_train, skew_test, skew_all], columns=["skew"], index=['train', 'test', "all"]).transpose()
        
        
        print("skew : ")
        print(df_concat_skew)
        
        if corr_flag==True:
            showCorr(df, str_value)
        
        
               
        #tmp_se = pd.Series( ["_"] * len(df_dist), columns=["dataset"] )
        #print(tmp_se.values)
        #df_dist.append(tmp_se)
        #df_dist["dataset"].apply(lambda x: "train" if pd.isna(x[self.str_target_value_]) == False else "test")
        #df_dist.plot(kind="kde", y=df_dist["dataset"])
        
        plt.figure(figsize=_fig_size)
        sns.distplot(df_train[str_value].dropna(),kde=True,label="train")
        sns.distplot(df_test[str_value].dropna(),kde=True,label="test")

        plt.title('distplot by {}'.format(str_value),size=20)
        plt.xlabel(str_value)
        plt.ylabel('prob')
        plt.legend() #実行させないと凡例が出ない。
        plt.tight_layout()
        path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_distplot_{}.png".format(str_value))
        #print("save: ", path_to_save)
        plt.savefig(path_to_save)
        plt.show(block=False) 
        plt.close()
        #sns.distplot(df_dist[str_value], hue=df_dist["dataset"])
        
        #visualize_distribution(df[str_value].dropna())
        #visualize_probplot(df[str_value].dropna())
        
#        plt.figure(figsize=(10,5))
#        
#        sns.distplot()
#        plt.show()
 
        if (str_value != str_target) and (str_target in df.columns):
            #plt.figure(figsize=(10,5))
            if regression_flag == 1:
                sns.jointplot(str_value, str_target, df_train)
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_jointplot_{}.png".format(str_value))
                #print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()
                
                df_train.plot.hexbin(x=str_value, y=str_target, gridsize=15, sharex=False)
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_hexbin_{}.png".format(str_value))
                #print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()

                
            #plt.show()
            else:
                
                df_small = df_train[[str_value, str_target]]
                print(df_small.groupby(str_target)[str_value].describe().T)
                
                type_val = df_small[str_target].unique()
                #print(type_val)
                plt.figure()
                for i, val in enumerate(type_val):
                    sns.distplot(df_small.loc[df_small[str_target]==val, str_value].dropna(),kde=True,label=str(val)) #, color=mycols[i%len(mycols)])
                plt.legend() #実行させないと凡例が出ない。
                plt.tight_layout()
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_distplot_target_{}.png".format(str_value))
                #print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()
                
                
                plt.figure(figsize=_fig_size)
                plt.xlabel(str_value, fontsize=9)
                for i, val in enumerate(type_val):
                    sns.kdeplot(df_small.loc[df_small[str_target] == val, str_value].dropna().values, bw=0.5,label='Target: {}'.format(val))
                    
                sns.kdeplot(df_test[str_value].dropna().values, bw=0.5,label='Test')
                path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_kde_target_{}.png".format(str_value))
                print("save: ", path_to_save)
                plt.savefig(path_to_save)
                plt.show(block=False) 
                plt.close()
                
        
        print("******\n")
        
        del df, df_train, df_test
        gc.collect()
        
        return
    
    
def showDetails(_df_train, _df_test, new_cols, target_val, debug=True, regression_flag=1, corr_flag=True,  obj_flag=False):
    
    df_train = _df_train.copy()
    
    
    
    if _df_test is None:
        df_test = pd.DataFrame(index=[0], columns=new_cols)
    else:
        df_test = _df_test.copy()
        

    

    for new_col in new_cols:
        
        if obj_flag:
            df_train[new_col] = df_train[new_col].astype("str")
            df_test[new_col] = df_test[new_col].astype("str")
        
        try:
            if df_train[new_col].dtype == "object":
                showValueCount(df_train, df_test, new_col, target_val, debug=debug, regression_flag=regression_flag)
            else:
                showJointPlot(df_train, df_test, new_col, target_val, debug=debug, regression_flag=regression_flag, corr_flag=corr_flag, empty_nums=[-999], log_flag=0)
        except Exception as e:
            print(e)
            print("******[error col : {}]******".format(new_col))
            
def interEDA(df_train, df_test, inter_col, new_cols, target_val, _fig_size=(10, 5)):

    df = pd.concat([df_train, df_test])
    elements = df[inter_col].unique()
    type_val = df[target_val].unique()

    for col in new_cols:
        plt.figure(figsize=_fig_size)
        plt.title('interaction kde of {}'.format(inter_col),size=20)
        plt.xlabel(col, fontsize=9)
        for e in elements:

            df_small = df_train.loc[df_train[inter_col] == e]
            for i, val in enumerate(type_val):
                sns.kdeplot(df_small.loc[df_small[target_val] == val, col].dropna().values, bw=0.5,label='Inter:{}, Target: {}'.format(e, val))
                
            sns.kdeplot(df_test.loc[df_test[inter_col]==e, col].dropna().values, bw=0.5,label='Inter:{}, Test'.format(e))
        path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_inter_kde_{}_vs_{}.png".format(inter_col, col))
        print("save: ", path_to_save)
        plt.savefig(path_to_save)
        plt.show(block=False) 
        plt.close()




def procEDA_(df_train, df_test):
    
    #df_train = df_train[df_train["y"] < (90)]
    
    new_col=["延床面積（㎡）"]#
    #new_col=df_train.columns
    showDetails(df_train, df_test, new_col, "y", corr_flag=False)
    
    sys.exit()
    #df["nom"]
    #print(df_train.loc[df_train["auto__nom_8_count_agg_by_nom_7"] > 30000, "auto__nom_8_count_agg_by_nom_7"])
    
    #showValueCount(df_train, df_test, "ord_5", "target", debug=True, regression_flag=0)
    for i in range(6):
        

        col_name = "ord_{}".format(i)
        df_train[col_name] /= df_train[col_name].max()  # for convergence
        df_test[col_name] /= df_test[col_name].max()
        
        
        new_name = "{}_sqr".format(col_name)
        df_train[new_name] = 4*(df_train[col_name] - 0.5)**2
        df_test[new_name] = 4*(df_test[col_name] - 0.5)**2
    #
    new_col=["ord_3", "ord_3_sqr"]#getColumnsFromParts(["ord_3"], df_train.columns)
    #showDetails(df_train, df_test, new_col, "target", corr_flag=False)
    for col in new_col:
        showJointPlot(df_train, df_test, col, "target", debug=True, regression_flag=0, corr_flag=False, empty_nums=[-999], log_flag=1)
    
        
    # new_cols=list(df_train.columns.values)
    # new_cols.remove("bin_3")
    # new_cols.remove("target")
    # #new_cols=["null_all_count"]
    # #new_cols = getColumnsFromParts(["bin_3"], df_train.columns)
    # #showDetails(df_train, df_test, new_cols, "target")
    # for col in new_cols:
    #     try:
    #         interEDA(df_train, df_test, col, ["bin_3"], "target")
    #     except Exception as e:
    #         print(e)
    #         print("******[inter error col : {}]******".format(col))

    sys.exit(0)
    
    
    # colums_parts=[]
    
    # parts_cols = getColumnsFromParts(colums_parts, df_train.columns)
    
    # new_cols = list(set(new_cols + parts_cols))

    use_columns=list(df_test.columns)
    bin_list = ["bin_{}".format(i) for i in range(5)]
    
    ord_list = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "ord_5_1", "ord_5_2"]

    nom_list = ["nom_{}".format(i) for i in range(10)] #getColumnsFromParts(["nom_"] , use_columns)
    #oh_nom_list = getColumnsFromParts(["OH_nom"] , use_columns)
    time_list = ["day", "month"]

    nan_pos_list = getColumnsFromParts(["_nan_pos"] , use_columns)
    count_list = getColumnsFromParts(["_count"], use_columns)
    #inter_list = getColumnsFromParts(["inter_"], use_columns)
    additional_list = ["day_cos", "day_sin", "month_cos", "month_sin"]


    embedding_features_list=time_list + ord_list + nom_list + bin_list
    continuous_features_list =  additional_list+count_list +nan_pos_list
    final_cols = embedding_features_list+continuous_features_list

    #adversarialValidation(df_train[final_cols], df_test[final_cols], drop_cols=[])
    adv2(df_train[final_cols], df_test[final_cols], drop_cols=[])
    
    return



def procSave():
    
    df_train, df_test = loadRaw()
    
    print("df_train_interm:{}".format(df_train.shape))
    print("df_test_interm:{}".format(df_test.shape))
    

    df_train["part"] = "train"
    df_test["part"] = "test"
    
    df = pd.concat([df_train, df_test])
    
    
    syurui_list = list(df["種類"].unique())

    
    for w in syurui_list:
        
        df_csv = df.loc[(df["種類"]==w)]
        df_csv.to_csv(PROC_DIR /"syurui_{}.csv".format(w), encoding='utf_8_sig')
        
        
def edaSeasonMatch(df_train, df_test):
    
    print(df_train.groupby("Season")["total_match_team1"].mean())    
    print(df_test.groupby("Season")["total_match_team1"].mean())    
    
    sys.exit()
    
    
def compare_sample(df_train, df_test):
    
    df_train_sample = pd.read_csv(PROC_DIR/"df_train_sample.csv", index_col=0)
    df_train_sample["org_ID"] = [f"{s}_{w}_{l}" for s, w, l in zip(df_train_sample["Season"], df_train_sample["TeamIdA"], df_train_sample["TeamIdB"])]
    df_train_sample.set_index("org_ID", inplace=True)
    df_train_sample = df_train_sample.loc[df_train_sample["Season"]<2015]
    
    
    df_train =df_train.loc[df_train["Season"]>=2003]
    # df_swap = df_train[["team1ID", "team2ID", "seed_num_diff", "Season"]]
    # df_swap["team2ID"] = df_train["team1ID"]
    # df_swap["team1ID"] = df_train["team2ID"]
    # df_swap["seed_num_diff"] = -1 * df_train["seed_num_diff"]
    
    
    # df_train = pd.concat([df_train[["team1ID", "team2ID", "seed_num_diff", "Season"]],df_swap])
    df_train["org_ID"] = [f"{s}_{w}_{l}" for s, w, l in zip(df_train["Season"], df_train["team1ID"], df_train["team2ID"])]
    df_train.set_index("org_ID", inplace=True)
    
    df_train_sample["diff"] = df_train_sample["SeedDiff"]- df_train["seed_num_diff"]
    print(df_train_sample["diff"].value_counts())
    
    print(df_train_sample["SeedDiff"].equals(df_train["seed_num_diff"]))
    print(df_train.shape)
    print(df_train_sample.shape)
    
    sys.exit()
    
def procEDA2(df_train, df_test):

    
    #compare_sample(df_train, df_test)
    #print(df_train.loc[(df_train["team1ID"]==1411)&(df_train["team2ID"]==1421)])
    #sys.exit()
    
    target_col="Pred"
    
    # other_list = ['FGM','FGA','FGM3','FGA3','FTM','FTA',
    #                                      'OR','DR','Ast','TO','Stl','Blk','PF',
    #                                       'goal_rate', '3p_goal_rate',
    #                                      'ft_goal_rate']
    # new_col= getColumnsFromParts(other_list, df_train.columns)
    
    new_col = getColumnsFromParts(["pair", ], df_train.columns)
    
    object_flag = 0
    
    if object_flag:
        for col in new_col:
            for tmp_df in [df_train, df_test]:
                tmp_df[col] = tmp_df[col].astype("object")

    
    #new_col=df_train.columns
    showDetails(df_train, df_test, new_col, target_col, corr_flag=False)  
    
def tmp1(df_train, df_test):
    
    area = ["千代田区", "新宿区", "江戸川区", "神津島村", "町田市"]
    tmp = df_train.loc[df_train["市区町村名"].isin(area), :].groupby("市区町村名")["地区名"].value_counts(dropna=False)
    print(tmp)
    
    tmp = df_train.loc[df_train["地区名"].isnull(), ["市区町村名", "種類", "地域"]]
    print(tmp)
    
    
    
def procAdv(df_train, df_test):
    
    drop_cols=[]
    rank_cols = getColumnsFromParts(["ranksystem_"], df_train.columns)
    df_train=df_train[rank_cols]
    df_test = df_test[rank_cols]
    adversarialValidation(df_train, df_test, drop_cols=drop_cols)
    
def eda_site(df_train, df_test):
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    df_sub = pd.read_csv(INPUT_DIR/'sample_submission.csv')

    for df in [df_train, df_test, df_sub]:
        df["site_id"] = df["site_path_timestamp"].map(lambda x: x.split("_")[0])
        df["path"] = df["site_path_timestamp"].map(lambda x: x.split("_")[1])
        df["timestamp"] = df["site_path_timestamp"].map(lambda x: x.split("_")[2])
        

        print(df.groupby("site_id")["path"].agg(["count", "nunique"]))

    n_split = 5

    df_test_info = df_test.groupby("site_id")["path"].agg(["count", "nunique"])

    for site_id, row in df_test_info.iterrows():
        count = row["count"]
        nunique = row["nunique"]

        path_list = df_train.loc[df_train["site_id"]==site_id, "path"].unique()
        random.shuffle(path_list)

        path_set_list= [t for t in zip(*[iter(path_list)]*nunique)]
        diff_dict = {}
        for i, path_set in enumerate(path_set_list):
            #print(f"{i} : {path_set}")
            train_path_count = df_train.loc[df_train["path"].isin(path_set), "path"].count()
            diff_count = abs(train_path_count-count)
            diff_dict[i] = diff_count
        
        sort_i_list = sorted(diff_dict.items(), key=lambda x:x[1])
        #print(sort_i_list)

        for k in range(n_split):
            sort_i = sort_i_list[k][0]

          
            path_set =path_set_list[sort_i]
            df_train.loc[df_train["path"].isin(path_set), "fold"] = k
            ##print(f"{sort_i}, {sort_i_list[k]}")
            #print(f"df_train fold k : {df_train.loc[df_train['fold']==k].shape}")
            #pdb.set_trace()


    for k in range(n_split):

        df_fold_t = df_train.loc[df_train["fold"]==k, ["site_id", "path"]]

        print(f"fold {k}:")
        print(df_fold_t.groupby("site_id")["path"].agg(["count", "nunique"]))





def eda_rssi(df_train, df_test):

    target_col="waypoint_x"
    #df_train[target_col] = df_train[target_col].astype(float)
    for col in ['timestamp', 'wifi_lastseen_ts', 'wifi_rssi',  "start_time",  'waypoint_timestamp', "waypoint_x", "waypoint_y"]:
        
        df_train[col] = df_train[col].astype(float)
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(float)

    df_train["diff_from_lastseen"] = df_train["wifi_lastseen_ts"] - df_train["timestamp"]
    df_test["diff_from_lastseen"] = df_test["wifi_lastseen_ts"] - df_test["timestamp"]

    eda_cols = ['timestamp' ,'wifi_ssid', 'wifi_bssid', 'wifi_rssi', 'wifi_frequency', "diff_from_lastseen", ]
    for i, site_id in enumerate(df_train["site_id"].unique()):

        print(f"site {i} : {site_id}")
        _df_train = df_train.loc[df_train["site_id"]==site_id]
        _df_test = df_test.loc[df_test["site_id"]==site_id]


        showDetails(_df_train, _df_test, eda_cols, target_col,debug=True, regression_flag=1, corr_flag=False)
    

def eda_test(df_train, df_test):

    print(df_train)
    df_train_old = pd.read_pickle(PROC_DIR / f'df_proc_train_nn_old.pkl')
    df_test_old = pd.read_pickle(PROC_DIR / f'df_proc_test_nn_old.pkl')


    pdb.set_trace()



def eda_site_id_bssid(df_train, df_test):
    print(df_train.columns)
    #pdb.set_trace()
    site_id_list = df_test["site_id"].unique()

    for s in site_id_list:
        print(f"site_id : {s}")
        df_tr = df_train.loc[df_train["site_id"]==s]
        df_te = df_test.loc[df_test["site_id"]==s]
        print(f"train : {df_tr['wifi_bssid'].nunique()}, test : {df_te['wifi_bssid'].nunique()}")
        path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_venn_{}.png".format(s))
   
        createVenn(train_set=set(df_tr['wifi_bssid'].unique()), 
                    test_set=set(df_te['wifi_bssid'].unique()), 
                    title_str=s, 
                    path_to_save=path_to_save,
                    train_label="train", test_label="test")

def createScatter(df_train, df_test, x_col, y_col, title_str, path_to_save):
    plt.figure(figsize=(20, 10))

    ax = df_train.plot.scatter(x=x_col, y=y_col, alpha=0.5, s=0.5, c='b')
    df_test.plot.scatter(x=x_col, y=y_col, c='r', s=0.5, alpha=0.5, ax=ax)

    
    plt.title(f'{title_str}',fontsize=10)
    #path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_venn_{}.png".format(title_str))
    #print("save: ", path_to_save)
    plt.savefig(path_to_save)
    plt.show(block=False) 
    plt.close()

def eda_last_seen(df_train, df_test):

    for col in ['timestamp', 'wifi_lastseen_ts', ]:
        
        df_train[col] = df_train[col].astype(float)
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(float)

    for col in ['timestamp', 'wifi_lastseen_ts',]:
        
        df_train[col] = df_train[col].astype(int)
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(int)

    df_train["site_path_timestamp"] = [f"{s}_{p}_{t}" for s, p, t in zip(df_train["site_id"], df_train["path"], df_train.index)]
    df_test["site_path_timestamp"] = [f"{s}_{p}_{t}" for s, p, t in zip(df_test["site_id"], df_test["path"], df_test.index)]
    df_train = df_train.set_index("site_path_timestamp")
    df_test = df_test.set_index("site_path_timestamp")

    print(df_train.columns)
    #
    site_id_list = df_test["site_id"].unique()

    for s in site_id_list:
        print(f"site_id : {s}")
        df_tr = df_train.loc[df_train["site_id"]==s]
        df_tr["f_p"] = df_tr["floor"]
        df_te = df_test.loc[df_test["site_id"]==s]
        df_te["floor"] = np.nan#df_te["path"]
        #pdb.set_trace()
        cols = ["wifi_lastseen_ts", "floor"]
        df_all = pd.concat([df_tr[cols], df_te[cols]])
        df_all = df_all.sort_values("wifi_lastseen_ts")
        df_all["floor"] = df_all["floor"].fillna(method='ffill')

        df_te["floor"] = df_all.loc[df_te.index, "floor"]
        


        tit = f"{s}_{df_tr['floor'].unique()}"
        path_to_save= os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_{}_wifi_lastseen_ts.png".format(s))
        createScatter(df_tr, df_te, x_col="floor", y_col="wifi_lastseen_ts", title_str=tit, path_to_save=path_to_save)
        #pdb.set_trace()
        showDetails(df_tr, df_te, new_cols=["wifi_lastseen_ts", "floor"], target_val="x", debug=True, regression_flag=1, corr_flag=False)
        
    
          

def proc(df_train, df_test):

    # train_site = set(list(df_train["site_id"].unique()))
    # test_site = set(list(df_test["site_id"].unique()))
    # pdb.set_trace()
    # df_train["site_id"] = df_train["site_id"].map(lambda x: x.name)
    # print(train_site)
    # print(test_site)
    # df_train.to_pickle(PROC_DIR/"df_train_wifi.pkl")
    # sys.exit()

    #eda_rssi(df_train, df_test)
    eda_last_seen(df_train, df_test)

    


def procEDA(_df_train, _df_test, eda_cols=[], convert_obj=False):
   
    

    target_col = "x"
    #eda_cols = ['wb_0']#, "diff_t1_wifi",] #'wifi_rssi', "wifi_frequency", 'wifi_lastseen_ts',  "start_time", ]
    #eda_cols = [ 'wifi_ssid', 'wifi_bssid', 'wifi_rssi', 'wifi_frequency']
    
    # print(_df_train.columns)
    rssi_list = getColumnsFromParts(["wr_"], _df_train.columns)
    eda_cols = rssi_list + ["cumcount"]
    # bssi_list = getColumnsFromParts(["wb_"], _df_train.columns)
    # print(set(_df_train.columns)-set(bssi_list))
    # sys.exit()
    
    if len(eda_cols) == 0:
        eda_cols = list(_df_train.columns)
        df_train = _df_train
        df_test = _df_test
        
        eda_cols.remove(target_col)
    else:
    
        tmp_cols = eda_cols + [target_col]
        df_train = _df_train[tmp_cols]
        df_test = _df_test[eda_cols]
    
    df_train[target_col] = df_train[target_col].astype(float)
    #df_test[target_col] = df_test[target_col].astype(float)
    print(df_train[target_col].describe())
    
    
    if convert_obj:
      for col in eda_cols:
          for tmp_df in [df_train, df_test]:
              
              if col in tmp_df.columns:
                  tmp_df[col] = tmp_df[col].astype(object) 

    #df_train = df_train.loc[(df_train["wifi_rssi"] < -30) & (df_train["wifi_rssi"] > -75)]
    #df_test = df_test.loc[(df_test["wifi_rssi"] < -30) & (df_test["wifi_rssi"] > -75)]
    #.loc[(df_train["diff_t1_wifi"] > 3000), "diff_t1_wifi"] = 3000
    #df_test.loc[(df_test["diff_t1_wifi"] > 3000), "diff_t1_wifi"] = 3000
    
    #eda_cols = ["prev_x", "prev_y",]#getColumnsFromParts(["x", "y", "floor"], _df_train.columns)
    showDetails(df_train, df_test, eda_cols, target_col,debug=True, regression_flag=1, corr_flag=False)
    
def procMeta(df_train, df_test):

    print(df_train.columns)
    print(df_test.columns)

    df_total_train = pd.read_pickle(INPUT_DIR / f'df_total_train.pkl')
    print(df_total_train.columns)
    df_total_test = pd.read_pickle(INPUT_DIR / f'df_total_test.pkl')
    print(df_total_test.columns)

    pdb.set_trace()
    #sys.exit()

    target_col="floor_num"

    eda_cols=['start_time', 
       'brand', 'model', 'android_name', 'api_level',
       'accelerometer_type', 'accelerometer_name', 'accelerometer_version',
       'accelerometer_vendor', 'accelerometer_resolution',
       'accelerometer_power', 'accelerometer_maximumRange', 'gyroscope_type',
       'gyroscope_name', 'gyroscope_version', 'gyroscope_vendor',
       'gyroscope_resolution', 'gyroscope_power', 'gyroscope_maximumRange',
       'magnetometer_type', 'magnetometer_name', 'magnetometer_version',
       'magnetometer_vendor', 'magnetometer_resolution', 'magnetometer_power',
       'magnetometer_maximumRange', 'accelerometer_uncalib_type',
       'accelerometer_uncalib_name', 'accelerometer_uncalib_version',
       'accelerometer_uncalib_vendor', 'accelerometer_uncalib_resolution',
       'accelerometer_uncalib_power', 'accelerometer_uncalib_maximumRange',
       'gyroscope_uncalib_type', 'gyroscope_uncalib_name',
       'gyroscope_uncalib_version', 'gyroscope_uncalib_vendor',
       'gyroscope_uncalib_resolution', 'gyroscope_uncalib_power',
       'gyroscope_uncalib_maximumRange', 'magnetometer_uncalib_type',
       'magnetometer_uncalib_name', 'magnetometer_uncalib_version',
       'magnetometer_uncalib_vendor', 'magnetometer_uncalib_resolution',
       'magnetometer_uncalib_power', 'magnetometer_uncalib_maximumRange',
       'version_name', 'version_code']
    showDetails(df_train, df_test, eda_cols, target_col,debug=True, regression_flag=0, corr_flag=False)
    



def loadRaw():
    
    df_train = pd.read_csv(INPUT_DIR / "train_all.csv")
    print(f"laod df_train : {df_train.shape}")
    df_test = pd.read_csv(INPUT_DIR / "test_all.csv")
    print(f"laod df_test : {df_test.shape}")


    return df_train, df_test

def loadProc(mode=None, decode_flag=False):

    mode_suf = f"_{mode}" if mode is not None else ""
   
    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train{mode_suf}.pkl')
    #print(f"load df_train : {df_train.shape}")
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test{mode_suf}.pkl')
    #print(f"load df_test : {df_test.shape}")
    
    if decode_flag:
        dec_dict = pickle_load(PROC_DIR / 'decode_dict.pkl')
        for col in df_test.columns:
            if col in dec_dict.keys():
                df_train[col].replace(dec_dict[col], inplace=True)
                df_test[col].replace(dec_dict[col], inplace=True)
            


    return df_train, df_test

def ts_plot(df_train, df_test):
    #df = pd.concat([df_train, df_test])
    
    cols = ["time_step", "pressure", "u_in", "u_out", "oof", "shift_p2_cumsum_time_lag_by_u_in"]
    df_train, df_test = AddScaleCols(df_train, df_test, cols)
    
    R_cols = [5, 20, 50]
    C_cols = [10, 20, 50]
    
    for r in R_cols:
        for c in C_cols:
            ppath_to_dir = PATH_TO_GRAPH_DIR/f"ts_plot/{r}_{c}"
            os.makedirs(ppath_to_dir, exist_ok=True)
            
            b_ids = sorted(df_train["breath_id"].unique())
            for b in b_ids[:100]:
                ppath_to_png = ppath_to_dir/f"{r}_{c}_train_{b}.png"
                eda_ts(df_train, b, ppath_to_png, add_col="shift_p2_cumsum_time_lag_by_u_in")
                #pdb.set_trace()
                
                
def AddScaleCols(df_train, df_test, cols):
    df = pd.concat([df_train[cols], df_test[cols]])
    
    mm_sc_ = preprocessing.MinMaxScaler()
    
    for col in cols:
        mm_sc_.fit(df.loc[:, col].values.reshape(-1, 1))
        df_train[f"scaled_{col}"] = mm_sc_.transform(df_train[col].values.reshape(-1, 1))
        df_test[f"scaled_{col}"] = mm_sc_.transform(df_test[col].values.reshape(-1, 1))
        
        
    return df_train, df_test
        
            
def eda_ts(df, b_id, ppath_to_png, ts="time_step", add_col=None):
    
    
    
    cols = [ts, "pressure", "u_in", "u_out", "oof"]
    cols = cols + [add_col] if add_col is not None else cols
    scaled_cols = [f"scaled_{c}" for c in cols]
    breath_1 = df.loc[df['breath_id'] == b_id, cols+scaled_cols]
    
    fig = plt.figure(figsize = (12, 24)) 
    
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = ax1.twinx()
    
    ax1.plot(breath_1[ts], breath_1['pressure'], 'r-', label='pressure')
    ax1.plot(breath_1[ts], breath_1['oof'], 'm-', label='oof')
    ax1.plot(breath_1[ts], breath_1['u_in'], 'g-', label='u_in')
    ax2.plot(breath_1[ts], breath_1['u_out'], 'b-', label='u_out')
    
    ax1.set_xlabel(ts)
    
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    
    
    if add_col is not None:
        ax3 = fig.add_subplot(3, 1, 2)
        ax4 = ax3.twinx()
        ax3.plot(breath_1[ts], breath_1[add_col], 'o-', label=add_col)
        ax4.plot(breath_1[ts], breath_1['u_out'], 'b-', label='u_out')
        
        ax3.set_xlabel(ts)
        
        ax3.legend(loc=0)
        ax4.legend(loc=0)
        
        
    scaled_cols = [f"scaled_{c}" for c in cols]
    ax5 = fig.add_subplot(3, 1, 3)
    
    ax5.plot(breath_1[f"scaled_{ts}"], breath_1['scaled_pressure'], 'r-', label='pressure')
    ax5.plot(breath_1[f"scaled_{ts}"], breath_1['scaled_u_in'], 'g-', label='u_in')
    ax5.plot(breath_1[f"scaled_{ts}"], breath_1['scaled_u_out'], 'b-', label='u_out')
    if add_col is not None:
        ax5.plot(breath_1[f"scaled_{ts}"], breath_1[f"scaled_{add_col}"], label=add_col)
        
    ax5.set_xlabel(ts)
    
    ax5.legend(loc=0)

        
    
    
    plt.show()
    fig.savefig(ppath_to_png)
           

def _eda_ts(df, b_id, ppath_to_png, ts="time_step", add_col=None):
    
    
    fig, ax1 = plt.subplots(figsize = (12, 8))
    
    
    cols = [ts, "pressure", "u_in", "u_out", ]
    cols = cols + [add_col] if add_col is not None else cols
    breath_1 = df.loc[df['breath_id'] == b_id, cols]
    ax2 = ax1.twinx()
    
    ax1.plot(breath_1[ts], breath_1['pressure'], 'r-', label='pressure')
    ax1.plot(breath_1[ts], breath_1['u_in'], 'g-', label='u_in')
    ax2.plot(breath_1[ts], breath_1['u_out'], 'b-', label='u_out')
    
    if add_col is not None:
        ax1.plot(breath_1[ts], breath_1[add_col], 'o-', label=add_col)
    
    ax1.set_xlabel(ts)
    
    ax1.legend(loc=(1.1, 0.8))
    ax2.legend(loc=(1.1, 0.7))
    plt.show()
    fig.savefig(ppath_to_png)


def eda_lag(df_train, df_test):
    
    col = "diff_time_step_p1_gp_by_breath_id"
    train_small_outlier_id = df_train.loc[(df_train[col]<0.01)&(df_train["time_step"]!=0), "breath_id"].unique()
    test_small_outlier_id = df_test.loc[df_test[col]<0.01, "breath_id"].unique()
    print(f"train_small_outlier_id: {train_small_outlier_id}")
    print(f"test_small_outlier_id: {test_small_outlier_id}")
    
    train_large_outlier_id = df_train.loc[(df_train[col]>0.1)&(df_train["time_step"]!=0), "breath_id"].unique()
    
    pdb.set_trace()
    
    target_val="pressure"

    new_cols = ["diff_time_step_p1_gp_by_breath_id"]#["cumsum_time_lag_by_u_in"]#getColumnsFromParts(["cumsum"], df_train.columns)
    
    
    df_train = df_train.loc[df_train["u_out"]==0, new_cols+[target_val]]
    df_test = df_test.loc[df_test["u_out"]==0 , new_cols]

    exclude = []#["breath_id", "time_step", "u_in"]
    for col in exclude:
        new_cols.remove(col)

    for c in []:
        df_train[c] = df_train[c].astype("str")

        if c in df_test.columns:
            df_test[c] = df_test[c].astype("str")

    
    showDetails(df_train, df_test, new_cols=new_cols, target_val=target_val, regression_flag=1, corr_flag=True)
    
def main(params):

    df_y_pred = pd.read_csv(OUTPUT_DIR/"20211003-004740_20211003_092614_SimpleLSTM_Wrapper--0.348461--_submission.csv", index_col=0)
    df_y_pred.index = df_y_pred.index.map(lambda x: f"test_{x}")
    df_oof = pd.read_csv(OUTPUT_DIR/"20211003-004740_20211003_092614_SimpleLSTM_Wrapper--0.348461--_oof.csv", index_col=0)    
    

    df_train, df_test = loadProc(mode=params["mode"], decode_flag=False)
    pdb.set_trace()
    df_train["oof"] = df_oof["pressure"]
    df_test["oof"] = 0
    df_test["pressure"]=df_y_pred["pressure"]
    #ts_plot(df_train, df_test)
    
    
   
    
    
    eda_lag(df_train, df_test)
     
    pdb.set_trace()
    sys.exit()
    
    #pdb.set_trace()


def argParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="lgb", choices=['lgb','exp','nn','graph', 'ave', 'stack'] )
    

    parser.add_argument('-full', '--full_load_flag', action="store_true")
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('-f', '--force', nargs='*')
    


    args=parser.parse_args()

    setting_params= vars(args)

    return setting_params

if __name__ == '__main__':
    
    setting_params=argParams()
    main(params = setting_params)

    sys.exit()

    #df_train, df_test = loadRaw()
    
    procEDA2(df_train, df_test)
    #procAdv(df_train, df_test)
    #tmp1(df_train, df_test)
    #procSave()
    