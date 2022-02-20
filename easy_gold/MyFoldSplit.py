import sys
from calendar import month_name

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, GroupKFold, KFold
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from dateutil.relativedelta import relativedelta
from collections import Counter, defaultdict
from utils import *

def stratified_group_k_fold(X, y, groups, k, seed=2021):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

class StratifiedKFoldWithGroupID():

    def __init__(self, group_id_col, stratified_target_id, n_splits, random_state=2021, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.group_id_col = group_id_col
        self.stratified_target_id = stratified_target_id
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, _df_X, _se_y, _group, *args, **kwargs):

        df = _df_X[[self.group_id_col]]

        if isinstance(_se_y, pd.DataFrame):
            df["group_target"] = _se_y[self.stratified_target_id]
        else:
            df["group_target"] = _se_y

        df["group_target"]  = procLabelEncToSeries(df["group_target"])
        df = df.reset_index()

        for train_idx, test_idx in stratified_group_k_fold(X=df, y=df["group_target"].values, groups=df[self.group_id_col].values, k=self.n_splits, seed=self.random_state):
            
            print(f"fold train :{df.iloc[train_idx]['group_target'].value_counts()}")
            print(f"fold valid :{df.iloc[test_idx]['group_target'].value_counts()}")
            #pdb.set_trace()

            yield train_idx, test_idx


class ssl1Fold():
    def __init__(self):
        self.n_splits = 1#n_splits
        

    def split(self, _df_X, _df_y, _group, *args, **kwargs):

        df = _df_X.reset_index()

        for i in range(1):

            train_idx = df.index
            test_idx = df.loc[df["train_flag"]==1].index

            yield train_idx, test_idx



class DummyKfold():
    def __init__(self, n_splits, random_state):
        self.n_splits = n_splits
        self.random_state = random_state


    def split(self, _df_X, _df_y, _group, *args, **kwargs):

        # num_data = len(_df_X)
        # idx_list = np.arange(num_data)
        # kf = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)



        for i in range(self.n_splits):
        #     print("TRAIN:", train_index, "TEST:", test_index)

        #     yield train_index, test_index
            yield [0], [0]




class SeasonKFold():
    """時系列情報が含まれるカラムでソートした iloc を返す KFold"""

    def __init__(self, n_splits, ts_column="Season", clipping=False, num_seasons=5,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # 時系列データのカラムの名前
        self.ts_column = ts_column
        # 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
        self.clipping = clipping
        
        self.num_seasons = num_seasons
        
        self.n_splits = n_splits

    def split(self, X, *args, **kwargs):
        # 渡されるデータは DataFrame を仮定する
        assert isinstance(X, pd.DataFrame)

        # clipping が有効なときの長さの初期値
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # 時系列のカラムを取り出す
        ts = X[self.ts_column]
        # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
        ts_df = ts.reset_index()
        
        ts_list = sorted(ts_df[self.ts_column].unique())
        

        
        for i in range(self.n_splits):
            # 添字を元々の DataFrame の iloc として使える値に変換する
            
            
           
            
            train_list = ts_list[:-self.num_seasons]
            test_list= ts_list[-self.num_seasons:]
            print(f"train season: {train_list}")
            print(f"test season: {test_list}")
            #pdb.set_trace()
            
            train_iloc_index = ts_df.loc[ts_df[self.ts_column].isin(train_list)].index
            test_iloc_index = ts_df.loc[ts_df[self.ts_column].isin(test_list)].index

            
            ts_list = train_list
            
            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])

class TournamentGroupKFold(GroupKFold):
    
    def __init__(self, group_id_col="Season", day_num_col="DayNum",tournament_start_daynum=133, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_id_col = group_id_col
        self.day_num_col = day_num_col
        self.tournament_start_daynum = tournament_start_daynum

        
    def split(self, _df_X, _df_y, _group, *args, **kwargs):
        
        
        df_tournament = _df_X.loc[_df_X[self.day_num_col]>=self.tournament_start_daynum]
        df_tournament_y = _df_y.loc[df_tournament.index]
        df_reg = _df_X.loc[~_df_X.index.isin(df_tournament.index)]
        
        reg_index = _df_X.index.get_indexer(df_reg.index).tolist()
        

        
        if _group is None:
            _group = df_tournament[self.group_id_col]

        for train_id_index, test_id_index in super().split(df_tournament, df_tournament_y, _group):
            
              
            # print(f"train_id_index:{train_id_index}")
            # print(f"test_id_index:{test_id_index}")
            # print(f"train season: {df_tournament.iloc[train_id_index]['Season'].unique()}")
            # print(f"test season: {df_tournament.iloc[test_id_index]['Season'].unique()}")
            
            train_set_ID = df_tournament.iloc[train_id_index].index
            test_set_ID = df_tournament.iloc[test_id_index].index
            
            train_index_list = _df_X.index.get_indexer(train_set_ID).tolist()
            test_index_list = _df_X.index.get_indexer(test_set_ID).tolist()

            
            yield train_index_list+reg_index, test_index_list

   
    
    


class VirtulTimeStampSplit():
    
    def __init__(self, n_splits, num_valid=2500000):
        self.num_valid = num_valid
        self.n_splits = n_splits
    
    def split_melt(self, _df_X, _df_y, _group):
        
        cpu_stats(f"in split")

        row_id_sequence =  _df_X.index.tolist()
        row_id_list = _df_X.index.unique().tolist()
        #print(row_id_list)

        all_train_idx = range(len(_df_X))

        print(f"row_id_sequence : {sys.getsizeof(row_id_sequence)}")
        print(f"row_id_list : {sys.getsizeof(row_id_list)}")
        print(f"all_train_idx : {sys.getsizeof(all_train_idx)}")
        cpu_stats(f"after creating all_train_idx")
        
        for n in range(self.n_splits):
            

            
            valid_row_id_list = row_id_list[-self.num_valid:]
            first_valid_row_id = valid_row_id_list[0]
            valid_id_from = row_id_sequence.index(first_valid_row_id)
            valid = all_train_idx[valid_id_from:]

            cpu_stats(f"mid yield")

            row_id_list = row_id_list[:-self.num_valid]
            all_train_idx = all_train_idx[:valid_id_from]#_df_X.index.get_loc(row_id_list)

            
            print(f"fold : {n}")
            print(f"train : {len(all_train_idx)},  {all_train_idx}")
            print(f"valid : {len(valid)},  {valid}")
            cpu_stats(f"before yield")
            
            yield all_train_idx, valid

    def split(self, _df_X, _df_y, _group):

        all_train_idx = range(len(_df_X))
        
        for n in range(self.n_splits):
            
            valid = all_train_idx[-self.num_valid:]
            all_train_idx = all_train_idx[:-self.num_valid]

            print(f"fold : {n}")
            print(f"train : {len(all_train_idx)},  {all_train_idx}")
            print(f"valid : {len(valid)},  {valid}")
            
            yield all_train_idx, valid


def return_debug_index(debug, _train_idx, _val_idx, rate=0.5):

    if debug:
        train_idx = random.sample(_train_idx, int(len(_train_idx)*rate))
        val_idx = random.sample(_val_idx, int(len(_val_idx)*rate))
    else:
        train_idx = _train_idx
        val_idx = _val_idx

    return train_idx, val_idx

            
class myGroupKFold(GroupKFold):

    def __init__(self, group_id_col, cut_uout_flag, debug, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_id_col = group_id_col
        self.cut_uout_flag = cut_uout_flag
        self.debug = debug

    def split(self, _df_X, _df_y, _group=None, *args, **kwargs):

        df_X = _df_X.reset_index()

        #if _group is None:
        group = df_X[self.group_id_col]
        

        for train_index, val_index in super().split(df_X, _df_y, group):
            
            if self.debug:
                train_gp_id_list = list(df_X.iloc[train_index][self.group_id_col].unique())
                val_gp_id_list = list(df_X.iloc[val_index][self.group_id_col].unique())
                train_gp_id_list, val_gp_id_list = return_debug_index(self.debug, train_gp_id_list, val_gp_id_list, rate=0.5)
                train_index = df_X.loc[df_X[self.group_id_col].isin(train_gp_id_list)].index
                val_index = df_X.loc[df_X[self.group_id_col].isin(val_gp_id_list)].index

            #.set_trace()
            if self.cut_uout_flag:
                df_train = df_X.iloc[train_index]
                new_train_index = df_train.loc[df_train["u_out"]==0].index
                #new_train_index=train_index


                df_val = df_X.iloc[val_index]
                new_val_index = df_val.loc[df_val["u_out"]==0].index
                print(f"train: {df_X.loc[train_index, 'u_out'].value_counts()}")
                print(f"test: {df_X.loc[val_index, 'u_out'].value_counts()}")
                print(f"test u_out 0 : {df_X.loc[new_val_index, 'u_out'].value_counts()}")
                yield new_train_index, new_val_index
            else:
                yield train_index, val_index

    

class TimeStampNewUserSplit(GroupKFold):
    
    def __init__(self, group_id_col, new_user_head_num=10, old_user_tail_num=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_id_col = group_id_col
        self.new_user_head_num = new_user_head_num
        self.old_user_tail_num = old_user_tail_num
        
    def split(self, _df_X, _df_y, _group, *args, **kwargs): # we assume df has already been sorted by timestamp

        df = _df_X[[self.group_id_col]]
        df[_df_y.columns] = _df_y
        df = df.reset_index()
        
        if _group is None:
            _group = df[self.group_id_col]
        
        fold_idx=1
        for old_id_index, new_id_index in super().split(df[self.group_id_col], df[_df_y.columns], _group):
            
              
            test_new_user_index = set(df.iloc[new_id_index].groupby(self.group_id_col).head(self.new_user_head_num).index)
            test_old_user_index = set(df.iloc[old_id_index].groupby(self.group_id_col).tail(self.old_user_tail_num).index)
            
            #print(f"test_new_user : {len(test_new_user)}, test_old_user : {len(test_old_user)}")
            #print(f"test_new_user : {len(test_new_user_index)}, test_old_user : {len(test_old_user_index)}")
            #print(f"train_old_user_index ; {len(train_old_user_index)}, add : {len(train_old_user_index) + len(test_old_user_index)}")
            #print(f"old_id_index : {len(old_id_index)}, new_id_index : {len(new_id_index)}")
            
            #print( df.iloc[new_id_index].groupby(self.group_id_col).head(self.new_user_head_num))
            #print(f"{df.iloc[test_old_user_index].groupby(self.group_id_col).count()}")
            
            
            cpu_stats(f"TimeStampNewUserSplit")
            fold_idx+=1
            
            yield list(set(old_id_index) - test_old_user_index), list(test_new_user_index|test_old_user_index)

   
            
        
class myStratifiedKFold(StratifiedKFold):
    
    def __init__(self, stratified_col, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stratified_col = stratified_col
        
    def split(self, _df_X, _df_y, dummy_group, *args, **kwargs):
        
        
        
        
        if  self.stratified_col in _df_y.columns:
            
            df_y = _df_y[[self.stratified_col]]
        else:  #self.stratified_col in _df_X.columns:
        
            df_y = _df_X[[self.stratified_col]]
            
        
        df_y = df_y.reset_index()
 
        for train_id_index, test_id_index in super().split(_df_X, df_y[self.stratified_col]):
            
            print(f"fold train :{train_id_index}")
            print(f"{df_y.iloc[train_id_index][self.stratified_col].value_counts()}")
            
            print(f"fold valid :{test_id_index}")
            print(f"{df_y.iloc[test_id_index][self.stratified_col].value_counts()}")
            
            # train_label_list = df_y.iloc[train_id_index][self.stratified_col].unique()
            # test_label_list = df_y.iloc[test_id_index][self.stratified_col].unique()
            # only_test_list = list(set(test_label_list) - set(train_label_list))
            # if len(only_test_list)>0:
            #     only_test_index = list(df_y.loc[df_y[self.stratified_col].isin(only_test_list)].index)
            #     train_id_index = np.array(train_id_index.tolist()+only_test_index)
                
            #     for num in only_test_index:
            #         new_test_id_index = test_id_index.tolist()
            #         new_test_id_index.remove(num)
            #         test_id_index = np.array(new_test_id_index)
            
            yield train_id_index, test_id_index

        


class siteStratifiedPathGroupKFold():

    def __init__(self, df_test, group_id_col, stratified_target_id, n_splits):

        self.group_id_col = group_id_col
        self.stratified_target_id = stratified_target_id
        self.n_splits = n_splits

        self.df_test_info = df_test.groupby(self.stratified_target_id)[self.group_id_col].agg(["count", "nunique"])


    def split(self, _df_X, _se_y, _group, *args, **kwargs):

        df_train = _df_X[[self.group_id_col, self.stratified_target_id]]
        df_train = df_train.reset_index()
        df_train["fold"]=self.n_splits

        for site_id, row in self.df_test_info.iterrows():
            count = row["count"]
            nunique = row["nunique"]

            path_list = df_train.loc[df_train[self.stratified_target_id]==site_id, self.group_id_col].unique()
            random.shuffle(path_list)

            path_set_list= [t for t in zip(*[iter(path_list)]*nunique)]
            diff_dict = {}
            for i, path_set in enumerate(path_set_list):
                #print(f"{i} : {path_set}")
                train_path_count = df_train.loc[df_train[self.group_id_col].isin(path_set), self.group_id_col].count()
                diff_count = abs(train_path_count-count)
                diff_dict[i] = diff_count
            
            sort_i_list = sorted(diff_dict.items(), key=lambda x:x[1])
            #print(sort_i_list)

            for k in range(self.n_splits):
                sort_i = sort_i_list[k][0]

            
                path_set =path_set_list[sort_i]
                df_train.loc[df_train[self.group_id_col].isin(path_set), "fold"] = k
                ##print(f"{sort_i}, {sort_i_list[k]}")
                #print(f"df_train fold k : {df_train.loc[df_train['fold']==k].shape}")
            #pdb.set_trace()

        for k in range(self.n_splits):

            #df_fold_t = df_train.loc[df_train["fold"]==k, ["site_id", "path"]]

            #print(f"fold {k}:")
            #print(df_fold_t.groupby("site_id")["path"].agg(["count", "nunique"]))


            yield list(df_train.loc[df_train["fold"]!=k].index), list(df_train.loc[df_train["fold"]==k].index)


            

    
class myStratifiedKFoldWithGroupID(StratifiedKFold):

    def __init__(self, group_id_col, stratified_target_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.group_id_col = group_id_col
        self.stratified_target_id = stratified_target_id

    def split(self, _df_X, _se_y, _group, *args, **kwargs):

        #_df_X["group_target"] = _se_y
        df = _df_X[[self.group_id_col, self.stratified_target_id]]
        #df["group_target"] = _se_y
        df = df.reset_index()
        gp = df.groupby(self.group_id_col)[self.stratified_target_id].apply(lambda x:x.mode()[0])
        
        #print(gp)
        #print(gp.index)
        #del df
        #gc.collect()

        fold_idx=1
        for train_id_index, test_id_index in super().split(gp.index, gp,  _group):
            #print(f"fold_idx : {fold_idx}")
            #print(f"train_id_index : {train_id_index}, test_id_index : {test_id_index}")
            #print(f"train_id : {gp.index[train_id_index]}, test_id : {gp.index[test_id_index]}")
            train_id_list = list(gp.index[train_id_index])
            test_id_list = list(gp.index[test_id_index])

            print(f"fold train :{df.loc[df[self.group_id_col].isin(train_id_list), self.stratified_target_id].value_counts()}")
            print(f"fold valid :{df.loc[df[self.group_id_col].isin(test_id_list), self.stratified_target_id].value_counts()}")

            #print(f"train_seq_id : {df.loc[df[self.group_id_col].isin(train_id_list)].index}, test_id : {df.loc[df[self.group_id_col].isin(test_id_list)].index}")
            fold_idx+=1

            yield list(df.loc[df[self.group_id_col].isin(train_id_list)].index), list(df.loc[df[self.group_id_col].isin(test_id_list)].index)




# class StratifiedKFoldWithGroupID(StratifiedKFold):

#     def __init__(self, group_id_col, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.group_id_col = group_id_col

#     def split(self, _df_X, _se_y, _group, *args, **kwargs):

#         #_df_X["group_target"] = _se_y
#         df = _df_X[[self.group_id_col]]
#         df["group_target"] = _se_y

#         df["group_target"]  = procLabelEncToSeries(df["group_target"])
#         df = df.reset_index()
#         gp = df.groupby(self.group_id_col)["group_target"].apply(lambda x:x.mode()[0])
#         #print(gp)
        
#         #pdb.set_trace()
#         #print(gp.index)
#         #del df
#         #gc.collect()

#         fold_idx=1
#         for train_id_index, test_id_index in super().split(gp.index, gp,  _group):
#             #print(f"fold_idx : {fold_idx}")
#             #print(f"train_id_index : {train_id_index}, test_id_index : {test_id_index}")
#             #print(f"train_id : {gp.index[train_id_index]}, test_id : {gp.index[test_id_index]}")
#             train_id_list = list(gp.index[train_id_index])
#             test_id_list = list(gp.index[test_id_index])

#             print(f"fold train :{df.loc[df[self.group_id_col].isin(train_id_list), 'group_target'].value_counts()}")
#             print(f"fold valid :{df.loc[df[self.group_id_col].isin(test_id_list), 'group_target'].value_counts()}")

#             #print(f"train_seq_id : {df.loc[df[self.group_id_col].isin(train_id_list)].index}, test_id : {df.loc[df[self.group_id_col].isin(test_id_list)].index}")
#             fold_idx+=1

#             yield list(df.loc[df[self.group_id_col].isin(train_id_list)].index), list(df.loc[df[self.group_id_col].isin(test_id_list)].index)

        

def testStr():

    dict_pd = {
        "seq_id":["id0_0", "id0_1", "id0_2", "id0_3", "id1_0", "id1_1", "id1_2", "id1_3", "id2_0", "id2_1", "id2_2", "id2_3", "id3_0", "id3_1", "id3_2", "id3_3"],
        "val":[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "id":["id0", "id0", "id0", "id0", "id1", "id1", "id1", "id1", "id2", "id2", "id2", "id2", "id3", "id3", "id3", "id3"],
        "cat_y":[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,]
    }

    df = pd.DataFrame(dict_pd)
    df = df.set_index("seq_id")
    print(df)

    fold = StratifiedKFoldWithGroupID(n_splits=2, group_id_col="id")

    for tr_idx, vl_idx in fold.split(df, df["cat_y"], None):
        print(f"tr_idx : {tr_idx}, vl_idx : {vl_idx}")

class Day28KFold(TimeSeriesSplit):
    def __init__(self, date_column, clipping=False, test_days=28, split_type=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 時系列データのカラムの名前
        self.date_column = date_column
        # 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
        self.clipping = clipping

        self.test_days = test_days

        self.split_type= split_type

    def split(self, _X, _y, _group, *args, **kwargs):
        # 渡されるデータは DataFrame を仮定する
        assert isinstance(_X, pd.DataFrame)
        X = _X[self.date_column].reset_index()

        # clipping が有効なときの長さの初期値
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # 時系列のカラムを取り出す
        time_duration_list = X[self.date_column].unique()
        #print(f"time_duration_list : {time_duration_list}")
        ts_df = pd.DataFrame(time_duration_list, columns=[self.date_column])
        #print(ts_df)



        # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
        #ts_df = ts.reset_index()
        # 時系列でソートする
        sorted_ts_df = ts_df.sort_values(by=self.date_column)

        del ts_df
        gc.collect()
        
        # スーパークラスのメソッドで添字を計算する
        fold_idx=1
        for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
            # 添字を元々の DataFrame の iloc として使える値に変換する

            if self.split_type==1:

                train_start_day = sorted_ts_df.iloc[train_index].min()[0]
                test_end_day = sorted_ts_df.iloc[test_index].max()[0]
                test_start_day = test_end_day - relativedelta(days=(self.test_days - 1))
                train_end_day = test_start_day - relativedelta(days=1)
            
            elif self.split_type == 2:

                last_day=sorted_ts_df[self.date_column].max()
                test_start_day = last_day - relativedelta(days=(self.test_days - 1) * fold_idx)
                test_end_day = test_start_day + relativedelta(days=(self.test_days - 1))
                train_end_day = test_start_day - relativedelta(days=1)
                train_start_day = sorted_ts_df[self.date_column].min()
                print(f"last_day :{last_day}")



            train_iloc_index = X[X[self.date_column] <= train_end_day].index
            test_iloc_index = X[(X[self.date_column] >= test_start_day) & (X[self.date_column] <= test_end_day)].index

            print(f"train {train_start_day} to {train_end_day}")
            print(f"test {test_start_day} to {test_end_day}")

            

            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

                print(f"train_fold_min_len : {train_fold_min_len}")
                print(f"test_fold_min_len : {test_fold_min_len}")

            print("********************")
            fold_idx+=1
            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])




        

class MovingWindowKFold(TimeSeriesSplit):
    """時系列情報が含まれるカラムでソートした iloc を返す KFold"""

    def __init__(self, ts_column, clipping=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 時系列データのカラムの名前
        self.ts_column = ts_column
        # 得られる添字のリストの長さを過去最小の Fold に揃えるフラグ
        self.clipping = clipping

    def split(self, X, *args, **kwargs):
        # 渡されるデータは DataFrame を仮定する
        assert isinstance(X, pd.DataFrame)

        # clipping が有効なときの長さの初期値
        train_fold_min_len, test_fold_min_len = sys.maxsize, sys.maxsize

        # 時系列のカラムを取り出す
        ts = X[self.ts_column]
        # 元々のインデックスを振り直して iloc として使える値 (0, 1, 2...) にする
        ts_df = ts.reset_index()
        # 時系列でソートする
        sorted_ts_df = ts_df.sort_values(by=self.ts_column)
        # スーパークラスのメソッドで添字を計算する
        for train_index, test_index in super().split(sorted_ts_df, *args, **kwargs):
            # 添字を元々の DataFrame の iloc として使える値に変換する
            train_iloc_index = sorted_ts_df.iloc[train_index].index
            test_iloc_index = sorted_ts_df.iloc[test_index].index

            if self.clipping:
                # TimeSeriesSplit.split() で返される Fold の大きさが徐々に大きくなることを仮定している
                train_fold_min_len = min(train_fold_min_len, len(train_iloc_index))
                test_fold_min_len = min(test_fold_min_len, len(test_iloc_index))

            yield list(train_iloc_index[-train_fold_min_len:]), list(test_iloc_index[-test_fold_min_len:])


def main():
    df = sns.load_dataset('flights')

    month_name_mappings = {name: str(n).zfill(2) for n, name in
                           enumerate(month_name)}
    df['month'] = df['month'].apply(lambda x: month_name_mappings[x])
    df['year-month'] = df.year.astype(str) + '-' + df.month.astype(str)
    df['year-month'] = pd.to_datetime(df['year-month'], format='%Y-%m')

    # データの並び順をシャッフルする
    df = df.sample(frac=1.0, random_state=42)

    # 特定のカラムを時系列としてソートした分割
    folds = MovingWindowKFold(ts_column='year-month', n_splits=5)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12))

    # 元々のデータを時系列ソートした iloc が添字として得られる
    for i, (train_index, test_index) in enumerate(folds.split(df)):
        print(f'index of train: {train_index}')
        print(f'index of test: {test_index}')
        print('----------')
        sns.lineplot(data=df, x='year-month', y='passengers', ax=axes[i], label='original')
        sns.lineplot(data=df.iloc[train_index], x='year-month', y='passengers', ax=axes[i], label='train')
        sns.lineplot(data=df.iloc[test_index], x='year-month', y='passengers', ax=axes[i], label='test')

    plt.legend()
    #plt.show()

    path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_MovingWindowKFold.png")
    plt.savefig(path_to_save)

def main2():

    df_train = pd.read_pickle(PROC_DIR / 'df_proc_train.pkl')

    folds = Day28KFold(date_column='date', n_splits=5)

    fig, axes = plt.subplots(5, 1, figsize=(12, 12))

    
    #df = df_train.loc[df_train["id"]=="HOBBIES_1_008_CA_1_validation", ["date", "demand"]]
    df = df_train[["date", "demand"]]

    # 元々のデータを時系列ソートした iloc が添字として得られる
    for i, (train_index, test_index) in enumerate(folds.split(df)):

        df_tr = df.iloc[train_index, :]
        df_val = df.iloc[test_index, :]
        train_min = df_tr["date"].min()
        train_max = df_tr["date"].max()
        test_min = df_val["date"].min()
        test_max = df_val["date"].max()

        print(f"train : {train_min} to {train_max}")
        print(f"test : {test_min} to {test_max}")

        print(f'index of train: {len(train_index)}')
        print(f'index of test: {len(test_index)}')
        print('----------')
    #     sns.lineplot(data=df, x='date', y='demand', ax=axes[i], label='original')
    #     sns.lineplot(data=df.iloc[train_index], x='date', y='demand', ax=axes[i], label='train')
    #     sns.lineplot(data=df.iloc[test_index], x='date', y='demand', ax=axes[i], label='test')

    # plt.legend()
    # #plt.show()

    # path_to_save = os.path.join(str(PATH_TO_GRAPH_DIR), datetime.now().strftime("%Y%m%d%H%M%S") + "_Day28KFold.png")
    # plt.savefig(path_to_save)

if __name__ == '__main__':
    testStr()