# coding:utf-8

from pathlib import Path

import numpy as np
import pandas as pd

import argparse
import inspect


from utils import *
#from eda import showDetails
from nlp_utils import *

    
    
def countNull(df, target_):
    
    list_col=list(df.columns)
    for del_col in target_:
        list_col.remove(del_col)
    
    df = addNanPos(df, list_col)
    
    df["null_all_count"] = df[list_col].isnull().sum(axis=1)
    
    #ord
    #df["null_ord_count"] = df[getColumnsFromParts(["ord_"], list(df.columns))].isnull().sum(axis=1)


    
    
    
    return df



def checkNan(df, target_, fill_val=-999):
    
    #df = df.replace([np.inf, -np.inf], np.nan)
    nan_cols = showNAN(df)
    
    for col in nan_cols:
        if not col in target_:
            if not ON_KAGGLE:
                print("fill na : ", col)
            #df[col].fillna(df[col].mode()[0], inplace=True)
            df[col].fillna(fill_val, inplace=True)
    
    return df
    
    
def removeColumns(df, drop_cols=[], not_proc_list=[]):
    
    
    
    #df_train, df_test = self.getTrainTest()
    #drop_cols += get_useless_columnsTrainTest(df_train, df_test, null_rate=0.95, repeat_rate=0.95)
    
    exclude_columns=not_proc_list + drop_cols
    dCol = checkCorreatedFeatures(df, exclude_columns=exclude_columns, th=0.99999)
    drop_cols.extend(dCol)
    
    
    final_drop_cols = []
    for col in drop_cols:
        if not col in not_proc_list and col in df.columns:
            if not ON_KAGGLE:
                print("remove : {}".format(col))
            final_drop_cols.append(col)
    
    df.drop(columns=final_drop_cols, inplace=True)
    return df
    


    
class Preprocessor:
    
    def __init__(self, _df_train, _df_test, index_col, _str_target_value, mode=None, _regression_flag=1):
        
        self.target_ = _str_target_value
        self.regression_flag_ = _regression_flag
        self.index_col = index_col
        self.mode = mode
        
        self.all_feature_func_dict = {t[0]:t[1] for t in inspect.getmembers(self, inspect.ismethod) if "fe__" in t[0] }
        self.all_proc_func_dict = {t[0]:t[1] for t in inspect.getmembers(self, inspect.ismethod) if "proc__" in t[0] }
        
        self.path_to_f_dir = PATH_TO_FEATURES_DIR
        
        if _df_test is not None:
            self.df_all_ = pd.concat([_df_train, _df_test], sort=False) #, ignore_index=True)
        else:
            self.df_all_ = _df_train
            
     

        self.original_columns_ = list(self.df_all_.columns)
        

        
        for del_col in self.target_:
            if del_col in self.original_columns_:
                self.original_columns_.remove(del_col)
        
        
        
        self.train_idx_ = _df_train.index.values
        self.test_idx_ = _df_test.index.values if _df_test is not None else []
        
        
        if not ON_KAGGLE:
            print(f"original_columns :{self.original_columns_}")
            print("df_train : ", _df_train.shape)
            print("self.train_idx_ : ", self.train_idx_)
            
            if _df_test is not None:
                print("df_test_ : ", _df_test.shape)
                print("self.test_idx_ : ", self.test_idx_)
        
        self.add_cols_ = []
        self.model = None
        
    def getTrainTest(self):
        
        
        if len(self.test_idx_) == 0:
            df_train = self.df_all_
        else:
            df_train = self.df_all_.loc[self.df_all_.index.isin(self.train_idx_)]
    
        
        df_test = self.df_all_.loc[self.df_all_.index.isin(self.test_idx_)]
        
        for col in self.target_:
            if col in df_test.columns:
                df_test.drop(columns=[col], inplace=True)
        
        
        
        if not ON_KAGGLE:
            pass
            # print("**** separate train and test ****")
            # print("df_train : ", df_train.shape)
            # print("self.train_idx_ : ", df_train.index)
            # print("df_test_ : ", df_test.shape)
            # print("self.test_idx_ : ", df_test.index)
        

        return df_train, df_test
    
    def fe__num_words(self, df):
        
        df["num_words"] = df["DESCRIPTION_TRANSLATED"].map(lambda x: len( x.split(" ")))
        
        
        return df
    
    def fe__COUNTRY_CURRENCY(self, df):

        df['COUNTRY_CURRENCY'] = df['COUNTRY_NAME'] + '_' + df['CURRENCY']


        return df
 

    def fe__TOWN_NAME_COUNTRY_NAME(self, df):

        df["TOWN_NAME"].fillna("None", inplace=True)
       
        
        df['TOWN_NAME_COUNTRY_NAME'] = df['TOWN_NAME'] + '_' + df['COUNTRY_NAME']


        return df

    def fe__CURRENCY_POLICY_CURRENCY_EXCHANGE_COVERAGE_RATE(self, df):

        df["CURRENCY_POLICY_CURRENCY_EXCHANGE_COVERAGE_RATE"] =  df["CURRENCY_POLICY"] + "_" + df["CURRENCY_EXCHANGE_COVERAGE_RATE"] + "_" + df["DISTRIBUTION_MODEL"]

        return df

    def fe__CURRENCY_POLICY_CURRENCY_EXCHANGE_COVERAGE_RATE_CURRENCY(self, df):

        df["CURRENCY_POLICY_CURRENCY_EXCHANGE_COVERAGE_RATE_CURRENCY"] = df["CURRENCY_POLICY_CURRENCY_EXCHANGE_COVERAGE_RATE"] + "_" + df["CURRENCY"]

        return df
    
    def loadFs(self, df, f_name_list):
        
        for f_name in f_name_list:
            force_flag = True if f_name in self.force_list else False
            df = self.loadF(df, f_name, force_flag)
            
        return df
    
    def loadF(self, df, f_name, force=False):
        
        if f_name in df.columns:
            print(f"{f_name} already exists in columns")
            return df
        
        import warnings
        warnings.filterwarnings('ignore')
        
        path_to_f = self.path_to_f_dir /f"{f_name}.pkl"
        
        if (path_to_f.exists()) & (force==False):
            
            df[f_name] = pd.read_pickle(path_to_f)
            print(f"load {f_name}")
            
        else:
            
            with timer2(f"create {f_name}"):
                df = self.all_feature_func_dict[f"fe__{f_name}"](df)

            df[f_name].to_pickle(path_to_f)
            
            #if not ON_KAGGLE:
                #df_train, df_test = self.getTrainTest()
                #showDetails(df_train, df_test, new_cols=[f_name], target_val=self.target_[0], corr_flag=False)
                
            
            
        return df
    
    
    def proc__0_eliminate_str(self, df):


        #前処理．明らかにおかしなものは修正．
        
        df.loc[df["DESCRIPTION_TRANSLATED"].isna(), "DESCRIPTION_TRANSLATED"] = df.loc[df["DESCRIPTION_TRANSLATED"].isna(), "DESCRIPTION"]

        
        df["DESCRIPTION_TRANSLATED"]  = df["DESCRIPTION_TRANSLATED"].str.replace("<br />", " ").values
        df["DESCRIPTION_TRANSLATED"]  = df["DESCRIPTION_TRANSLATED"].str.replace("\\\\t", " ").values
        df["DESCRIPTION_TRANSLATED"]  = df["DESCRIPTION_TRANSLATED"].str.replace("\\\\u200e", " ").values
        df["DESCRIPTION_TRANSLATED"]  = df["DESCRIPTION_TRANSLATED"].str.replace("\\u200b", " ").values
        df.loc[1772417, "DESCRIPTION_TRANSLATED"] = "Anara is 35 years old, married, and has 2 sons. Anara has a secondary education. She has been breeding livestock as the main income for her family since 2010. Thanks to her hard work and responsible approach to the business, Anara and her husband were able to develop their farm up to 5 cows and a horse. To further develop her farm, Anara applied to Bai Tushum Bank for a loan of 180,000 KGS to purchase livestock in order to increase her income. Anara plans to invest the loan income in the further development of her farm."
        
        

        df.loc[1667738, "DESCRIPTION_TRANSLATED"] =df.loc[1667738, "DESCRIPTION_TRANSLATED"].replace("13,0000", "13000")
        
        
        
        
        return df
        
        
        

    def proc__TAG(self, df):

        #TAGSのOne-Hot化
        
        train_rep = {' #Biz Durable Asset': '#Biz Durable Asset',
                     ' #First Loan': '#First Loan',
                     ' volunteer_pick': 'volunteer_pick',
                     ' #Single Parent': '#Single Parent',
                     ' #Refugee': '#Refugee',
                     ' #Repeat Borrower': '#Repeat Borrower',
                     ' #Technology': '#Technology',
                     ' volunteer_like': 'volunteer_like',
                     ' #US immigrant': '#US immigrant',
                     ' user_favorite': 'user_favorite',
                     ' #Vegan': '#Vegan',
                     ' #Single': '#Single',
                     ' #Animals': '#Animals',
                     ' #Schooling': '#Schooling',
                     ' #Health and Sanitation': '#Health and Sanitation',
                     ' #Widowed': '#Widowed',
                     ' #Repair Renew Replace': '#Repair Renew Replace',
                     ' #Sustainable Ag': '#Sustainable Ag',
                     ' #Parent': '#Parent',
                     ' #Trees': '#Trees',
                     ' #Unique': '#Unique',
                     ' #US Black-Owned Business': '#US Black-Owned Business',
                     ' #Elderly': '#Elderly',
                     ' #Orphan': '#Orphan',
                     ' #Supporting Family': '#Supporting Family',
                     ' #Job Creator': '#Job Creator',
                     ' #Female Education': '#Female Education',
                     ' #Fabrics': '#Fabrics',
                     ' #Eco-friendly': '#Eco-friendly',
                     ' #Woman-Owned Business': '#Woman-Owned Business'}
        
        df['TAGS'] = df['TAGS'].replace(train_rep, regex=True)
        tags_df2 = df['TAGS'].str.get_dummies(sep=',').add_prefix('tag_')
        
        df = pd.concat([df, tags_df2], axis=1)
        
        df.drop(columns=["TAGS"], inplace=True)
        
        return df

        
    
    def proc__CURRENCY_EXCHANGE_COVERAGE_RATE(self, df):

        df["CURRENCY_EXCHANGE_COVERAGE_RATE"].fillna("None", inplace=True)
        df["CURRENCY_EXCHANGE_COVERAGE_RATE"]=df["CURRENCY_EXCHANGE_COVERAGE_RATE"].astype(str)
        
        


        return df
    
    def proc__delite(self, df):
        
        drop_cols = ["COUNTRY_CODE"]
        
        df.drop(columns=drop_cols, inplace=True)
        
        return df

    
    def proc__inter(self, df):

        #組み合わせ特徴を作成
        
        self.loadFs(df, f_name_list=["TOWN_NAME_COUNTRY_NAME"])
        
        cols = ["TOWN_NAME_COUNTRY_NAME", "ACTIVITY_NAME", "SECTOR_NAME", "COUNTRY_NAME",
                "LOAN_USE", "REPAYMENT_INTERVAL", "ORIGINAL_LANGUAGE", "CURRENCY"
                ]
        
        df = interactionFE(df, cols, inter_nums=[2])
        
        return df


    
    def proc__use_features(self, df):
        
        #Universal Sentence EncoderによるDESCRIPTION_TRANSLATEDの文章ベクトル化
        
        df_use = getUniversalSentenceEncoder(se_text=df["DESCRIPTION_TRANSLATED"], output_feature_dim=1000)

            
        df = pd.concat([df, df_use], axis=1)
        
        return df
    

    
    def proc__w2v(self, df):
        
        #GloVeで単語ベクトルを取得→SWEMで文章ベクトル化
        
        df_w2v =  GetWord2VecEmbeddings(se_text=df["DESCRIPTION_TRANSLATED"], output_feature_dim=1000)
   
            
        df = pd.concat([df, df_w2v], axis=1)
        
        return df
    
    def proc__bow(self, df):

        #BM25によるベクトル化→SVDで50次元に圧縮
        df_bow_bm25 = GetFeaturesOfBOW(se_text=df["DESCRIPTION_TRANSLATED"], output_feature_dim=50, vectorizer_type="bm25", compress_type="svd")
        
        
        df = pd.concat([df, df_bow_bm25], axis=1)
        
        return df
        
    
    
    def proc(self, params):
        
        
        
    
        
        all_feature_names = [name.replace("fe__", "") for name in  self.all_feature_func_dict.keys()]
        self.force_list = []
        if params["force"] is not None:
            if len(params["force"]) == 0:
                
                self.force_list = all_feature_names
            else:
                self.force_list = params["force"]
        
        load_list = all_feature_names

        for k, v in self.all_proc_func_dict.items():
            print(f"proc : {k}")
            self.df_all_ = v(self.df_all_)
        
        for f_name in load_list:
            print(f"feature proc : {f_name}")
            force_flag = True if f_name in self.force_list else False
            self.df_all_ = self.loadF(self.df_all_, f_name, force=force_flag)
            
            
            
        self.df_all_ =checkNan(self.df_all_, target_=self.target_, fill_val=-999)
        
        
        exclude_label= ['DESCRIPTION', 'DESCRIPTION_TRANSLATED', "LOAN_USE"]
        self.df_all_, decode_dict = proclabelEncodings(self.df_all_, not_proc_list=exclude_label+self.target_)
        pickle_dump(decode_dict, PROC_DIR / 'decode_dict.pkl')
        
        #self.df_all_ =  removeColumns(self.df_all_, drop_cols=[], not_proc_list=self.target_)


    

def procMain(df_train, df_test, index_col, target_col, setting_params):
    

            
    myPre = Preprocessor(df_train, df_test, index_col, target_col, mode=None)
    myPre.proc(params=setting_params)
    df_proc_train, df_proc_test = myPre.getTrainTest()

    return df_proc_train, df_proc_test




def main(setting_params):

    
    target_col = ['LOAN_AMOUNT']
    index_col = "LOAN_ID"


    full_load_flag=setting_params["full_load_flag"]
    save_pkl=1
    if ON_KAGGLE==True or full_load_flag==1:
        

        
        df_train = pd.read_csv(INPUT_DIR / f'train.csv')
        df_test = pd.read_csv(INPUT_DIR / f'test.csv')
        
        
        df_train = df_train.set_index(index_col)
        df_test = df_test.set_index(index_col)
        df_train = reduce_mem_usage(df_train)
        df_test = reduce_mem_usage(df_test)
        
        
        
        
        
        
        if save_pkl and ON_KAGGLE==False:
            df_train.to_pickle(INPUT_DIR / 'train.pkl')
            df_test.to_pickle(INPUT_DIR / 'test.pkl')
        
    else:
        
        
        df_train = pd.read_pickle(INPUT_DIR / 'train.pkl')
        df_test = pd.read_pickle(INPUT_DIR / 'test.pkl')

    

    print("df_train:{}".format(df_train.shape))
    print("df_test:{}".format(df_test.shape))
    

    df_proc_train, df_proc_test = procMain(df_train, df_test, index_col, target_col, setting_params) 
    df_proc_train = reduce_mem_usage(df_proc_train)
    df_proc_test = reduce_mem_usage(df_proc_test)

    df_proc_train.to_pickle(PROC_DIR / f'df_proc_train.pkl')
    df_proc_test.to_pickle(PROC_DIR / f'df_proc_test.pkl')

    
    print(f"df_proc_train:{df_proc_train.columns}")
    print("df_proc_train:{}".format(df_proc_train.shape))
    
    print(f"df_proc_test:{df_proc_test.columns}")
    print("df_proc_test:{}".format(df_proc_test.shape))
    
    


        
def argParams():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-m', '--mode', default="lgb", choices=['lgb','exp','nn','graph', 'ave', 'stack'] )
    
    
    parser.add_argument('-stack_dir', '--stacking_dir_name', type=int, )
    parser.add_argument('-full', '--full_load_flag', action="store_true")
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('-f', '--force', nargs='*')



    args=parser.parse_args()

    setting_params= vars(args)

    return setting_params

if __name__ == '__main__':
    setting_params=argParams()
    
    print(setting_params["force"])

    
    main(setting_params = setting_params)









