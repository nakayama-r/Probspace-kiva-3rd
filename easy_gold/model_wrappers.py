# coding:utf-8

import os
import pathlib
import numpy as np
import pandas as pd

import lightgbm as lgb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


from DNNmodel import *
from utils import *



def extractModelParameters(original_param, model):
    
    model_params_keys = model.get_params().keys()
    model_params = {}
    for k, v in original_param.items():
        if k in model_params_keys:
            model_params[k] = v
    print(model_params)
    return model_params

def extractModelParametersWithStr(original_param, exclude_str="__"):

    
    model_params = {}
    for k, v in original_param.items():
        if not exclude_str in k:
            model_params[k] = v
    print(model_params)
    return model_params

class Averaging_Wrapper(object):
    
    def __init__(self, df_train, df_test, target_col, path_to_meta_feature_dir, rate_list=None):
        self.model = None
        self.initial_params = {
            "random_seed_name__":"random_state",
            
        }
        self.target_col = target_col
        self.rate_list_ = rate_list
        
        self.best_iteration_ = 1


        self.df_meta_train, self.df_meta_test = self.setMetaFromFiles(path_to_meta_feature_dir)
        idx1 = set(self.df_meta_train.index)
        idx2 = set(df_train.index)
        print(f"df_meta_train : {self.df_meta_train.shape}")
        print(f"df_train : {df_train.shape}")

        


        self.df_meta_train[target_col] = df_train.loc[self.df_meta_train.index, target_col]

        
        
    def setMeta(self, df_train, df_test, target_col:str, y_pred_list:list, oof_list:list):
        
        np_y_pred = np.concatenate(y_pred_list, 1)
        np_oof = np.concatenate(oof_list, 1)
        print(np_y_pred.shape)
        print(np_oof.shape)
        self.df_meta_test = pd.DataFrame(np_y_pred, index=df_test.index)
        self.df_meta_train = pd.DataFrame(np_oof, index=df_train.index)

        self.df_meta_train[target_col] = df_train[target_col]
        
    
    def setMetaFromFiles(self, path_to_meta_feature_dir):
        pp_dir = pathlib.Path(path_to_meta_feature_dir)
        
        y_pred_list=[]
        oof_list = []
        name_list = []
        for f in pp_dir.glob('*--_oof.csv'):
            oof_f_name = f.name
            name_list.append(oof_f_name)
            print(oof_f_name)
            
            df_oof = pd.read_csv(str(f.parent/oof_f_name), index_col=0)[self.target_col]

            
            
            print(f"df_oof : {df_oof}")
            oof_list.append(df_oof)


            
            
            pred_f_name = oof_f_name.replace("oof", "submission")
            print(pred_f_name)
            
            df_pred = pd.read_csv(str(f.parent/pred_f_name), index_col=0)[self.target_col]
            
            print(f"df_pred : {df_pred}")
            y_pred_list.append(df_pred)
        
        df_oof = pd.concat(oof_list, axis=1)
        df_oof.columns = name_list
        #df_oof.columns=[i for i in range(0, len(df_oof.columns))]
        
        has_null_row_index = df_oof.loc[df_oof.isnull().any(axis=1)].index
        df_oof = df_oof.loc[~df_oof.index.isin(has_null_row_index)]
        
        df_pred = pd.concat(y_pred_list, axis=1)
        df_pred.columns = name_list
        #df_pred.columns=[i for i in range(0, len(df_pred.columns))]
        self.name_list = name_list

        #pdb.set_trace()
            
        
        return df_oof, df_pred
    

    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
            
        se_rate = pd.Series(self.rate_list_, index=self.name_list)
        
        ppath_to_rate_file = ppath_to_save_dir / "rate.csv"
        if ppath_to_rate_file.exists():
            
            df_rate = pd.read_csv(ppath_to_rate_file, index_col=0)
            
            se_rate.name = f"fold{df_rate.shape[1]}"
            df_rate = pd.concat([df_rate, se_rate], axis=1)
        else:
            se_rate.name = "fold0"
            df_rate = pd.DataFrame(se_rate)
            
        df_rate.to_csv(ppath_to_rate_file)
        print("######################")
        print("###  rate          ###")
        print(df_rate.mean(axis=1))
        #pdb.set_trace()
        
        

        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        best_score_dict={}
        eval_metric_func_dict = params["eval_metric_func_dict__"]
        print(X_train)
        
        if self.rate_list_ == None:
            
            f = list(eval_metric_func_dict.values())[0]

            
            def calc_loss_f(_rate_list, df_input_X, y_true):
                
                this_pred = np.zeros(len(df_input_X))
                for c, r in zip(df_input_X.columns, _rate_list):
                    this_pred += (df_input_X[c] * r).values

                score  = f(y_pred=this_pred, y_true=y_true)
                
                return score
            
            initial_rate_list = [0.5] * X_train.shape[1]
            loss_partial = partial(calc_loss_f, df_input_X=X_train, y_true=y_train.values)
           
            opt_result = sp.optimize.minimize(loss_partial, initial_rate_list, method='nelder-mead')
            
            self.rate_list_ = opt_result["x"].tolist()
            print(f"*****  opt result : {self.rate_list_} *****")
            print()
                
            #pdb.set_trace()
        
            
        
        
        y_train_pred = self.predict(X_train) #X_train.mean(axis=1).values
        print(y_train_pred)

        
        best_score_dict["train"]=calcEvalScoreDict(y_true=y_train.values, y_pred=y_train_pred, eval_metric_func_dict=eval_metric_func_dict)

        if X_valid is not None:
            #y_valid_pred = self.model.predict(X_valid)
            y_valid_pred = self.predict(X_valid) #X_valid.mean(axis=1).values
            best_score_dict["valid"] = calcEvalScoreDict(y_true=y_valid.values, y_pred=y_valid_pred, eval_metric_func_dict=eval_metric_func_dict)
            
        if X_holdout is not None:
            
            #y_holdout_pred = self.model.predict(X_holdout)
            y_holdout_pred = self.predict(X_holdout)#X_holdout.mean(axis=1).values
            best_score_dict["holdout"] = calcEvalScoreDict(y_true=y_holdout.values, y_pred=y_holdout_pred, eval_metric_func_dict=eval_metric_func_dict)


        self.best_score_ = best_score_dict
        print(self.best_score_)
        
        self.setFeatureImportance(X_train.columns)

    def predict(self, X_test, oof_flag=True):
        
        if self.rate_list_ == None:
            print(f"X_test : {X_test}")
            print(f"X_test.mean(axis=1) : {X_test.mean(axis=1)}")
            return X_test.mean(axis=1).values
        else:
            pred = np.zeros(len(X_test))
            for c, r in zip(X_test.columns, self.rate_list_):
                pred += (X_test[c] * r).values
            return pred.reshape(-1, 1)

    def setFeatureImportance(self, columns_list):
        self.feature_importances_ = np.zeros(len(columns_list))
    



class PytrochLightningBase():

    def __init__(self):
        super().__init__()

        self.initial_params = {

                "eval_max_or_min": "min",
                "val_every":1,
                "dataset_params":{},
                "random_seed_name__":"random_state",
                'num_class':1, #binary classification as regression with value between 0 and 1
                "use_gpu":1,
                'multi_gpu':False,
                }
        self.edit_params = {}

        self.best_iteration_ = 1

        self.reload_flag = False

        self.model = None
  


    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        params = prepareModelDir(params, self.__class__.__name__)

        self.edit_params = params
        pl.seed_everything(params[params["random_seed_name__"]]) 
        torch.backends.cudnn.enabled = True

        self.model.setParams(params)


        batch_size = params["batch_size"]
        if batch_size < 0:
            batch_size = X_train.shape[0]


        data_set_train = params["dataset_class"](X_train, y_train, params["dataset_params"], train_flag=True)
        dataloader_train = torch.utils.data.DataLoader(data_set_train, batch_size=batch_size, shuffle=True, collate_fn=params["collate_fn"],num_workers=params["num_workers"]) #, sampler=ImbalancedDatasetSampler(data_set_train))
        

        dataloader_val = None
        if X_valid is not None:
            data_set_val = params["dataset_class"](X_valid, y_valid, params["dataset_params"], train_flag=True)
            dataloader_val = torch.utils.data.DataLoader(data_set_val, batch_size=batch_size, shuffle=False, collate_fn=params["collate_fn"],num_workers=params["num_workers"])



        wandb_logger=None
        if (not ON_KAGGLE) and (params["no_wandb"]==False):
            wandb_run = wandb.init(project=PROJECT_NAME, group=params["wb_group_name"], reinit=True, name=params["wb_run_name"] )
            wandb_logger = WandbLogger(experment=wandb_run)
            wandb_logger.log_hyperparams(params)
            
    
        early_stop_callback = EarlyStopping(
                                monitor=f'val_{params["eval_metric"]}',
                                min_delta=0.00,
                                patience=params['early_stopping_rounds'],
                                verbose=True,
                                mode=params['eval_max_or_min']
                            )



        checkpoint_callback = ModelCheckpoint(
                                dirpath=PATH_TO_MODEL_DIR / params["model_dir_name"],
                                filename=params["path_to_model"].stem,
                                verbose=True,
                                monitor=f'val_{params["eval_metric"]}',                                
                                mode=params['eval_max_or_min'],
                                save_weights_only=True,
                                )

        callbacks_list = []
        if params['early_stopping_rounds'] <= self.initial_params["epochs"]:
            callbacks_list = [early_stop_callback, checkpoint_callback]


        metrics_callback = MetricsCallback(monitor_metric=f'val_{params["eval_metric"]}',min_max_flag=params['eval_max_or_min'])
        callbacks_list.append(metrics_callback)
                                

        self.trainer = pl.Trainer(
                        num_sanity_val_steps=0,
                        gpus=self.initial_params["use_gpu"], 
                        check_val_every_n_epoch=self.initial_params["val_every"],
                        max_epochs=self.initial_params["epochs"],
                        accumulate_grad_batches=self.initial_params["accumulate_grad_batches"],
                        callbacks=callbacks_list,#early_stop_callback, checkpoint_callback],
                        logger=wandb_logger,  

        )   

        self.trainer.fit(self.model, dataloader_train, dataloader_val)

        if X_valid is None:

            new_path = checkExistsAndAddVnum(params["path_to_model"])
            self.trainer.save_checkpoint(str(new_path))
         

        self.best_score_, self.best_iteration_ = metrics_callback.getScoreInfo()
        print(self.best_score_)
        
        self.feature_importances_ = np.zeros(len(X_train.columns))


    def predict(self, X_test, oof_flag=False):
        
        num_tta = self.edit_params["num_tta"]
        

        batch_size=self.edit_params["batch_size"] 
        dummy_y = pd.DataFrame(np.zeros((X_test.shape[0], 1)), index=X_test.index)
        data_set_test = self.edit_params["dataset_class"](X_test, dummy_y, self.edit_params["dataset_params"], train_flag=(num_tta>1))
        dataloader_test = torch.utils.data.DataLoader(data_set_test, batch_size=batch_size, shuffle=False, collate_fn=self.edit_params["collate_fn"],num_workers=self.edit_params["num_workers"])
        
        self.model.oof_prediction = oof_flag
        if self.model.oof_prediction==False:
            self.trainer.logger = None 


        tta_list = []
        for tta_i in range(num_tta):
            print(f"tta : {tta_i+1}th")
            if self.reload_flag:
                
                self.trainer.test(test_dataloaders=dataloader_test, model=self.model)
            else:
                
                self.trainer.test(test_dataloaders=dataloader_test, ckpt_path='best')
            final_preds = self.model.final_preds
            tta_list.append(final_preds)

        #pdb.set_trace()

        

        return np.array(tta_list).mean(axis=0) 

    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
            
        ppath_to_model = ppath_to_save_dir / f"model__{prefix}__{model_dir_name}__{self.__class__.__name__}.pkl"
        torch.save(self.model.state_dict(), str(ppath_to_model))

        print(f'Trained nn model was saved! : {ppath_to_model}')
        
        with open(str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json"), 'w') as fp:
            json.dump(bs, fp)


    def procLoadModel(self, model_dir_name, prefix, params):
        self.edit_params = params

        self.model.cuda()

        
        if params["multi_gpu"]:
            self.model = nn.DataParallel(self.model)
        
        ppath_to_save_dir = PATH_TO_UPLOAD_MODEL_PARENT_DIR / model_dir_name
        print(f"ppath_to_save_dir : {ppath_to_save_dir}")
        print(list(ppath_to_save_dir.glob(f'model__{prefix}*')))
        #print(list(ppath_to_save_dir.iterdir()))
        
        name_list = list(ppath_to_save_dir.glob(f'model__{prefix}*'))
        if len(name_list)==0:
            print(f'[ERROR] Trained nn model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]

        ppath_to_ckpt_model = searchCheckptFile(ppath_to_save_dir, ppath_to_model, prefix)
        
        #pdb.set_trace()
        self.model.load_state_dict(torch.load(str(ppath_to_ckpt_model))["state_dict"])
        #self.model.load_state_dict(torch.load(str(ppath_to_model)))
        print(f'Trained nn model was loaded! : {ppath_to_ckpt_model}')
        
        a = int(re.findall('iter_(\d+)__', str(ppath_to_model))[0])
        
        
   
        #print(self.model.best_iteration_ )
        #self.model.best_iteration_ 
        self.best_iteration_ = a
        
        path_to_json = str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json")
        if not os.path.exists(path_to_json):
            print(f'[ERROR] Trained nn json was NOT EXITS! : {path_to_json}')
            return -1
        with open(path_to_json) as fp:
            self.best_score_ = json.load(fp)
            #self.model._best_score = json.load(fp)

        

        self.trainer = pl.Trainer(
                        num_sanity_val_steps=0,
                        gpus=self.edit_params["use_gpu"], 
                        check_val_every_n_epoch=self.edit_params["val_every"],
                        max_epochs=self.edit_params["epochs"],
        )
        self.reload_flag = True

        return 0




class BERT_Wrapper(PytrochLightningBase):
    def __init__(self, 
                model_name,
                auxiliary_loss_flag,
                text_features_list, 
                embedding_category_features_list,
                continuous_features_list,
                label_list,
                weight_list=None,
                max_token_len=200, #128,#162,#200,#256, #,
                _type="regression",
                exp="None",
                ):


        super().__init__()

        if model_name == "bert":
            
            path_of_pretrained_model = 'bert-base-cased'
        elif model_name == "bert-tiny":
            path_of_pretrained_model = "prajjwal1/bert-tiny"
        elif model_name == "finbert":
            path_of_pretrained_model = "ipuneetrathore/bert-base-cased-finetuned-finBERT"
        elif model_name == "funnel":
            path_of_pretrained_model = "funnel-transformer/small"
        elif model_name == "bart":
            path_of_pretrained_model = 'facebook/bart-base'
            
        elif model_name == "distilbert":
            path_of_pretrained_model = 'distilbert-base-cased'
        elif model_name == "multilingual":
            path_of_pretrained_model = "bert-base-multilingual-cased"
        elif model_name == "roberta-base":
            path_of_pretrained_model = "roberta-base"
        elif model_name == "bert-large":
            path_of_pretrained_model = "bert-large-cased"
        elif model_name == "gpt2":
            path_of_pretrained_model = "gpt2"


        self.initial_params["dataset_class"] = BertDataSet
        self.initial_params["collate_fn"] = None#collate_fn_LSTM
        
        n_classes = len(label_list) #if _type=="regression" else 400

        if auxiliary_loss_flag:
            aux_loss_list = ["loan_in_currency2"]
            continuous_features_list = [c for c in continuous_features_list if c not in aux_loss_list ]
        else:
            aux_loss_list=[]

        self.initial_params["dataset_params"] = {
            "embedding_category_features_list":embedding_category_features_list, 
            "continuous_features_list":continuous_features_list,
            "text_feature":text_features_list[0],
            "weight_list":weight_list,
            #"use_feature_cols":use_feature_cols,
            #"last_query_flag":last_query_flag,
            "max_token_len":max_token_len,
            "label_col":label_list + aux_loss_list, #["LOAN_AMOUNT"] + aux_loss_list,
            "path_of_pretrained_model":path_of_pretrained_model,

        }


        ppath_to_decode_dict = PROC_DIR / 'decode_dict.pkl'
        if ppath_to_decode_dict.exists():
            dec_dict = pickle_load(ppath_to_decode_dict)
        else:
            dec_dict=None



        emb_dim_pairs_list = []
        print("embedding_category_features_list")
        for col in embedding_category_features_list:
            if col == "ORIGINAL_LANGUAGE":
                dim = 5
                emb_dim = 3
            elif col == "ACTIVITY_NAME":
                dim = 161
                emb_dim = 50
            elif col == "SECTOR_NAME":
                dim = 15
                emb_dim = 8
            elif col == "COUNTRY_NAME":
                dim = 61
                emb_dim = 30
            elif col == "CURRENCY":
                dim = 51
                emb_dim = 25
            elif col == "REPAYMENT_INTERVAL":
                dim = 3
                emb_dim = 2
            # elif col == "DISTRIBUTION_MODEL":
            #     dim = 2
            #     emb_dim = 2
            elif col == "COUNTRY_CURRENCY":
                dim = 75
                emb_dim = 38
            elif col == "TOWN_NAME_COUNTRY_NAME":
                dim = 2822
                emb_dim = 50
            else:
                if (dec_dict is not None) & (col in dec_dict.keys()):
                
                    dim = len(dec_dict[col].keys())
                    emb_dim = min(dim//2, 50)
                else:
                    continue
            #dim = int(df_all[col].nunique())
            pair = (dim, emb_dim)
            emb_dim_pairs_list.append(pair)
            print("{} : {}".format(col, pair))

        

        self.model = myBert(
                            model_name=model_name,
                            path_of_pretrained_model=path_of_pretrained_model,
                            n_numerical_features=len(continuous_features_list), 
                            n_emb_features = len(embedding_category_features_list),
                            emb_dim_pairs_list=emb_dim_pairs_list,
                            n_classes=n_classes,
                            n_classes_aux=len(aux_loss_list),
                            _type = _type,
                            exp=exp,
                            )

        print(self.model)


    def predict(self, X_test, oof_flag=False):
        
        num_tta = self.edit_params["num_tta"]

        assert num_tta==1

        if num_tta == 1:

            np_pred = super().predict(X_test, oof_flag)
        
            #TODO: get final attention

        return np_pred

    

def prepareModelDir(params, prefix):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / params["model_dir_name"]
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()

        params["path_to_model"] = ppath_to_save_dir / f"{prefix}_train_model.ckpt"

        return params


def checkExistsAndAddVnum(ppath_to_model):

    v_num = 0
    cur_path = ppath_to_model
    stem = cur_path.stem

    while cur_path.exists():
        v_num+=1
        cur_path = cur_path.parent/f"{stem}-v{v_num}.ckpt"

    return cur_path



class LGBWrapper_Base(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = None
        
        self.initial_params = {
            
            

                'n_jobs': -1,
                #"device":"gpu",

                'boosting_type': 'gbdt',
                "random_seed_name__":"random_state",
                "deal_numpy":False,
                "first_metric_only": True,
                
                'max_depth': 6,
                #'max_bin': 300,
                #'bagging_fraction': 0.9,
                #'bagging_freq': 1, 
                'colsample_bytree': 0.9,
                #'colsample_bylevel': 0.3,
                #'min_data_per_leaf': 2,
                "min_child_samples":12, #8,
                
                'num_leaves': 120,#240,#2048,#1024,#31,#240,#120,#32, #3000, #700, #500, #400, #300, #120, #80,#300,
                'lambda_l1': 0.9,#0.5,
                'lambda_l2': 0.9,#0.5,

                
                }
        
    def getWeight(self, X_train, params):
        
        if isinstance(X_train, pd.DataFrame):
            
            weight_list = params["weight_list"]
            
            if len(weight_list) == 0:
                np_weight=None
            elif len(weight_list)==1:
                np_weight = X_train[weight_list[0]].values
                X_train = X_train.drop(columns=weight_list)
            else:
                raise Exception("set only one weight col to weight_list")
            
        else:
            raise Exception("not implemented error : weight col for numpy X_train")


        return X_train, np_weight

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        self.edit_params=params
        metric_name = list(params["eval_metric_func_dict__"].keys())[0]
        print(f"metric_name : {metric_name}")
        eval_metric = params["eval_metric_func_dict__"][metric_name]  
        params["metric"] = "None"
        print(f"----------eval_metric:{callable(eval_metric)}")
        
        
        X_train, np_weight = self.getWeight(X_train, params)

        eval_set = [(X_train.values, y_train.values)] if isinstance(X_train, pd.DataFrame) else [(X_train, y_train)]
        eval_names = ['train']
  
        self.model = self.model.set_params(**extractModelParametersWithStr(params, exclude_str="__"))

        if X_valid is not None:
            if isinstance(X_valid, pd.DataFrame):
                eval_set.append((X_valid.values, y_valid.values))  
            else:
                eval_set.append((X_valid, y_valid))  

            eval_names.append('valid')


        if X_holdout is not None:
            if isinstance(X_holdout, pd.DataFrame):
                eval_set.append((X_holdout.values, y_holdout.values))
            else:
                eval_set.append((X_holdout, y_holdout))  
            eval_names.append('holdout')



        categorical_columns = 'auto'


        call_back_list = []
        if (not ON_KAGGLE) and (params["no_wandb"]==False):
            wandb.init(project=PROJECT_NAME, group=params["wb_group_name"], reinit=True, name=params["wb_run_name"] )
            wandb.config.update(params,  allow_val_change=True)
            call_back_list.append(wandb_callback())


        
        self.model.fit(X=X_train, y=y_train, sample_weight=np_weight,
                    eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,
                    verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                    categorical_feature=categorical_columns,
                    callbacks=call_back_list,
                    )
        print(self.model)
        self.best_score_ = self.model.best_score_
        print(self.best_score_)
        self.feature_importances_ = self.model.feature_importances_
        self.best_iteration_ = self.model.best_iteration_

    def predict(self, X_test, oof_flag=True):
        
        X_test, np_weight = self.getWeight(X_test, self.edit_params)
        
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_).reshape(-1, 1)
    
    def procModelSaving(self, model_dir_name, prefix, bs):

        ppath_to_save_dir = PATH_TO_MODEL_DIR / model_dir_name
        if not ppath_to_save_dir.exists():
            ppath_to_save_dir.mkdir()
            
        ppath_to_model = ppath_to_save_dir / f"model__{prefix}__{model_dir_name}__{self.__class__.__name__}.pkl"
        pickle.dump(self.model, open(ppath_to_model, 'wb'))
        print(f'Trained LGB model was saved! : {ppath_to_model}')
        
        with open(str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json"), 'w') as fp:
            json.dump(bs, fp)
        
    def procLoadModel(self, model_dir_name, prefix, params):
        
        ppath_to_save_dir = PATH_TO_UPLOAD_MODEL_PARENT_DIR / model_dir_name
        print(f"ppath_to_save_dir : {ppath_to_save_dir}")
        print(list(ppath_to_save_dir.glob(f'model__{prefix}*')))
        #print(list(ppath_to_save_dir.iterdir()))
        
        name_list = list(ppath_to_save_dir.glob(f'model__{prefix}*'))
        if len(name_list)==0:
            print(f'[ERROR] Trained LGB model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]
        # if not os.path.exists(ppath_to_model):
        #     print(f'[ERROR] Trained LGB model was NOT EXITS! : {ppath_to_model}')
        #     return -1

        self.model = pickle.load(open(ppath_to_model, 'rb'))
        print(f'Trained LGB model was loaded! : {ppath_to_model}')
        
        a = int(re.findall('iter_(\d+)__', str(ppath_to_model))[0])
        
        
        self.model._best_iteration= a
        #print(self.model.best_iteration_ )
        #self.model.best_iteration_ 
        self.best_iteration_ = self.model.best_iteration_
        
        path_to_json = str(ppath_to_model).replace("model__", "bs__").replace("pkl", "json")
        if not os.path.exists(path_to_json):
            print(f'[ERROR] Trained LGB json was NOT EXITS! : {path_to_json}')
            return -1
        with open(path_to_json) as fp:
            self.best_score_ = json.load(fp)
            #self.model._best_score = json.load(fp)

        return 0

    
class LGBWrapper_regr(LGBWrapper_Base):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        self.model = lgb.LGBMRegressor()
        print(f"lgb version : {lgb.__version__}")
        
        self.initial_params['objective'] = 'regression' #torch_rmse #'regression'
        self.initial_params['metric'] = 'mae'
        
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        
        if isinstance(y_train, pd.DataFrame):
            assert y_train.shape[1] == 1
            y_train = y_train.iloc[:,0]
        
        
        
        if y_valid is not None:
            if isinstance(y_valid, pd.DataFrame):
                assert y_valid.shape[1] == 1
                y_valid = y_valid.iloc[:,0]
        
                
        if y_holdout is not None:
            if isinstance(y_holdout, pd.DataFrame):
                assert y_holdout.shape[1] == 1
                y_holdout = y_holdout.iloc[:,0]
                
        #pdb.set_trace()
        super().fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_holdout=X_holdout, y_holdout=y_holdout, params=params)



class LGBWrapper_cls(LGBWrapper_Base):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        super().__init__()
        
        self.model = lgb.LGBMClassifier()

        self.initial_params['num_leaves'] = 3
        self.initial_params['max_depth'] = 8
        self.initial_params['min_data_in_leaf'] = 3
        


    def proc_predict(self, X_test, oof_flag=False):

        if oof_flag:

            pred = self.model.predict(X_test, num_iteration=self.model.best_iteration_)
        else:
            pred = self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)
            

        return pred



    #def predict_proba(self, X_test):
    def predict(self, X_test, oof_flag=False):
        if (self.model.objective == 'binary') :
            #print("X_test b:", X_test)
            #print("X_test:shape b", X_test.shape)
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]
        else:
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)
            #pred = self.proc_predict(X_test, oof_flag)

            #return self.model.predict(X_test, num_iteration=self.model.best_iteration_)
    
    def predict_proba(self, X_test):
        #print("X_test:", X_test)
        #print("X_test:shape", X_test.shape)
        return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)
 