

from utils import *
import torch
import torch.utils.data

from image_utils import *
from transformers import AutoTokenizer




class BertDataSet(torch.utils.data.Dataset):

    def __init__(self, df_train_X, df_train_y, dataset_params, train_flag):

  
        
        self.max_token_len = dataset_params["max_token_len"]
        self.tokenizer = AutoTokenizer.from_pretrained(dataset_params["path_of_pretrained_model"])
        self.text_feature = dataset_params["text_feature"]
        self.embedding_category_features_list = dataset_params["embedding_category_features_list"]
        self.continuous_features_list = dataset_params["continuous_features_list"]
        self.weight_list = dataset_params["weight_list"]
        #self.use_feature_cols = dataset_params["use_feature_cols"]
        self.label_col = dataset_params["label_col"]
        self.row_index_name = df_train_X.index.name

        print(df_train_X.columns)
        print(df_train_y.columns)
        print(self.text_feature)
        


        for col in self.label_col:
            if train_flag:
                if col in df_train_y.columns:
                    df_train_X[col]=df_train_y[col]
            else:
                df_train_X[col]=0


        self.encoding = self.tokenizer(
            df_train_X[self.text_feature].tolist(),
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        table_cols = self.embedding_category_features_list + self.continuous_features_list
        self.np_table_X = df_train_X[table_cols].values
        
        self.np_y = df_train_X[self.label_col].values
        self.np_w = df_train_X[self.weight_list].values if self.weight_list is not None else None
        


    def __len__(self):
        return self.np_y.shape[0]

    def __getitem__(self, idx):


        return [self.encoding["input_ids"][idx], self.encoding["attention_mask"][idx], self.np_table_X[idx]], self.np_y[idx]

