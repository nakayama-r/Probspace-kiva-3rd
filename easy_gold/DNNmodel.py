
from utils import *
from image_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


from DataSet import *
from Loss import *

import transformers
from transformers import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(torch.__version__)

def set_seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



set_seed_torch(SEED_NUMBER)



class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, monitor_metric, min_max_flag):
        super().__init__()

        self.monitor_metric = monitor_metric
        self.min_max_flag = min_max_flag
        self.last_score = -10000000000 if self.min_max_flag == "max" else 10000000000
        self.best_score_dict = {"train":{}, "valid":{}}

    def checkBest(self, score_dict):

    
        if self.monitor_metric in score_dict.keys():
            score = score_dict[self.monitor_metric].item()
            if ((self.min_max_flag=="max") and (self.last_score < score)) or ((self.min_max_flag=="min") and (self.last_score > score)):
                self.last_score = score

                self.setBestScore(score_dict)
                print(f"renew best!! : {self.best_epoch}")
        else:
            print(f"[Warning] There is no validation score, so best val score is not logged!!")

    def setBestScore(self, score_dict):

        for k, v in score_dict.items():
            if "step" in k:
                continue

            if k == "current_epoch":
                self.best_epoch = int(v)

            if "val" in k:
                self.best_score_dict["valid"][k.replace("val_", "")] = v.item()
            else:
                self.best_score_dict["train"][k.replace("train_", "")] = v.item()

        
        


    def getScoreInfo(self):

        return self.best_score_dict, self.best_epoch #, self.best_score_dict




    def on_train_epoch_end(self, trainer, pl_module):

        #print(f"metrics on_train_epoch_end : {trainer.callback_metrics}")

        self.checkBest(trainer.callback_metrics)


        

class BertSequenceVectorizer:
    def __init__(self, model_name="bert-base-uncased", max_len=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = max_len

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor, output_attentions=True)
        pdb.set_trace()
        seq_out, pooled_out, atten_tuple = bert_out['last_hidden_state'], bert_out['pooler_output'], bert_out["attentions"]
        #atten_tuple : [1, num_head, num_token, num_token] x num_layer

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()

def calcBatchMeanEvalScoreDictFromEvalScoreDictList(eval_score_dict_list, not_proc_cols=[]):


    batch_mean_eval_score_dict={}
    for eval_score_dict in eval_score_dict_list:

        for name, score in eval_score_dict.items():
            if name in not_proc_cols:
                continue

            if name in batch_mean_eval_score_dict.keys():
                batch_mean_eval_score_dict[name].append(eval_score_dict[name])
            else:
                batch_mean_eval_score_dict[name] = [eval_score_dict[name]]
    
    for name in batch_mean_eval_score_dict.keys():
        if name in not_proc_cols:
            continue
        batch_mean_eval_score_dict[name] = torch.tensor(batch_mean_eval_score_dict[name]).mean().item()
    
    return batch_mean_eval_score_dict

class SaveOutput:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out): 
        #print(f"module_out : {module_out}")
        #pdb.set_trace()
        #self.outputs.append(module_out.detach()) 
        self.outputs = module_out.detach()
    def clear(self): 
        self.outputs = []

class PytorchLightningModelBase(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.score_dict = {}
        self.best_score_dict = {}
        self.best_epoch = 1


    def init_weights(self):
        initrange = 0.5

        if self.n_emb_features > 0:
            for i, l in enumerate(self.emb_layers):
                l.weight.data.uniform_(-initrange, initrange)

       

        #cited from https://gist.github.com/thomwolf/eea8989cab5ac49919df95f6f1309d80
        
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        #nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)
            
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    


    

    def criterion(self, y_true, y_pred, weight=None):

        #print(f"y_true : {y_true.shape}")
        #print(f"y_pred : {y_pred.shape}")
        #print(f"weight : {weight.shape}")
        #pdb.set_trace()

        if weight is None:
            return nn.MSELoss()(y_pred[:, :2].float(), y_true[:, :2].float())
        else:
            return weighted_mse_loss(input=y_pred[:, :2].float(), target=y_true[:, :2].float(), weight=weight)


    def loadBackbone(self, ppath_to_backbone_dir, fold_num):
        
       
        prefix = f"fold_{fold_num}__iter_"
        name_list = list(ppath_to_backbone_dir.glob(f'model__{prefix}*.pkl'))
        if len(name_list)==0:
            print(f'[ERROR] Pretrained nn model was NOT EXITS! : {prefix}')
            return -1
        ppath_to_model = name_list[0]

        prefix=f"fold_{fold_num}"
        ppath_to_ckpt_model = searchCheckptFile(ppath_to_backbone_dir, ppath_to_model, prefix)
        tmp_dict = torch.load(str(ppath_to_ckpt_model))["state_dict"]
        print(tmp_dict.keys())

       
        self.load_state_dict(tmp_dict, strict=True)
        print(f"load backbone : {ppath_to_ckpt_model}")

    def setParams(self, _params):

        self.learning_rate = _params["learning_rate"]
        self.eval_metric_func_dict = _params["eval_metric_func_dict__"]
        self.monitor=_params["eval_metric"]
        self.mode=_params['eval_max_or_min']

        self.last_score = -10000000000 if self.mode == "max" else 10000000000

        print("show eval metrics : ")
        print(self.eval_metric_func_dict)

        if _params["pretrain_model_dir_name"] is not None:
            ppath_to_backbone_dir = PATH_TO_MODEL_DIR/_params["pretrain_model_dir_name"]
            self.loadBackbone(ppath_to_backbone_dir, fold_num=_params["fold_n"])


    def _proc_forward(self, batch, test_flag=False):

        out = self.forward(batch)

        if test_flag:
            return out
        else:

            loss = self.criterion(y_pred=out, y_true=batch[-1], weight=batch[0][-1])
            batch_eval_score_dict=calcEvalScoreDict(y_true=batch[-1], y_pred=out, eval_metric_func_dict=self.eval_metric_func_dict)

            ret_dict = {"loss":loss,}
            
            return ret_dict, batch_eval_score_dict


    def training_step(self, batch, batch_idx):


        ret_dict, train_batch_eval_score_dict = self._proc_forward(batch)
        
        
        for k, v in ret_dict.items():
            self.log(f'{k}_step', v, logger=False)
        ret = ret_dict

        for k, v in train_batch_eval_score_dict.items():
            self.log(f"train_{k}_step", v, logger=False)
            ret[f"{k}"] = v

        return ret



    def validation_step(self, batch, batch_idx):


        ret_dict, val_batch_eval_score_dict =  self._proc_forward(batch)

        for k, v in ret_dict.items():
            self.log(f'val_{k}_step', v, logger=False)
        ret = ret_dict


        # self.log('val_loss_step', loss, logger=False)
        # ret = {'loss': loss}

        for k, v in val_batch_eval_score_dict.items():
            self.log(f"val_{k}_step", v, logger=False)
            ret[f"{k}"] = v
        

        #self.log('val_loss', loss)


        return ret

    def training_epoch_end(self, outputs):
        
        
        score_dict = calcBatchMeanEvalScoreDictFromEvalScoreDictList(outputs, not_proc_cols=["loss"])
        score_dict["loss"] = torch.stack([o['loss'] for o in outputs]).mean().item()

        
        self.log("current_epoch", float(self.current_epoch), logger=False)
        print_text = f"Epoch {self.current_epoch}, "
        for name, score in score_dict.items():
            self.log(f"train_{name}", score)
            print_text += f"{name} {score}, "
        print(print_text)

        #
        #self.score_dict["train"] = score_dict
        #self.checkBest()
        #pdb.set_trace()
        #

        # if self._on_training_epoch_end is not None:
        #     for f in self._on_training_epoch_end:
        #         f(self.current_epoch, loss)

    def validation_epoch_end(self, outputs):

        score_dict = calcBatchMeanEvalScoreDictFromEvalScoreDictList(outputs, not_proc_cols=["loss"])
        score_dict["loss"] = torch.stack([o['loss'] for o in outputs]).mean().item()

        #pdb.set_trace()

        print_text = f"Epoch {self.current_epoch}, "
        for name, score in score_dict.items():
            self.log(f"val_{name}", score)
            print_text += f"val_{name} {score}, "
        print(print_text)

        #self.score_dict["valid"] = score_dict
        
        #loss = torch.stack([o['val_loss'] for o in outputs]).mean()

        # if self._on_validation_epoch_end is not None:
        #     for f in self._on_validation_epoch_end:
        #         f(self.current_epoch, loss)

    

    def test_step(self, batch, batch_idx):

        out = self._proc_forward(batch, test_flag=True)
        
        np_out = out.data.cpu().detach().numpy()
        del out
        torch.cuda.empty_cache()
        #out = self.forward(batch)
        #pdb.set_trace()
        return {'out': np_out}

    def test_epoch_end(self, outputs):
        #pdb.set_trace()  
        self.final_preds = np.concatenate([o['out'] for o in outputs])#torch.cat([o['out'] for o in outputs])


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=self.learning_rate*0.1, last_epoch=-1)

        return [optimizer], [scheduler]

class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
                         
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        #[batch, length, feature_dim]
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out

class myBert(PytorchLightningModelBase):

    def __init__(self, 
                model_name, 
                path_of_pretrained_model,
                n_numerical_features, 
                n_emb_features,
                emb_dim_pairs_list,
                d_model=256,
                n_classes=1,
                n_classes_aux=0,
                _type="regression",
                exp="None",
                ) -> None:
        super().__init__()

        self.model_name = model_name
        self.path_of_pretrained_model = path_of_pretrained_model

        self.n_numerical_features = n_numerical_features
        self.n_emb_features = n_emb_features
        self.d_model = d_model

        self.n_classes = n_classes
        self.n_classes_aux = n_classes_aux

        self.type_ = _type
        self.exp = exp
        
        table_tmp_num = 0
        if self.n_emb_features>0:
            table_tmp_num+=1

            self.emb_layers = nn.ModuleList([nn.Embedding(m, d) for m, d in emb_dim_pairs_list])
            self.total_cat_emb_dim = sum([d for m, d in emb_dim_pairs_list])
            #self.cat_drop = nn.Dropout(0.3)
            self.categorical_proj = nn.Sequential(
                nn.Linear(self.total_cat_emb_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(), 
            )

        if self.n_numerical_features >0:
            table_tmp_num+=1
            self.continuout_emb_dim = 2
            self.conti_embed_layers = nn.ModuleList([nn.Linear(1,self.continuout_emb_dim ,bias=False) for n in range(n_numerical_features)])
            #self.conti_drop = nn.Dropout(0.3)
            self.continuous_proj = nn.Sequential(
                nn.Linear(self.continuout_emb_dim  *n_numerical_features, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(), 
            )

        self.table_src_dim = self.d_model * table_tmp_num


        if exp != "table":
            self.bert = AutoModel.from_pretrained(self.path_of_pretrained_model, output_attentions=True, output_hidden_states=True)
        
        
            
        
        if exp == "f_conv1d":

            self.cnn1 = nn.Conv1d(self.bert.config.hidden_size, 256, kernel_size=2, padding=1)
            self.cnn2 = nn.Conv1d(256, 128, kernel_size=2, padding=1)

            regressor_input_num = 128 + self.table_src_dim
        elif exp == "f_lstm":
            hiddendim_lstm = 256
            self.lstm_pooler = LSTMPooling(self.bert.config.num_hidden_layers, self.bert.config.hidden_size, hiddendim_lstm)
            regressor_input_num = hiddendim_lstm + self.table_src_dim
        elif exp == "table":
            regressor_input_num = self.table_src_dim
        else:
            #regressor_input_num=self.bert.config.hidden_size + self.table_src_dim
            self.num_bert_concat = 12#6
            if self.model_name == "distilbert":
                self.num_bert_concat = 6
            elif self.model_name == "bart":
                self.num_bert_concat = 1


            regressor_input_num=self.bert.config.hidden_size * self.num_bert_concat+ self.table_src_dim
        

        self.regressor =  nn.Sequential(
                nn.Linear(regressor_input_num, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, n_classes)
                )#

        #pdb.set_trace()
        if self.n_classes_aux > 0:
            self.regressor_aux =  nn.Sequential(
                nn.Linear(self.bert.config.hidden_size + self.table_src_dim, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, n_classes_aux)
                )#


        if self.model_name != "bart":
            self.init_weights() #this contains bug????

        num_reinit_layers=3
       

        if exp!="table":
            
            for param in self.bert.parameters():
                param.requires_grad = False

            if self.model_name == "bert":
                for param in self.bert.encoder.layer[-num_reinit_layers:].parameters():
                    param.requires_grad = True
            elif self.model_name == "distilbert":
                for param in self.bert.transformer.layer[-1:].parameters():
                    param.requires_grad = True

            elif self.model_name == "funnel":
                #pdb.set_trace()
                for param in self.bert.decoder.layers[-1:].parameters():
                    param.requires_grad = True
            elif self.model_name == "bert-tiny":
                for param in self.bert.encoder.layer[-1:].parameters():
                    param.requires_grad = True
            elif self.model_name == "finbert":
                
                for param in self.bert.encoder.layer[-1:].parameters():
                    param.requires_grad = True
                
            elif self.model_name == "bart":
                #pdb.set_trace()
                for param in self.bert.encoder.layers[-1:].parameters():
                    param.requires_grad = True

                #pdb.set_trace()
                # for param in self.bert.decoder.layers[-1:].parameters():
                #     param.requires_grad = True
            elif self.model_name == "multilingual":
                for param in self.bert.encoder.layer[-4:].parameters():
                    param.requires_grad = True
            elif self.model_name == "roberta-base":
                for param in self.bert.encoder.layer[-1:].parameters():
                    param.requires_grad = True
            elif self.model_name == "bert-large":
                for param in self.bert.encoder.layer[-1:].parameters():
                    param.requires_grad = True
            elif self.model_name == "gpt2":
                #pdb.set_trace()
                for param in self.bert.h[-1:].parameters():
                    param.requires_grad = True
            
        

            

    def reinit_bert(self, _bert_model, num_reinit_layers):
        for layer in _bert_model.encoder.layer[-num_reinit_layers:]:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=_bert_model.config.initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.normal_(mean=0.0, std=_bert_model.config.initializer_range)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
        return _bert_model
        

    def _forward(self, batch):

        input_ids = batch[0][0]
        attention_mask = batch[0][1]

        if self.exp != "table":
            output = self.bert(input_ids, attention_mask=attention_mask)

            #torch.Size([128, 12, 128, 128]) #[batch_size, num_head, token_len, token_len]

            if self.model_name == "bart":
                attention = None#output.decoder_attentions[-1].sum(1)[:, 0, :] #[batch_size, token_len] num_head方向にsumをとった
            else:
                attention = output.attentions[-1].sum(1)[:, 0, :]

        else:
            attention=None
                #pdb.set_trace()


        ###############
        #  table data
        ###############
        src = batch[0][2]
        table_concat_list = []
        if self.n_emb_features>0:
            emb_list = [emb_layer(src[...,i].long()) for i, emb_layer in enumerate(self.emb_layers)]
            emb_concat_org = torch.cat(emb_list, axis=-1)
            emb_concat = self.categorical_proj(emb_concat_org)
            table_concat_list.append(emb_concat)

        if self.n_numerical_features >0:
            conti_emb_list = [emb_layer(src[...,i+self.n_emb_features].unsqueeze(-1).float()) for i, emb_layer in enumerate(self.conti_embed_layers)]
            conti_emb_concat_org = torch.cat(conti_emb_list, axis=-1)
            conti_emb_concat = self.continuous_proj(conti_emb_concat_org)
            table_concat_list.append(conti_emb_concat)

        if len(table_concat_list)==1:
            table_src=table_concat_list[0]
        elif len(table_concat_list)>1:
            table_src = torch.cat(table_concat_list, axis=-1)
        else:
            table_src = None
        
        

        if self.exp == "f_conv1d":
            last_hidden_state = output['last_hidden_state'].permute(0, 2, 1)
            cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
            cnn_embeddings = self.cnn2(cnn_embeddings)
            sequence_output, _ = torch.max(cnn_embeddings, 2)
            table_nlp_src = torch.cat([sequence_output,  table_src], axis=-1)
            #pdb.set_trace()
        elif self.exp == "f_lstm":
            all_hidden_states = torch.stack(output[2])
            lstm_pooling_embeddings  = self.lstm_pooler(all_hidden_states)
            table_nlp_src = torch.cat([lstm_pooling_embeddings,  table_src], axis=-1)
        elif self.exp == "table":
            table_nlp_src = table_src
        else:
            #table_nlp_src = torch.cat([output.last_hidden_state[:, 0,:],  table_src], axis=-1)
            #if self.exp == "f_concat":
            if self.model_name == "bart":
                #decoder_sequence_output = torch.cat([output["decoder_hidden_states"][-1*i][:,0] for i in range(1, 6+1)], dim=1)
                #encoder_sequence_output = torch.cat([output["encoder_hidden_states"][-1*i][:,0] for i in range(1, 6+1)], dim=1)

                if table_src is not None:
                    #table_nlp_src = torch.cat([decoder_sequence_output,  encoder_sequence_output, table_src], axis=-1)
                    table_nlp_src = torch.cat([output.last_hidden_state[:, 0,:],  table_src], axis=-1)
                else:
                    table_nlp_src = output.last_hidden_state[:, 0,:] #torch.cat([decoder_sequence_output,  encoder_sequence_output], axis=-1) 
                
                 

            else:
                sequence_output = torch.cat([output["hidden_states"][-1*i][:,0] for i in range(1, self.num_bert_concat+1)], dim=1)
                table_nlp_src = torch.cat([sequence_output,  table_src], axis=-1)

        #
        preds = self.regressor(table_nlp_src)

        if self.n_classes_aux > 0:

            preds_aux = self.regressor_aux(table_nlp_src)
            preds = (preds, preds_aux)

        return preds, attention


    def forward(self, batch):
        
        out, attention = self._forward(batch)

        self.step_attention = attention

        return out

    def criterion_classification(self, y_true, y_pred, weight=None):


        f_loss = FocalLossWithOneHot(gamma=1)
        loss = f_loss(input=y_pred, target_one_hot=y_true)



        return loss

    def criterion_expectation(self, y_true, y_pred):

        rank = torch.Tensor([i for i in range(y_true.shape[1])]).cuda()
        y_pred_expectation = (torch.sum(y_pred*rank, dim=1) + 1)*25

        y_true_idx = (y_true.argmax(axis=1)+1)*25

        loss_expectation = nn.L1Loss()(y_pred_expectation,  y_true_idx)

        #pdb.set_trace()

        return loss_expectation



    def criterion(self, y_true, y_pred, weight=None):


        if self.model_name != "bart":
            loss = PseudHuberLoss_torch(y_true, y_pred)

            loss = loss.mean()
        else:
            
            loss = nn.L1Loss()(y_pred,  y_true) 


        
        return loss


    def criterion_aux(self, y_true, y_pred, weight=None):

        loss = nn.L1Loss()(y_pred,  y_true) 


        return loss
    
    
    def _proc_forward(self, batch, test_flag=False):

        

        out = self.forward(batch)

        if self.n_classes_aux>0:
            out_loan = out[0]
            out_aux = out[1]

            y_true_loan = batch[-1][:,0].view(-1, 1)
            y_true_aux = batch[-1][:,1:]
            

        else:
            out_loan = out
            y_true_loan = batch[-1]

        
            

        
        if test_flag:
            


            return out_loan

        else:

            if self.type_ == "classification":
                loss = self.criterion_classification(y_pred=out_loan, y_true=y_true_loan, weight=batch[0][-1])
                ret_dict = {"loss":loss}

                if self.exp=="DLDL":

                    loss_expectation = self.criterion_expectation(y_true=y_true_loan, y_pred=out_loan)
                    ret_dict["loss"] = loss_expectation+loss
                    ret_dict["loss_expectation"] = loss_expectation.item()
                    ret_dict["loss_LDL"] = loss.item()

                
                #pdb.set_trace()
                out_loan_idx = (out_loan.argmax(axis=1) + 1)*25
                y_true_loan_idx = (y_true_loan.argmax(axis=1) + 1)*25
                #pdb.set_trace()
                batch_eval_score_dict=calcEvalScoreDict(y_true=y_true_loan_idx.float(), y_pred=out_loan_idx.float(), eval_metric_func_dict=self.eval_metric_func_dict)
                #pdb.set_trace()

            else:

            
                if self.n_classes == 1:

                    loss = self.criterion(y_pred=out_loan, y_true=y_true_loan, weight=batch[0][-1])
                    ret_dict = {"loss":loss}
                else:

                    ret_dict = {}
                    for i in range(self.n_classes):
                        loss = self.criterion(y_pred=out_loan[:,i], y_true=y_true_loan[:,i], weight=batch[0][-1])
                        if "loss" in ret_dict.keys():
                            ret_dict["loss"] += loss
                        else:
                            ret_dict["loss"] = loss
                        
                        ret_dict[f"loss_{i}"] = loss.item()

                    

                    idx = y_true_loan.argmax(1, keepdim=True)
                    out_loan = out_loan.gather(1, idx)

                    y_true_loan =  y_true_loan.max(axis=1)[0].reshape(-1, 1)





                if self.n_classes_aux>0:
                    loss_aux = self.criterion_aux(y_pred=out_aux, y_true=y_true_aux, weight=batch[0][-1])
                    loss_loan = loss
                    loss = loss_loan + loss_aux*0.001

                    ret_dict["loss"] = loss
                    ret_dict["loss_loan"] = loss_loan.detach()
                    ret_dict["loss_aux"] = loss_aux.detach()

                batch_eval_score_dict=calcEvalScoreDict(y_true=y_true_loan, y_pred=out_loan, eval_metric_func_dict=self.eval_metric_func_dict)
                
                
            
            return ret_dict, batch_eval_score_dict




    
    def test_step(self, batch, batch_idx):

        out_dict = super().test_step(batch, batch_idx)

        #out_dict["attention"] = self.step_attention

        return out_dict

    def test_epoch_end(self, outputs):
        
        super().test_epoch_end(outputs)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.model_name == "bart":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience =30, min_lr=1e-6)

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_mae'
            }
        else: 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=self.learning_rate*0.1, last_epoch=-1)

            return [optimizer], [scheduler]

