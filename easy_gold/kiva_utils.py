# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 23:20:16 2022

@author: r00526841
"""

from utils import *
from eda import *
from nlp_utils import *
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import clip
from PIL import Image


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


 
def proc2_addLoanInCurrency2(df, emb_sentence):
    
    df["row_emb_similarity"] = df["row_emb"].map(lambda x: getMaxSimilarity(x, emb_sentence)[1] if isinstance(x, np.ndarray) else x)
    
    #pdb.set_trace()
    
    df["max_similarity_row_num"] = df["row_emb_similarity"].map(lambda x: np.argmax(x) if isinstance(x, np.ndarray) else x)
    df["max_similarity_row"] = [r[int(n)] if pd.notna(n) else n for r, n in zip(df["row_list"], df["max_similarity_row_num"])]
    df["max_similarity_value"] = [r[int(n)] if pd.notna(n) else n for r, n in zip(df["row_emb_similarity"], df["max_similarity_row_num"])]
    
    
    df["loan_in_row"] = df["max_similarity_row"].map(lambda x: max([float(n) for n in re.findall(r'\d+',re.sub(r"(\d),(\d)", r"\1\2", x))]) if pd.notna(x) else x)
    
    return df

def proc_addLoanInCurrency2(df, emb_sentence, embedder):
    
    
    df["row_list"] = df["DESCRIPTION_TRANSLATED"].map(lambda x: [r for r in x.replace(".” ", ". ").split(". ") if any(map(str.isdigit, r))])
    
    
    
    
    df["row_emb"]= df["row_list"].map(lambda x: np.stack([embedder(x_str).numpy().reshape(-1) for x_str in x]) if len(x) > 0 else np.nan)
    
    
    df = proc2_addLoanInCurrency2(df, emb_sentence)
    
    
    return df
    

def procFindSimilar(df_train, df_test, loan_sentence_dict, cur):
    example_str = loan_sentence_dict[cur]
    se_text = pd.Series([example_str], index=["test_col"])

    emb_sentence = getUniversalSentenceEncoder(se_text, output_feature_dim=512)
    
    
    df_tmp_train = df_train.loc[(df_train["CURRENCY"]==cur)]
    df_tmp_test = df_test.loc[(df_test["CURRENCY"]==cur)]
    
    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    df_tmp_train = proc_addLoanInCurrency2(df_tmp_train, emb_sentence, embedder)
    df_tmp_test = proc_addLoanInCurrency2(df_tmp_test, emb_sentence, embedder)
    
    
    
   
    
    
    return df_tmp_train, df_tmp_test

loan_sentence_dict = {
    "PHP": "requested a PHP 25,000 loan through NWTF to purchase",
    'COP': "requested a 25,000 loan through NWTF to purchase",
    "HNL": "requesting a loan of 15,000 Lempiras in order to",
    "KES": "seeking a loan of 40,000 KES to buy",
    "USD": "would like a loan of $1,000 USD through KIF to",
    "RWF": "requesting a loan of 200,000 RWF for",
    "PYG": "requested a 25,000 loan through NWTF to purchase",
    "PEN": "requested a 25,000 soles loan to purchase",
    "VND": "asked for a loan of 30,000,000 VND",
    "MGA": "requested a 25,000 loan to purchase",
    "INR": "applied for a loan of INR 20,000 from Prayas, one of Milaap's field partners",
    "MZN": "requested a 25,000 loan to purchase",
    "PKR": "requested an amount of PKR 70,000 for",
    "JOD": "requesting a loan of 500 JOD to",
    "XOF": "requested a 25,000 loan to purchase",
    "LRD": "requesting a Kiva loan of 45,000 LRD from BRAC Liberia to buy",
    "UGX": "requested a loan of 800,000 UGX from",
    "GTQ": "requested a 25,000 loan to purchase",
    "MXN": "applying for a 10,000 MXN loan",
    "KHR": "applied for a loan from Kiva of 6,000,000 KHR to",
    "TJS": "requesting a loan of 4,000 TJS to",
    "HTG": "wants a loan of 15,000 HTG to",
    "EGP": "requesting a loan in the amount of 30,000 EGP to",
    "IDR": "has asked for a loan of 8,700,000 Indonesian rupiahs (IDR) from KBMI",
    "NIO": "needs a loan of 35,000 Nicaraguan cordobas (NIO) to",
    "GHS": "requested a 25,000 loan to purchase",
    "FJD": "seeks a loan of FJD 1,500 to",
    "BOB":  "requested a 25,000 loan to purchase",
    "SBD": "applied for a 5,800 SBD loan to",
    "WST": "applied for a 1,250 WST loan to",
    "BRL": "requested a 25,000 loan to purchase",
    "NGN": "requested a 25,000 loan to purchase",
    "XAF": "requested a 25,000 loan to purchase",
    "EUR": "requesting a 1,200 EUR loan to",
    "KGS": "requesting a loan of 170,000 KGS for",
    "MWK": "requested a 25,000 loan to purchase",
    "GEL": "is asking for a loan from Kiva and Kiva's field partner, Credo Bank, in the amount of 750 GEL for tuition",
    "ALL": "The loan I am applying for now, 100,000 Albanian Lek (ALL)",
    "ZMW": "she has applied for a 14,000 ZMW loan to increase her float",
    "CRC": "requested a 25,000 loan to purchase",
    "TOP": "Kolotita has applied for a 1,500 TOP loan",
    "MDL": "requested a 25,000 loan to purchase",
    "LSL": "requested a 25,000 loan to purchase",
    "DOP": "requested a 25,000 loan to purchase", 
    "SLL": "requested a 25,000 loan to purchase", 
    "TRY": "requested a 25,000 loan to purchase",  
    "NPR": "requested a 25,000 loan to purchase",
    "THB": "requested a 25,000 loan to purchase", 
    "PGK": "requested a 25,000 loan to purchase", 
    "ILS": "requested a 25,000 loan to purchase", 
    "AMD": "requested a 25,000 loan to purchase", 
    
}

th_dict = {
    "PHP": 0.2,
    "COP": 1.0,#0.4,
    "HNL":0.2,
    "KES":0.2,
    "USD":0.2,
    "RWF":0.3,
    "PYG":0.3,
    "PEN":0.3,
    "VND":0.4,
    "MGA":0.3,
    "INR":0.3,
    "MZN":0.5,
    "PKR":0.2,
    "JOD":0.2,
    "XOF":0.5,
    "LRD":0.5,
    "UGX":0.2,
    "GTQ":0.5,
    "MXN":0.4,
    "KHR":0.2,
    "TJS":0.2,
    "HTG":0.5,
    "EGP":0.4,#0.3,
    "IDR":0.3,
    "NIO":0.4,
    "GHS":0.4,
    "FJD":0.3,#0.2,
    "BOB":0.4,
    "SBD":0.4,
    "WST":0.3,
    "BRL":0.3,
    "NGN":0.3,
    "XAF":0.4,
    "EUR":0.2,
    "KGS":0.2,
    "MWK":0.4,
    "GEL":0.3,
    "ALL":0.4,
    "ZMW":0.4,
    "CRC":0.4,
    "TOP":0.5,
    "MDL":0.4,
    "LSL":0.3,
    "DOP":1.0,#0.3,
    "SLL":0.3,
    "TRY":0.3,
    "NPR":0.3,
    "THB":0.4,
    "PGK":0.3,
    "ILS":0.5, #u200e try again!
    "AMD":0.2,
}

def addLoanInCurrency2():

    # 1. DESCRIPTION_TRANSLATEDから数字が入っているsentenceだけを取り出し抽出候補とする．
	# 2. CURRENCYごとに希望金額が書かれたsentenceの一例（loan_sentence_dict参照）を決めておき，候補のsentenceとのcosine類似度をUniversal Sentence Encoderの文章ベクトルから算出する．
	# 3. 類似度がしきい値（th_dict参照）よりも大きかったsentenceに書かれた数値を希望金額（loan_in_currency2）として抽出する．
    # 4. 学習データの LOAN_AMOUNTと希望金額を比較し，USDへの変換レート（tmp_scale）を算出
    # 5. tmp_scaleの中央値をとってその通貨のUSDへの変換レート（scale2）とする．
    # 6. 希望金額（loan_in_currency2）に変換レート（scale2）をかけてUSDでの希望金額（scaled_loan_amount2）に変換．
    # 7. scaled_loan_amount2を$25刻みに変換するしたもの（scaled_loan_amount_round2）も算出する．

    
    df_train_proc = pd.read_pickle(PROC_DIR / f'df_proc_train.pkl')
    df_test_proc = pd.read_pickle(PROC_DIR / f'df_proc_test.pkl')
   
    
    dec_dict = pickle_load(PROC_DIR / 'decode_dict.pkl')
    dec_currency = dec_dict["CURRENCY"]
    
    
    
    
    df_train_proc["CURRENCY"] = df_train_proc["CURRENCY"].map(dec_currency)
    df_test_proc["CURRENCY"] = df_test_proc["CURRENCY"].map(dec_currency)
    
    #
    
    currency_list = df_train_proc["CURRENCY"].unique()

    
    for cur_str in currency_list:
    
        df_tmp_train, df_tmp_test = procFindSimilar(df_train_proc, df_test_proc, loan_sentence_dict, cur=cur_str)
        
        
        df_tmp_train["loan_in_currency2"] = [l if (pd.notna(v)) & (v > th_dict[cur_str]) else np.nan for l, v in zip(df_tmp_train["loan_in_row"], df_tmp_train["max_similarity_value"])]
        df_tmp_test["loan_in_currency2"] = [l if (pd.notna(v)) & (v > th_dict[cur_str]) else np.nan for l, v in zip(df_tmp_test["loan_in_row"], df_tmp_test["max_similarity_value"])]
        
        df_tmp_train["tmp_scale"] = df_tmp_train["LOAN_AMOUNT"]/df_tmp_train["loan_in_currency2"]
        
        
        scale = df_tmp_train["tmp_scale"].median()
        
        print(f"{cur_str}: {scale}/{len(df_tmp_train)}")
        
        df_tmp_train["scale2"] =scale
        df_tmp_test["scale2"] =scale
    
    
        df_tmp_train["scaled_loan_amount2"] = df_tmp_train["scale2"] * df_tmp_train["loan_in_currency2"]
        df_tmp_test["scaled_loan_amount2"] = df_tmp_test["scale2"] * df_tmp_test["loan_in_currency2"]
        
        
        loan_amount_unique = np.arange(25, 10025, 25)
    
        df_tmp_train['scaled_loan_amount_round2'] = df_tmp_train["scaled_loan_amount2"].map(lambda x: loan_amount_unique[np.abs(loan_amount_unique-x).argmin()] if pd.notna(x) else 0)
        df_tmp_test['scaled_loan_amount_round2'] = df_tmp_test["scaled_loan_amount2"].map(lambda x: loan_amount_unique[np.abs(loan_amount_unique-x).argmin()] if pd.notna(x) else 0)
    
        
        use_cols = ["scaled_loan_amount2", "scaled_loan_amount_round2", "scale2", ]
    
        for col in use_cols:
    
            df_tmp_train[col].fillna(0, inplace=True)
            df_tmp_test[col].fillna(0, inplace=True)
    
    
            df_train_proc.loc[df_tmp_train.index, col] = df_tmp_train[col]
            df_test_proc.loc[df_tmp_test.index, col] = df_tmp_test[col]
            
        use_cols = ["row_list", "row_emb", "row_emb_similarity", "max_similarity_row_num", 
                "max_similarity_row", "max_similarity_value", "loan_in_row","loan_in_currency2"]
    
        for col in use_cols:
    
    
            df_train_proc.loc[df_tmp_train.index, col] = df_tmp_train[col]
            df_test_proc.loc[df_tmp_test.index, col] = df_tmp_test[col]
        
    
    reverse_dec_currency = {}
    for k, v in dec_currency.items():
        reverse_dec_currency[v] = k
    
    df_train_proc["CURRENCY"].replace(reverse_dec_currency, inplace=True)
    df_test_proc["CURRENCY"].replace(reverse_dec_currency, inplace=True)
    
    
    df_train_proc.to_pickle(PROC_DIR/"df_proc_train.pkl")
    df_test_proc.to_pickle(PROC_DIR/"df_proc_test.pkl")
    

def myClipPreprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        #_convert_image_to_rgb,
        #ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def proc_addClip(df):

    #画像をOpenAI CLIP(https://github.com/openai/CLIP)のimage encoderに入力して画像ベクトル化

    model, preprocess = clip.load("ViT-B/32")
    #preprocess = myClipPreprocess(224)
    model.cuda().eval()
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(f"clip.available_models() {clip.available_models()}")
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)




    images = []
    no_image=[]
    for idx, (img_id, tr_te) in tqdm(enumerate(
        zip(
            df["IMAGE_ID"].values,
            df["tr_te"].values,
        )
    ), total=len(df)):


        ppath_to_img = INPUT_DIR/f'{tr_te}_images/{img_id}.jpg'
        if ppath_to_img.exists():
            image = Image.open(ppath_to_img).convert("RGB")
            #torch_img = torchvision.transforms.functional.to_tensor(image)
            images.append(preprocess(image))
            no_image.append(0)
        else:
            pilImg = Image.fromarray(np.uint8(np.zeros((224, 224, 3))))
            #torch_img = torchvision.transforms.functional.to_tensor(pilImg)
            images.append(preprocess(image))
            no_image.append(1)


    batch_size=64
    image_features = []

    for i in range(0, len(images), batch_size):
        batch_img = images[i:i+batch_size] # the result might be shorter than batchsize at the end
        batch_img = (torch.stack(batch_img))
        batch_img = batch_img.cuda()

        #
        with torch.no_grad():
            image_feature = model.encode_image(batch_img).float()
            #image_features.append(image_feature)

            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            #text_feature /= text_feature.norm(dim=-1, keepdim=True)

            np_imgs = image_feature.cpu().numpy()
            #print(np_imgs.shape)
            image_features.append(np_imgs)
            

    np_img_fs = np.concatenate(image_features)
    col_names = [f"clipped_feature_{i}" for i in range(np_img_fs.shape[1])]
    df_clip = pd.DataFrame(np_img_fs, index=df.index, columns=col_names)

    df["no_image"] = no_image
    df = pd.concat([df, df_clip], axis=1)

    return df, col_names


def addClip():


    df_train_proc_lgb = pd.read_pickle(PROC_DIR / f'df_proc_train.pkl')
    df_test_proc_lgb = pd.read_pickle(PROC_DIR / f'df_proc_test.pkl')

    df_train_proc_lgb["tr_te"]="train"
    df_test_proc_lgb["tr_te"]="test"

    df = pd.concat([df_train_proc_lgb, df_test_proc_lgb])

    df = df[["LOAN_USE", "IMAGE_ID", "tr_te"]]
    

    df, col_names = proc_addClip(df)
    

    df.loc[df["tr_te"]=="train", col_names].reset_index().to_csv(PROC_DIR/"clip_train2.csv", index=False)
    df.loc[df["tr_te"]=="test", col_names].reset_index().to_csv(PROC_DIR/"clip_test2.csv", index=False)

def _proc_preprocess_DESCRIPTION(df):
  
    df["DESCRIPTION"]  = df["DESCRIPTION"].str.replace("<br />", " ").values
    df["DESCRIPTION"]  = df["DESCRIPTION"].str.replace("\\\\t", " ").values
    df["DESCRIPTION"]  = df["DESCRIPTION"].str.replace("\\\\u200e", " ").values
    df["DESCRIPTION"]  = df["DESCRIPTION"].str.replace("\\u200b", " ").values

    if 1667738 in df.index:
        df.loc[1667738, "DESCRIPTION"] =df.loc[1667738, "DESCRIPTION"].replace("13,0000", "13000")


    return df

def _proc_addOtherLanguage(df_train, df_test):

    dec_dict = pickle_load(PROC_DIR / 'decode_dict.pkl')
    dec_language = dec_dict["ORIGINAL_LANGUAGE"]

    for k, v in dec_language.items():
        if v == "English":
            eng_num = k 

    print(f"num train sample: {df_train.shape[0]}, num train unique id : {df_train.index.nunique()}")
    print(f"num test sample: {df_test.shape[0]}, num test unique id : {df_test.index.nunique()}")

    df_otherLang_train = df_train.loc[df_train["ORIGINAL_LANGUAGE"]!=eng_num]
    df_otherLang_test = df_test.loc[df_test["ORIGINAL_LANGUAGE"]!=eng_num]


    print(f"num other language train sample: {df_otherLang_train.shape[0]}, num other language train unique id : {df_otherLang_train.index.nunique()}")
    print(f"num other language test sample: {df_otherLang_test.shape[0]}, num other language test unique id : {df_otherLang_test.index.nunique()}")

    df_otherLang_train["DESCRIPTION_TRANSLATED"] = df_otherLang_train["DESCRIPTION"]
    df_otherLang_test["DESCRIPTION_TRANSLATED"] = df_otherLang_test["DESCRIPTION"]


    df_train = pd.concat([df_train, df_otherLang_train]).reset_index()
    df_test = pd.concat([df_test, df_otherLang_test]).reset_index()

    print(f"num train sample: {df_train.shape[0]}, num train unique id : {df_train['LOAN_ID'].nunique()}")
    print(f"num test sample: {df_test.shape[0]}, num test unique id : {df_test['LOAN_ID'].nunique()}")

    return df_train, df_test



def addOtherLanguage():


    df_train, df_test = loadProc(mode="nn", decode_flag=False)

    df_train =  _proc_preprocess_DESCRIPTION(df_train)
    df_test = _proc_preprocess_DESCRIPTION(df_test)

    df_train, df_test = _proc_addOtherLanguage(df_train, df_test)


def labelDistributionTarget(df_train, df_test, target_col_list, setting_params):

    assert target_col_list[0] == "LOAN_AMOUNT"

    idx = (df_train["LOAN_AMOUNT"].values / 25).astype(int)-1
    num_class = 400

    label = [[normal_sampling(int(age), i) for i in range(num_class)] for age in idx]
    np_label = np.array(label)

    target_col_list = [f"target_{i}" for i in range(num_class)]
    df_target = pd.DataFrame(np_label, columns=target_col_list, index=df_train.index)
    df_train = pd.concat([df_train, df_target], axis=1)
    setting_params["num_class"] = len(target_col_list)

    return df_train, df_test, target_col_list, setting_params



def OneHotClassificationTarget(df_train, df_test, target_col_list, setting_params):

    assert target_col_list[0] == "LOAN_AMOUNT"

    idx = (df_train["LOAN_AMOUNT"].values / 25).astype(int)-1

    num_class = 400
    np_one_hot = np.identity(num_class)[idx]

    target_col_list = [f"target_{i}" for i in range(num_class)]
    df_target = pd.DataFrame(np_one_hot, columns=target_col_list, index=df_train.index)
    df_train = pd.concat([df_train, df_target], axis=1)
    setting_params["num_class"] = len(target_col_list)

    return df_train, df_test, target_col_list, setting_params

def returnClassificationTarget(df_y_pred, df_oof, df_train, df_test, target_col_list, setting_params):

    #pdb.set_trace()

    df_oof["LOAN_AMOUNT"] = (df_oof[target_col_list].values.argmax(axis=1) + 1) * 25
    df_y_pred["LOAN_AMOUNT"] = (df_y_pred[target_col_list].values.argmax(axis=1) + 1) * 25

    target_col_list=["LOAN_AMOUNT"]
    setting_params["num_class"] = 1

    return df_y_pred, df_oof,target_col_list, setting_params

def OneHotRegressionTarget(df_train, df_test, target_col_list, one_hot_col, target_col):

    train_one_hot = pd.get_dummies(df_train[one_hot_col], prefix="target")

    np_tmp = train_one_hot.values * df_train[target_col].values.reshape(-1, 1)

    df_target = pd.DataFrame(np_tmp, columns=train_one_hot.columns, index=train_one_hot.index)

    target_col_list = list(df_target.columns)
    #df_train =df_train.drop(columns=[target_col])
    #df_test =df_test.drop(columns=[target_col])


    df_train = pd.concat([df_train, df_target], axis=1)

    return df_train, df_test, target_col_list

def returnTarget(df_y_pred, df_oof, df_train, df_test, target_col):

    np_oof = df_oof.values
    df_oof["CURRENCY"] = df_train["CURRENCY"]

    

    train_idx = df_oof["CURRENCY"].values

    df_oof[target_col] =  np.take_along_axis(np_oof, train_idx.reshape(-1, 1), 1)

    df_oof = df_oof[[target_col]]


    np_pred = df_y_pred.values
    df_y_pred["CURRENCY"] = df_test["CURRENCY"]

    test_idx = df_y_pred["CURRENCY"].values

    df_y_pred[target_col] = np.take_along_axis(np_pred, test_idx.reshape(-1, 1), 1)

    df_y_pred = df_y_pred[[target_col]]



    return df_y_pred, df_oof





def simAug(df_train, df_test, add_col, nth=5, separate_col="CURRENCY"):

    #1. separate_colの値ごとにDESCRIPTION_TRANSLATEDのUniversal Sentence Encoderによる文章ベクトルを取得し自分とそれ以外の文章とのcosine類似度を算出．  
	#2. 学習データでは，類似度の大きい順にnth個の文章を選びその中からランダムに1つ選んでその文章のadd_colを特徴に追加．
	#3. テストデータでは，類似度の大きいnth個の文章のadd_colをそれぞれ追加（テストデータはnth倍されます．）
	 
    
    test_conc_list = []
    for cur in df_train[separate_col].unique():
        print(f"{cur}")
        df_tmp_train = df_train.loc[df_train[separate_col]==cur]
        df_tmp_test = df_test.loc[df_test[separate_col]==cur]

        
        if df_tmp_test.shape[0]>0:
            df_sim_test, similarities_test, max_vector_test, id_vector_test = calcAndSetSimilarity(sim_cols=add_col, df_train=df_tmp_train, df=df_tmp_test, return_all=True, nth=nth)

            conc_list = []
            for c in add_col+["value"]:
                use_simi_cols = [f"similarity_{i}_{c}" for i in range(id_vector_test.shape[1])]
                df_stack = df_sim_test[use_simi_cols].stack()
                df_stack.index = df_stack.index.droplevel(-1)
                df_stack.name = f"neighbor_{separate_col}_{c}"
                df_stack = df_stack.reset_index()
                df_stack["cum_id"] = df_stack.groupby("LOAN_ID").cumcount()
                #df_stack["cum_id"] = [f"{l}_{i}" for l, i in zip(df_stack["LOAN_ID"], df_stack["cum_id"])]
                df_stack = df_stack.set_index(["LOAN_ID","cum_id"])

                conc_list.append(df_stack)

            df_test_new_cols = pd.concat(conc_list, axis=1)
            test_conc_list.append(df_test_new_cols)
            #pdb.set_trace()




        df, similarities, max_vector, id_vector = calcAndSetSimilarity(sim_cols=add_col, df_train=df_tmp_train, df=None, return_all=True, nth=nth)



        df["ranodm_i"] = np.random.randint(0, id_vector.shape[1], df.shape[0]) 
        for c in add_col+["value"]:
            use_simi_cols = [f"similarity_{i}_{c}" for i in range(id_vector.shape[1])]
            np_use_simi = df[use_simi_cols].values
            selected_simi =  np.take_along_axis(np_use_simi, df["ranodm_i"].values.reshape(-1,1), axis=1) 
            df[f"neighbor_{separate_col}_{c}"] = selected_simi
            df_train.loc[df.index, f"neighbor_{separate_col}_{c}"] = df[f"neighbor_{separate_col}_{c}"]
            
    df_test_conc = pd.concat(test_conc_list)
    
    additional_cols = add_col+["value"]
    additional_cols = [f"neighbor_{separate_col}_{c}" for c in additional_cols]
    df_train_conc  = df_train[additional_cols]
    
    #pdb.set_trace()
    #df_test = df_test_conc.join(df_test)

    return df_train_conc, df_test_conc

def proc_simAug():

    df_train = pd.read_pickle(PROC_DIR / f'df_proc_train.pkl')
    df_test = pd.read_pickle(PROC_DIR / f'df_proc_test.pkl')
    
    #pdb.set_trace()

    add_col=["LOAN_AMOUNT"]
    separate_cols=["CURRENCY", "ACTIVITY_NAME", "CURRENCY_POLICY_CURRENCY_EXCHANGE_COVERAGE_RATE_CURRENCY"]

    for separate_col in separate_cols:
        df_train_conc, df_test_conc =  simAug(df_train, df_test, add_col=add_col , nth=5, separate_col=separate_col)

        df_train_conc.to_csv(PROC_DIR/f"df_train_simAug_{separate_col}.csv")
        df_test_conc.to_csv(PROC_DIR/f"df_test_simAug_{separate_col}.csv")


        
def argParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="loan", choices=['loan', 'clip', "simAug"] )


    args=parser.parse_args()

    setting_params= vars(args)

    return setting_params

if __name__ == '__main__':


    setting_params=argParams()

    if setting_params["mode"] == "clip":
        addClip()
    elif setting_params["mode"] == "simAug":
        proc_simAug()

    else:
        addLoanInCurrency2()

    

