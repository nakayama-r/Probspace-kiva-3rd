# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:48:23 2022

@author: r00526841
"""

from utils import *
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency



from gensim.models import KeyedVectors
import gensim.downloader as api


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.mixture import GaussianMixture

from DNNmodel import BertSequenceVectorizer
from sklearn.metrics.pairwise import cosine_similarity



import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub


def checkCurrencyString(query_str, df_train, df_test, text_col):
    
    
    
    flag_train = df_train[text_col].map(lambda x: True if query_str in x else False)
    flag_test = df_test[text_col].map(lambda x: True if query_str in x else False)
    
    print(f"train : {len(df_train.loc[flag_train])}, test : {len(df_test.loc[flag_test])}")
    
    #text = re.sub(r'\?month', 'August', text, flags=re.IGNORECASE)
    
    return flag_train, flag_test


def getMaxSimilarity(np_query, np_db=None, nth=1):
    similarities = cosine_similarity(np_query, np_db)
    
    if np_db is None:
        np.fill_diagonal(similarities, 0)
    max_vector = np.sort(similarities, axis=1)[:, -nth:][:,::-1] #similarities.max(axis=1) 
    id_vector = np.argsort(similarities)[:, -nth:][:,::-1] #similarities.argmax(axis=1)
    #print('pairwise dense output:\n {}\n'.format(similarities.shape))
    #print('max_vector:\n {}\n'.format(max_vector.shape))
    #print('id_vector:\n {}\n'.format(id_vector.shape))
    
    
    return similarities, max_vector, id_vector


def setSimilarColumns(df, max_vector, id_vector,  columuns, df_train):
    
    
    for i in range(max_vector.shape[1]):
    
        df[f"similarity_{i}_value"] = max_vector[:,i]
        
        for c in columuns:
            df[f"similarity_{i}_{c}"] = df_train.iloc[id_vector[:,i]][c].values
            
        
    return df

def calcAndSetSimilarity(sim_cols, df_train, df=None, return_all=False, nth=1):

    
    use_list = getColumnsFromParts(["USE_embedding_",], df_train.columns)
    df_train_f = df_train[use_list]
    
    if df is None:
        df = df_train
        df_f=df_train_f
        similarities, max_vector, id_vector = getMaxSimilarity(df_f.values, None, nth=nth)
    else:
        df_f = df[use_list]
        similarities, max_vector, id_vector = getMaxSimilarity(df_f.values, df_train_f.values, nth=nth)
    
   

    df = setSimilarColumns(df, max_vector, id_vector,  sim_cols, df_train)
    
    if return_all:
        return df, similarities, max_vector, id_vector
    else:
    
        return df

def getUniversalSentenceEncoder(se_text, output_feature_dim=512, compress_type="svd"):
    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    
    
    #tqdm.pandas()
    features = np.stack(
        se_text.map(lambda x: embedder(x).numpy().reshape(-1)).values
    )
    
    print(f"USE feature : {features.shape}")
    
    features = GetCompressedFeature(np_features=features, output_feature_dim=output_feature_dim, compress_type=compress_type)
    
    new_cols = [f"USE_embedding_{i}" for i in range(features.shape[1])]
    
    df_new = pd.DataFrame(features, columns=new_cols, index=se_text.index)
    
    
    return df_new

class SCDVEmbedder(TransformerMixin, BaseEstimator):
    def __init__(self, w2v, tokenizer, k=5):
        self.w2v = w2v
        self.vocab = set(self.w2v.vocab.keys())
        self.tokenizer = tokenizer
        self.k = k
        self.topic_vector = None
        self.tv = TfidfVectorizer(tokenizer=self.tokenizer)

    def __assert_if_not_fitted(self):
        assert self.topic_vector is not None, \
            "SCDV model has not been fitted"

    def __create_topic_vector(self, corpus: pd.Series):
        self.tv.fit(corpus)
        self.doc_vocab = set(self.tv.vocabulary_.keys())

        self.use_words = list(self.vocab & self.doc_vocab)
        self.use_word_vectors = np.array([
            self.w2v[word] for word in self.use_words])
        w2v_dim = self.use_word_vectors.shape[1]
        self.clf = GaussianMixture(
            n_components=self.k, 
            random_state=42,
            covariance_type="tied")
        self.clf.fit(self.use_word_vectors)
        word_probs = self.clf.predict_proba(
            self.use_word_vectors)
        world_cluster_vector = self.use_word_vectors[:, None, :] * word_probs[:, :, None]

        doc_vocab_list = list(self.tv.vocabulary_.keys())
        use_words_idx = [doc_vocab_list.index(w) for w in self.use_words]
        idf = self.tv.idf_[use_words_idx]
        topic_vector = world_cluster_vector.reshape(-1, self.k * w2v_dim) * idf[:, None]
        topic_vector = np.nan_to_num(topic_vector)

        self.topic_vector = topic_vector
        self.vocabulary_ = set(self.use_words)
        self.ndim = self.k * w2v_dim

    def fit(self, X, y=None):
        self.__create_topic_vector(X)

    def transform(self, X):
        tokenized = X.fillna("").map(lambda x: self.tokenizer(x))

        def get_sentence_vector(x: list):
            embeddings = [
                self.topic_vector[self.use_words.index(word)]
                if word in self.vocabulary_
                else np.zeros(self.ndim, dtype=np.float32)
                for word in x
            ]
            if len(embeddings) == 0:
                return np.zeros(self.ndim, dtype=np.float32)
            else:
                return np.mean(embeddings, axis=0)
        return np.stack(
            tokenized.map(lambda x: get_sentence_vector(x)).values
        )


def GetBertFeatures(se_text):
    
    BSV = BertSequenceVectorizer(model_name="bert-base-multilingual-uncased", max_len=128)
    features = np.stack(
                        se_text.fillna("").map(lambda x: BSV.vectorize(x).reshape(-1)).values
                    )
    
    print(f"bert feature : {features.shape}")
    
    new_cols = [f"bert_{i}" for i in range(features.shape[1])]
    
    df_new = pd.DataFrame(features, columns=new_cols, index=se_text.index)
    
    
    return df_new

    
    

def GetWord2VecEmbeddings(se_text, output_feature_dim=50, compress_type="svd"):
    

    w2v_model = api.load('glove-twitter-25')
    
    #pdb.set_trace()
    
    tokenizer = SimpleTokenizer()
    swem = SWEM(w2v=w2v_model, tokenizer=tokenizer)


    # def f(x):
    #     embeddings = [
    #         w2v_model.get_vector(word)
    #         if word in w2v_model.key_to_index
    #         else np.zeros(ndim, dtype=np.float32)
    #         for word in x.split()
    #     ]
        
        
    #     if len(embeddings) == 0:
    #         return np.zeros(ndim, dtype=np.float32)
    #     else:
    #         return np.mean(embeddings, axis=0)


    features = np.stack(
        se_text.str.replace("\n", "").map(
            lambda x: swem.average_pooling(x)
            #lambda x: swem.max_pooling(x)
            #lambda x: swem.concat_average_max_pooling(x)
            #lambda x: swem.hierarchical_pooling(x, n=3)
        ).values
    )
    
    print(f"w2v : {features.shape}")
    features = GetCompressedFeature(np_features=features, output_feature_dim=output_feature_dim, compress_type=compress_type)
    
    new_cols = [f"w2v_{i}" for i in range(features.shape[1])]
    
    df_new = pd.DataFrame(features, columns=new_cols, index=se_text.index)

    
    return df_new
    

class SimpleTokenizer:
    def tokenize(self, text: str):
        return text.split()


class SWEM():
    """
    Simple Word-Embeddingbased Models (SWEM)
    https://arxiv.org/abs/1805.09843v1
    """

    def __init__(self, w2v, tokenizer, oov_initialize_range=(-0.01, 0.01)):
        self.w2v = w2v
        self.tokenizer = tokenizer
        self.vocab = set(self.w2v.key_to_index.keys())
        self.embedding_dim = self.w2v.vector_size
        self.oov_initialize_range = oov_initialize_range

        if self.oov_initialize_range[0] > self.oov_initialize_range[1]:
            raise ValueError("Specify valid initialize range: "
                             f"[{self.oov_initialize_range[0]}, {self.oov_initialize_range[1]}]")

    def get_word_embeddings(self, text):
        np.random.seed(abs(hash(text)) % (10 ** 8))

        vectors = []
        for word in self.tokenizer.tokenize(text):
            if word in self.vocab:
                vectors.append(self.w2v[word])
            else:
                vectors.append(np.random.uniform(self.oov_initialize_range[0],
                                                 self.oov_initialize_range[1],
                                                 self.embedding_dim))
        return np.array(vectors)

    def average_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.mean(word_embeddings, axis=0)

    def max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.max(word_embeddings, axis=0)

    def concat_average_max_pooling(self, text):
        word_embeddings = self.get_word_embeddings(text)
        return np.r_[np.mean(word_embeddings, axis=0), np.max(word_embeddings, axis=0)]

    def hierarchical_pooling(self, text, n):
        word_embeddings = self.get_word_embeddings(text)

        text_len = word_embeddings.shape[0]
        if n > text_len:
            print(text)
            word_embeddings = np.concatenate([word_embeddings, np.random.uniform(self.oov_initialize_range[0],self.oov_initialize_range[1],(n-text_len,self.embedding_dim))])
            print(word_embeddings.shape)
            text_len = n
            
            #pdb.set_trace()
            #raise ValueError(f"window size must be less than text length / window_size:{n} text_length:{text_len}")
        window_average_pooling_vec = [np.mean(word_embeddings[i:i + n], axis=0) for i in range(text_len - n + 1)]

        return np.max(window_average_pooling_vec, axis=0)
    
    

# reference: https://github.com/arosh/BM25Transformer/blob/master/bm25.py
class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)
    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        return X
    
    
def GetCompressedFeature(np_features, output_feature_dim, compress_type="svd"):
    
    if np_features.shape[1] <= output_feature_dim:
        print(f"no compress: {np_features.shape} <= output_feature_dim : {output_feature_dim}")
        return np_features
    
    if compress_type == "svd":
        compress  = TruncatedSVD(n_components=output_feature_dim, random_state=42)
    elif compress_type == "nmf":
        compress  = NMF(n_components=output_feature_dim, random_state=42)
        
    elif compress_type == "lda":
        compress  = LatentDirichletAllocation(n_components=output_feature_dim, random_state=42)
    
    
    new_mtrx = compress.fit_transform(np_features)
    
    print(f"reduce dim : {np_features.shape} -> {new_mtrx.shape}")
    
    
    return new_mtrx
    
    
    
    


def GetFeaturesOfBOW(se_text, output_feature_dim, vectorizer_type="count", compress_type="svd"):
    
    
    if vectorizer_type == "count":
        bow = CountVectorizer()
       
    
    elif vectorizer_type == "tfidf":
        
        bow = TfidfVectorizer()
        
        
    elif vectorizer_type == "bm25":
        
        bow = Pipeline(steps=[
            ("CountVectorizer", CountVectorizer()),
            ("BM25Transformer", BM25Transformer())
        ])
    
    bow_features = bow.fit_transform(se_text)
    

    
    new_mtrx = GetCompressedFeature(np_features=bow_features, output_feature_dim=output_feature_dim, compress_type=compress_type)
    
    new_cols = [f"vectorize_{vectorizer_type}_compress_{compress_type}_{i}" for i in range(output_feature_dim)]
    
    df_new = pd.DataFrame(new_mtrx, columns=new_cols, index=se_text.index)

    
    return df_new
    
    