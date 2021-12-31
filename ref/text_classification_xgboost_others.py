#!/usr/bin/env python
# coding: utf-8

# #  Getting the data

# In[14]:


import pandas as pd
bbc_text_df = pd.read_csv('../data/bbc-text.csv')
bbc_text_df.head()


# #  Data Exploration & Visualisation

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
sns.countplot(x=bbc_text_df.category, color='green')
plt.title('BBC text class distribution', fontsize=16)
plt.ylabel('Class Counts', fontsize=16)
plt.xlabel('Class Label', fontsize=16)
plt.xticks(rotation='vertical');


# In[16]:


from gensim import utils
import gensim.parsing.preprocessing as gsp

filters = [
           gsp.strip_tags, 
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords, 
           gsp.strip_short, 
           gsp.stem_text
          ]
#   去除标签; 英文字母小写化; 去除标点; 去除多余空格; 去除整数、数字; 
# 去除停用词（如and、to、the等）
# from gensim.parsing.preprocessing import strip_short
# strip_short("salut les amis du 59")
# u'salut les amis'
# strip_short("one two three four five six seven eight nine ten", minsize=5)
# u'three seven eight'
# 词干提取（将单词转换至词源形式）
def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s


# In[17]:


bbc_text_df.iloc[2,1]


# In[18]:


clean_text(bbc_text_df.iloc[2,1])


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')

from wordcloud import WordCloud

def plot_word_cloud(text):
    wordcloud_instance = WordCloud(width = 800, height = 800, 
                                   background_color ='black', 
                                   stopwords=None,
                                   min_font_size = 10).generate(text) 
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud_instance) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()


# In[20]:


texts = ''
for index, item in bbc_text_df.iterrows():
    texts = texts + ' ' + clean_text(item['text'])
    
plot_word_cloud(texts)


# In[21]:


def plot_word_cloud_for_category(bbc_text_df, category):
    text_df = bbc_text_df.loc[bbc_text_df['category'] == str(category)]
    texts = ''
    for index, item in text_df.iterrows():
        texts = texts + ' ' + clean_text(item['text'])
    
    plot_word_cloud(texts)


# In[22]:


plot_word_cloud_for_category(bbc_text_df,'tech')


# In[23]:


plot_word_cloud_for_category(bbc_text_df,'sport')


# In[24]:


plot_word_cloud_for_category(bbc_text_df,'politics')


# In[25]:


df_x = bbc_text_df['text']
df_y = bbc_text_df['category']


# In[26]:


df_x[0]


# # Building the Machine Learning model & pipeline

# ## Converting to Doc2Vec

# In[27]:


from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm

import multiprocessing
import numpy as np

class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, window=5, 
                 epochs=100, min_count=4, workers=4, alpha=0.025, 
                 min_alpha=0.025):
        # self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1
        self.window = window
        self.min_count = min_count
        self.alpha = alpha
        self.min_alpha = min_alpha

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(clean_text(row).split(), [index]) 
                    for index, row in enumerate(df_x)]
        model = Doc2Vec(documents=tagged_x, 
                        vector_size=self.vector_size, 
                        workers=self.workers)

        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), 
                        total_examples=len(tagged_x), 
                        epochs=1)
            #model.alpha -= self.learning_rate
            #model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(clean_text(row).split())
                                     for index, row in enumerate(df_x)]))


# In[28]:


doc2vec_trf = Doc2VecTransformer()
doc2vec_features = doc2vec_trf.fit(df_x).transform(df_x)


# In[29]:


import numpy as np
#np.eigen(doc2vec_features)


# ## Pipeline with Doc2Vec & LogisticRegression

# In[30]:


doc2vec_features.shape
from sklearn.ensemble import RandomForestClassifier as rfc
rfc_scores = []
for i in [5, 10, 20]:
    rfc_model = rfc(random_state=0, oob_score=True,
                   max_features = i + 1)
    _ = rfc_model.fit(doc2vec_features, bbc_text_df.category)
    rfc_score = rfc_model.oob_score_
    rfc_scores.append(rfc_score)


# In[31]:


fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel("p")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs p for randomforest")
ax.plot([10, 20, 30], rfc_scores, marker='o', label="rfc",
        drawstyle="steps-post")
ax.legend()
plt.show()


# In[32]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

pl_log_reg = Pipeline(steps=[('doc2vec',Doc2VecTransformer()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', 
                                                            solver='saga', max_iter=1000))])
scores = cross_val_score(pl_log_reg, df_x, df_y, cv=5,scoring='accuracy')
print('Accuracy for Logistic Regression: ', scores.mean())


# ## Pipeline with Doc2Vec & RandomForest

# In[33]:


from sklearn.ensemble import RandomForestClassifier

pl_random_forest = Pipeline(steps=[('doc2vec',Doc2VecTransformer()),
                                   ('random_forest', RandomForestClassifier())])
scores = cross_val_score(pl_random_forest, df_x, df_y, cv=5,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())


# ## Pipeline with Doc2Vec & XGBoost

# In[34]:


import xgboost as xgb

pl_xgb = Pipeline(steps=[('doc2vec',Doc2VecTransformer()),
                         ('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
scores = cross_val_score(pl_xgb, df_x, df_y, cv=5)
print('Accuracy for XGBoost Classifier : ', scores.mean())


# ## Converting to Tf-Idf

# In[35]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X)


# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer

class Text2TfIdfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        df_x = df_x.apply(lambda x : clean_text(x))
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)


# In[37]:


tfidf_transformer = Text2TfIdfTransformer()
tfidf_vectors = tfidf_transformer.fit(df_x).transform(df_x)


# In[38]:


tfidf_vectors.shape


# In[39]:


type(tfidf_vectors)


# ## Pipeline with Tf-Idf & LogisticRegression

# In[40]:


pl_log_reg_tf_idf = Pipeline(steps=[('tfidf',Text2TfIdfTransformer()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=100))])
scores = cross_val_score(pl_log_reg_tf_idf, df_x, df_y, cv=5,scoring='accuracy')
print('Accuracy for Tf-Idf & Logistic Regression: ', scores.mean())


# ## Pipeline with Tf-Idf & RandomForest

# In[41]:


pl_random_forest_tf_idf = Pipeline(steps=[('tfidf',Text2TfIdfTransformer()),
                                   ('random_forest', RandomForestClassifier())])
scores = cross_val_score(pl_random_forest_tf_idf, df_x, df_y, cv=5,scoring='accuracy')
print('Accuracy for Tf-Idf & RandomForest : ', scores.mean())


# ## Pipeline with Tf-Idf & XGBoost

# In[42]:


pl_xgb_tf_idf = Pipeline(steps=[('tfidf',Text2TfIdfTransformer()),
                         ('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
scores = cross_val_score(pl_xgb_tf_idf, df_x, df_y, cv=5)
print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())


# In[ ]:




