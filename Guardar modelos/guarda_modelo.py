#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[2]:


df=pd.read_csv('churn.csv')
df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_columns=list(df.dtypes[df.dtypes=='object'].index)
for i in categorical_columns:
    df[i] = df[i].str.lower().str.replace(' ','_')
df.totalcharges=pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges=df.totalcharges.fillna(0)
df.churn=(df.churn == 'yes').astype(int)


# In[3]:


df_train_full,df_test=train_test_split(df,test_size=0.2,random_state=1)
df_train,df_val=train_test_split(df_train_full,test_size=0.25,random_state=1)

df_train=df_train.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)

y_train=df_train.churn.values
y_val=df_val.churn.values
y_test=df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

numerical=['tenure', 'monthlycharges', 'totalcharges']
categorias=['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

dv=DictVectorizer(sparse=False)
train_dicts=df_train[numerical + categorias].to_dict(orient='records')
X_train= dv.fit_transform(train_dicts)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

dicts_val = df_val[categorias+numerical].to_dict(orient='records')
X_val=dv.fit_transform(dicts_val)

y_pred=model.predict_proba(X_val)[:,1]
((y_pred >= 0.52).astype(int)==y_val).astype(int).mean()


# In[4]:


scores=[]
for j in np.linspace(0,1,21):
    decision=(y_pred >= j)
    score= (y_val == decision).mean()
    scores.append(score)


# In[5]:


plt.plot(np.linspace(0,1,21), scores)


# In[6]:


from sklearn.metrics import accuracy_score


# In[7]:


scores=[]
for j in np.linspace(0,1,21):
    score=accuracy_score(y_val, y_pred >= j)
    scores.append(score)


# In[8]:


from collections import Counter


# Nuestro modelo es 80% pero si considero descartar todos es decir j=1.0, tengo que el modelo le pega al 73%, es decir que nuestro modelo arregla un 7% de lo que es el "modelo estupido", esto muestra un desbalance entre los datos ya que casi 3:1 son falsos

# In[9]:


actual_positive = (y_val == 1)
actual_negative = (y_val == 0)
t=0.5
pred_positive = (y_pred >= t)
pred_negative = (y_pred < t)


# In[10]:


tp=(actual_positive & pred_positive).sum()
tn=(actual_negative & pred_negative).sum()
fp=(pred_positive & actual_negative).sum()
fn=(pred_negative & actual_positive).sum()


# In[ ]:





# In[11]:


confusion_matrix= np.array([
    [tn,fp],
    [fn,tp]
])
confusion_matrix


# In[12]:


confusion_matrix_norm=confusion_matrix / confusion_matrix.sum()


# In[13]:


confusion_matrix_norm


# In[14]:


precision= tp/(tp+fp)


# In[15]:


recall=tp/(tp+fn)


# In[16]:


tpr=tp/(tp+fn)


# In[17]:


fpr=fp/(tn+fp)


# In[18]:


parametro= np.linspace(0,1,101)
scores=[]
for j in parametro:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    pred_positive = (y_pred >= j)
    pred_negative = (y_pred < j)
    tp=(actual_positive & pred_positive).sum()
    tn=(actual_negative & pred_negative).sum()
    fp=(pred_positive & actual_negative).sum()
    fn=(pred_negative & actual_positive).sum()
    scores.append((float(j),float(tp),float(fp),float(fn),float(tn)))


# In[19]:


df_scor=pd.DataFrame(scores,columns=['par','tp','fp','fn','tn'])


# In[20]:


df_scor


# In[21]:


df_scor['tpr']=df_scor.tp/(df_scor.tp+df_scor.fn)
df_scor['fpr']=df_scor.fp/(df_scor.tn+df_scor.fp)
df_scor


# In[22]:


plt.plot(df_scor.par,df_scor.tpr, label='TPR')
plt.plot(df_scor.par,df_scor.fpr, label='FPR')
plt.legend()


# Lo que buscamos es que TPR sea máximo y FPR mínimo así que hay que hallar los valores que convienen

# In[23]:


np.random.seed(1)
y_rand=np.random.uniform(0,1,size=len(y_val))


# In[24]:


def fpr_tpr_df(y_val,y_pred):
    parametro= np.linspace(0,1,101)
    scores=[]
    for j in parametro:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)
        pred_positive = (y_pred >= j)
        pred_negative = (y_pred < j)
        tp=(actual_positive & pred_positive).sum()
        tn=(actual_negative & pred_negative).sum()
        fp=(pred_positive & actual_negative).sum()
        fn=(pred_negative & actual_positive).sum()
        scores.append((float(j),float(tp),float(fp),float(fn),float(tn)))
    df_scor=pd.DataFrame(scores,columns=['par','tp','fp','fn','tn'])
    df_scor['tpr']=df_scor.tp/(df_scor.tp+df_scor.fn)
    df_scor['fpr']=df_scor.fp/(df_scor.tn+df_scor.fp)
    return df_scor


# Todo esto se calcula automatico con scikitlearn

# In[25]:


model_ran=fpr_tpr_df(y_val,y_rand)


# In[26]:


model_ran


# In[27]:


plt.plot(model_ran.fpr,model_ran.tpr, label='random')
plt.plot(df_scor.fpr,df_scor.tpr, label='modelo')
#plt.plot(model_ran.var,model_ran.fpr, label='FPR')
plt.legend()


# In[28]:


from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[29]:


auc(df_scor.fpr,df_scor.tpr)


# In[30]:


fpr, tpr, par=roc_curve(y_val, y_pred)


# In[31]:


roc_auc_score(y_val, y_pred)


# In[32]:


def train(df,y_train,C=1.0):
    dicts = df[categorias+numerical].to_dict(orient='records')
    dv=DictVectorizer(sparse=False)
    X_train= dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=10000)
    model.fit(X_train, y_train)

    return dv, model


# In[33]:


dv,model=train(df_train,y_train, C=0.001)


# In[34]:


def predict(df, dv, model):
    dicts = df[categorias+numerical].to_dict(orient='records')
    X=dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred


# In[35]:


y_pred = predict(df_val, dv, model)


# In[36]:


from sklearn.model_selection import KFold


# In[37]:


kfold= KFold(n_splits=10, shuffle=True, random_state=1)


# In[38]:


for C in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    scores=[]
    for train_idx, val_idx in kfold.split(df_train_full):
        df_train = df_train_full.iloc[train_idx]
        df_val = df_train_full.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv,model = train(df_train,y_train,C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val,y_pred)
        scores.append(auc)
    print('%s %.3f +- %.3f' % (C,np.mean(scores),np.std(scores)))


# In[39]:


C=1.0
dv,model = train(df_train_full,df_train_full.churn,C=1.0)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test,y_pred)
auc


# In[40]:


import pickle
output_file= 'model_c=%s.bin' % C
output_file


# In[41]:


f_out= open(output_file, 'wb')
pickle.dump((dv,model), f_out)
f_out.close()


# In[1]:


import pickle


# In[2]:


model_file='model_c=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv,model=pickle.load(f_in)


# In[3]:


dv,model


# In[ ]:




