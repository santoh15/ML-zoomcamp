# In[2]:


import requests


# In[3]:


url= 'http://localhost:6565/predict'


# In[15]:

customer_id = '0002-ORFBO'
customer = {
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "no",
  "dependents": "no",
  "tenure": 108,
  "phoneservice": "yes",
  "multiplelines": "no",
  "internetservice": "dsl",
  "onlinesecurity": "yes",
  "onlinebackup": "no",
  "deviceprotection": "yes",
  "techsupport": "yes",
  "streamingtv": "yes",
  "streamingmovies": "yes",
  "contract": "one_year",
  "paperlessbilling": "yes",
  "paymentmethod": "bank_transfer_(automatic)",
  "monthlycharges": 79.85,
  "totalcharges": 330.75
}


# In[18]:


respuesta = requests.post(url, json=customer).json()
print(respuesta)

# In[19]:


if respuesta['churn'] == True:
    print('sending promo email to %s' %(customer_id))
else:
    print('no email sent to %s' %(customer_id))


# In[ ]:




