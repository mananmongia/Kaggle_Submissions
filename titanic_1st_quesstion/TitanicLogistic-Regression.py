
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[3]:


train_data.head()


# In[4]:


test_data.head()


# In[11]:


X_train = train_data.iloc[:,2:]
Y_train = train_data.iloc[:,1]
X_test = test_data.iloc[:,1:]


# In[12]:


X_train.head()


# In[13]:


X_test.head()


# In[14]:


del X_train['Cabin']
del X_test['Cabin']


# In[16]:


del X_train['Ticket']
del X_test['Ticket']


# In[17]:


del X_train['Fare']
del X_test['Fare']


# In[18]:


del X_train['Name']
del X_test['Name']


# In[23]:


X_train['Age'].fillna( (X_train['Age'].mean() + X_test['Age'].mean())/2 ,inplace = True)
X_test['Age'].fillna( (X_train['Age'].mean() + X_test['Age'].mean())/2 ,inplace = True)


# In[28]:


X_train.isnull().sum()


# In[29]:


X_test.isnull().sum()


# In[30]:


X_train['Embarked'].fillna('Q',inplace = True)


# In[31]:


from sklearn.preprocessing import LabelEncoder
sexe_le = LabelEncoder()
embark_le = LabelEncoder()


# In[33]:


X_train = X_train.values
X_test = X_test.values


# In[34]:


X_train[:,1] = sexe_le.fit_transform(X_train[:,1])
X_test[:,1] = sexe_le.transform(X_test[:,1])


# In[35]:


X_train[:,5] = embark_le.fit_transform(X_train[:,5])
X_test[:,5] = embark_le.transform(X_test[:,5])


# In[37]:


from sklearn.preprocessing import OneHotEncoder
embark_ohe = OneHotEncoder(categorical_features = [5])
X_train = embark_ohe.fit_transform(X_train)
X_test = embark_ohe.transform(X_test)


# In[40]:


X_train = X_train.toarray()
X_test = X_test.toarray()


# In[42]:


X_train = X_train[:,1:]
X_test = X_test[:,1:]


# In[44]:


print(X_train.shape)
print(X_test.shape)


# In[45]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,Y_train)


# In[46]:


Y_pred = clf.predict(X_test)


# In[48]:


def print_ans(arr):
    j = 892
    for i in arr:
        print(str(j) + "," + str(i))
        j = j+1


# In[49]:


print_ans(Y_pred)


# In[72]:


X_modified = np.append(np.array(X_train),np.array(X_test),axis = 0)
X_modified.shape


# In[75]:


Y_modified = np.append(np.array(Y_train),np.array(Y_pred), axis = 0)
Y_modified.shape


# In[76]:


modified_clf = LogisticRegression(random_state = 0)
modified_clf.fit(X_modified,Y_modified)


# In[77]:


Y_pred_modified = modified_clf.predict(X_test)


# In[78]:


print_ans(Y_pred_modified)


# In[79]:


Y_pred_modified.shape

