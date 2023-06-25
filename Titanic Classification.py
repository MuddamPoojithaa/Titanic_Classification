#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns,set()


# In[7]:


titanic_data=pd.read_csv("C:/Users/dell/Downloads/train.csv")
titanic_test=pd.read_csv("C:/Users/dell/Downloads/test.csv")


# In[8]:


titanic_data.head()


# In[12]:


titanic_data.shape


# In[13]:


titanic_data.columns


# In[14]:


titanic_data.isnull().sum()


# In[15]:


sns.countplot(x='Survived', data=titanic_data)


# In[16]:


sns.countplot(x='Survived',hue='Sex', data=titanic_data)


# In[ ]:


sns.countplot(x='Survived',hue='Pclass', data=titanic_data)


# In[18]:


def add_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return titanic_data[titanic_data['Pclass']==1]['Age'].mean()
        elif Pclass==2:
            return titanic_data[titanic_data['Pclass']==2]['Age'].mean()
        elif Pclass==3:
            return titanic_data[titanic_data['Pclass']==3]['Age'].mean()
    else:
        return Age


# In[19]:


df=titanic_data


# In[20]:


df['Age']=df[['Age','Pclass']].apply(add_age,axis=1)


# In[21]:


df.Sex=df.Sex.map({'female':0, 'male':1})
df.Embarked=df.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})


# In[22]:


df.drop('Cabin',axis=1,inplace=True)


# In[23]:


df.dropna(inplace=True)


# In[24]:


df.drop(['Name', 'PassengerId', 'Ticket'], axis = 1, inplace = True)


# In[25]:


min_age=min(df.Age)
max_age=max(df.Age)
min_fare=min(df.Fare)
max_fare=max(df.Fare)


# In[26]:


df.Age = (df.Age-min_age)/(max_age-min_age)
df.Fare = (df.Fare-min_fare)/(max_fare-min_fare)


# In[27]:


df.head()


# In[28]:


x_data=df.drop('Survived',axis=1)
y_data=df['Survived']


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2, random_state=0, stratify=y_data)


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model = LogisticRegression()


# In[33]:


model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)


# In[34]:


from sklearn.metrics import classification_report


# In[35]:


print(classification_report(y_test_data, predictions))


# In[36]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test_data, predictions))


# In[37]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_data, predictions)


# In[38]:


cf_matrix=confusion_matrix(y_test_data, predictions)


# In[39]:


import seaborn as sns

sns.heatmap(cf_matrix, annot=True)


# In[40]:


df1=titanic_test


# In[41]:


df1.head()


# In[42]:


df1.isnull().sum()


# In[43]:


df1['Age']=df1[['Age','Pclass']].apply(add_age,axis=1)


# In[44]:


df1['Fare']=df1['Fare'].fillna(df1['Fare'].median())


# In[45]:


df1.Sex=df1.Sex.map({'female':0, 'male':1})
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})


# In[46]:


min_age1=min(df1.Age)
max_age1=max(df1.Age)
min_fare1=min(df1.Fare)
max_fare1=max(df1.Fare)


# In[47]:


df1.Age = (df1.Age-min_age1)/(max_age1-min_age1)
df1.Fare = (df1.Fare-min_fare1)/(max_fare1-min_fare1)


# In[48]:


df1.drop(['Cabin','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[49]:


df1.head()


# In[50]:


prediction=model.predict(df1)
prediction


# In[53]:


test=pd.read_csv("C:/Users/dell/Downloads/test.csv")


# In[54]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": prediction})
submission.to_csv('submission.csv', index=False)


# In[55]:


pred_df = pd.read_csv('submission.csv')


# In[56]:


sns.countplot(x='Survived', data=pred_df)


# In[9]:


sns.countplot(titanic_data['SibSp'])


# In[10]:


sns.distplot(titanic_data['Age'])


# In[11]:


sns.distplot(titanic_data['Fare'])


# In[12]:


sns.barplot(data=titanic_data, x='Pclass', y='Fare', hue='Survived')


# In[13]:


sns.barplot(data=titanic_data, x='Survived', y='Fare', hue='Pclass')


# In[ ]:




