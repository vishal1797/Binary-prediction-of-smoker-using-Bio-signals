#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading datasets

# In[3]:


train=pd.read_csv(r'train.csv')
test=pd.read_csv(r'test.csv')
sub_data=pd.read_csv(r'sample_submission.csv')


# In[4]:


train.drop('id',axis=1,inplace=True)
test.drop('id',axis=1,inplace=True)


# In[5]:


train.head()


# # EDA

# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# In[12]:


train.describe()


# In[13]:


train.info()


# In[14]:


train=train.drop_duplicates()
test=test.drop_duplicates()

age : 5-years gap
height(cm)
weight(kg)
waist(cm) : Waist circumference length
eyesight(left)
eyesight(right)
hearing(left)
hearing(right)
systolic : Blood pressure
relaxation : Blood pressure
fasting blood sugar
Cholesterol : total
triglyceride
HDL : cholesterol type
LDL : cholesterol type
hemoglobin
Urine protein
serum creatinine
AST : glutamic oxaloacetic transaminase_type
ALT : glutamic oxaloacetic transaminase_type
Gtp : γ-GTP
dental caries  
# In[15]:


df = train.copy()

def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str = 'Train correlation') -> None:
    excluded_columns = ['id']
    columns_without_excluded = [col for col in df.columns if col not in excluded_columns]
    corr = df[columns_without_excluded].corr()
    
    fig, axes = plt.subplots(figsize=(14, 10))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='mako', annot=True, annot_kws={"size": 6})
    plt.title(title_name, color='white')
    plt.show()

# Plot correlation heatmap for encoded dataframe
plot_correlation_heatmap(df, 'Dataset Correlation')


# In[16]:


# histogram of the 'smoking' column in the train dataset 

train['smoking'].hist(bins=3, color='k')

plt.xticks(ticks=[0, 1], labels=['No Smoking', 'Smoking'])

plt.ylabel('Frequency')

plt.title('Smoking Distribution in Train Dataset')

plt.show()


# In[ ]:





# # Hyperparameter Tunning

# In[46]:


#problem is dataset is too large and it getting too long to train the model thats why we are using smaller part
# of the dataset to get the hyperparameters


# In[59]:


train_data_subset = train.sample(frac=0.1)


# In[60]:


train_data_subset.shape


# In[61]:


xx = train_data_subset.iloc[:,:-1]
yy=train_data_subset['smoking']


# In[ ]:





# In[62]:


#we already have the testing the datset seperately thats why we aregoing split the dataset into training data and validation data


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_val=train_test_split(xx,yy,test_size=0.5)


# In[50]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()


# In[53]:


from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score


# In[54]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


# In[40]:


# k-fold cross-validation

cv = KFold(n_splits=5, shuffle=True, random_state=42)


# In[42]:


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv, n_jobs=-1)

# Fit the model to the training data
grid_search.fit(x_train, y_train)


# In[55]:


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:", best_params)


# In[ ]:





# # now Model Traning

# In[63]:


x= train.iloc[:,:-1]
y=train['smoking']


# In[65]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_val=train_test_split(x,y,test_size=0.5)


# In[67]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    max_depth=20,
    max_features='auto',
    min_samples_leaf=4,
    min_samples_split=5,
    n_estimators=200
)


# In[68]:


model.fit(x_train,y_train)


# In[70]:


model.score(x_test,y_val)


# In[ ]:





# # AUC - ROC

# In[72]:


from sklearn.metrics import roc_auc_score

y_pred= model.predict(x_test)

roc_auc_score(y_val, y_pred)


# # Result

# In[73]:


prediction= model.predict(test)


# In[75]:


results= pd.DataFrame( {'Predicted_Probabilities': prediction})

results.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")s


# In[ ]:




