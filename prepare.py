#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import acquire


# Use the function defined in acquire.py to load the iris data.

# In[2]:


df_iris = acquire.get_iris_data()


# In[ ]:





# In[3]:


df_iris.columns


# In[ ]:





# Clean up the column names - replace the period with an underscore and lowercase.

# In[4]:


# df_iris.colums.sts.replace('.','_', regex=False).str.lower()


# Drop the species_id and measurement_id columns.

# In[5]:


df_iris = df_iris.drop(columns='species_id')


# In[6]:


df_iris = df_iris.drop(columns='measurement_id')


# In[7]:


df_iris


# Rename the species_name column to just species.

# In[8]:


df_iris = df_iris.rename(columns= {'species_name':'species'})


# In[9]:


df_iris


# Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

# In[41]:


def prep_iris(df):
    '''this should be the clean and fix data set maybe could be improve '''
    
#     this is going to drop all the columns 
    df = df.drop(columns=['species_id','measurement_id'])
#    this is rename one of the colums
    df = df.rename(columns= {'species_name':'species'})
    
    return df 


# In[27]:


fresh_iris = acquire.get_iris_data()


# Using the Telco dataset
# 
# Use the function defined in acquire.py to load the Telco data.

# In[28]:


df_telco = acquire.get_telco_data()
df_telco


# Drop any unnecessary, unhelpful, or duplicated columns. This could mean dropping foreign key columns but keeping the corresponding string values, for example.
# 
# Handle null values.

# In[29]:


# look at data first before just droping columns 


# In[30]:


df_telco = df_telco.drop(columns='payment_type_id') 
#put it into a list to drop more[]


# In[31]:


df_telco = df_telco.drop(columns='internet_service_type_id')


# In[32]:


df_telco = df_telco.drop(columns='contract_type_id')


# In[33]:


# has no null vaule 
df_telco.isnull().sum()


# In[34]:


df_telco = df_telco.total_charges.str.replace(' ','0')


# Create a function named prep_telco that accepts the raw telco data, and returns the data with the transformations above applied.

# In[35]:


def prep_telco(df):

    '''this should be the first split the data with basic input
    also change some of the colums to 1 being yes or 0 being no'''    
    df = df.drop(columns='payment_type_id')
    df = df.drop(columns='internet_service_type_id')
    df = df.drop(columns='contract_type_id')
#     check to see if there any null
    df.isnull().sum()
#     get rid of the blank space
    df.total_charges = df.total_charges.str.replace(' ', '0.0')
    return df


# In[ ]:

def telco_int(train):
    '''  this is making the gender colums more into a int so it better to grapgh
     making partner into a int 
     maybe churn into a int
     making the service type into a int and making it easlier to read
    '''
    train.loc[:, 'is_female'] = train['gender'].replace({'Female': 1, 'Male': 0})
    train.loc[:, 'has_partner'] = train['partner'].map({'Yes': 1, 'No': 0})
    train.loc[:,'churn'] = train['churn'].replace({'No': 0,'Yes': 1})
    train.loc[:,'internet_service_type'] = train['internet_service_type'].replace({'Fiber optic': 1,'DSL': 2, 'None':0})
    return train



# split the data

# In[36]:


from sklearn.model_selection import train_test_split


# Write a function to split your data into train, test and validate datasets. Add this function to prepare.py.
# 
# 

# In[47]:


def splitting_data(df, col):
    
    '''this should be the first split the data with basic input
    also change some of the colums to 1 being yes or 0 being no'''
    
    train, validate_test = train_test_split(df,
                     train_size=0.6,
                     random_state=123,
                     stratify=df[col]
                    )
 
    
    
#     second split
    validate, test = train_test_split(validate_test,
                                     train_size=0.5,
                                      random_state=123,
                                      stratify=validate_test[col])
    
    return train,validate,test



# In[48]:


iris_df = acquire.get_iris_data()


# In[49]:


iris_df.head()


# In[50]:


iris_df = prep_iris(iris_df)


# In[51]:


iris_df


# In[52]:


# train_iris, validate_iris, test_iris = splitting_data(iris_df, 'species')


# In[53]:


# print(train_iris.shape)
# print(validate_iris.shape)
# print(test_iris.shape)


# In[ ]:





# Run the function in your notebook on the Iris dataset, returning 3 datasets, train_iris, validate_iris and test_iris.
# 
# 

# In[ ]:


# species


# Run the function on the Titanic dataset, returning 3 datasets, train_titanic, validate_titanic and test_titanic.
# 
# 

# In[ ]:





# Run the function on the Telco dataset, returning 3 datasets, train_telco, validate_telco and test_telco.
# 
# 

# In[ ]:

def clean_titanic(df):
    """
    students - write docstring- no :)
    """
    #drop unncessary columns
    df = df.drop(columns=['embarked', 'age','deck', 'class'])
    
    #made this a string so its categorical
    df.pclass = df.pclass.astype(object)
    
    #filled nas with the mode
    df.embark_town = df.embark_town.fillna('Southampton')
    
    return df


def preprocess_telco(train_df, val_df, test_df):
    '''
    preprocess_telco will take in three pandas dataframes
    of our telco data, using the clean data from the 
    prepare only not the before 
    this will turn most of the colums into int types making
    it ready for the modling phase
    '''
    # with a looping structure:
    # go through the three dfs, set the index to customer id
    for df in [train_df, val_df, test_df]:
        df = df.set_index('customer_id')
        df['total_charges'] = df['total_charges'].astype(float)
    # initialize an empty list to see what needs to be encoded:
    encoding_vars = []
    # loop through the columns to fill encoded_vars with appropriate
    # datatype field names
    for col in train_df.columns:
        if train_df[col].dtype == 'O':
            encoding_vars.append(col)
    encoding_vars.remove('customer_id')
    encoding_vars.remove('total_charges')
    # initialize an empty list to hold our encoded dataframes:
    encoded_dfs = []
    for df in [train_df, val_df, test_df]:
        df_encoded_cats = pd.get_dummies(
            df[encoding_vars],
              drop_first=True).astype(int)
        encoded_dfs.append(pd.concat(
            [df,
            df_encoded_cats],
            axis=1).drop(columns=encoding_vars))
    return encoded_dfs
        
    

def compute_class_metrics(y_train, y_pred):

                counts = pd.crosstab(y_train, y_pred)
                TP = counts.iloc[1,1]
                TN = counts.iloc[0,0]
                FP = counts.iloc[0,1]
                FN = counts.iloc[1,0]


                all_ = (TP + TN + FP + FN)

                accuracy = (TP + TN) / all_

                TPR = recall = TP / (TP + FN)
                FPR = FP / (FP + TN)

                TNR = TN / (FP + TN)
                FNR = FN / (FN + TP)

                precision =  TP / (TP + FP)
                f1 =  2 * ((precision * recall) / ( precision + recall))

                support_pos = TP + FN
                support_neg = FP + TN

                print(f"Accuracy: {accuracy}\n")
                print(f"True Positive Rate/Sensitivity/Recall/Power: {TPR}")
                print(f"False Positive Rate/False Alarm Ratio/Fall-out: {FPR}")
                print(f"True Negative Rate/Specificity/Selectivity: {TNR}")
                print(f"False Negative Rate/Miss Rate: {FNR}\n")
                print(f"Precision/PPV: {precision}")
                print(f"F1 Score: {f1}\n")
                print(f"Support (0): {support_pos}")
                print(f"Support (1): {support_neg}")

    


