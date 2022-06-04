# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name # set the region of the instance

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")



try:
   d= pd.read_csv('s3://mybucketmlproject/sample_data1.csv',index_col=0)
   print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
    
    
    d.head()
print(d.shape)
d.isnull().sum()
x=d.iloc[:,:31].values
y=d[['Result'].values
print(x.shape)
print(y.shape)
print(type(d))
print(type(x))
print(type(y))

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.40,random_state=4)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
print(ypred.shape)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred)*100)
