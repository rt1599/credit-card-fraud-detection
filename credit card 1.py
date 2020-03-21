import numpy as np
import pandas as pd
import matplotlib.pyplot as pit
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
data=pd.read_csv('C:/Users/rishbh/Desktop/machine learning/creditcard.csv')
#grab a peek at the data
print(data.head())
#describe the data
print(data.columns)
print(data.shape)
print(data.describe())
#show graphs of datasets 
data.hist(figsize=(20,20))
#pit.show()
#determine the no of fraudalent cases
Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]
outlier_fraction=len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('fraud cases:{}' . format(len(Fraud)))
print('valid cases:{}' . format(len(Valid)))
print('%Amount details of the fraudulent transaction') 
print(Fraud.Amount.describe())
print('%details of valid transaction') 
print(Valid.Amount.describe()) 
#correlation matrix
corrmat=data.corr()
fig=pit.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
pit.show()
'''#get all the columns from the dataframe
columns=data.columns.tolist()
#filter the columns to remove the data we do not want
columns=[c for c in columns if c not in ['Class']]
#store the variables we will be predicting on
target ='Class'
x=data[columns]
y=data[target]
#print the shape of x and y
print(x.shape)
print(y.shape)
#define a random state
state=1
#define the outlier detection method
classifiers={
    "Isolation Forest":IsolationForest(max_samples=len(x),contamination=outlier_fraction,random_state=state),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)
    }
#fill the model
n_outlier=len(Fraud)
for i, (clf_name,clf)in enumerate(classifiers.items()):
    #fill the data and tag outliers
    if clf_name=="Local Outlier Factor":
        y_pred=clf.fit_predict(x)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred=clf.decision_function(x)
        y_pred=clf.predict(x)


#reshape the prediction value to 0 for valid,1 for fraud
y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1
n_errors=(y_pred!=y).sum()
#run classification matrices
print('{}:{}'. format(clf_name,n_errors))
print(accuracy_score(y,y_pred))
print(classification_report(y,y_pred))

'''

