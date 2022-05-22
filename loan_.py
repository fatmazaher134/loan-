from tkinter import Y

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('I://تطبيق machine//train_u6lujuX_CVtuZ9i (1).csv')
print(data.head())
data.info()
# print(data.nunique())

plt.figure()
sns.heatmap(data.corr(),annot=True,fmt='.2f')
data.plot('Loan_Status')
plt.show()
print(data.isna().any())
print(data.duplicated().sum())

# column loanamount
data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace=True)
# column credit_history
data['Credit_History'].fillna(data['Credit_History'].mode()[0],inplace=True)
# column gender
data['Gender'].fillna(data['Gender'].mode()[0],inplace=True)
# column loan_amount_term
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0],inplace=True)
# column dependents
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace=True)
# column self employed
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace=True)
# column married
data=data[data['Married'].notna()]

# print(data.info())

plt.figure(figsize=(10,6))
sns.boxplot(x='Gender',y='LoanAmount',hue='Education',data=data)

plt.figure(figsize=(10,6))
sns.boxplot(x='Gender',y='LoanAmount',hue='Married',data=data)

plt.figure()
sns.countplot(data['Education'])
plt.show()

# column output
data['Loan_Status'].replace('N',0)
data['Loan_Status'].replace('Y',1)
# print(data['Loan_Status'])


# change coulmns

data.replace({'Married':{'No':0,'Yes':1},
'Gender':{'Male':1,'Female':0},
'Self_Employed':{'No':0,'Yes':1},
'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},
'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)
data.replace({"Dependents":{'3+':'3'}},inplace=True)




# print(data.head())
# split the data into x and y
d=data.sample(n=200)
# print(np.shape(d))
y=d['Loan_Status']
x=d.drop(['Loan_Status','Loan_ID'],axis=1)

# print(x.head(5))
# split the data into traning and test
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)

# print(y_train)
# create the model
from sklearn.linear_model import LogisticRegression 
# train and create prediction
model=LogisticRegression()
model.fit(x_train,y_train,sample_weight=None)
predict=model.predict(x_test)

# calculate performance
from sklearn.metrics import classification_report
print(classification_report(y_test,predict))