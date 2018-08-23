from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data=input("enter the name of the dataset file:")

df=pd.read_csv(data)

number=LabelEncoder()
df['class']=number.fit_transform(df['class'].astype('str'))


X=df.iloc[:,:df.shape[1]-1]

Y=df.iloc[:,df.shape[1]-1]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=41)

regr = linear_model.LinearRegression()
regr.fit(X_train,Y_train)
sc=regr.score(X_test,Y_test)
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test,y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test,y_pred))

print('Accuracy: \n', sc)





