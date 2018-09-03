from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)

data=input("enter the name of the dataset file:")

df=pd.read_csv(data)

number=LabelEncoder()
df['class']=number.fit_transform(df['class'].astype('str'))


X=df.iloc[:,:df.shape[1]-1]
Y=df.iloc[:,df.shape[1]-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1000)

model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X_train, Y_train, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print("The summary:")
model.summary()