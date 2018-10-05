from __future__ import division
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math
kk=3
class heapobj:

	def __init__(self,dis,y):
		self.eq_dist=dis
		self.y_val=y


def predict(X_train,X_test,Y_train,Y_test):
	truep=0.01
	truen=0.01
	falsep=0.01
	falsen=0.01
	for i in range(X_test.shape[0]):
		L=[]
		KN=[]
		test_row=list(X_test.iloc[i])
		actual=Y_test.iloc[i]
		dist=0
		freq0=0
		
		for j in range(X_train.shape[0]):
			curr_row=list(X_train.iloc[j])
			val=Y_train.iloc[j]
			for k in range(len(curr_row)):
				dist+=pow((curr_row[k]-test_row[k]),2)
			dist=math.sqrt(dist)
			L.append(heapobj(dist,val))
		L=sorted(L,key=lambda x:x.eq_dist)
		for m in range(kk):
			if L[m].y_val==0:
				freq0=freq0+1
		if freq0>=kk//2:
			pred=0
		else:
			pred=1

		if(pred==actual and pred==1):
			truep+=1
		if(pred!=actual and pred==1):
			falsep+=1
		if(pred!=actual and pred==0):
			falsen+=1
		if(pred==actual and pred==0):
			truen+=1
	return (truep+truen)/X_test.shape[0],(truep/(truep+falsep)),(truep/(truep+falsen))

df=pd.read_csv("SPECT.csv")

number=LabelEncoder()
df['class']=number.fit_transform(df['class'].astype('str'))


X=df.iloc[:,:df.shape[1]-1]

Y=df.iloc[:,df.shape[1]-1]

part=int(df.shape[0]/10)
acc=0
prec=0
rec=0
for i in range(10):
	X_test=X.iloc[i*part:(i+1)*part,:]
	Y_test=Y.iloc[i*part:(i+1)*part]
	X_train=X
	Y_train=Y
	accx,precx,recx=predict(X_train,X_test,Y_train,Y_test)
	acc+=accx
	prec+=precx
	rec+=recx
print("Accuracy is:{} \nPrecision is:{}\nRecall is:{}".format(acc/10,prec/10,rec/10))