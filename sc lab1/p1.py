from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

data=input("enter the name of the dataset file:")

df=pd.read_csv(data)

number=LabelEncoder()
df['class']=number.fit_transform(df['class'].astype('str'))


X=df.iloc[:,:df.shape[1]-1]

Y=df.iloc[:,df.shape[1]-1]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=41)

def train(X_train,Y_train):
	weights=[1/(X_train.shape[1]+1) for i in range(X_train.shape[1])]
	loop=100
	rate=0.1
	while(loop!=0):
		for i in range(X_train.shape[0]):
			cr=list(X_train.iloc[i])
			y=Y_train.iloc[i]
			cost=0
			for j in range(len(cr)):
				cost+=weights[j]*cr[j]
			if(cost>0):
				cost=1
			else:
				cost=0
			error=y-cost
			if(error!=0):
				for j in range(len(cr)):
					weights[j]+=rate*error*cr[j]
		loop=loop-1

	return weights

def testing(X_test,Y_test,weights):
	size=X_test.shape[0]
	count=0
	for i in range(X_test.shape[0]):
		cost=0
		cr=list(X_test.iloc[i])
		y=Y_test.iloc[i]
		for j in range(len(weights)):
			cost+=weights[j]*cr[j]
		if(cost>0):
			cost=1
		else:
			cost=0
		string="Expected:"+str(cost)+" Actual:"+str(y)
		print(string)
		error=y-cost
		if(error==0):
			count+=1
	return float(count/size)
w=train(X_train,Y_train)
accuracy=testing(X_test,Y_test,w)
print("Accuracy is:")
print(accuracy)



