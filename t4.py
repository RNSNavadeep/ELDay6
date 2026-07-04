import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn import metrics
df=pd.read_csv('data.csv')
print("first five row of the data:",df.head())
print("shape of the data:",df.shape)
print("columns of the data:",df.columns)
print("info of the data:",df.info())
print("null values in the data:",df.isnull().sum())
df.drop("Unnamed: 32",axis=1,inplace=True)
print("number of duplicated values:",df.duplicated().sum())
le=LabelEncoder()
df["diagnosis"]=le.fit_transform(df[["diagnosis"]])
zval=zscore(df)
print("number of zval greater than 3:", (zval>3).sum())
#df=df[abs(zval)<3]
print("shape of the data after removing outliers:",df.shape)
plt.boxplot(df["radius_mean"])
plt.title("Boxplot of radius_mean")
plt.xlabel("radius_mean")
plt.ylabel("Values")
plt.show()
x=df.drop("diagnosis",axis=1)
y=df["diagnosis"]
ss=StandardScaler()
x=ss.fit_transform(x)
x=pd.DataFrame(x)
print("first five row of the data:",x.head())
print("first five row of the data:",y.head())
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
mo=LogisticRegression()
mo.fit(xtr,ytr)
pred=mo.predict(xte)
acc=metrics.accuracy_score(yte,pred)
predc=metrics.precision_score(yte,pred)
rec=metrics.recall_score(yte,pred)
con=metrics.confusion_matrix(yte,pred)
print("accuracy score:",acc)
print("precision score:",predc)
print("recall score:",rec)
print("confusion matrix:",con)
sns.heatmap(con,annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()