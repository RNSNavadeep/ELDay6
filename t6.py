import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from yellowbrick.model_selection import ValidationCurve
df=pd.read_csv("Iris.csv")
print("first 20 rows of the data:",df.head(20))
print("info of the data:",df.info())
print("null values in the data:",df.isnull().sum())
print("duplicated values in the data:",df.duplicated().sum())
print("unique values in the target column:",df["Species"].unique())
print("shape of the data:",df.shape)
le=LabelEncoder()
df["Species"]=le.fit_transform(df[["Species"]])
print("data after label encoding:",df.head())
print("unique values in the target column after label encoding:",df['Species'].unique())
x=df.drop("Species",axis=1)
y=df["Species"]
plt.boxplot(x)
plt.xticks(rotation=90)
plt.show()
plt.hist(x)
plt.show()

ss=StandardScaler()
x_scaled=ss.fit_transform(x)
x=pd.DataFrame(x_scaled,columns=x.columns)
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
k=math.floor(math.sqrt(len(x)))
if k%2==0:
    k=k+1
model=KNeighborsClassifier(n_neighbors=k)
model.fit(xtr,ytr)
pred=model.predict(xte)
ac=accuracy_score(yte,pred)
cr=classification_report(yte,pred)
cm=confusion_matrix(yte,pred)
print("Accuracy Score:",ac)
print("Classification Report:\n",cr)
print("Confusion Matrix:\n",cm)
values=np.arange(1,25)
visual=ValidationCurve(model,
                      
                      param_name='n_neighbors',
                      param_range=values,
                       scoring='accuracy',
                       cv=5)
visual.fit(xtr,ytr)
visual.show()