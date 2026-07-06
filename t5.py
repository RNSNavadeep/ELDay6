import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
df=pd.read_csv("heart.csv")
print("first 20 row of the data:",df.head(20))
print("info of the data:",df.info())
print("null values in the data:",df.isnull().sum())
print("duplicated values in the data:",df.duplicated().sum())
print("shape of the data:",df.shape)
df.drop_duplicates(keep="first",inplace=True)
print("number of duplicated values after removing:",df.duplicated().sum())
print("unique values in the target column:",df["target"].unique())
print("shape of the data:",df.shape)
ss=StandardScaler()
for col in df:
  if col != "target":
    df[col]=ss.fit_transform(df[[col]])
print("data after scaling:",df.head())
plt.boxplot(df)
plt.xticks(rotation=90)
plt.show()
sns.heatmap(df.corr(),annot=True)
plt.show()
x=df.drop("target",axis=1)
y=df["target"]
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
dtm=DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=42)
dtm.fit(xtr,ytr)
dtmpred=dtm.predict(xte)
rfm=RandomForestClassifier(n_estimators=100,max_depth=3,criterion='gini',random_state=42)
rfm.fit(xtr,ytr)
rfmpred=rfm.predict(xte)
dt_train_acc = metrics.accuracy_score(ytr, dtm.predict(xtr))
dt_test_acc = metrics.accuracy_score(yte, dtm.predict(xte))
rf_train_acc = metrics.accuracy_score(ytr, rfm.predict(xtr))
rf_test_acc = metrics.accuracy_score(yte, rfm.predict(xte))
print("Decision Tree: Accuracy on training Data: {:.3f}".format(dt_train_acc))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(dt_test_acc))
print("Random Forest: Accuracy on training Data: {:.3f}".format(rf_train_acc))
print("Random Forest: Accuracy on test Data: {:.3f}".format(rf_test_acc))
print("for decision tree classifier")
dt_importance = pd.Series(
    dtm.feature_importances_,
    index=x.columns
)

print(dt_importance.sort_values(ascending=False))
print("for random forrest classifier")
rf_importance=pd.Series(
    rfm.feature_importances_,
    index=x.columns
)
print(rf_importance.sort_values(ascending=False))
dt_scores = cross_val_score(
    dtm,
    x,
    y,
    cv=5,
    scoring="accuracy"
)

print(dt_scores)
print("Decision Tree: Average Cross-Accounting {:.3f}".format(dt_scores.mean()))
rf_scores=cross_val_score(
    rfm,x,y,cv=5,scoring="accuracy"
)
print(rf_scores)
print("Random Forest: Average Cross-Accounting {:.3f}".format(rf_scores.mean())
)