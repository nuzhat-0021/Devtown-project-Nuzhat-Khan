import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
df=pd.read_csv("customer_retail.csv")
df=df.dropna()
df=df[['Quantity','UnitPrice','Country']]
encoder=LabelEncoder()
df['Country_encoded']=encoder.fit_transform(df['Country'])
plt.figure(figsize=(8,5))
plt.scatter(df['Quantity'],df['UnitPrice'])
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title('Scatter Plot of Quantity vs UnitPrice')
x=df[['Quantity','UnitPrice']]
y=df['Country_encoded']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Logistic Regression")
log_model=LogisticRegression(max_iter=100)
log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
accuracy1=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy1)
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
print("Decision Tree")
dt_model=DecisionTreeClassifier()
dt_model.fit(x_train,y_train)
y_pred=dt_model.predict(x_test)
accuracy2=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy2)
print(confusion_matrix(y_test,y_pred))
print("KNN")
knn_model=KNeighborsClassifier()
knn_model.fit(x_train,y_train)
y_pred=knn_model.predict(x_test)
accuracy3=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy3)
print(confusion_matrix(y_test,y_pred))
models=[
    'Logistic Regression',
    'Decision Tree',
    'KNN'
]
accuracies=[
    accuracy1,
    accuracy2,
    accuracy3
]
plt.figure(figsize=(8,5))
plt.bar(models,accuracies)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()