# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Load & Preprocess: Read placement CSV, drop "sl_no", "salary", encode categorical columns.
2. Split Data: Extract features (X) and target (y), split into train/test (20% test).
3. Train & Predict: Train Logistic Regression model, predict test set outcomes.
4. Evaluate: Compute accuracy, confusion matrix, print classification report, predict sample.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Hari Nivedhan P
RegisterNumber: 212224220031

import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)#removes the specified row or column
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) , True
#accuracy_score(y_true, y_pred, normalize=False)
#Normalize : It contains the boolean value(True/False).If False, return the number of cor
#Otherwise, it returns the fraction of correctly confidential samples.
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions, 5+3=8 incorrect predictions
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
 
*/
```

## Output:
![Screenshot 2025-05-12 101535](https://github.com/user-attachments/assets/d44d37f6-4c73-4e5a-b9cb-734763b14d00)
![Screenshot 2025-05-12 101557](https://github.com/user-attachments/assets/abe15dd2-2a0f-4c29-ba0a-60a0b4a7e87b)
![Screenshot 2025-05-12 102347](https://github.com/user-attachments/assets/dd3d6e49-5d91-49ac-830f-213b5c9ca36f)
![Screenshot 2025-05-12 102400](https://github.com/user-attachments/assets/deba6e7b-c6e0-4582-932c-b473d5484b36)
![Screenshot 2025-05-12 102414](https://github.com/user-attachments/assets/45162753-b0c0-448d-9239-e209f65758dd)
![Screenshot 2025-05-12 102424](https://github.com/user-attachments/assets/b995cbda-c7ad-491b-bbe1-6c0acd0a959a)
![Screenshot 2025-05-12 102431](https://github.com/user-attachments/assets/ada392bb-000a-4a9c-b69c-9075ae0a9d98)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
