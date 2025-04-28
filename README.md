# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start

Step 2: Import Required Libraries
Import numpy for numerical operations.
Import pandas for data handling and manipulation.

Step 3: Load and Explore the Dataset
Load the dataset into a Pandas DataFrame using pd.read_csv().

Step 4: Data Preprocessing
Check for missing or null values and handle them if necessary (fill or remove).

Step 5: Split the Data

Step 6: Create and Train the Logistic Regression Model
Create an instance of LogisticRegression().

Step 7: Predict the Results

Step 8: Evaluate the Model
Calculate the Accuracy Score to measure the model's overall performance.

Step 9: Visualize Results (Optional but Recommended)

Step 10: Interpret Results

Step 11: End

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Manisha selvakumari.S.S.
RegisterNumber: 212223220055 
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("Placement_Data.csv")
print("Name: Manisha selvakumari.S.S.")
print("Reg No: 212223220055")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)

```

## Output:
![Screenshot (252)](https://github.com/user-attachments/assets/fc1c02ad-0a9e-4dc0-82b8-097937dd75c8)

![Screenshot (253)](https://github.com/user-attachments/assets/3c5b3ed6-e577-4000-a7aa-e967ff40563e)

![Screenshot 2025-04-28 221703](https://github.com/user-attachments/assets/ca79aff4-440c-460e-b1e3-6e386fad04b9)

![Screenshot 2025-04-28 221719](https://github.com/user-attachments/assets/ef4deb58-0667-4fa5-a1b1-a8332c273d79)

![Screenshot (254)](https://github.com/user-attachments/assets/a799e371-e84a-4374-9574-31dc2b767f7a)

![Screenshot 2025-04-28 222158](https://github.com/user-attachments/assets/7cc1195d-b25b-42c8-8d67-7fe798f479e7)

![Screenshot 2025-04-28 222203](https://github.com/user-attachments/assets/73b176a3-50f6-4527-b66d-65ed9a70f42a)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
