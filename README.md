# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the Iris dataset.

2.Split the dataset into training and testing data.

3.Apply feature scaling using StandardScaler.

4.Train the model using SGDClassifier with training data.

5.Predict the test data and evaluate accuracy, classification report, and confusion matrix.
 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: GAYATHRI S
RegisterNumber: 212224230073 
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Output:
<img width="622" height="401" alt="image" src="https://github.com/user-attachments/assets/4c28351f-ad4f-4706-b342-7148d1738493" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
