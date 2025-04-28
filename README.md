# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries for numerical operations, data handling, and preprocessing.

2.Load the startup dataset (50_Startups.csv) using pandas.

3.Extract feature matrix X and target vector y from the dataset.

4.Convert feature and target values to float and reshape if necessary.

5.Standardize X and y using StandardScaler.

6.Add a column of ones to X to account for the bias (intercept) term.

7.Initialize model parameters (theta) to zeros.

8.Perform gradient descent to update theta by computing predictions and adjusting for error.

9.Input a new data point, scale it, and add the intercept term.

10.Predict the output using learned theta, then inverse-transform it to get the final result.
 
## Program
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sridharan J
RegisterNumber: 212222040158
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)

X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)

theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:

### Data information and value of X:
![image](https://github.com/user-attachments/assets/fa3f7c54-7fe2-4302-b311-701dd70c8f2d)
![image](https://github.com/user-attachments/assets/e66f2f32-d841-4002-a6ae-cbfd98b8c77d)

### Value of Y:
![image](https://github.com/user-attachments/assets/bccfc6fd-9682-4b59-88f6-b460d8dd8553)

### Value of X1 Scaled:
![image](https://github.com/user-attachments/assets/727e97a5-79ff-4f14-a8d8-6293e2eac037)

### Value of Y1 Scaled:
![image](https://github.com/user-attachments/assets/87b9461c-0339-48b4-84f3-2cc358837595)

### Predicted Value
![image](https://github.com/user-attachments/assets/f15b1528-6825-4b96-af30-87d31480a8b6)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
