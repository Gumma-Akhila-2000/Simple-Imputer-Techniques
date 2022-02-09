import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\G AKHILA\Desktop\Datasets\Machine Learning\Simple Imputer (Mean,Median & Mode)\Mean.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Missing value treatment using simple imputer (1st strategy - Mean)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

#--------------------------------------

# Fit & Transform methods - Filled Missing Values of Numerical Data

imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:,1:3])

#--------------------------------------

# Encoding Categorical Data using LABEL ENCODER Technique

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])

X[:,0] = labelencoder_X.fit_transform(X[:,0])

#--------------------------------------

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

#--------------------------------------

# Splitting Training & Testing Phase

from sklearn.model_selection import train_test_split

# testing the model with 20%
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0) 

#--------------------------------------

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
