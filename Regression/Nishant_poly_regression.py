#polynomial regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
'''#splitting data set in two
from sklearn.model_selection import train_test_split 
#nowwe use model selection in place of sklearn.crossvalidation
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=0)'''
#fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
#fitting polynomial regressiont to dataset
#3 lines transforming matrix to X_poly
from sklearn.preprocessing import PolynomialFeatures#this automatically add coulmn of 1 
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)

lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)
#visualising linear results
plt.scatter(X,y,color='red')#real sallaries
plt.plot(X,lin_reg.predict(X),color='blue')#using linear reg
plt.title('Truth of bluff(linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
#visualising poly regression
plt.scatter(X,y,color='red')#real sallaries
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')#using poly reg
plt.title('Truth of bluff(poly regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()