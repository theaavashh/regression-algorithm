## import requirement libraries
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

## making random dataset with one independent and dependent feature
X,y=make_regression(n_samples=30,n_features=1,n_targets=1,random_state=3)


# spliting the dataset in training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)


## class for simple linear regression
class SimpleLinear:

 ## constructor
 def __init__(self):
   self.coeff_ = 0
   self.intercept = 0

## training function
 def fit(self,X_train,y_train):
   x_mean_value,y_mean_value=0,0

   for values in X_train:
     x_mean_value+=values
    
   x_mean_value=x_mean_value/len(X_train)

   for values in y_train:
     y_mean_value+=values

   y_mean_value=y_mean_value/len(y_train)
     

   numerator=0
   denomenator=0
 
   for x,y in zip(X_train,y_train):
     numerator+=(x-x_mean_value)*(y-y_mean_value)
     denomenator+=(x-x_mean_value)**2

   self.coeff_=numerator/denomenator
   self.intercept=y_mean_value - self.coeff_*x_mean_value
     
## prediction function
 def predict(self,X_test):
   X_test=X_test.ravel()
   y_predict=self.coeff_*X_test + self.intercept
   return y_predict
 

## performance metrices
 def mean_square_error(self,y_true,y_predict):
   mean_square=0
   n=y_true.shape[0]
   
   for y_true,y_pred in zip(y_true,y_predict):
    mean_square+=(y_true - y_pred)**2

   print(mean_square/n)
    
   

   
    



   



   
   

simple_linear=SimpleLinear()
simple_linear.fit(X_train,y_train)
y_predict=simple_linear.predict(X_test)
print(y_predict)

simple_linear.mean_square_error(y_test,y_predict)

