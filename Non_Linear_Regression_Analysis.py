# Objectives¶
# After completing this lab you will be able to:

# Differentiate between linear and non-linear regression
# Use non-linear regression model in Python
# If the data shows a curvy trend, then linear regression will not produce very accurate results when compared to a non-linear regression since linear regression presumes that the data is linear. Let's learn about non linear regressions and apply an example in python. In this notebook, we fit a non-linear model to the datapoints corrensponding to China's GDP from 1960 to 2014.

# Importing required libraries

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Although linear regression can do a great job at modeling some datasets, it cannot be used for all datasets. First recall how linear regression, models a dataset. It models the linear relationship between a dependent variable y and the independent variables x. It has a simple equation, of degree 1, for example y =  2𝑥  + 3
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Non-linear regression is a method to model the non-linear relationship between the independent variables  𝑥  and the dependent variable  𝑦 . Essentially any relationship that is not linear can be termed as non-linear, and is usually represented by the polynomial of  𝑘  degrees (maximum power of  𝑥 ). For example:

#  𝑦=𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑 
 
# Non-linear functions can have elements like exponentials, logarithms, fractions, and so on. For example:
# 𝑦=log(𝑥)
 
# We can have a function that's even more complicated such as :
# 𝑦=log(𝑎𝑥3+𝑏𝑥2+𝑐𝑥+𝑑)
 
# Let's take a look at a cubic function's graph.
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# As you can see, this function has  𝑥3  and  𝑥2  as independent variables. Also, the graphic of this function is not a straight line over the 2D plane. So this is a non-linear function.

# Some other types of non-linear functions are:

# Quadratic¶
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Exponential¶
# An exponential function with base c is defined by
# 𝑌=𝑎+𝑏𝑐𝑋
 
# where b ≠0, c > 0 , c ≠1, and x is any real number. The base, c, is constant and the exponent, x, is a variable.
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

y= np.exp(x)
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Logarithmic
# The response  𝑦  is a results of applying the logarithmic map from the input  𝑥  to the output  𝑦 . It is one of the simplest form of log(): i.e.
# 𝑦=log(𝑥)
 
# Please consider that instead of  𝑥 , we can use  𝑋 , which can be a polynomial representation of the  𝑥  values. In general form it would be written as
# 𝑦=log(𝑋)
x = np.arange(-5.0, 5.0, 0.1)

y = np.log(x)
y_noise = 0.5* np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Sigmoidal/Logistic
x = np.arange(-5.0, 5.0, 0.1)
y = 1-4/(1+np.power(3, x-2))
y_noise = 0.5* np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# Non-Linear Regression example
# For an example, we're going to try and fit a non-linear model to the datapoints corresponding to China's GDP from 1960 to 2014. We download a dataset with two columns, the first, a year between 1960 and 2014, the second, China's corresponding annual gross domestic income in US dollars for that year.
import numpy as np
import pandas as pd

#downloading dataset
!wget -nv -O china_gdp.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv
    
df = pd.read_csv("china_gdp.csv")
df.head(10)

# Plotting the Dataset
# This is what the datapoints look like. It kind of looks like an either logistic or exponential function. The growth starts off slow, then from 2005 on forward, the growth is very significant. And finally, it decelerates slightly in the 2010s.
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
# plt.xlabel('Year')
plt.show()

# Choosing a model
# From an initial look at the plot, we determine that the logistic function could be a good approximation, since it has the property of starting with a slow growth, increasing growth in the middle, and then decreasing again at the end; as illustrated below:
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

# The formula for the logistic function is the following:

# $$ \hat{Y} = \frac1{1+e^{\beta\_1(X-\beta\_2)}}$$

# $\beta\_1$: Controls the curve's steepness,

# $\beta\_2$: Slides the curve on the x-axis.

The formula for the logistic function is the following:

$$ \hat{Y} = \frac1{1+e^{\beta\_1(X-\beta\_2)}}$$

$\beta\_1$: Controls the curve's steepness,

$\beta\_2$: Slides the curve on the x-axis.
The formula for the logistic function is the following:
​
$$ \hat{Y} = \frac1{1+e^{\beta\_1(X-\beta\_2)}}$$
​
$\beta\_1$: Controls the curve's steepness,
​
$\beta\_2$: Slides the curve on the x-axis.
# ​
# Building The Model
# Now, let's build our regression model and initialize its parameters.
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
   
#  Lets look at a sample sigmoid line that might fit with the data:
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Our task here is to find the best parameters for our model. Lets first normalize our x and y:
# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

#### How we find the best parameters for our fit line?

# we can use **curve_fit** which uses non-linear least squares to fit our sigmoid function, to data. Optimize values for the parameters so that the sum of the squared residuals of sigmoid(xdata, \*popt) - ydata is minimized.

# popt are our optimized parameters.
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Now we plot our resulting regression model.
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Practice¶
# Can you calculate what is the accuracy of our model?
# write your code here
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
