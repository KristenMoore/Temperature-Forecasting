import json
from pprint import pprint
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from scipy.optimize import leastsq
from sklearn import datasets, linear_model, metrics
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import pylab as pl
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.pipeline import make_pipeline


# Use 9 days of hourly observed weather/temperature data to predict the next 24 hours
# load & visualise data
indir = 'Data Scientist Exercise'
time_list=[] 
hour_list=[]
tempm_list=[]
humidity_list=[]
meantempm_list=[]
i=0;
for root, dirs, filenames in os.walk(indir):
    for filename in filenames:
        with open(os.path.join(root, filename)) as data_file:
             data = json.load(data_file)
             if len((data['history']['observations']))>0:
                 # neighborhood_list =(data['location']['nearby_weather_stations']['pws']['station']) # Don't think this is used, seems to be all SFO without station info
                 observation_list = (data['history']['observations'])
                 dailysummary_list= (data['history']['dailysummary'])
                 for m in dailysummary_list:
                     meantempm_list.append((m['meantempm']))
                 for o in observation_list:
                          time_list.append(int((o['date']['hour']))+24*i)  # quick-fix
                          hour_list.append(int((o['date']['hour'])))
                          tempm_list.append(float((o['tempm'])))
                          humidity_list.append(int((o['hum'])))
                          i+=1;
                          
plt.plot(time_list, tempm_list)
plt.title("Tempm by hour")
plt.xlabel("Time (hours)")
plt.ylabel("Temperature (tempm)")
plt.show()
day_list=range(1,len(meantempm_list)+1)
plt.plot(day_list, meantempm_list)
plt.title("Daily average tempm")
plt.xlabel("Day")
plt.ylabel("Mean Temperature")
plt.show()


# ----- As a first attempt, try fitting a sin function ----
def sin_func(x, a, b, c):
     return b*np.sin(np.divide(np.multiply(2*np.pi,x),24)+np.ones(len(time_list))*c) + a
fitted_sin, fitted_covariance = curve_fit(sin_func, time_list, tempm_list, p0=(np.mean(tempm_list),3*np.std(tempm_list)/(2**0.5),10))
fitted_sin_list=[]
for x in time_list:
    fitted_sin_list.append(sin_func(x,fitted_sin[0],fitted_sin[1],fitted_sin[2]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(time_list, tempm_list, color='red')
ax.plot(time_list, fitted_sin_list, color='green')
plt.show()

# ----- scikit-learn polynomial regression ------
regr = linear_model.LinearRegression(normalize=True,copy_X=True)
xx = np.array(time_list[0:24]).reshape(24,1)
print type(xx)
X=xx
for i in range(2,5):
   X=np.append(X, X**i, 1)
yy = np.array(tempm_list[0:24]).reshape(24,1)

regr.fit(X, yy)
print regr.score(X,yy)
pl.scatter(xx, yy,  color='black')
pl.plot(xx, regr.predict(X), color='blue',
        linewidth=3)
pl.title("Prediction using polynomial regression")
pl.xticks(())
pl.yticks(())
pl.show()
# ------ It looks like this overfit the data, so let's add regularization ----
coefs = []
alphas=[.001,.003,.006,.01,.03,.06,1]
for a in alphas:
    regularized_regr=linear_model.Ridge(copy_X=True)
    regularized_regr.set_params(alpha=a)
    coefs.append(metrics.mean_squared_error(yy,regularized_regr.fit(X, yy).predict(X)))
print coefs    



regularized_regr=linear_model.Ridge(alpha=.5,copy_X=True)
regularized_regr.fit(X,yy)
print metrics.mean_squared_error(yy,regularized_regr.predict(X))
print regularized_regr.score(X,yy)


EN_regularized_regr=linear_model.ElasticNet(alpha=0.5)
EN_regularized_regr.fit(X,yy)
print EN_regularized_regr.score(X,yy)
pl.scatter(xx, yy,  color='black')
pl.plot(xx, regr.predict(X), color='blue',
        linewidth=3)
pl.plot(xx, EN_regularized_regr.predict(X), color='green',linewidth=3)
pl.plot(xx, regularized_regr.predict(X), color='red',linewidth=3)
pl.xticks(())
pl.yticks(())
pl.show()

# ------ scikit-learn Decision Tree -----
clf_1 = DecisionTreeRegressor(max_depth=8)
clf_2 = DecisionTreeRegressor(max_depth=5)
clf_1.fit(xx, yy)
clf_2.fit(xx, yy)
# Predict & plot results
X_test = np.arange(0.0, 1500.0, 1)[:, np.newaxis]
y_1 = clf_1.predict(X_test)
y_2 = clf_2.predict(X_test)
import pylab as pl

pl.figure()
pl.scatter(xx, yy, c="k", label="data")
pl.plot(X_test, y_1, c="g", label="max_depth=2", linewidth=2)
pl.plot(X_test, y_2, c="r", label="max_depth=5", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Decision Tree Regression")
pl.legend()
pl.show()


# fit cubic polynomial to each day & plot them side by side
for k in range(0,9):
   z=np.polyfit(time_list[24*k:24*(k+1)], tempm_list[24*k:24*(k+1)], 3)
   def poly_func(x,a,b,c,d):
       return a+b*x+c*np.square(x)+d*np.power(x,3)
   fitted_poly_list=[]
   for x in time_list[24*k:24*(k+1)]:
       fitted_poly_list.append(poly_func(x,z[3],z[2],z[1],z[0]))
   ax = plt.subplot(9,1,k+1)
   ax.scatter(time_list[24*k:24*(k+1)], tempm_list[24*k:24*(k+1)], color='red')
   ax.plot(time_list[24*k:24*(k+1)], fitted_poly_list, color='green')
plt.show()


# Moving On To Time Series Analysis. Using ARMA: try (2,2) and (2,0)
arma_model22=sm.tsa.ARMA(tempm_list,(2,2)).fit()
arma_model20 =sm.tsa.ARMA(tempm_list, (2,0)).fit()
print arma_model22.params, arma_model22.aic, arma_model22.bic, arma_model22.hqic 
print arma_model20.params, arma_model20.aic, arma_model20.bic, arma_model20.hqic

# plot the residuals for ARMA(2,2) and ARMA(2,0)
resid20 = arma_model20.resid
resid22 = arma_model22.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(time_list,resid20, color='red')
ax.plot(time_list,resid22, color='green')
plt.title("Residual Plot")
plt.show()

# Check residuals
print stats.normaltest(resid20)  # check if residuals are normally distributed
print stats.normaltest(resid22)  # returns s^2 + k^2, and p-value. Here s=z-score for skewtest, k=z-score for kurtosistest, p=2-sided chi squared prob 
# Plot the residuals for the ARMA(2,2) model Vs quantiles
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid22, line='q', ax=ax, fit=True)
plt.title("Residuals Vs Quantiles")
plt.show()

# predict next 24 hours tempm and plot prediction
predict_time_list=range(0, 25)
predict_tempm = arma_model22.predict(224, 248, dynamic=True)
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(predict_time_list,predict_tempm, color='red')
plt.title("Pedicted tempm")
plt.show()
