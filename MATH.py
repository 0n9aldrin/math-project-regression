import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('3d_csv.csv')
print(df.head())

threedee = plt.figure().gca(projection='3d')
threedee.scatter(df['Win percentage'], df['Ace'], df['Return'])
threedee.set_xlabel('Win')
threedee.set_ylabel('Ace')
threedee.set_zlabel('Return')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

######################################## Data preparation #########################################

df = pd.read_csv('3d_csv.csv')

X = df[['Ace', 'Return']].values.reshape(-1,2)
Y = df['Win percentage']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(2, 25, 30)      # range of Ace
y_pred = np.linspace(27, 50, 30)  # range of Return
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

############################################## Evaluate ############################################

r2 = model.score(X, Y)

coef = model.coef_
intercept = model.intercept_
print(coef)
print(intercept)

############################################## Plot ################################################

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]



for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.plot([1.9], [47.5], [83.2], color='r', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Ace (%)', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_zlabel('Win (%)', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')


ax1.text2D(0.2, 0.32, '', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, '', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, '', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=27, azim=112)
ax2.view_init(elev=16, azim=-51)
ax3.view_init(elev=60, azim=165)

fig.tight_layout()




for ii in np.arange(0, 360, 1):
    ax1.view_init(elev=27, azim=ii)
    ax2.view_init(elev=16, azim=ii)
    ax3.view_init(elev=60, azim=ii)
    fig.savefig('gif_image%d.png' % ii)
    print(ii)



ax1.set_title('$R^2 = %.2f$' % r2, fontsize=20)
fig.suptitle('z = ' + str(round(coef[0], 2)) + 'x + ' + str(round(coef[1], 2)) + 'y ' + str(round(intercept, 2)), fontsize=20)

fig.tight_layout()

import statsmodels.api as sm
model = sm.OLS(Y, X).fit()
print(model.summary())











af = pd.read_csv('ALL.csv')
print(af.head())
plt.figure()
plt.scatter(af['Win percentage'], af['Return'])
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()


plt.scatter(af['Return'],af['Win percentage'])

plt.title('Win % vs return %')

plt.xlabel('Return')
plt.ylabel('Win')

plt.show()

ex = af['Return'].values.reshape(-1,1)
why = af['Win percentage']

# try_y = 4 + 1.3 * ex

plt.scatter(ex, why, color = 'blue')
# plt.scatter(36.56, 51.56, color = 'red')
# plt.plot(ex, try_y, color = 'k')
plt.title('Return % vs Win %')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()



# Plot for simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
happy = lin_reg.fit(ex, why)
why_pred1 = lin_reg.predict(ex)

plt.scatter(ex, why, color = 'blue')
plt.plot(ex, lin_reg.predict(ex), color = 'k')
plt.title('Return % vs Win %')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.scatter(36.56, 51.56, color = 'red')
plt.show()


print(happy.coef_)
print(happy.intercept_)

# Plot for poly linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(ex)
lin_reg_2 = LinearRegression()
what = lin_reg_2.fit(X_poly, why)
why_pred = lin_reg_2.predict(X_poly)

print(what.coef_)
print(what.intercept_)

X_grid = np.arange(min(ex), max(ex), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ex, why, color = 'blue')
plt.scatter(36.56, 51.56, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.metrics import r2_score
print(r2_score(why, why_pred))
print(r2_score(why, why_pred1))



log_ex = af['Return'].values.reshape(-1,1)
log_why = af['Win percentage'].values.reshape(-1,1)
#Logarithmic

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer



# # General Functions
# def func_exp(x, a, b, c):
#     """Return values from a general exponential function."""
#     return a * np.exp(b * x) + c


# # Helper
# def generate_data(func, *args, jitter=0):
#     """Return a tuple of arrays with random data along a general function."""
#     xs = np.linspace(1, 5, 50)
#     ys = func(xs, *args)
#     noise = jitter * np.random.normal(size=len(xs)) + jitter
#     xs = xs.reshape(-1, 1)                                  # xs[:, np.newaxis]
#     ys = (ys + noise).reshape(-1, 1)
#     return xs, ys

transformer = FunctionTransformer(np.log, validate=True)

# Data
x_samp = log_ex
y_samp = transformer.fit_transform(log_why)

# Regression
regressor = LinearRegression()
results = regressor.fit(x_samp, y_samp)                # 2
model = results.predict
y_fit = model(x_samp)


print(results.coef_)
print(results.intercept_)

# Visualization
plt.scatter(log_ex, log_why)
plt.plot(log_ex, np.exp(y_fit), "k", label="Fit")     # 3
plt.title("Exponential Fit")



# try_log_y = 0.00000001405*np.exp(0.5063*X_grid) + 48.55
log_pred = 0.00000001405*np.exp(0.5063*X_grid) + 48.55
loggy = 0.00000001405*np.exp(0.5063*ex) + 48.55

# try_log_y = 20.42306*np.power(1.02513, X_grid)

plt.scatter(ex, why, color = 'blue')
plt.scatter(36.56, 51.56, color = 'red')
plt.plot(X_grid, log_pred, color = 'k')
plt.title('Return % vs Win % (Exponential)')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()

print(r2_score(why, loggy))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

import scipy
file = pd.read_csv('ALL.csv')

r = []
a = scipy.stats.pearsonr(file['Return'], file['Win percentage'])
r.append(a[0])
a = scipy.stats.pearsonr(file['Ace'], file['Win percentage'])
r.append(a[0])
a = scipy.stats.pearsonr(file['DF'], file['Win percentage'])
r.append(a[0])
a = scipy.stats.pearsonr(file['Serve Percentage'], file['Win percentage'])
r.append(a[0])

plt.scatter(file['Return'], file['Win percentage'], color='blue')
plt.title('Return % vs Win %')
# plt.text(28, 80, 'r = ' + str(r[0]), fontsize=12)
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()

plt.scatter(file['Ace'], file['Win percentage'], color='blue')
plt.title('Ace % vs Win %')
plt.text(12.5, 80, 'r = ' + str(r[1]), fontsize=12)
plt.xlabel('Ace %')
plt.ylabel('Win %')
plt.show()

plt.scatter(file['DF'], file['Win percentage'], color='blue')
plt.title('Double Fault % vs Win %')
plt.text(5, 80, 'r = ' + str(r[2]), fontsize=12)
plt.xlabel('Double Fault %')
plt.ylabel('Win %')
plt.show()

plt.scatter(file['Serve Percentage'], file['Win percentage'], color='blue')
plt.title('First serve % vs Win %')
plt.text(51.5, 75.5, 'r = ' + str(r[3]), fontsize=12)
plt.xlabel('First Serve %')
plt.ylabel('Win %')
plt.show()

# Plot for Line of best fit
ex = file['Return'].values.reshape(-1,1)
why = file['Win percentage']

# random_x = np.linspace(27.5, 48, 30)
# reshaped_x = random_x.reshape(-1,1)
try_y = 4 + 1.3 * ex

plt.scatter(ex, why, color = 'blue')
plt.scatter(36.56, 51.56, color = 'red')
plt.scatter(41.9, 77.4, color = 'c')
plt.plot(ex, try_y, color = 'k')
plt.title('Return % vs Win % (Eye)')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()

# Plot for simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
happy = lin_reg.fit(ex, why)
why_pred1 = lin_reg.predict(ex)

plt.scatter(ex, why, color = 'blue')
plt.plot(ex, lin_reg.predict(ex), color = 'k')
plt.scatter(41.9, 77.4, color = 'c')
plt.title('Return % vs Win % (Linear regression)')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.scatter(36.56, 51.56, color = 'red')
plt.show()

#Plot for poly regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(ex)
lin_reg_2 = LinearRegression()
what = lin_reg_2.fit(X_poly, why)
why_pred = lin_reg_2.predict(X_poly)

X_grid = np.arange(min(ex), max(ex), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(ex, why, color = 'blue')
plt.scatter(36.56, 51.56, color = 'red')
plt.scatter(41.9, 77.4, color = 'c')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('Return % vs Win % (Polynomial Regression)')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()

#Plot for exponential regression
log_pred = 0.00000001405*np.exp(0.5063*X_grid) + 48.55
loggy = 0.00000001405*np.exp(0.5063*ex) + 48.55

plt.scatter(ex, why, color = 'blue')
plt.scatter(36.56, 51.56, color = 'red')
plt.scatter(41.9, 77.4, color = 'c')
plt.plot(X_grid, log_pred, color = 'k')
plt.title('Return % vs Win % (Exponential)')
plt.xlabel('Return %')
plt.ylabel('Win %')
plt.show()

#Plot for multiple linear regression

######################################## Data preparation #########################################


X = file[['Ace', 'Return']].values.reshape(-1,2)
Y = file['Win percentage']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(2, 23, 30)      # range of Ace
y_pred = np.linspace(27, 43, 30)  # range of Return
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

############################################## Plot ################################################

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]



for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.plot([8.7], [41.9], [77.4], color='r', zorder=15, linestyle='none', marker='s', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Ace (%)', fontsize=12)
    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_zlabel('Win (%)', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')


ax1.text2D(0.2, 0.32, '', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, '', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, '', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=27, azim=112)
ax2.view_init(elev=16, azim=-51)
ax3.view_init(elev=60, azim=165)

fig.tight_layout()




plt.style.use('default')

ax1 = plt.figure(figsize=(4, 4))

ax1 = fig.add_subplot(projection='3d')


ax1.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
ax1.plot([8.7], [41.9], [77.4], color='r', zorder=15, linestyle='none', marker='s', alpha=0.5)
ax1.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
ax1.set_xlabel('Ace (%)', fontsize=12)
ax1.set_ylabel('Return (%)', fontsize=12)
ax1.set_zlabel('Win (%)', fontsize=12)
ax1.locator_params(nbins=4, axis='x')
ax1.locator_params(nbins=5, axis='x')



ax1.text2D(0.2, 0.32, '', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)


for ii in np.arange(80, 180, 1):
    ax1.view_init(elev=27, azim=ii)
    fig.savefig('gif_image%d.png' % ii)
    print(ii)
    
# ax1.view_init(elev=27, azim=150)


fig.tight_layout()
