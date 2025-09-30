# load and import packages
from matplotlib.pyplot import subplots
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from patsy import dmatrices

import sklearn.linear_model as skl
import sklearn.model_selection as skm 
from sklearn.preprocessing import StandardScaler
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from ISLP.models import \
(Stepwise ,
sklearn_selected ,
sklearn_selection_path)
from statsmodels.api import OLS
from functools import partial
from sklearn.pipeline import Pipeline

def RMSE(y,yhat):
    mse=np.mean((y-yhat)**2)
    return mse**(1/2)

# load the data
data = pd.read_csv('data/diabetes.csv')

# train and test
rng=np.random.default_rng(1315)
train_ind = rng.choice(data.index, size=int(len(data)/2), replace=False)
data_train=data.loc[train_ind]
data_test=data.drop(data_train.index).reset_index(drop=True)

d_train = MS(data_train.columns.drop('y')).fit_transform(data_train)
d_test=MS(data_train.columns.drop('y')).fit_transform(data_test)

# training / test data for elasticnet
# note: elasticnet will add an intercept, therefore we remove it here
x_train=np.asarray(d_train.drop('intercept',axis=1))
y_train=np.asarray(data_train['y'])
x_test=np.asarray(d_test.drop('intercept',axis=1))
y_test=np.asarray(data_test['y'])

# forward seletion
def negAIC(estimator, X, Y):
    "Negative AIC"
    n, p = X.shape
    Yhat = estimator.predict(X)
    MSE = np.mean((Y - Yhat)**2)
    return n + n * np.log(MSE) + 2 * (p + 1)
def nCp(sigma2 , estimator , X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat)**2)
    return -(RSS + 2 * p * sigma2) / n

d_train_tmp = MS(data_train.columns.drop('y')).fit(data_train)
strategy = Stepwise.first_peak(d_train_tmp,direction='forward',max_terms=len(d_train_tmp.terms))

forward_aic=sklearn_selected(OLS,strategy,scoring=negAIC)
forward_aic.fit(d_train,y_train)
forward_aic.selected_state_

# test error
RMSE(y_test, forward_aic.predict(d_test))

# lasso path and trace plot

## scaling (manual or using scaler)
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
Xs = x_train - x_train.mean(0)[None ,:]
X_scale = x_train.std(0)
Xs = Xs / X_scale[None ,:]
Xs[0:2,0:3]
x_train_s[0:2,0:3]

lambdas, soln_array = skl.ElasticNet.path(x_train_s,y_train,l1_ratio=1,n_alphas=100)[0:2]
soln_array.shape

soln_path = pd.DataFrame(soln_array.T,columns=d_train.columns.drop("intercept"),index=-np.log(lambdas))
path_fig, ax=subplots(figsize=(8,8))
soln_path.plot(ax=ax,legend=False)
ax.set_xlabel('$-\log(\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
path_fig.show()

## same results using pipeline
beta_hat=soln_path.iloc[39]
beta_hat.head(n=10)
(lambdas[39], np.linalg.norm(beta_hat))
np.sqrt(np.sum(beta_hat**2))

## same results but using pipeline
lasso = skl.ElasticNet(alpha=lambdas[39], l1_ratio=1)
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])
pipe.fit(x_train, y_train)
(lasso.alpha,np.linalg.norm(lasso.coef_))

# Parameter tuning by estimating the test error
validation = skm.ShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
param_grid={'lasso__alpha':lambdas}
grid = skm.GridSearchCV(pipe,
                        param_grid,
                        cv=validation,
                        scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
grid.best_params_['lasso__alpha'] 
grid.best_score_

## test error plot
lasso_fig, ax = subplots(figsize=(8,8))
ax.plot(-np.log(lambdas),-grid.cv_results_['mean_test_score'])
ax.set_xlabel('$-\log(\lambda)$',fontsize=20) 
ax.set_ylabel('Test MSE',fontsize=20)
lasso_fig.show()

# Run elasticnet pipeline: scaling and lasso
cvfolds=10
sc = StandardScaler()
lassoCV = skl.ElasticNetCV(n_alphas=100,l1_ratio=1,cv=cvfolds)
pipeCV = Pipeline(steps=[('scaler', sc),('lasso', lassoCV)])
pipeCV.fit(x_train, y_train)

# cv plot
tuned_lasso = pipeCV.named_steps['lasso']
lassoCV_fig , ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(tuned_lasso.alphas_),tuned_lasso.mse_path_.mean(1),yerr=tuned_lasso.mse_path_.std(1) / np.sqrt(cvfolds))
ax.axvline(-np.log(tuned_lasso.alpha_), c='k', ls='--')
lassoCV_fig.show()

np.min(tuned_lasso.mse_path_.mean(1))

# test error
pred=pipeCV.predict(x_test)
RMSE(y_test, pipeCV.predict(x_test))

# intermezzo on scaling
import statsmodels.api as sm 
from statsmodels.api import OLS 

## linear regression
fit0 = sm.OLS(y_train, MS(['age','sex','bmi']).fit_transform(d_train)).fit()
fit0.params
fit0.predict()[0:5]
pred=MS(['age','sex','bmi']).fit_transform(d_train)@fit0.params
pred[0:5]

## lasso without scaling
lasso = skl.ElasticNet(alpha=0, l1_ratio=1)
fit1=lasso.fit(x_train[:,0:3],y_train)
(fit1.intercept_,fit1.coef_)
pred1=lasso.intercept_+x_train[:,0:3]@lasso.coef_
pred1[0:5]
lasso.predict(x_train[:,0:3])[0:5]

## lasso with manual scaling
scales=x_train.std(0)[0:3]
means=x_train.mean(0)[0:3]
fit2=lasso.fit(x_train_s[:,0:3],y_train)
(fit2.intercept_,fit2.coef_)
(fit2.intercept_ - np.sum((fit2.coef_ * means) / scales),fit2.coef_/scales)

## lasso with pipeline
scaler = StandardScaler(with_mean=True,  with_std=True)
pipe = Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])
pipe.fit(x_train[:,0:3], y_train)
(lasso.intercept_,lasso.coef_)
lasso.predict(x_train[:,0:3])[0:5] ### wrong 
lasso.predict(x_train_s[:,0:3])[0:5] ### correct
pred3=lasso.intercept_+x_train_s[:,0:3]@lasso.coef_ ### correct
pred3[0:5]
pred4=pipe.predict(x_train[:,0:3])### correct
pred4[0:5]
