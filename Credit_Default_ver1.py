#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time ,datetime
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# In[2]:


train = pd.read_csv('UCI_Credit_Card.csv')

# In[3]:


train.isnull().sum()

# In[4]:


train.head()

# In[5]:


# pay0 should be renamed as pay1 and default.payment.next.month as Default_pay
train.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
train.rename(columns={'default.payment.next.month': 'Default_pay'}, inplace=True)
train.head(3)
train['EDUCATION'].replace([0, 6], 5, inplace=True)
train['MARRIAGE'].replace(0, 3, inplace=True)

# In[6]:


train.describe().T

# In[7]:


# X = train.drop(columns = ['ID','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','Default_pay'])
X = train.drop(columns=['ID', 'Default_pay'])
y = train['Default_pay']

# In[8]:


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
X_scaled.shape
std_train = pd.DataFrame(X_scaled)
std_train
train_data = std_train.rename(
    columns={0: 'LIMIT_BAL', 1: 'SEX', 2: 'EDUCATION', 3: 'MARRIAGE', 4: 'AGE', 5: 'PAY_1', 6: 'PAY_2', 7: 'PAY_3',
             8: 'PAY_4', 9: 'PAY_5', 10: 'PAY_6', 11: 'BILL_AMT1', 12: 'BILL_AMT2', 13: 'BILL_AMT3', 14: 'BILL_AMT4',
             15: 'BILL_AMT5', 16: 'BILL_AMT6', 17: 'PAY_AMT1', 18: 'PAY_AMT2', 19: 'PAY_AMT3', 20: 'PAY_AMT4',
             21: 'PAY_AMT5', 22: 'PAY_AMT6'})
train_data.head(2)

# In[9]:





def vif_score(x):
    scalar = StandardScaler()
    arr = scalar.fit_transform(x)
    return pd.DataFrame([[x.columns[i], variance_inflation_factor(arr, i)] for i in range(arr.shape[1])],
                        columns=["Feature", "VIF Score"])


vif_score(X)

# In[10]:


train_data.head(2)
y.head(2)

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(train_data, y, test_size=0.25, random_state=355)


# In[12]:


# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, train_data, Y_train, cv):
    # One Pass
    model = algo.fit(train_data, Y_train)
    acc = round(model.score(train_data, Y_train) * 100, 2)

    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo,
                                                   train_data,
                                                   Y_train,
                                                   cv=cv,
                                                   n_jobs=-1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(Y_train, train_pred) * 100, 2)

    return train_pred, acc, acc_cv


# In[13]:


# Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(),
                                                  x_train,
                                                  y_train,
                                                  10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))

# In[14]:


# Decision Tree Classifier
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = fit_ml_algo(DecisionTreeClassifier(),
                                               x_train,
                                               y_train,
                                               10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))

# In[15]:


# Gradient Boosting Trees
start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = fit_ml_algo(GradientBoostingClassifier(),
                                                  x_train,
                                                  y_train,
                                                  10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))

# In[16]:


# Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = fit_ml_algo(GaussianNB(),
                                                                 x_train,
                                                                 y_train,
                                                                 10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))

# In[17]:


# Stochastic Gradient Descent
start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd = fit_ml_algo(SGDClassifier(),
                                                  x_train,
                                                  y_train,
                                                  10)
sgd_time = (time.time() - start_time)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))

# In[18]:


# Random Forest Classifier
start_time = time.time()
train_pred_rfc, acc_rfc, acc_cv_rfc = fit_ml_algo(RandomForestClassifier(),
                                                  x_train,
                                                  y_train,
                                                  10)
rfc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_rfc)
print("Accuracy CV 10-Fold: %s" % acc_cv_rfc)
print("Running Time: %s" % datetime.timedelta(seconds=rfc_time))

# In[19]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Gradient Boosting Trees', 'Naive Bayes',
              'Stochastic Gradient Decent', 'Random Forest Classifier'],
    'Score': [acc_log, acc_dt, acc_gbt, acc_gaussian, acc_sgd, acc_rfc]})
print("---Regular Accuracy Scores---")
models.sort_values(by='Score', ascending=False)

# In[20]:


cv_models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Gradient Boosting Trees', 'Naive Bayes',
              'Stochastic Gradient Decent', 'Random Forest Classifier'],
    'Score': [acc_cv_log, acc_cv_dt, acc_cv_gbt, acc_cv_gaussian, acc_cv_sgd, acc_cv_rfc]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)

# here, I will choose Random Forest Classifier as it's performing good

# In[21]:


test_error = []
for n in range(1, 20):
    # Use n random trees
    model = RandomForestClassifier(n_estimators=n, max_features='auto')
    model.fit(x_train, y_train)
    test_preds = model.predict(x_test)
    test_error.append(1 - accuracy_score(test_preds, y_test))

# Clearly there are diminishing returns, on such a small dataset, we've pretty much extracted all the information we can after about 2 trees

# In[23]:


n_estimators = [100, 200]
max_features = [6, 10, 12]
bootstrap = [True, False]
param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'bootstrap': bootstrap}
rfc = RandomForestClassifier()
grid = GridSearchCV(rfc, param_grid)

# In[24]:


grid.fit(x_train, y_train)

# In[25]:


grid.best_params_

# here, we got best parameter so fitting model on best parameters

# In[26]:


rfClassifier = RandomForestClassifier(bootstrap=True,
                                      max_depth=4,
                                      min_samples_leaf=100,
                                      min_samples_split=200,
                                      max_features=10,
                                      n_estimators=100).fit(x_train, y_train)

# In[27]:


predic = rfClassifier.predict(x_test)

# In[28]:


predic = grid.predict(x_test)

# In[29]:


print(classification_report(y_test, predic))

# In[30]:


print(confusion_matrix(y_test, predic))

# In[31]:


print(accuracy_score(y_test, predic))

# In[34]:


import pickle

filename = 'model.pkl'
pickle.dump(rfClassifier, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
model_score = load_model.score(x_test, y_test)

