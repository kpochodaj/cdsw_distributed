# DASK - import helper libraries
import cdsw_dask_utils
import cdsw

# DASK - Run a Dask cluster with three workers and return an object containing
# a description of the cluster. 
# 
# Using helper library 
#
# Note that the scheduler will run in the current session, and the Dask
# dashboard will become available in the nine-dot menu at the upper
# right corner of the CDSW app.

cluster = cdsw_dask_utils.run_dask_cluster(
  n=2, \
  cpu=1, \
  memory=2, \
  nvidia_gpu=0
)

# DASK - Get the Dask Scheduler UI
import os 
engine_id = os.environ.get('CDSW_ENGINE_ID')
cdsw_domain = os.environ.get('CDSW_DOMAIN')

from IPython.core.display import HTML
HTML('<a  target="_blank" rel="noopener noreferrer" href="http://read-only-{}.{}">http://read-only-{}.{}</a>'
     .format(engine_id,cdsw_domain,engine_id,cdsw_domain))

# DASK - Connect a Dask client to the scheduler address in the cluster

from dask.distributed import Client
# client = Client(cluster["scheduler_address"])
client = Client(cluster["scheduler_address"])

# END OF DASK SET-UP, search for DASK to see the next bit

# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# READ SOURCE DATA
df = pd.read_csv('data/application_train.csv')
print('Training data shape: ', df.shape)
df.head()

# PRE-PROCESS DATA
# drop SK_ID_CURR
df = df.drop(df.columns[0], axis=1)

# EDA
df['TARGET'].value_counts()
df['TARGET'].astype(int).plot.hist();
df.dtypes.value_counts()

# ONE-HOT ENCODING
df = pd.get_dummies(df)
df.head()

# SPLIT DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split

y = df.take([0], axis=1)
X = df.drop(df.columns[0], axis=1)

y.head()
X.head()

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42)

# IMPUTE MISSING VALUES
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

X_train.head()

# RANDOM FOREST
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# model parameters
model_params = {
  'n_estimators' : [20, 30]
}

# create random forest classifier model
rf = RandomForestClassifier(random_state=1)

# set up grid search meta-estimator
clf = GridSearchCV(rf, model_params, cv=3)

# DASK - TRAIN MODEL
# ### Fit Model with Dask
from joblib import Parallel, parallel_backend
with parallel_backend('dask'):
  model = clf.fit(X_train, y_train)
  
## Optional - print dask cluster config 

import json
print(json.dumps(client.scheduler_info(), indent=4))

## stop CDSW workers
import cdsw
cdsw.stop_workers()

# print winning set of hyperparameters
from pprint import pprint
pprint(model.best_estimator_.get_params())

# generate predictions using the best-performing model
predictions = model.predict(X_test)
print(predictions)






