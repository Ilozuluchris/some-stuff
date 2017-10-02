import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from first import X_train,X_test,y_train,y_test
from first import scoring_SN,X_scoring

print scoring_SN.head()
print X_scoring.head()

'''
l1_space = np.linspace(0,1,100)
param_grid = {'l1_ratio':l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet(normalize=True)

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net,param_grid,cv=10)

# Fit it to the training data
gm_cv.fit(X_train,y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test,y_pred)
mse = mean_squared_error(y_pred,y_test)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
#print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet RMSE: {}".format(np.sqrt(mse)))
'''