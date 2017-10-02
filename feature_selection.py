from sklearn.feature_selection import RFE
from first import X,y,X_train,X_test,y_train,y_test
from first import scoring_SN,X_scoring
from sklearn.linear_model import BayesianRidge,RidgeCV,SGDRegressor,ElasticNetCV
from sklearn.model_selection import cross_validate,GridSearchCV

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
alpha_space = np.linspace(0,10,100)
#l_space = np.linspace(0,1,100)
#params =  {'l1_ratio':l_space}
l1_space = np.linspace(0,1,100)
param_grid = {'l1_ratio':l1_space}

# Instantiate the ElasticNet regressor: elastic_net

# Setup the GridSearchCV object: gm_cv
estimator = ElasticNetCV(l1_ratio=l1_space,alphas=alpha_space,cv=10) #7 =10.7828467602


# Fit
#estimator = RidgeCV(alphas=alpha_space,normalize=True,cv=10,gcv_mode='auto')#7=10.8156374958
#selector = cross_validate(es,X=X_train,y=y_train,cv=10)
#estimator = BayesianRidge(normalize=True)
selector = RFE(estimator, 7, step=1)
selector.fit(X, y.ravel())
print selector.support_
#print selector.ranking_
y_scoring = selector.predict(X_scoring)
#r2 = selector.score(X_test,y_test)
#mse = mean_squared_error(y_test,y_pred)
#print("Tuned ElasticNet l1 ratio: {}".format(selector.best_params_))
#print("Tuned ElasticNet R squared: {}".format(r2))
#print("Tuned ElasticNet RMSE: {}".format(np.sqrt(mse)))
#print ("Best Score is : {}".format(selector.best_score_))



def write_to_csv(filename,y):
    jh = np.array(zip(scoring_SN, y))

    new = pd.DataFrame(jh)
    print new.head()
    new.to_csv(filename, sep=',', encoding='UTF-8', header=['S/N', 'Scores'], index=False)

write_to_csv('scoring_with_feature_selection.csv',y_scoring)