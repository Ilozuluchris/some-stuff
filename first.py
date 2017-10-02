import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

#importing csv
df = pd.read_csv('Train.csv').dropna().drop_duplicates()
#assigning new column names
new_columns = ['S/N','Gender','Age','Location','famsize','Pstatus','Medu','Fedu','traveltime','studytime',     'failures','schoolsup','famsup',
'paid','activities','nursery','higher','internet','famrel','freetime','health','absences','Scores']
#todo: paid => did they pay for extra classes
df.columns = new_columns


#converting to categorical
categorical_variable_list = ['Gender','Location','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet']
df_cleaned =  pd.get_dummies(df,columns=categorical_variable_list,drop_first=True)#todo:train both models using drop_first=True and False
#print (df_cleaned.info())
#df_cleaned has 200 rows and 23 columns

#creating target and feature variables
y = df_cleaned['Scores'].values.reshape(-1,1)#200 rows,1 coulmn
X = df_cleaned.drop(['Scores','S/N'],axis=1).as_matrix() #200 rows,22 columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
from sklearn.linear_model import LinearRegression
reg = LinearRegression(normalize=True)
reg.fit(X_train,y_train)
print reg
y_pred = reg.predict(X_test)
from sklearn.metrics import  mean_squared_error
print np.sqrt(mean_squared_error(y_test,y_pred))
#print reg.intercept_
#print reg.coef_
#print zip(X.columns,reg.coef_)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
'''l1_space = np.linspace(0, 100, 60)
param_grid = {'l1_ratio':l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(reg,cv=5)

# Fit it to the training data
gm_cv.fit(X_train,y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(y_test,y_pred)
mse = mean_squared_error(y_pred,y_test)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
'''