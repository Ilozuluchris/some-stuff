import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def scoring_csv():
    """
    :return: serial number of observations(Scoring_SN) and features to predict on(X_Scoring)
    """
    scoring_csv = pd.read_csv('scoring.csv')

    scoring_columns = ['S/N', 'Gender', 'Age', 'Location', 'famsize', 'Pstatus', 'Medu', 'Fedu','reason', 'traveltime', 'studytime','failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'famrel', 'freetime', 'health', 'absences',]
    scoring_csv.columns = scoring_columns

    scoring_csv = scoring_csv.drop(['reason'], axis=1)


    categorical_variable_list = ['Gender','Location','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet']


    scoring_csv_cleaned=  pd.get_dummies(scoring_csv,columns=categorical_variable_list,drop_first=True)#todo:train both models

    scoring_SN = scoring_csv_cleaned['S/N']
    X_scoring = scoring_csv_cleaned.drop(['S/N'],axis=1)
    return scoring_SN,X_scoring


scoring_SN,X_scoring= scoring_csv()

def write_csv_to_submit(scoring_SN,y_pred):
    """

    :param scoring_SN:serial number of observations
    :param y_pred: predicted y values
    :return: write_csv_to_submit.csv file to submit
    """
    #write_csv_to_submit.csv = pd.t

#print (df_cleaned.info())
#df_cleaned has 200 rows and 23 columns

#importing csv

df = pd.read_csv('Train.csv').dropna().drop_duplicates()
#assigning new column names
new_columns = ['S/N','Gender','Age','Location','famsize','Pstatus','Medu','Fedu','traveltime','studytime','failures','schoolsup','famsup',
'paid','activities','nursery','higher','internet','famrel','freetime','health','absences','Scores']
#todo: paid => did they pay for extra classes
df.columns = new_columns


#converting to categorical
categorical_variable_list = ['Gender','Location','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet']


df_cleaned =  pd.get_dummies(df,columns=categorical_variable_list,drop_first=True)#todo:train both models



#creating target and feature variables
y = df_cleaned['Scores'].values.reshape(-1,1)#200 rows,1 coulmn
X = df_cleaned.drop(['Scores','S/N'],axis=1) #200 rows,22 columns
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
'''print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
'''
if __name__ == '__main__':
    reg = LinearRegression(normalize=True)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    from sklearn.metrics import  mean_squared_error
    print np.sqrt(mean_squared_error(y_test,y_pred)) #todo : this equals to 11.8393014174

#print reg.intercept_
#print reg.coef_
#print zip(X.columns,reg.coef_)