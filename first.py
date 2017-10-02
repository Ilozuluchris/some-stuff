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
print (df_cleaned.info())
#df_cleaned has 200 rows and 23 columns

#creating target and feature variables
y = df_cleaned['Scores'] #200 rows,1 coulmn
X = df_cleaned.drop('Scores',axis=1)#200 rows,22 columns

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)