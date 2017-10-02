import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
import seaborn as sns

#importing csv
df = pd.read_csv('Train.csv')

#assigning new column names
new_columns = ['S/N','Gender','Age','Location','famsize','Pstatus','Medu','Fedu','traveltime','studytime',     'failures','schoolsup','famsup',
'paid','activities','nursery','higher','internet','famrel','freetime','health','absences','Scores']
#todo: paid => did they pay for extra classes
df.columns = new_columns


#converting to categorical
categorical_variable_list = ['Gender','Location','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet']
df_set = pd.get_dummies(df,columns=categorical_variable_list,drop_first=True)#todo:train both models using drop_first=True and False
print (df_set.info())