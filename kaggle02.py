import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error


#loading train and test data
train = pd.read_csv("tcd-ml-1920-group-income-train.csv")
test = pd.read_csv("tcd-ml-1920-group-income-test.csv")

#renaming column
rename_cols = {"Total Yearly Income [EUR]":'Income'}
train = train.rename(columns=rename_cols)
test = test.rename(columns=rename_cols)

# merging train and test data
data = pd.concat([train,test],ignore_index=True)

#data imputation
fill_col_dict = {'Year of Record': 1975,
 'Gender':'f',
 'Satisfation with employer': 'Average',
 'Profession': 'payment analyst',
 'University Degree': 0,
 'Hair Color': 'Black'}
for col in fill_col_dict.keys():
    data[col] = data[col].fillna(fill_col_dict[col])
    
data.head(20)

#function to do frequency encoding
def create_cat_con(df,cats,cons,normalize=True):
    for i,cat in enumerate(cats):
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = cat + '_FE_FULL'
        df[nm] = df[cat].map(vc)
        df[nm] = df[nm].astype('float32')
        for j,con in enumerate(cons):
#             print("cat %s con %s"%(cat,con))
            new_col = cat +'_'+ con
            print('timeblock frequency encoding:', new_col)
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)
            temp_df = df[new_col]
            fq_encode = temp_df.value_counts(normalize=True).to_dict()
            df[new_col] = df[new_col].map(fq_encode)
            df[new_col] = df[new_col]/df[cat+'_FE_FULL']
    return df

#declaring categorical and continuous columns respectively
cats = ['Year of Record', 'Housing Situation','Crime Level in the City of Employement','Work Experience in Current Job [years]','Satisfation with employer','Gender', 'Country',
        'Profession', 'University Degree','Wears Glasses','Hair Color','Age']
cons = ['Size of City','Body Height [cm]','Yearly Income in addition to Salary (e.g. Rental Income)']

#Calling function to do frequency encoding
data = create_cat_con(data,cats,cons)

# Label encoding categorical variables
for col in train.dtypes[train.dtypes == 'object'].index.tolist():
    feat_le = LabelEncoder()
    feat_le.fit(data[col].unique().astype(str))
    data[col] = feat_le.transform(data[col].astype(str))

del_col = set(['Income','Instance'])
features_col =  list(set(data) - del_col)
features_col

# splitting data into train and test 
X_train,X_test = data[features_col].iloc[:1048573],data[features_col].iloc[1048574:]
Y_train = data['Income'].iloc[:1048573]
X_test_id = data['Instance'].iloc[1048574:]
x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)

#specifying hyperparameters for lightgbm algorithm as a dictionary
params = {
          'max_depth': 20,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_seed": 10,
          "metric": 'mse',
          "verbosity": -1,
         }

# create dataset for lightgbm
trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)
# training model
clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
#predicting income
pre_test_lgb = clf.predict(X_test)
'done'

#calculating mean absolute error

pre_val_lgb = clf.predict(x_val)
val_mae = mean_absolute_error(y_val,pre_val_lgb)
val_mae

# writing results in csv file
sub_df = pd.DataFrame({'Instance':X_test_id,
                       'Total Yearly Income [EUR]':pre_test_lgb})
sub_df.head()

sub_df.to_csv("submission8.csv",index=False)
'done'
