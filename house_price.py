import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.float_format', lambda x : '{:.2f}'.format(x))

### Data Processing ###

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# save the Id Column before delete
train_id = df_train.Id
test_id = df_train.Id

df_train.drop('Id', axis=1, inplace=True)
df_test.drop('Id', axis=1, inplace=True)
#Check Null Data
#print(((df_train.isnull().sum()) / len(df_train)).sort_values(ascending=True)[:40])
#print(((df_train.isnull().sum()) / len(df_train)).sort_values(ascending=True)[40:80])
#print(((df_test.isnull().sum()) / len(df_test)).sort_values(ascending=True)[:40])
#print(((df_test.isnull().sum()) / len(df_test)).sort_values(ascending=True)[40:80])

"""
Visualization of Data Correlation
 
f, ax = plt.subplots()
ax.scatter(x=df_train.GrLivArea, y= df_train.SalePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
fig = plt.gcf()
plt.show()
fig.savefig('GrLiveArea & SalePrice.pdf')

# Delete outlier
df_train = df_train.drop(df_train[(df_train.GrLivArea>4000) & (df_train.SalePrice<300000)].index)

sns.distplot(df_train.SalePrice)
plt.ylabel('Frequency')

fig =plt.figure()
res = stats.probplot(df_train.SalePrice, plot=plt)
plt.show()

# for balance of 'SalePrice' data
df_train.SalePrice = np.log(df_train.SalePrice)

"""

### Feature Engineering ###
dtrain = df_train.shape[0]
dtest = df_test.shape[0]
y_train = df_train.SalePrice.values

combine = pd.concat((df_train, df_test)).reset_index(drop=True)
combine.drop(['SalePrice'], axis=1, inplace=True)
#print(combine.shape)

combine_nan = (combine.isnull().sum() / len(combine)) * 100
combine_nan = combine_nan.drop(combine_nan[combine_nan == 0].index).sort_values(ascending=False)[:30]
NAN_data = pd.DataFrame({'NAN Data': combine_nan})
#print(NAN_data.head(30))
"""
# Display NAN Data

f, ax = plt.subplots(figsize=(15,12))
sns.barplot(x=combine_nan.index, y=combine_nan)
plt.xticks(rotation='70')
plt.yticks(rotation='40')
plt.xlabel('Feature')
plt.ylabel('Nan data')
plt.show()

# Visualization between 'SalePrice' and other Features

correlation = df_train.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(correlation, vmax=1.0, annot=True, annot_kws={'size':5}, square=True, fmt='.2f')
plt.show()

"""

# Input Missing Data

for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual',
            'GarageCond', 'GarageFinish','GarageYrBlt', 'GarageType', 'BsmtExposure',
            'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType'):
    combine[col] = combine[col].fillna('None')

for col in ('MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF',
            'TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFinSF2', 'BsmtFinSF1'):
    combine[col] = combine[col].fillna(0)

combine['LotFrontage'] = combine.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
combine['MSZoning'] = combine['MSZoning'].fillna(combine['MSZoning'].mode()[0])
combine = combine.drop(['Utilities'], axis=1)
combine["Functional"] = combine["Functional"].fillna("Typ") # reference by description
combine['Electrical'] = combine['Electrical'].fillna(combine['Electrical'].mode()[0])
combine['Exterior1st'] = combine['Exterior1st'].fillna(combine['Exterior1st'].mode()[0])
combine['Exterior2nd'] = combine['Exterior2nd'].fillna(combine['Exterior2nd'].mode()[0])
combine['KitchenQual'] = combine['KitchenQual'].fillna(combine['KitchenQual'].mode()[0])
combine['SaleType'] = combine['SaleType'].fillna(combine['SaleType'].mode()[0])
combine['MSSubClass'] = combine['MSSubClass'].fillna("None")
combine['TotalSF'] = combine['TotalBsmtSF'] + combine['1stFlrSF'] + combine['2ndFlrSF'] # Create Total House Size index

# MSSubClass=The building class
combine.MSSubClass = combine.MSSubClass.apply(str)
# Changing OverallCond into a categorical variable
combine.OverallCond = combine.OverallCond.apply(str)
# # Year and Month sold are transformed into categorical features
combine.YrSold = combine.YrSold.astype(str)
combine.MoSold = combine.MoSold.astype(str)

cols = (
'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageYrBlt',
'GarageType', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType', 'YrSold', 'MoSold')

for c in cols:
    label = LabelEncoder()
    label.fit(list(combine[c].values))
    combine[c] = label.transform(list(combine[c].values))


# Check Skewness in the Numeric Data
numeric_feats = combine.dtypes[combine.dtypes != "object"].index

skewed_feats = combine[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
skewness = pd.DataFrame({'Skew': skewed_feats})
#print(skewness.head(10))
skewness = skewness[abs(skewness) > 0.75 ]
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    combine[i] = boxcox1p(combine[i], lam)

combine = pd.get_dummies(combine)

df_train = combine[:dtrain]
df_test = combine[dtrain:]

### Modeling ###

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)
    rmse = np.sqrt(-cross_val_score(model, df_train.values, y_train, scoring='neg_mean_squared_error', cv=kf))
    return (rmse)

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4,
                                   max_features='sqrt', min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05,
                             max_depth=3, min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213,
                             silent=1, random_state=7, nthread=-1)
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05,
                              n_estimators=720, max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, baggging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf =11)
score_lasso = rmsle_cv(lasso)
print("\nLasso score:{:.4f} ({:.4f})\n".format(score_lasso.mean(), score_lasso.std()))
score_ENet = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score_ENet.mean(), score_ENet.std()))
score_KRR = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score_KRR.mean(), score_KRR.std()))
score_xgb = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score_xgb.mean(), score_xgb.std()))
score_lgb = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score_lgb.mean(), score_lgb.std()))

models = pd.DataFrame({
    'Model' : ['lasso', 'ENet', 'KRR','model_xgb', 'model_lgb'],
    'Score' : ['score_lasso', 'score_ENet', 'score_KRR', 'score_xgb', 'score_lgb']})
print(models.sort_values(by='Score', ascending=True))

