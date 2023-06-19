conda update --all
conda install -c conda-forge jupyterthemes
git checkout 0d1d7fc32 # revert to previous commit temporarily
import matplotlib.pyplot as plt
feature_flag = features.isna().sum() > features.shape[0] * 0.2
features_omit = list(feature_flag[feature_flag == True].index)

num_features = []
str_features = []
for i in features.columns:
    if pd.api.types.is_string_dtype(features[i]):
        str_features.append(i)
    if pd.api.types.is_numeric_dtype(features[i]):
        num_features.append(i)
for i in str_features:
    features.loc[features[i].isna(), i] = "NAN"
for i in num_features:
    features[i].fillna(features[i].median(), inplace=True)

onehot_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Cabin_deck", "Cabin_side"]
for i in onehot_cols:
    features = pd.get_dummies(features, columns=[i], prefix=[i])

estimators = [("rf", gd.best_estimator_), ("gbm", gd_gbm.best_estimator_)]
reg = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())

##################################
# titanic
##################################
data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data_df.replace({'Title': mapping}, inplace=True)
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

data_df['Fare'].fillna(data_df['Fare'].median(), inplace = True)
pd.qcut(data_df['Fare'], 5)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label = LabelEncoder()
label.fit_transform(data_df['FareBin'])

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
std_scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size,
               'n_neighbors': n_neighbors}
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, verbose=True,
                cv=10, scoring = "roc_auc")
gd.fit(X, y)
y_pred = gd.best_estimator_.predict(X_test)

##################################
# Ame Housing
##################################
np.log1p(train["SalePrice"])
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

from scipy.special import boxcox1p
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
boxcox1p(all_data[feat], lam) # lam 0.15, sometimes use np.log1p(target)

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)

#####################################
# space titanic
#####################################
from pandas_profiling import ProfileReport
profile = ProfileReport(train_df, title="Profiling Report")
col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train_df['SumSpends'] = train_df[col_to_sum].sum(axis=1)

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("imp", SimpleImputer(strategy='mean'), null_cols)])
train_df[null_cols] = ct.fit_transform(train_df[null_cols])
train_test_split(X, y, random_state=23)

from sklearn.feature_selection import SequentialFeatureSelector

model_fs = CatBoostClassifier(verbose=False)
sf = SequentialFeatureSelector(model_fs, scoring='accuracy', direction = 'backward')
sf.fit(X,y)
 list(sf.get_feature_names_out())

########################################
# store sales
########################################
pd.read_csv(path + 'oil.csv', parse_dates=['date'], infer_datetime_format=True, index_col='date')
data_oil['dcoilwtico'].rolling(7).mean()
calendar.merge(data_oil, how='left', left_index=True, right_index=True)
calendar['ma_oil'].fillna(method='ffill', inplace=True)
calendar['dofw'] = calendar.index.dayofweek
df_hev.groupby(df_hev.index).first()
df_train.date = df_train.date.dt.to_period('D')
df_train = df_train.set_index(['store_nbr', 'family', 'date']).sort_index()

y = df_train.unstack(['store_nbr', 'family']).loc[sdate:edate]
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess, Fourier
# Fourier features
fourier = CalendarFourier(freq='W', order=4)
dp = DeterministicProcess(index=y.index,
                          constant=False,
                          order=1,
                          seasonal=False,
                          additional_terms=[fourier],
                          drop=True)
X = dp.in_sample()
X_test = dp.out_of_sample(steps=16)
calendar.loc[stest:etest]['ma_oil'].values

X['oil']  = calendar.loc[sdate:edate]['ma_oil'].values
model = Ridge(fit_intercept=True, solver='auto', alpha=0.4, random_state=SEED)
model.fit(X, y)
y_pred.stack(['store_nbr', 'family']).reset_index()
from sklearn.metrics import mean_squared_log_error
y_target.groupby('family').apply(lambda r: mean_squared_log_error(r['sales'], r['sales_pred']))
df_train.unstack(['store_nbr', 'family']).loc['2014':].loc(axis=1)['sales', :, 'SCHOOL AND OFFICE SUPPLIES'].plot(legend=None)

r1 = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, random_state=SEED)
r2 = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED)
b1 = BaggingRegressor(base_estimator=r1,
                      n_estimators=10,
                      n_jobs=-1,
                      random_state=SEED)
b2 = BaggingRegressor(base_estimator=r2,
                      n_estimators=10,
                      n_jobs=-1,
                      random_state=SEED)
model = VotingRegressor([('et', b1), ('rf', b2)])
np.stack(y_pred, axis=1)

########################################
# benetech_MakeGraphsAccessible
########################################

model = VisionEncoderDecoderModel.from_pretrained(CFG.model_dir)
model.eval()

device = torch.device("cuda:0")

model.to(device)
decoder_start_token_id = model.config.decoder_start_token_id
processor = DonutProcessor.from_pretrained(CFG.model_dir)

ids = ds["id"]
ds.set_transform(partial(preprocess, processor=processor))

data_loader = DataLoader(
    ds, batch_size=CFG.batch_size, shuffle=False
)


all_generations = []
for batch in tqdm(data_loader):
    pixel_values = batch["pixel_values"].to(device)

    batch_size = pixel_values.shape[0]

    decoder_input_ids = torch.full(
        (batch_size, 1),
        decoder_start_token_id,
        device=pixel_values.device,
    )

    try:
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=CFG.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=2,       #1 int    (1 - 10)
            temperature=.9,     #1 float  (0 -  ) less div - more div
            top_k=1,           #1 int    (1 -  ) less div - more div
            top_p=.4,           #1 float (0 - 1) more div - less div
            return_dict_in_generate=True,
        )

        all_generations.extend(processor.batch_decode(outputs.sequences))

    except:
        all_generations.extend([""]*batch_size)

# standard models
# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.linear_model import Perceptron
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV

#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
