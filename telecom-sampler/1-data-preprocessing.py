import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

pd.set_option('display.max_columns', 50)
data_dir = r'data/'
df = pd.read_csv(data_dir + 'telecom_customer_churn.csv')
zipcode_population = pd.read_csv(data_dir + 'telecom_zipcode_population.csv')

def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)
    
    plt.figure(figsize = (16,12))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


# print(df);
# print(df.head());

# print(df['Customer Status'].value_counts(1));

# print(df.groupby('Customer Status')['Tenure in Months'].mean());
# print(df.groupby('Customer Status')['Tenure in Months'].max());
# print(df.groupby('Customer Status')['Tenure in Months'].min());

# print(df.groupby('Customer Status')['Total Charges'].mean());

list_of_leak_columns = ['Total Charges', 'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Revenue'];

print(list_of_leak_columns);

for column in list_of_leak_columns:
    df[column] = df[column]/df['Tenure in Months'];

df = df.drop(columns=['Tenure in Months']);

# print(df.head())

# 1) data error or measurement error - i.e. Age
# 2) meaningful null values - LIke missing offer vs having a valid offer, a null is valid value.

# check which columsn have NA values in the columns
# print(df.isna().sum());
# print(df['Offer'].value_counts(dropna=False));
# clean the Offer NA values
df['Offer'] = df['Offer'].fillna('No Offer');

df['Avg Monthly Long Distance Charges'] = df['Total Long Distance Charges'].fillna(0);

df['Multiple Lines'] = df['Multiple Lines'].fillna('Not Applicable');
df['Internet Type'] = df['Internet Type'].fillna('Not Applicable');
df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(0);

internet_related_object_columns = ['Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data'];

for column in internet_related_object_columns:
    df[column] = df[column].fillna('Not Applicable');


#--- data encoding -----
# 1) one hot enconding

one_hot_columns = ['Offer', 'Multiple Lines', 'Payment Method'] + internet_related_object_columns

print(one_hot_columns);
for column in one_hot_columns:
    pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1);

# drop the hot encoded columns
df = df.drop(columns=one_hot_columns);

# 2) label encoding
df['Gender'] = np.where(df['Gender'] == 'Female', 1, 0)
df['Married'] = np.where(df['Married'] == 'Yes', 1, 0)
df['Phone Service'] = np.where(df['Phone Service'] == 'Yes', 1, 0)
df['Internet Service'] = np.where(df['Internet Service'] == 'Yes', 1, 0)
df['Paperless Billing'] = np.where(df['Paperless Billing'] == 'Yes', 1, 0)

df['Internet Type'] = np.where(df['Internet Type'] == 'Not Applicable', 0, 
                               np.where(df['Internet Type'] == 'DSL', 1,
                                        np.where(df['Internet Type'] == 'Cable', 2, 4)));

df['Contract'] = np.where(df['Contract'] == 'Month-to-Month', 0, 
                             np.where(df['Contract'] == 'One Year', 1, 2));


df = df.merge(zipcode_population, how='left', on='Zip Code');

df = df.drop(columns=['Zip Code', 'Latitude', 'Longitude']);

df['is Los Angeles'] = np.where(df['City'] == 'Los Angeles', 1, 0);
df['is San Diego'] = np.where(df['City'] == 'San Diego', 1, 0);
df['is San Jose'] = np.where(df['City'] == 'San Jose', 1, 0);
df['is Sacramento'] = np.where(df['City'] == 'Sacramento', 1, 0);
df['is San Francisco'] = np.where(df['City'] == 'San Francisco', 1, 0);
df['is Fresno'] = np.where(df['City'] == 'Fresno', 1, 0);
df['is Long Beach'] = np.where(df['City'] == 'Fresno', 1, 0);
df['is Escondido'] = np.where(df['City'] == 'Escondido', 1, 0);
df['is Oakland'] = np.where(df['City'] == 'Oakland', 1, 0);
df['is Stockton'] = np.where(df['City'] == 'Stockton', 1, 0);

df = df.drop(columns=['City']);

pd.set_option('display.max_columns', 70)
print(df.shape)

print(df['Customer Status'].value_counts());
# print(df.head());

df_joined = df[df['Customer Status'] == 'Joined'];
df_churned_stayed = df[df['Customer Status'] != 'Joined'];


# df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True);

# sns.displot(df['Avg Monthly Long Distance Charges'], kde=True);
# plt.title('Avg Monthly Long Distance Charges');
# plt.show();
# print(df.head())

# --------- Train Test Split ------------
feature_columns = [col for col in df_churned_stayed.columns if col not in ['Customer Status', 'Churn Category', 'Churn Reason', 'Customer ID']];
target_column = ['Customer Status']

# Predictor variables
X = df_churned_stayed[feature_columns]

# Target variable
y = df_churned_stayed[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# Define a list of continuous and categorical features so that they can be scaled.
continous_and_ordinal_columns = ['Age','Number of Dependents','Number of Referrals',
                                 'Avg Monthly Long Distance Charges','Internet Type',
                                 'Avg Monthly GB Download','Contract','Monthly Charge',
                                 'Total Charges','Total Refunds', 'Total Extra Data Charges',
                                 'Total Long Distance Charges', 'Total Revenue','Population']

# for col in continous_and_ordinal_columns:
#     print(col)
#     sns.boxplot(X_train.reset_index()[col])
#     plt.show()

# --- Eliminate outliers ------
# Number of referrals < 11
# Number of dependents < 4
# Total Extra Data charges < 20
referral_mask = (X_train['Number of Referrals'] < 11)
dependents_mask = (X_train['Number of Dependents'] < 4)
extra_data_mask = (X_train['Total Extra Data Charges'] < 20)

# filter out the data
X_train_outliers_eliminated = X_train[referral_mask & dependents_mask & extra_data_mask]

scaler = MinMaxScaler()
scaler.fit(X_train_outliers_eliminated)
X_train_scaled = X_train_outliers_eliminated.copy()
X_train_scaled[X_train_scaled.columns] = scaler.transform(X_train_outliers_eliminated[X_train_outliers_eliminated.columns])
X_test_scaled = X_test.copy()
X_test_scaled[X_test_scaled.columns] = scaler.transform(X_test[X_test.columns])

y_train_narrowed = y_train.loc[X_train_scaled.index]
print(y_train.value_counts());
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
# train with scaled featurea and labels
# print(y_train_narrowed);
print(y_train_narrowed.shape);
lr_classifier.fit(X_train_scaled, y_train_narrowed.values.ravel())

# Predict on test model 
y_pred_lr = lr_classifier.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))
print("F1 score is: ", f1_score(y_test, y_pred_lr, pos_label='Churned'))


#KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
# train with scaled features and labels
knn_classifier.fit(X_train_scaled, y_train_narrowed.values.ravel())

# Predict on test model
y_pred_knn = knn_classifier.predict(X_test_scaled.values)
print(classification_report(y_test, y_pred_knn))
print("F1 score is: ", f1_score(y_test, y_pred_knn, pos_label='Churned'))

# Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=6)
# train with scaled features and labels
dt_classifier.fit(X_train, y_train)
# Predict on test model
y_pred_dt = dt_classifier.predict(X_test)
print(classification_report(y_test, y_pred_dt))
print("F1 score is: ", f1_score(y_test, y_pred_dt, pos_label='Churned'))



# Random Forest Classifier
rf_classifier = RandomForestClassifier(max_depth=10)
# train with scaled features and labels
rf_classifier.fit(X_train, y_train)
# Predict on test model
y_pred_rf = rf_classifier.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("F1 score is: ", f1_score(y_test, y_pred_rf, pos_label='Churned'))

# Light GBM Classifier
lgb_classifier = lgb.LGBMClassifier(max_depth=4)

# Set categorical features for LightGBM
categorical_features = [col for col in X_train.columns if col not in ['Age','Number of Dependents','Number of Referrals',
                                                                      'Avg Monthly Long Distance Charges',
                                                                      'Avg Monthly GB Download','Monthly Charge',
                                                                      'Total Charges','Total Refunds','Total Extra Data Charges',
                                                                      'Total Long Distance Charges','Total Revenue',
                                                                      'Population','Revenue per Tenure Months']]
# train with scaled features and labels
lgb_classifier.fit(X_train, y_train, categorical_feature=categorical_features)
# Predict on test model
y_pred_lgb = lgb_classifier.predict(X_test)
print(classification_report(y_test, y_pred_lgb))
print("F1 score is: ", f1_score(y_test, y_pred_lgb, pos_label='Churned'))


# XGBoost Classifier

# Set categorical features for XGBoost
# Change categorical features' types
X_train[categorical_features] = X_train[categorical_features].astype('category')
X_test[categorical_features] = X_test[categorical_features].astype('category')

y_train_encoded = y_train.copy()
y_test_encoded = y_test.copy()

y_train_encoded['Customer Status'] = np.where(y_train['Customer Status'] == 'Churned',1,0)
y_test_encoded['Customer Status'] = np.where(y_test['Customer Status'] == 'Churned',1,0)
xgb_classifier = xgb.XGBClassifier(max_depth=3, enable_categorical=True)

# train with scaled features and labels
xgb_classifier.fit(X_train, y_train_encoded)
y_pred_xgb = xgb_classifier.predict(X_test)
print("F1 score is: ", f1_score(y_test_encoded, y_pred_xgb))

#PLot lgbm feature importances
lgb_importances = lgb_classifier.feature_importances_
features_names = X_train.columns
# plot_feature_importances(lgb_importances, features_names)

df_importances_first_35 = pd.DataFrame(lgb_importances, index=features_names).reset_index().sort_values(by=0, ascending=False).head(35);

feature_columns_top_importance = list(df_importances_first_35['index']);

X_train_top_columns = X_train[feature_columns_top_importance]
X_test_top_columns = X_test[feature_columns_top_importance]

categorical_features_narrowed = [col for col in categorical_features if col in X_train_top_columns.columns]

lgb_classifier.fit(X_train_top_columns, y_train, categorical_feature=categorical_features_narrowed)
y_pred_lgb_top = lgb_classifier.predict(X_test_top_columns)
print("F1 score is: ", f1_score(y_test, y_pred_lgb_top, pos_label='Churned'))

# Change categorical features' types
# df_joined_updated = df_joined[categorical_features].astype('category')
# df_joined[categorical_features] = df_joined[categorical_features].astype('category')

# joined_customers_preds = lgb_classifier.predict(df_joined[feature_columns_top_importance])

# df_joined['predictions'] = joined_customers_preds

# df_joined.head(10)




# -- MUltiple Classifier ------
target_column_multiclass = ['Churn Category']
X_multiclass = df_churned_stayed[feature_columns]
y_multiclass = df_churned_stayed[target_column_multiclass]
# fill the NA category with a filler 'Stayed'
y_multiclass = y_multiclass.fillna('Stayed')
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(X_multiclass, y_multiclass, test_size=0.30, random_state=42)
# Define a list of continuous and categorical features so that they can be scaled.
lgbm_classifier_multiclass = lgb.LGBMClassifier(max_depth=4, objective='multiclass', num_class=4)
lgbm_classifier_multiclass.fit(X_train_multiclass, y_train_multiclass, categorical_feature=categorical_features)

y_pred_lgbm_multiclass = lgbm_classifier_multiclass.predict(X_test_multiclass)
print(classification_report(y_test_multiclass, y_pred_lgbm_multiclass))
print(y_pred_lgbm_multiclass)