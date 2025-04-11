import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
data_dir = r'data/'
df = pd.read_csv(data_dir + 'telecom_customer_churn.csv')
zipcode_population = pd.read_csv(data_dir + 'telecom_zipcode_population.csv')


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


print(df.head());


# df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True);

# sns.displot(df['Avg Monthly Long Distance Charges'], kde=True);
# plt.title('Avg Monthly Long Distance Charges');
# plt.show();
# print(df.head())