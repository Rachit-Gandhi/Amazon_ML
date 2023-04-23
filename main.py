import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from joblib import parallel_backend
import numpy as np
import tensorflow as tf

# Define data generator class
class DataGenerator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps_per_epoch = len(data) // batch_size
    
    def __len__(self):
        return self.steps_per_epoch
    
    def __getitem__(self, idx):
        batch_data = self.data[idx * self.batch_size:(idx + 1) * self.batch_size].copy()
        X = batch_data[['title_length', 'bullet_points_length', 'description_length', 'PRODUCT_TYPE_ID']].values
        y = batch_data['PRODUCT_LENGTH'].values
        return X, y

# Load the data
train_data = pd.read_csv('/kaggle/input/amazon-product-length-prediction-dataset/dataset/train.csv')

# Drop rows with missing values
train_data = train_data.dropna()

test_data = pd.read_csv('/kaggle/input/amazon-product-length-prediction-dataset/dataset/test.csv')

# Create an instance of SimpleImputer with the mean strategy
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the PRODUCT_LENGTH column of train data
imputer.fit(train_data[['PRODUCT_LENGTH']])

# Transform the PRODUCT_LENGTH column of train data by filling in missing values with the mean
train_data['PRODUCT_LENGTH'] = imputer.transform(train_data[['PRODUCT_LENGTH']])

# Transform the PRODUCT_LENGTH column of test data by filling in missing values with the mean
test_data['PRODUCT_LENGTH'] = imputer.transform(test_data[['PRODUCT_LENGTH']])

# Data preprocessing
unique_product_type_ids = train_data['PRODUCT_TYPE_ID'].unique()

# One-hot encode categorical variables in train data
train_data = pd.get_dummies(train_data, columns=['PRODUCT_TYPE_ID'], prefix='PRODUCT_TYPE_ID')

# One-hot encode categorical variables in test data
test_data = pd.get_dummies(test_data, columns=['PRODUCT_TYPE_ID'], prefix='PRODUCT_TYPE_ID')

print("Data preprocessing done")

# Feature engineering and selection
train_data['title_length'] = train_data['TITLE'].apply(len)
train_data['bullet_points_length'] = train_data['BULLET_POINTS'].apply(len)
train_data['description_length'] = train_data['DESCRIPTION'].apply(len)

test_data['title_length'] = test_data['TITLE'].apply(len)
test_data['bullet_points_length'] = test_data['BULLET_POINTS'].apply(len)
test_data['description_length'] = test_data['DESCRIPTION'].apply(len)

train_data.drop(['DESCRIPTION', 'TITLE', 'BULLET_POINTS'], axis=1, inplace=True)
test_data.drop(['DESCRIPTION', 'TITLE', 'BULLET_POINTS'], axis=1, inplace=True)

train_data.to_csv("./dataset/train_processed.csv", sep="\t")
test_data.to_csv("./dataset/test_processed.csv", sep="\t")

print("Feature engineering done")

X_train = train_data[['title_length', 'bullet_points_length', 'description_length', 'PRODUCT_TYPE_ID']]
y_train = train_data['PRODUCT_LENGTH']

X_test = test_data[['title_length', 'bullet_points_length', 'description_length', 'PRODUCT_TYPE_ID']]

# Define batch size and data generators
batch_size = 64
train_gen = DataGenerator(pd.concat([X_train, y_train], axis=1), batch_size)

# Model selection and
train_data.dropna(inplace=True) # remove missing data

model = RandomForestRegressor(random_state=42)
param_dist = {
'n_estimators': [50, 100, 200, 500],
'max_depth': [10, 20, 30, 50, None],
'min_samples_split': [2, 5, 10, 20],
'min_samples_leaf': [1, 2, 4, 8],
'max_features': ['sqrt', 'log2', None],
'bootstrap': [True, False]
}

with parallel_backend('multiprocessing'):
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
random_search.fit(X_train, y_train)
model = random_search.best_estimator_

print("Model training done")

test_data = pd.read_csv('/kaggle/input/amazon-product-length-prediction-dataset/dataset/test.csv')

test_data['title_length'] = test_data['TITLE'].apply(len)
test_data['bullet_points_length'] = test_data['BULLET_POINTS'].apply(len)
test_data['description_length'] = test_data['DESCRIPTION'].apply(len)

test_data = pd.get_dummies(test_data, columns=['PRODUCT_TYPE_ID'], prefix='PRODUCT_TYPE_ID')
test_data.drop(['DESCRIPTION', 'TITLE', 'BULLET_POINTS'], axis=1, inplace=True)
test_data['PRODUCT_LENGTH'] = imputer.transform(test_data[['PRODUCT_LENGTH']])
predictions = model.predict(test_data[['title_length', 'bullet_points_length', 'description_length', 'PRODUCT_TYPE_ID']])
submission = pd.DataFrame({'PRODUCT_ID': test_data['PRODUCT_ID'], 'PRODUCT_LENGTH': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created")
