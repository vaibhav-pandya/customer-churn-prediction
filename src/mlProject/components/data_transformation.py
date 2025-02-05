import os
from mlProject import logger
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from mlProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def transform_data(self, data):
        """Applies preprocessing transformations to the dataset."""
        
        # Convert TotalCharges to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

        # Define target and features
        X = data.drop(columns=['customerID', 'Churn'])  # Drop non-useful and target column
        y = data['Churn'].map({'No': 0, 'Yes': 1})  # Convert Churn to 0 & 1

        # Define column categories
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        binary_categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        multi_categorical_features = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                      'Contract', 'PaymentMethod']

        # Define preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Fill missing values
            ('scaler', StandardScaler())  # Standardize numerical data
        ])

        binary_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='if_binary', dtype=int))  # Convert binary to 0 & 1
        ])

        multi_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(drop='first', dtype=int))  # One-Hot Encode multi-category columns
        ])

        # Combine all transformations
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_categorical_features),
            ('multi', multi_transformer, multi_categorical_features)
        ])

        # Apply transformations
        X_transformed = preprocessor.fit_transform(X)

        # Convert transformed data into a DataFrame
        feature_names = preprocessor.get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        return X_transformed_df, y

    def train_test_splitting(self):
        """Splits the dataset into train and test sets after transformation."""
        # Load raw data
        data = pd.read_csv(self.config.data_path)

        # Apply transformations
        X_transformed, y = self.transform_data(data)

        # Train-test split (75% train, 25% test)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.25, random_state=42)

        # Save transformed train and test data
        train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info("Data transformed and split into training and test sets")
        logger.info(f"Train shape: {train_data.shape}")
        logger.info(f"Test shape: {test_data.shape}")

        print("Train shape:", train_data.shape)
        print("Test shape:", test_data.shape)
        