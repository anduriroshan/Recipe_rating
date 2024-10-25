import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data_source', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocess_crimes(self, crime_df):
        """
        This function handles the preprocessing of crime data.
        """
        try:
            date_format = '%m/%d/%Y %I:%M:%S %p'

            # Replace None and empty values with NaN, then fill NaN with 0
            crime_df = crime_df.replace({None: np.nan, "": np.nan}).fillna(value=0)

            # Clean 'Victim_Age' column by replacing values < 0 with 0
            crime_df['Victim_Age'] = crime_df['Victim_Age'].apply(lambda x: 0 if x < 0 else x)

            # Drop rows where both 'Latitude' and 'Longitude' are 0
            crime_df.drop((crime_df[(crime_df['Latitude'] == 0) & (crime_df['Longitude'] == 0)].index), inplace=True)

            # Remove outliers for 'Latitude' and 'Victim_Age'
            for col in ['Latitude', 'Victim_Age']:
                Q1 = crime_df[col].quantile(0.25)
                Q3 = crime_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                crime_df = crime_df[(crime_df[col] > lower_bound) & (crime_df[col] < upper_bound)]

            # Add new features
            crime_df['Is_CrossStreet'] = crime_df['Cross_Street'].apply(lambda x: 0 if x == 0 else 1)
            crime_df['Victim_Present'] = crime_df['Victim_Age'].apply(lambda x: 0 if x <= 0 else 1)

            # Convert date columns to datetime and calculate new date-related features
            crime_df['Date_Reported'] = pd.to_datetime(crime_df['Date_Reported'], format=date_format)
            crime_df['Date_Occurred'] = pd.to_datetime(crime_df['Date_Occurred'], format=date_format)
            crime_df['Days_Lapsed'] = (crime_df['Date_Reported'] - crime_df['Date_Occurred']).dt.days
            crime_df['Day_Occurred'] = crime_df['Date_Occurred'].dt.day
            crime_df['WeekDay_Occurred'] = crime_df['Date_Occurred'].dt.weekday
            crime_df['Month'] = crime_df['Date_Occurred'].dt.month
            crime_df['Day_Rep'] = crime_df['Date_Reported'].dt.day
            crime_df['DayOfYear'] = crime_df['Date_Occurred'].dt.dayofyear

            # Handle other numeric transformations
            crime_df['Reporting_District_no'] = crime_df['Reporting_District_no'].astype(int)
            crime_df['Rep_Dist_no'] = crime_df['Reporting_District_no'].apply(lambda x: int(str(x)[-2:]))
            crime_df['Hour_Occ'] = crime_df['Time_Occurred'].apply(lambda x: int(x / 100))
            crime_df['Part 1-2'] = crime_df['Part 1-2'].astype(int).apply(lambda x: 0 if x == 1 else 1)
            crime_df['PCode'] = crime_df['Premise_Code'].apply(lambda x: int(x / 100))
            crime_df['IsArrest'] = crime_df['Status'].apply(lambda status: 1 if status in ['AA', 'JA'] else 0)

            # Drop redundant columns
            columns_to_drop = ['Location', 'Area_Name', 'Premise_Description', 'Status_Description',
                               'Date_Reported', 'Date_Occurred', 'Cross_Street', 'Weapon_Description', 'Time_Occurred']
            crime_df = crime_df.drop(columns=columns_to_drop, axis=1)

            return crime_df

        except Exception as e:
            raise CustomException(e, sys)

    def process_modus_operandi(self, data, vectorizer=None):
        """
        Processes the 'Modus_Operandi' column and returns binary-encoded data along with the CountVectorizer.
        """
        try:
            data['Modus_Operandi'] = data['Modus_Operandi'].astype(str).replace({None: np.nan, "": np.nan}).fillna(value="0")

            if vectorizer is None:
                vectorizer = CountVectorizer(binary=True)
                modus_matrix = vectorizer.fit_transform(data['Modus_Operandi'])
            else:
                modus_matrix = vectorizer.transform(data['Modus_Operandi'])

            modus_columns = ['Modus_' + word for word in vectorizer.get_feature_names_out()]
            modus_df = pd.DataFrame(modus_matrix.toarray(), columns=modus_columns, dtype='int8')

            processed_data = pd.concat([data.reset_index(drop=True), modus_df.reset_index(drop=True)], axis=1)
            processed_data = processed_data.drop(['Modus_Operandi'], axis=1)

            return processed_data, vectorizer

        except Exception as e:
            raise CustomException(e, sys)

    def normalize_data(self, train_data, test_data, numerical_cols):
        """
        Normalizes specified numerical columns using MinMaxScaler.
        """
        try:
            scaler = MinMaxScaler()
            train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
            test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])

            return train_data, test_data, scaler

        except Exception as e:
            raise CustomException(e, sys)

    def convert_categorical_to_numerical(self, train_data, test_data, categorical_cols):
        """
        Converts categorical columns into numerical using one-hot encoding and aligns columns in train and test sets.
        """
        try:
            train_dummies = pd.get_dummies(train_data[categorical_cols], columns=categorical_cols, dtype="int8")
            test_dummies = pd.get_dummies(test_data[categorical_cols], columns=categorical_cols, dtype="int8")

            train_dummies, test_dummies = train_dummies.align(test_dummies, join='outer', axis=1, fill_value=0)

            transformed_train = pd.concat([train_data, train_dummies], axis=1)
            transformed_test = pd.concat([test_data, test_dummies], axis=1)

            transformed_train = transformed_train.drop(categorical_cols, axis=1)
            transformed_test = transformed_test.drop(categorical_cols, axis=1)

            return transformed_train, transformed_test

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Initial columns in train_df: %s", train_df.columns.tolist())
            
            logging.info("Preprocessing crime data")
            train_df = self.preprocess_crimes(train_df)
            test_df = self.preprocess_crimes(test_df)
            logging.info("Columns after preprocess_crimes: %s", train_df.columns.tolist())
            
            logging.info("Handling 'Modus_Operandi' column")
            train_df, vectorizer = self.process_modus_operandi(train_df)
            test_df, _ = self.process_modus_operandi(test_df, vectorizer)
            logging.info("Columns after process_modus_operandi: %s", train_df.columns.tolist())
            
            logging.info("Normalizing numerical columns")
            numerical_cols = ['Area_ID', 'Latitude', 'Longitude', 'Victim_Age', 'DayOfYear', 'Days_Lapsed',
                            'Reporting_District_no', 'Premise_Code', 'Weapon_Used_Code', 'Day_Occurred',
                            'Day_Rep', 'PCode', 'Rep_Dist_no', 'WeekDay_Occurred', 'Hour_Occ', 'Month']
            # Check if all numerical columns exist
            missing_num_cols = [col for col in numerical_cols if col not in train_df.columns]
            if missing_num_cols:
                logging.warning("Missing numerical columns: %s", missing_num_cols)
                numerical_cols = [col for col in numerical_cols if col in train_df.columns]
                
            train_df, test_df, scaler = self.normalize_data(train_df, test_df, numerical_cols)
            logging.info("Columns after normalize_data: %s", train_df.columns.tolist())
            
            logging.info("Converting categorical columns to numerical")
            categorical_cols = ['Victim_Sex', 'Victim_Descent', 'Status']
            # Check if all categorical columns exist
            missing_cat_cols = [col for col in categorical_cols if col not in train_df.columns]
            if missing_cat_cols:
                logging.warning("Missing categorical columns: %s", missing_cat_cols)
                categorical_cols = [col for col in categorical_cols if col in train_df.columns]
                
            train_df, test_df = self.convert_categorical_to_numerical(train_df, test_df, categorical_cols)
            logging.info("Columns after convert_categorical_to_numerical: %s", train_df.columns.tolist())
            
            # Check if Crime_Category exists before attempting to split
            if "Crime_Category" not in train_df.columns:
                logging.error("Crime_Category column not found. Available columns: %s", train_df.columns.tolist())
                raise ValueError("Crime_Category column not found in the dataset")
                
            # Print column names right before the split
            logging.info("Final columns before split: %s", train_df.columns.tolist())
            '''
            target_column_name = "Crime_Category"
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            '''
            logging.info("Data preprocessing complete")
            
            return train_df,test_df, self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            logging.error("Error in initiate_data_transformation: %s", str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Path to your train and test CSV files
        # File paths
        train_path = os.path.join(os.getcwd(), "data_source", "train.csv")
        test_path = os.path.join(os.getcwd(), "data_source", "test.csv")

        
        # Create an instance of DataTransformation
        data_transformer = DataTransformation()
        
        # Run the data transformation process
        train_df,test_df, preprocessor_path = data_transformer.initiate_data_transformation(train_path, test_path)
        
        print("Data Transformation Completed.")
        print("Train data shape:", train_df.shape)
        print("Test data shape:", test_df.shape)
        print("Preprocessor saved at:", preprocessor_path)

    except Exception as e:
        print(f"Error: {e}")
