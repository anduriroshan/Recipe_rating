import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os
import scipy.sparse as sp

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data_source', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates a ColumnTransformer with pipelines for different data types.
        """
        try:
            numerical_columns = ["ID", "UserReputation", "ThumbsUpCount", "ThumbsDownCount", "BestScore"]
            categorical_columns = ["RecipeNumber", "ReplyCount"]
            text_column = "Recipe_Review"
            
            # Pipeline for numerical features
            num_pipeline = Pipeline([
                ("scaler", MinMaxScaler())
            ])

            # Pipeline for categorical features - using dense output
            cat_pipeline = Pipeline([
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Pipeline for text features
            text_pipeline = Pipeline([
                ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", min_df=3))
            ])

            # Column Transformer without sparse parameter
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns),
                ("text_pipeline", text_pipeline, text_column)
            ])

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object: %s", str(e))
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully.")
            # Define columns
            numerical_columns = ["ID", "UserReputation", "ThumbsUpCount", "ThumbsDownCount", "BestScore"]
            categorical_columns = ["RecipeNumber", "ReplyCount"]
            text_column = "Recipe_Review"
            target_column_name = "Rating"
            columns_to_drop = ["RecipeName", "CommentID", "UserID", "CreationTimestamp"]
            # Handle missing values
            train_df['Recipe_Review'].fillna('missing', inplace=True)
            test_df['Recipe_Review'].fillna('missing', inplace=True)
            train_df[numerical_columns] = train_df[numerical_columns].fillna(0)
            test_df[numerical_columns] = test_df[numerical_columns].fillna(0)
            train_df[categorical_columns] = train_df[categorical_columns].fillna('missing')
            test_df[categorical_columns] = test_df[categorical_columns].fillna('missing')

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            logging.info("Preprocessing object obtained.")

            # Define target and columns to drop
            target_column_name = "Rating"
            columns_to_drop = ["RecipeName", "CommentID", "UserID", "CreationTimestamp"]

            # Separate input and target features
            X_train = train_df.drop(columns=[target_column_name] + columns_to_drop, axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[target_column_name] + columns_to_drop, axis=1)
            y_test = test_df[target_column_name]

            logging.info("Applying preprocessing object to training and testing data.")

            # Transform the feature data
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Convert sparse matrices to dense arrays if necessary
            if sp.issparse(X_train_transformed):
                logging.info("Converting sparse training matrix to dense array")
                X_train_transformed = X_train_transformed.toarray()
            if sp.issparse(X_test_transformed):
                logging.info("Converting sparse testing matrix to dense array")
                X_test_transformed = X_test_transformed.toarray()

            # Convert target variables to numpy arrays and reshape
            y_train_arr = np.array(y_train).reshape(-1, 1)
            y_test_arr = np.array(y_test).reshape(-1, 1)

            # Log shapes for debugging
            logging.info(f"X_train_transformed shape: {X_train_transformed.shape}")
            logging.info(f"y_train_arr shape: {y_train_arr.shape}")
            logging.info(f"X_test_transformed shape: {X_test_transformed.shape}")
            logging.info(f"y_test_arr shape: {y_test_arr.shape}")

            # Combine features with targets
            train_arr = np.hstack((X_train_transformed, y_train_arr))
            test_arr = np.hstack((X_test_transformed, y_test_arr))

            logging.info(f"Final training array shape: {train_arr.shape}")
            logging.info(f"Final testing array shape: {test_arr.shape}")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing object saved successfully.")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error("Error in initiate_data_transformation: %s", str(e))
            raise CustomException(e, sys)