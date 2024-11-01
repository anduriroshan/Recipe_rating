import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components import data_transformation
from src.components import model_trainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data_source', 'train.csv')
    validation_data_path: str = os.path.join('data_source', 'validation.csv')
    test_data_path: str = os.path.join('data_source', 'test.csv')
    raw_data_path: str = os.path.join('data_source', 'data.csv')
    mysql_uri: str = 'mysql+pymysql://root:root@localhost:3306/recipe_data'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.engine = self.connect_to_mysql()

    def connect_to_mysql(self):
        try:
            engine = create_engine(self.ingestion_config.mysql_uri)
            logging.info('Connected to MySQL database using SQLAlchemy')
            return engine
        except Exception as e:
            logging.error(f"Error while connecting to MySQL: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion method')
        try:
            # Fetching train data
            logging.info('Fetching train data from MySQL')
            train_query = "SELECT * FROM train"
            train_df = pd.read_sql(train_query, self.engine)
            logging.info('Train data fetched and converted to DataFrame')

            # Save full raw data and create train-validation split
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train-validation split initiated")

            train_set, validation_set = train_test_split(train_df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)

            # Fetching test data
            logging.info('Fetching test data from MySQL')
            test_query = "SELECT * FROM test"
            test_df = pd.read_sql(test_query, self.engine)
            logging.info('Test data fetched and converted to DataFrame')

            # Save test data
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

        
if __name__ == "__main__":
    # Initialize Data Ingestion
    ingestion = DataIngestion()
    train_data_path, validation_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Initialize Data Transformation
    data_transformer = data_transformation.DataTransformation()
    train_arr, validation_arr, preprocessor_path = data_transformer.initiate_data_transformation(train_data_path, validation_data_path)

    # Extract features and target variable for training and validation datasets
    # Assuming the last column in the array is the target variable
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]  # Features and target for training
    X_validation, y_validation = validation_arr[:, :-1], validation_arr[:, -1]  # Features and target for validation

    # Initialize Model Trainer with Stacking Classifier
    model_trainer = model_trainer.ModelTrainer()
    
    # Train the model and obtain accuracy
    accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_validation, y_validation)
    
    print(f"Stacking Classifier Accuracy: {accuracy:.2f}")

