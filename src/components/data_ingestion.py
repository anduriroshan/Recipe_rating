import os
import sys
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components import data_transformation
from src.components import model_trainer

@dataclass
class DataIngestionConfig:
    #username: str = os.getenv('MONGODB_USERNAME')
    #password: str = os.getenv('MONGODB_PASSWORD')
    logging.info('Got those var')
    train_data_path: str = os.path.join('data_source', 'train.csv')
    validation_data_path: str = os.path.join('data_source', 'validation.csv')
    test_data_path: str = os.path.join('data_source', 'test.csv')
    raw_data_path: str = os.path.join('data_source', 'data.csv')
    mongodb_uri: str = f'mongodb+srv://roshanandhuri:MXBQlqXEd3iC5ebk@cluster0.dqh7v.mongodb.net/'
    database_name: str = 'recipe_data'

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.client = self.connect_to_mongodb()
        self.db = self.client[self.ingestion_config.database_name]

    def connect_to_mongodb(self):
        try:
            client = MongoClient(self.ingestion_config.mongodb_uri)
            logging.info('Connected to MongoDB database')
            return client
        except Exception as e:
            logging.error(f"Error while connecting to MongoDB: {e}")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion method')
        try:
            # Fetching train data
            logging.info('Fetching train data from MongoDB')
            train_collection = self.db['train']
            train_data = list(train_collection.find({}, {'_id': 0}))  # Exclude MongoDB _id field
            train_df = pd.DataFrame(train_data)
            logging.info('Train data fetched and converted to DataFrame')

            # Save full raw data and create train-validation split
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train-validation split initiated")

            train_set, validation_set = train_test_split(train_df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)

            # Fetching test data
            logging.info('Fetching test data from MongoDB')
            test_collection = self.db['test']
            test_data = list(test_collection.find({}, {'_id': 0}))  # Exclude MongoDB _id field
            test_df = pd.DataFrame(test_data)
            logging.info('Test data fetched and converted to DataFrame')

            # Save test data
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            
            # Close MongoDB connection
            self.client.close()
            logging.info('MongoDB connection closed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            # Ensure MongoDB connection is closed even if an error occurs
            self.client.close()
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initialize Data Ingestion
    ingestion = DataIngestion()
    train_data_path, validation_data_path, test_data_path = ingestion.initiate_data_ingestion()

    # Initialize Data Transformation
    data_transformer = data_transformation.DataTransformation()
    train_arr, validation_arr, preprocessor_path = data_transformer.initiate_data_transformation(train_data_path, validation_data_path)

    # Extract features and target variable for training and validation datasets
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]  # Features and target for training
    X_validation, y_validation = validation_arr[:, :-1], validation_arr[:, -1]  # Features and target for validation

    # Initialize Model Trainer with Stacking Classifier
    model_trainer = model_trainer.ModelTrainer()
    
    # Train the model and obtain accuracy
    accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_validation, y_validation)
    
    print(f"Stacking Classifier Accuracy: {accuracy:.2f}")