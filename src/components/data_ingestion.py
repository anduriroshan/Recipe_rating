import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data_source', 'train.csv')
    validation_data_path: str = os.path.join('data_source', 'validation.csv')
    test_data_path: str = os.path.join('data_source', 'test.csv')
    raw_data_path: str = os.path.join('data_source', 'data.csv')
    mysql_uri: str = 'mysql+pymysql://root:root@localhost:3306/crime_data'

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
            
            logging.info('Fetching train data from MySQL')
            query = "SELECT * FROM train"
            df = pd.read_sql(query, self.engine)
            logging.info('Data fetched and converted to DataFrame')

            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train-test split initiated")
            
            train_set, validation_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            validation_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)

            logging.info('Fetching test data from MySQL')
            query = "SELECT * FROM test"
            df = pd.read_sql(query, self.engine)
            logging.info('Data fetched and converted to DataFrame')
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Train-test split initiated")

            logging.info('Data ingestion completed')
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()
