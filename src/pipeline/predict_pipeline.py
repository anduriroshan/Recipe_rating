import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Load model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Apply preprocessing to the input features
            data_scaled = preprocessor.transform(features)
            
            # Predict and return results
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, ID, RecipeNumber, UserReputation, ThumbsUpCount, ThumbsDownCount, BestScore, ReplyCount, Recipe_Review):
        self.ID = ID
        self.RecipeNumber = RecipeNumber
        self.UserReputation = UserReputation
        self.ThumbsUpCount = ThumbsUpCount
        self.ThumbsDownCount = ThumbsDownCount
        self.BestScore = BestScore
        self.ReplyCount = ReplyCount
        self.Recipe_Review = Recipe_Review

    def get_data_as_data_frame(self):
        """
        Prepares a single record as a DataFrame to be fed into the pipeline.
        """
        try:
            # Create a dictionary with the feature values
            custom_data_input_dict = {
                "ID": [self.ID],
                "RecipeNumber": [self.RecipeNumber],
                "UserReputation": [self.UserReputation],
                "ThumbsUpCount": [self.ThumbsUpCount],
                "ThumbsDownCount": [self.ThumbsDownCount],
                "BestScore": [self.BestScore],
                "ReplyCount": [self.ReplyCount],
                "Recipe_Review": [self.Recipe_Review]
            }

            # Convert to DataFrame for compatibility with preprocessing
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
