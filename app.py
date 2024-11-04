import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData  # assuming the PredictPipeline and CustomData are in predict_pipeline.py
import os
from src.exception import CustomException

# Set up the Streamlit app layout
st.title("Recipe Rating Prediction App")
st.write("This app predicts the rating of a recipe based on user inputs.")

# Helper function to display prediction
def display_prediction(predictions):
    for i, pred in enumerate(predictions):
        st.write(f"Prediction {i + 1}: {pred}")

# Instantiate the PredictPipeline
predict_pipeline = PredictPipeline()

# Option to manually enter data
st.header("Manual Input")
with st.form("manual_input_form"):
    ID = st.number_input("ID", min_value=0, step=1, help="Example: 86")
    RecipeNumber = st.text_input("Recipe Number", help="Example: 45")
    UserReputation = st.number_input("User Reputation", min_value=0, help="Example: 1")
    ThumbsUpCount = st.number_input("Thumbs Up Count", min_value=0, help="Example: 0")
    ThumbsDownCount = st.number_input("Thumbs Down Count", min_value=0, help="Example: 0")
    BestScore = st.number_input("Best Score", min_value=0, help="Example: 100")
    ReplyCount = st.number_input("Reply Count", min_value=0, help="Example: 0")
    Recipe_Review = st.text_area("Recipe Review", help="Example: This was so good!! The pumpkin was perfect as a thickening agent. I'm putting this on my Halloween dinner party menu!!")
    
    submit_manual = st.form_submit_button("Predict Manually")
    
    if submit_manual:
        try:
            # Prepare data as CustomData and convert to DataFrame
            input_data = CustomData(
                ID=ID,
                RecipeNumber=RecipeNumber,
                UserReputation=UserReputation,
                ThumbsUpCount=ThumbsUpCount,
                ThumbsDownCount=ThumbsDownCount,
                BestScore=BestScore,
                ReplyCount=ReplyCount,
                Recipe_Review=Recipe_Review
            )
            features = input_data.get_data_as_data_frame()
            
            # Predict
            predictions = predict_pipeline.predict(features)
            display_prediction(predictions)
        except CustomException as e:
            st.error(f"Error in prediction: {e}")

# Option to upload CSV file for batch prediction
st.header("Batch Prediction via CSV")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)
        
        # Display file data
        st.write("Uploaded Data:")
        st.write(data.head())
        
        # Predict on CSV data
        predictions = predict_pipeline.predict(data)
        data['Predicted Rating'] = predictions
        
        # Display predictions
        st.write("Predictions:")
        st.write(data[['Predicted Rating']])
        
        # Option to download predictions
        csv_output = data.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv"
        )
    except CustomException as e:
        st.error(f"Error in batch prediction: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
