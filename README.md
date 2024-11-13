# Recipe_rating
Try it on : https://reciperating.streamlit.app/
## Project Overview
- This project aims to build a machine learning model to predict the rating of recipes based on various features, such as recipe metadata, user engagement, and textual reviews. 

- The model performs best on the majority class (Rating 5), achieving an F1-score of 0.88 and a very high recall of 0.98.The overall **accuracy** of the model is **0.77**.

- The project includes several key components, including **data ingestion**, **data preprocessing**, **model training**, and a **prediction web app**. The data ingestion process fetches data from a **MongoDB database** and splits it into train, validation, and test sets. The data preprocessing pipeline combines various feature engineering techniques to handle different data types, including numerical, categorical, and text features (using **TfidfVectorizer**). 

- The model training component uses a **stacked ensemble** of several classifiers, including **XGBoost**, **LightGBM**, **Random Forest**, and **Logistic Regression**, to achieve good performance on the validation set, with the weighted F1-score used as the primary evaluation metric. 

- Finally, the project includes a Streamlit-based web application that allows users to input data manually or through a CSV file and get predictions, making the model accessible to end-users.

## Data Analysis
![image](https://github.com/user-attachments/assets/bb6b430a-5c7c-4b79-9f4d-6217815c9eef)


Key Insights

- **ID Distribution:** The distribution of unique IDs is right-skewed, indicating that most IDs have lower values, with a few higher ID values.
- **Recipe Number Distribution:** The distribution of recipe numbers has a prominent peak around 70-80, suggesting a large number of recipes with these numbers, followed by a gradual decline.
- **Recipe Code Distribution:** The distribution of recipe codes has a bimodal shape, with peaks around 20,000 and 150,000, indicating two predominant groups of recipe codes.
- **User Reputation Distribution:** The distribution of user reputation scores is highly right-skewed, with a large number of users having low reputation scores and a smaller number of users with very high reputation scores.
- **Reply Count Distribution:** The distribution of the number of replies per recipe is right-skewed, with most recipes having 0 or 1 reply, and a smaller number of recipes with higher reply counts.
- **Thumbs Up Count Distribution:** The distribution of the number of upvotes (thumbs up) per recipe has a prominent peak around 20-30 thumbs up, followed by a gradual decline.
- **Thumbs Down Count Distribution:** The distribution of the number of downvotes (thumbs down) per recipe is right-skewed, with most recipes having 0 to 25 downvotes, and a smaller number of recipes with higher downvote counts.
- **Best Score Distribution:** The distribution of the "best score" metric, which likely represents an overall quality or popularity score for the recipes, is highly right-skewed, with a large number of recipes having low best scores and a smaller number of recipes with very high best scores.
