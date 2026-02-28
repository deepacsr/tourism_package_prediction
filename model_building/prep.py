# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for imputation
from sklearn.impute import SimpleImputer

# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN_TOURIST"))
DATASET_PATH = "hf://datasets/deepacsr/tourism-package-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Check to remove any columns wihout labels
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=56
)

# For any missing values let us impute the values
#IMPUTATION DONE AFTER TRAIN/TEST SPLIT TO AVOID LEAKAGE

# Separate column based on types

numeric_cols = [
    'Age',     # Customer's age
    'CityTier', # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'NumberOfPersonVisiting', # Total number of people accompanying the customer on the trip
    'PreferredPropertyStar',  # Preferred hotel rating by the customer
    'NumberOfTrips',     # Average number of trips the customer takes annually
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome', # Gross monthly income of the customer
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch
    'NumberOfFollowups', # Total number of follow-ups by the salesperson after the sales pitch
    'DurationOfPitch', # Duration of the sales pitch delivered to the customer
    'Passport', # Whether the customer holds a valid passport     
    'OwnCar' # Whether the customer owns a car 
]
# Passport and Owncar which are Binary categorical columns here are numeric in nature

# List of categorical features in the dataset
categorical_cols = [
    'TypeofContact', # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation', # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender', # Gender of the customer (Male, Female)
    'ProductPitched', # The type of product pitched to the customer
    'MaritalStatus', # Marital status of the customer (Single, Married, Divorced)
    'Designation'# Customer's designation in their current organization
    
]

# Imputation done based on column type

# For Numeric columns median value is taken
num_imputer = SimpleImputer(strategy="median")
Xtrain[numeric_cols] = num_imputer.fit_transform(Xtrain[numeric_cols])

# For Test data only transformation is done and NOT FIT_TRANSFORM
Xtest[numeric_cols] = num_imputer.transform(Xtest[numeric_cols])

# For Categorical value most frequent occuring value is taken
cat_imputer = SimpleImputer(strategy="most_frequent")
Xtrain[categorical_cols] = cat_imputer.fit_transform(Xtrain[categorical_cols])
Xtest[categorical_cols] = cat_imputer.transform(Xtest[categorical_cols])

# Encoding categorical columns done during preprocessor stage of ML Pipeline

#Save the Train , Test data into .csv files
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

# Uploading 
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="deepacsr/tourism-package-prediction",
        repo_type="dataset",
    )
