from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py

# Data is located at:
# "https://raw.githubusercontent.com/fnakashima/nd00333_AZMLND_C2/master/starter_files/dataset/train_u6lujuX_CVtuZ9i.csv"

web_path = ['https://raw.githubusercontent.com/fnakashima/nd00333_AZMLND_C2/master/starter_files/dataset/train_u6lujuX_CVtuZ9i.csv']
ds = TabularDatasetFactory.from_delimited_files(path=web_path)

def clean_data(data):
    # Dict for cleaning data
    dependents = {"0":0, "1":1, "2":2, "3+":3}
    property_areas = {"Urban":1, "Semiurban":2, "Rural":3}

    # Clean and one hot encode data
    x_df = data.dropna()
    x_df.drop("Loan_ID", axis=1, inplace=True)

    # # Filtering "True", "Yes", "Y" won't work as it will be recoginised as a boolean value automatically by dataset framework
    x_df.loc[:,('Gender')] = x_df.Gender.apply(lambda s: 1 if s == "Male" else 2)
    x_df.loc[:,('Married')] = x_df.Married.apply(lambda s: 1 if s else 0)
    x_df.loc[:,('Dependents')] = x_df.Dependents.map(dependents)
    x_df.loc[:,('Education')] = x_df.Education.apply(lambda s: 1 if s == "Graduate" else 0)
    x_df.loc[:,('Self_Employed')] = x_df.Self_Employed.apply(lambda s: 1 if s else 0)
    x_df.loc[:,('Property_Area')] = x_df.Property_Area.map(property_areas)

    y_df = x_df.pop("Loan_Status").apply(lambda s: 1 if s else 0)
    return x_df, y_df

ds.to_pandas_dataframe()
x, y = clean_data(ds)

# Split data into train and test sets.
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# Default test_size: 0.25
x_train, x_test, y_train, y_test = train_test_split(x, y)

run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength(C)", np.float(args.C))
    run.log("Max iterations", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    # Save model
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.joblib')

if __name__ == '__main__':
    main()