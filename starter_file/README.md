# Customer Loan Status Prediction

This is the final project of the Udacity Azure ML Nanodegree.  
In this project, we build two models: one using Azure AutoML and one customized model whose hyperparameters are tuned using HyperDrive.  
After building the models, we compare the performance of both the models.  
Finally, we deploy the best performing model and test the endpoint.

## Architectural Diagram
Here is an architectual diagram of this project.
![Project Architectural Diagram](/starter_file/images/ProjectArchitecturalDiagram.PNG)

## Dataset

### Overview
In this project, we use a [loan prediction problem dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle. The dataset contains 13 features including a target column **Loan_Status**.

### Task
With this dataset, we predict a loan status (Yes or No). Therefore, we can categorise this problem as a binary classification problem. 

The dataset contains 12 features but we drop one column "Loan_ID" as it is just an identifier and not relevant to a prediction result.
In addition, as some features have string values, we update the data values into integer values by using dictionaries.
These tasks are done in the data clean up process defined in [train.py](./train.py).

### Access
We access the dataset directly from Kaggle by using [from_delimited_files](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py#from-delimited-files-path--validate-true--include-path-false--infer-column-types-true--set-column-types-none--separator------header-true--partition-format-none--support-multi-line-false--empty-as-string-false--encoding--utf8--) of [TabularDatasetFactory](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py)

## Automated ML
Since the target problem is to predict customer's loan status (binary prediction: 0 or 1), the task type is **classification** and the target label column is **Loan_Status**.  

We use **accuracy** as a primary metric and set **30** as experiment_timeout_minutes to limit experiment running duration.  
We set **10** as max_concurrent_iterations to run iterations in the experiment in parallel. featurization is set to **auto** to enable featurization step to be done automatically.  

We also set **enable_onnx_compatible_models** True to convert a model to ONNX later for the Standout suggestions.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
Here are models trained by Automated ML. The best accuracy was 0.80833 by [SparseNormalizer](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.shared.model_wrappers.sparsenormalizer?view=azure-ml-py), [XGBoostClassifier](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.shared.model_wrappers.xgboostclassifier?view=azure-ml-py)
![Models trained by Automated ML](/starter_file/images/AutoML_Models.PNG)

Here are details and some metrics of the model.
![Details of the best model](/starter_file/images/AutoML_BestModelDetails.PNG)
![Details of the best model](/starter_file/images/AutoML_BestModelMetrics.PNG)

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
