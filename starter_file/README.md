# Customer Loan Status Prediction

## Overview
This is the final project of the Udacity Azure ML Nanodegree.  
In this project, we build two models: one using Azure AutoML and one customized model whose hyperparameters are tuned using HyperDrive.  
After building the models, we compare the performance of both the models.  
Finally, we deploy the best performing model and test the endpoint.

## Architectural Diagram
Here is an architectual diagram of this project.
![Project Architectural Diagram](/starter_file/images/ProjectArchitecturalDiagram.PNG)

## Dataset

### Overview
In this project, we use a [loan prediction problem dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset) from Kaggle. The dataset contains 13 features including a target column ``Loan_Status``.

### Task
With this dataset, we predict a loan status (Yes or No). Therefore, we can categorise this problem as a binary classification problem. 

The dataset contains 12 features but we drop one column "Loan_ID" as it is just an identifier and not relevant to a prediction result.
In addition, as some features have string values, we update the data values into integer values by using dictionaries.
These tasks are done in the data clean up process defined in [train.py](./train.py).

### Access
We access the dataset stored online by using [from_delimited_files](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py#from-delimited-files-path--validate-true--include-path-false--infer-column-types-true--set-column-types-none--separator------header-true--partition-format-none--support-multi-line-false--empty-as-string-false--encoding--utf8--) of [TabularDatasetFactory](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.dataset_factory.tabulardatasetfactory?view=azure-ml-py).

## Automated ML
Since the target problem is to predict customer's loan status (binary prediction: 0 or 1), the task type is ``classification`` and the target label column is ``Loan_Status``.  

We use ``accuracy`` as a primary metric and set ``30`` as experiment_timeout_minutes to limit experiment running duration.  
We set ``10`` as max_concurrent_iterations to run iterations in the experiment in parallel. featurization is set to ``auto`` to enable featurization step to be done automatically.  

We also set **enable_onnx_compatible_models** True to convert a model to ONNX later for the Standout suggestions.
![Automated ML settings and configuration](/starter_file/images/AutoML_Settings.PNG)

### Results
Here are models trained by Automated ML. The best accuracy was ``0.80833`` by [SparseNormalizer](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.shared.model_wrappers.sparsenormalizer?view=azure-ml-py), [XGBoostClassifier](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.shared.model_wrappers.xgboostclassifier?view=azure-ml-py)
![Models trained by Automated ML](/starter_file/images/AutoML_Models.PNG)

Here are details and some metrics of the best model.
![Details of the best model](/starter_file/images/AutoML_BestModelDetails.PNG)
![Metircs of the best model](/starter_file/images/AutoML_BestModelMetrics.PNG)

Here is a result of RunDetails widget. (See [automl.ipynb](./automl.ipynb) for more details)
![RunDetails result for AutoML run](/starter_file/images/AutoML_RunDetails2.PNG)

The metrics also can be confirmed by `get_metrics()` of the best run as below.
![Result of get_metrics](/starter_file/images/AutoML_BestModel.PNG)

Since we only have limited time for running our lab and also have limited dataset (`615`), the result was not really great.
We could improve it by increasing volume of training dataset and tarining duration.
We could also try to select other metrics as a primary metric such as `AUC weighted`.

## Hyperparameter Tuning
In this project, we use the [Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) as a classification algorithm.

We specify two hyperparameters, one is the inverse of regularization strength(``C``) and another is the maximum number of iterations to converge(``max_iter``).

In terms of parameter sampling, we use [Random Parameter Sampling](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py). The random sampling supports early termination of low performance runs, therefore, we can save time for training and cost for computing resource and this is good especially for the initial search. This time, the choice of ``6`` values for the parameter C, and the choice of ``8`` values for the parameter max_iter are applied.

Regarding an early termination policy, we use [Bandit Policy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py). This policy ends runs when the primary metric isn't withing the specified slack factor/amount of the most successful run.

In Hyperdrive configuration, we specify ``Accuracy`` as a primary metric which is the same as AutoML project and the primary metric goal is ``PrimaryMetricGoal.MAXIMIZE`` to maximize the primary metric.

We also specify the following two parameters to limit iterations.

max_total_runs: ``1000`` (The maximum total number of runs to create)
max_concurrent_runs: ``10`` (The maximum number of runs to execute concurrently)

### Results
The best accuracy was ``0.8916666666666667`` and it was better than the result of Automated ML model.
Here are models tuned by HyperDrive.
![HyperDrive runs](/starter_file/images/HyperDrive_ChildRuns.PNG)

Here are details and some metrics of the best model.
![Details of the HyperDrive model](/starter_file/images/HyperDrive_BestModel3.PNG)
![Metircs of the HyperDrive model](/starter_file/images/HyperDrive_ChildRunMetrics.PNG)

Here is a result of RunDetails widget. (See [hyperparameter_tuning.ipynb](./hyperparameter_tuning.ipynb) for more details)
![RunDetails result for HyperDrive run](/starter_file/images/HyperDrive_RunDetails.PNG)

The metrics also can be confirmed by `get_metrics()` of the best run as below.
![Result of get_metrics for HyperDrive run](/starter_file/images/HyperDrive_BestModel.PNG)

Fur further improvement, we could try different patterns of parameter sampling although we have already tested some patterns.
Increasing max_total_runs with more data could also be an option.

## Model Deployment
Since the result of HyperDrive model was better than the one from Automated ML, we deploy the HyperDrive model.
Here is a deployed model.
![Deployed endpoint](/starter_file/images/DeployModel_Endpoint.PNG)

We use test data split from the train data to test the endpoint and check results.
We can use ```run``` method of the service endpoint with the input parameter to send a test request, and the service returns a response as below.
![Test endpoint](/starter_file/images/TestEndpoint.PNG)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
### Convert your model to ONNX format
In the Automated AL project, we successfully retrieved the ONNX model from the best run by specifying return_onnx_model option as below.
![Retrive ONNX model](/starter_file/images/StandoutSuggestion_ONNX.PNG)

To get this model, ```enable_onnx_compatible_models``` needs to be set true in AutoMLConfig.

After retrieving the model, it can be saved by using [OnnxConverter](https://docs.microsoft.com/en-us/python/api/azureml-automl-runtime/azureml.automl.runtime.onnx_convert.onnx_converter.onnxconverter?view=azure-ml-py).
![Save ONNX model](/starter_file/images/StandoutSuggestion_ONNX2.PNG)

### Enable logging in your deployed web app
In the HyperDrive project, we successfully enabled ApplicationInsights for detailed logging in the deployed endpoint as below.
![Enabled ApplicationInsights](/starter_file/images/StandoutSuggestion_AppInsightsEnabled.PNG)
![Enabled ApplicationInsights2](/starter_file/images/StandoutSuggestion_AppInsights.PNG)

By doing this, more useful logs can be seen through ApplicationInsights and can be filtered by query.
![Detailed loggs in ApplicationInsights](/starter_file/images/StandoutSuggestion_AppInsights2.PNG)
![Query in ApplicationInsights](/starter_file/images/StandoutSuggestion_AppInsights3.PNG)
