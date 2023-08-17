*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Project Title: Creating Car hierarchical decision model from evaluation cars database

*TODO:* short introduction to this project: 

 Create a multi-class classification model using two methods: AutoML and Hyperparameter Tuning, based on a database containing 7 features of passenger cars and their corresponding evaluation classes. Deploy one of these models as a web service for utilization.

## Project Set Up and Installation
*OPTIONAL:* 

 Upload the starter files "automl.ipynb," "hyperparameter_tuning.ipynb," and "train.py" to the AzureML notebook folder.

## Dataset: 
 
### Overview
*TODO*: Explain about the data you are using and where you got it from:


 The dataset comes from https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data.
 This Car Evaluation Database was derived from a simple hierarchical decision model. 
 The model evaluates cars(class) according to the six input attributes: buying, maint, doors, persons, lug_boot, safety.
  -Number of Instances: 1728 (instances completely cover the attribute space)
  -Number of Attributes: 6
  -Attribute Values:
    buying       v-high, high, med, low
    maint        v-high, high, med, low
    doors        2, 3, 4, 5-more
    persons      2, 4, more
    lug_boot     small, med, big
    safety       low, med, high
 -Missing Attribute Values: none
 -Class Distribution (number of instances per class)
    class      N          N[%]
    -----------------------------
    unacc     1210     (70.023 %) 
    acc        384     (22.222 %) 
    good        69     ( 3.993 %) 
    v-good      65     ( 3.762 %) 


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it:

 The class distribution is imbalanced to a significant degree. 
 The AUC-weighted metric is chosen as the primary metric for this.

### Access
*TODO*: Explain how you are accessing the data in your workspace:

 I have read the CSV file into a DataFrame and added column names since they were missing. Additionally, as the features are in textual format, I converted them to integer values for hyperparameter tuning.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment:

- The automl_settings contain the configuration parameters for AutoML, including experiment timeout, early stopping, iteration timeout, concurrency settings, primary metric selection, featurization, verbosity, and code generation.
- The automl_config is the configuration for the AutoML experiment. It specifies the task as classification, uses a debug log, specifies the compute target, enables ONNX compatible models, defines the training data, specifies the label column, sets the project folder path, and incorporates the automl_settings.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?:

RESULT; The best AUC weighted is 0.9988 by VotingEnsemble. 
Confusion Matrix;
                     predicted label
                     Vgood   good    acc   unacc
true label Vgood       63      1       1      0
           good         0     60       9      0
           acc          1      2     379      2
           unacc        0      0      15   1194

PARAMETERS; The top 4 features of importance are "safety", "persons capasity", "buying price" and "maint. price"

IMPROVE; The metrics and confusion matrics are very good though "Class balancing detection" alert is shown. 
To improve more, I would like to reengineer features or create new feature. This database has only three categories, cost, size, safety. But owner may consider the performance, entertaiment, styling, and so on. 
In case I stay with this database, I would like to try other metrics rather than auc-weighted,and to change validation setting to improve model stability and generalization performance.


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

model:RamdomForests
Random Forests is an ensemble learning method that combines multiple decision trees. It can exhibit strong classification performance on datasets that include categorical data. By combining multiple decision trees, it helps mitigate overfitting and improves generalization performance.

hyperparameters:
n_estimators: the model's complexity and expressive power. choice(100, 500, 1000)
min_samples_split: the minimum number of samples required for a split node. affect to the model's generalization performance.choice(2, 10, 20)
min_samples_leaf: the minimum number of samples required for a leaf node.choice(1, 5, 10)


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

RESULT: the best auc_weighted is 0.9997879003389194

HYPER PARAMETERS: the best Run Hyperparameters:
--min_samples_leaf
1
--min_samples_split
2
--n_estimators
500

IMPROVING: The metrics is very high. To improve it, add more data again.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

OVERVIEW; The deployed model predicts car acceptability. The model is deployed as a web service using Azure Container Instances, allowing users to make predictions based on the trained model.

INSTRUCTION;
- Get the Scoring URI; this is the endpoint you will be sending requests to.
- Create a sample Input Data; This should match the format of the train data.
- Convert Input to JSON; Convert the input data dictionary into a JSON string using the json.dumps() function.
- Set Headers and Make Request; Set the headers for the request, which should include the content type as 'application/json'. Then, use the requests.post() method to send a POST request to the scoring URI with the JSON input data.
- Get Predictions; The response object contains the predictions returned by the deployed model. You can extract the predictions using the .json() method.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

https://youtu.be/bMNHnhi48K0


## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

Convert a model to ONNX format;
Machine learning models have different file formats for each algorithm. In AzureML's AutoML, since it's not possible to identify the specific algorithm in advance, ONNX format is utilized to save the created model.
