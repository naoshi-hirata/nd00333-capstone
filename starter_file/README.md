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

The best metric is 0.9988 by MaxAbsScaler LightGBM.
AutoML実験の詳細な結果の表示。
モデルの一覧から、特定のモデルの詳細を選択。
選択したモデルの詳細ページで、モデルのアルゴリズム、ハイパーパラメータ、特徴量の変換、トレーニング時間、使用されたコンピューティングリソースなどの情報が含まれる。
具体的な手順はAzure Machine LearningのUIやコンソールを使用

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
