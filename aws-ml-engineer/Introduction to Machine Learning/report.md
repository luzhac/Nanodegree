# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NAME HERE

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The score is not good, required improvements.

### What was the top ranked model that performed?
WeightedEnsemble_L3 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
To improve the model, additional features like:
Year
Month 
Weekday (to account for weekly variations)
Weather-related features 

### How much better did your model preform after adding additional features and why do you think that is?
The score is more better  than before.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
After attempting hyperparameter tuning, the score did not improve. Due to limited computational resources, I focused on lightweight hyperparameter tuning rather than more extensive or resource-intensive methods.

### If you were given more time with this dataset, where do you think you would spend more time?
Test individual algorithm types (KNN, Neural Nets, RF, XGBoost,Deep learning)  

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|LightGBM_BAG_L2|RandomForestMSE_BAG_L2|LightGBMXT_BAG_L1|1.8|
|add_features|LightGBM_BAG_L2|RandomForestMSE_BAG_L2|LightGBMXT_BAG_L1|0.46|
|hpo|GBM|CAT|RF|0.49|


### Create a line plot showing the top model score for the three (or more) training runs during the project.

 

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.



![model_test_score.png](img/model_test_score.png)

## Summary
When the initial model score is not satisfactory, significant improvements can be achieved by:
Feature Engineering: Adding relevant features (year, month, weather) that capture meaningful patterns.
Hyperparameter Tuning: Fine-tuning model parameters to optimize performance.
