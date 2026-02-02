# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation

Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset

The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access

Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning

What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- ![ Include a screenshot of completed training jobs](images/112613.png)  
- Logs metrics during the training process:
  test_loss = running_loss / len(test_loader.dataset)
  test_acc = running_corrects/ len(test_loader.dataset)
- Tune at least two hyperparameters:
  hyperparameter_ranges = {"lr": ContinuousParameter(0.001, 0.01),"batch-size": CategoricalParameter([5, 10]),}
- Retrieve the best best hyperparameters from all your training jobs:
   'batch-size': '"10"','lr': '0.004341608444196659',

## Debugging and Profiling

****: Give an overview of how you performed model debugging and profiling in Sagemaker

 Set up debugging and profiling rules and hooks first, then run the estimator.
### Results

****: What are the results/insights did you get by profiling/debugging your model?

The validation loss diverges significantly, which mean Overfitting.
Try bigger Batch Size,lower Learning Rate.

**** Remember to provide the profiler html/pdf file in your submission.

![ profiler html](profiler-report.html)  



## Model Deployment
****: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Deploy the model using the tuner object; query the endpoint using predictor.predict, but got error.

**** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

![ a screenshot of the deployed active endpoint in Sagemaker](images/135849.png)  

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
