![alt text](https://github.com/kirahman2/tensorflow_classification/blob/master/images/tensorflow.png)

## Introduction
The data set in this project has already been preprocessed. It comes from a previous project I did on Fraudulent Credit Card Transactions. [Click here to see how the data was processed](https://github.com/kirahman2/fraud_detection/blob/master/README.md)

The goal of this project is to discover the power of deep learning using Tensorflow and Keras. Can a deep learning model out perform the Sklearn and CatBoost models from the fraud_detection notebook? 

## Data
In this project several variables were previously imputed and new features were created. The dataset was also upsampled and downsampled to balance the dataset. Please refer to the link in the introduction for further explanation.

## Modeling
During the modeling phase, I tested single layer and multilayer nets with varying numbers of neurons. After testing several neural net combinations, I found the best model to have 8 inner layers with 256 neurons. In addition to 8 inner layers, adding L2 regularization along with Dropout reduced overfitting which improved the model score. 

While fitting the model with 100 epochs, I saved each checkpoint and automatically tested and determined the best threshold (down to the thousandth place, .001), which produced the best AUC score for each checkpoint on the test dataset. 

After each checkpoint and threshold was saved, the script automatically stored the threshold and best performing model in Tensorflow format (.tf). Using the load_saved_model method from the SaveModel class, the script loads the model and is able to  make predictions on new data. 

## Results
| Model   | Performance | Performance with Tuning | 
| :------------- |:-------------|:-----|
| Neural Net | 0.000 AUC| 0.871 AUC|
| Logistic Regression | 0.720 AUC| 0.798 AUC|
| Random Forest Classifier | 0.846 AUC| 0.861 AUC|
| XGB Classifier     | 0.819 AUC| 0.880 AUC|
| Cat Boost Classifier | 0.887 AUC| 0.908 AUC|

## Conclusion
I was surprised to see that the neural net did not outperform Catboost. Given the amount of data, I was convinced the neural net would beat CatBoost. Despite testing and combining the best regularization techniques, I couldn’t attain a better score. I spent weeks testing different neural net models, which helped. When I decided to tune each threshold to the thousandth place, I certainly saw an improvement in the neural net’s performance. Ultimately, the way to improve this score would be through creating new features and feeding the model additional data. 

