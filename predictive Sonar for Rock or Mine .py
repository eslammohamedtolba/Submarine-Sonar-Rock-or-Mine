# import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# uploading dataset to a Pandas DataFrame
Sonar_data = pd.read_csv("sonar.csv",header=None)



# print dataset
Sonar_data.head()
# print dataset number of rows and column
Sonar_data.shape
# get some statistical info about the dataset
Sonar_data.describe()
# how many values for mines and how many values for rocks from the output
Sonar_data[60].value_counts()
# discover the relation between and output and each column
Sonar_data.groupby(60).mean()



# serparate data into input and label
X = Sonar_data.drop(columns=60,axis=1)
Y = Sonar_data[60]
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.1,random_state=2)

# training the logistic regression model with training data
LRModel = LogisticRegression()
LRModel.fit(x_train,y_train)




# accuracy on training data
x_train_prediction = LRModel.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
# accuracy on test data
x_test_prediction = LRModel.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print(training_data_accuracy,test_data_accuracy)




# Making a predictive System
input_data=(0.0131,0.0387,0.0329,0.0078,0.0721,0.1341,0.1626,0.1902,0.2610,0.3193,0.3468,0.3738,0.3055,0.1926,0.1385,0.2122,
            0.2758,0.4576,0.6487,0.7154,0.8010,0.7924,0.8793,1.0000,0.9865,0.9474,0.9474,0.9315,0.8326,0.6213,0.3772,0.2822,
            0.2042,0.2190,0.2223,0.1327,0.0521,0.0618,0.1416,0.1460,0.0846,0.1055,0.1639,0.1916,0.2085,0.2335,0.1964,0.1300,
            0.0633,0.0183,0.0137,0.0150,0.0076,0.0032,0.0037,0.0071,0.0040,0.0009,0.0015,0.0085) # input Data to predict
# changing input data into numpy array
input_data_array = np.asarray(input_data)
# reshape the 1D input data array into 2D to predict
reshaped_input_data_array = input_data_array.reshape(1,-1)
# predict the output of the user input
prediction = LRModel.predict(reshaped_input_data_array)
if prediction[0]=='R':
    print("The predicted object is Rock")
else:
    print("The predicted object is Mine")

