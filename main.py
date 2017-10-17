import pandas as pd
from keras.models import Sequential
from keras.layers import *

############## Train Data ################################################
train_data = pd.read_csv('sales_data_training_scaled.csv')

X = train_data.drop('total_earnings', axis=1).values  # array
y = train_data[['total_earnings']].values

################ Define Model #############################################
model = Sequential()

model.add(Dense(50, input_dim=9, activation='relu')) # 50 nodes ,9 input( wiz features)
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))              # final node for output

model.compile(loss='mean_squared_error', optimizer='adam')

################# Train Model ##############################################
model.fit(
    X,
    y,
    epochs=50,
    shuffle=True,
    verbose=2
)
################### Test data  #############################################
test_data = pd.read_csv('sales_data_testing_scaled.csv')

X_test = test_data.drop('total_earnings', axis=1).values
y_test = test_data[['total_earnings']].values

##################  Evaluation  ############################################
test_error_rate = model.evaluate(X_test,y_test,verbose=0)
print("MSE for test data set {}".format(test_error_rate))

####################### Prediction ###########################################

X_predict  = pd.read_csv("proposed_new_product.csv").values

prediction = model.predict(X_predict)
# keras give multiple prediction in form of array
# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0] ## a number (not array)
'''
print("length prediction :",len(prediction))
print("prediction[0] :",prediction[0])
print("prediction[0][0] :",prediction[0][0])
'''

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range

# from preprocessing we know ,
# total_earnings values were scaled by multiplying by 0.0000036968 and adding -0.115913

prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("Earnings Prediction for Proposed Product - ${}".format(prediction))

############## SAVING MODEL TO DISK ##############################################
model.save("trained_model.h5")
print("Model SAVED !")

############## USING TRAINED MODEL EXAMPLE #######################################
'''
from keras.models import load_model
import pandas as pd

model = load_model('trained_model.h5')

X_predict = pd.read_csv("proposed_new_product.csv").values
prediction = model.predict(X_predict)

prediction = prediction[0][0] ## a number (not array)

prediction = prediction + 0.1159
prediction = prediction / 0.0000036968

print("earning: ${}".format(prediction))
'''