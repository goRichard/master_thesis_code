import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


# train the first 100 hours simulation first
# read the data from .csv file

## read the flow rate data and combine it
flow_rate_0 = pd.read_csv('flow_rate_0.csv')
flow_rate_1 = pd.read_csv('flow_rate_0_multipliers_changed_random_dma_1.csv')
flow_rate_2 = pd.read_csv('flow_rate_0_multipliers_changed_random_dma_2.csv')
flow_rate_3 = pd.read_csv('flow_rate_0_multipliers_changed_random_dma_3.csv')
flow_rate_4 = pd.read_csv('flow_rate_0_multipliers_changed_random_dma_4.csv')
flow_rate_5 = pd.read_csv('flow_rate_0_multipliers_changed_random_dma_5.csv')
flow_rate_6 = pd.read_csv('flow_rate_0_multipliers_changed_up_dma_1.csv')
flow_rate_7 = pd.read_csv('flow_rate_0_multipliers_changed_up_dma_2.csv')
flow_rate_8 = pd.read_csv('flow_rate_0_multipliers_changed_up_dma_3.csv')
flow_rate_9 = pd.read_csv('flow_rate_0_multipliers_changed_up_dma_4.csv')
flow_rate_10 = pd.read_csv('flow_rate_0_multipliers_changed_down_dma_5.csv')
flow_rate_11 = pd.read_csv('flow_rate_0_multipliers_changed_down_dma_1.csv')
flow_rate_12 = pd.read_csv('flow_rate_0_multipliers_changed_down_dma_2.csv')
flow_rate_13 = pd.read_csv('flow_rate_0_multipliers_changed_down_dma_3.csv')
flow_rate_14 = pd.read_csv('flow_rate_0_multipliers_changed_down_dma_4.csv')
flow_rate_15 = pd.read_csv('flow_rate_0_multipliers_changed_down_dma_5.csv')

flow_rate_frames = [flow_rate_0, flow_rate_1, flow_rate_2, flow_rate_3, flow_rate_4, flow_rate_5,
                    flow_rate_6, flow_rate_7, flow_rate_8, flow_rate_9, flow_rate_10, flow_rate_11,
                    flow_rate_12, flow_rate_13, flow_rate_14, flow_rate_15]
flow_rate = pd.concat(flow_rate_frames)
control_variable = flow_rate.loc[:, ['PU1', 'PU2', 'PU4', 'PU5', 'PU6', 'PU7', 'PU8', 'V2']].values
# on or off later




## read the head data and combine it
head_0 = pd.read_csv('head_0.csv')
head_1 = pd.read_csv('head_0_multipliers_changed_random_dma_1.csv')
head_2 = pd.read_csv('head_0_multipliers_changed_random_dma_2.csv')
head_3 = pd.read_csv('head_0_multipliers_changed_random_dma_3.csv')
head_4 = pd.read_csv('head_0_multipliers_changed_random_dma_4.csv')
head_5 = pd.read_csv('head_0_multipliers_changed_random_dma_5.csv')
head_6 = pd.read_csv('head_0_multipliers_changed_up_dma_1.csv')
head_7 = pd.read_csv('head_0_multipliers_changed_up_dma_2.csv')
head_8 = pd.read_csv('head_0_multipliers_changed_up_dma_3.csv')
head_9 = pd.read_csv('head_0_multipliers_changed_up_dma_4.csv')
head_10 = pd.read_csv('head_0_multipliers_changed_up_dma_5.csv')
head_11 = pd.read_csv('head_0_multipliers_changed_down_dma_1.csv')
head_12 = pd.read_csv('head_0_multipliers_changed_down_dma_2.csv')
head_13 = pd.read_csv('head_0_multipliers_changed_down_dma_3.csv')
head_14 = pd.read_csv('head_0_multipliers_changed_down_dma_4.csv')
head_15 = pd.read_csv('head_0_multipliers_changed_down_dma_5.csv')


head_frames = [head_0, head_1, head_2, head_3, head_4, head_5,
               head_6, head_7, head_8, head_9, head_10, head_11,
               head_12, head_13, head_14, head_15]

head = pd.concat(head_frames)
state_variable = head.loc[:, ['T1', 'T2', 'T3', 'T4', 'T5', 'T7']].values


# normalised the data before putting it to NN

state_variable_normalised = np.zeros(state_variable.shape)
control_variable_normalised = np.zeros(control_variable.shape)



# concatenate control variable and state variable into one matrix

normalised_data = np.concatenate((state_variable_normalised, control_variable_normalised), axis=1)
print(normalised_data.shape)

# split the data into training part and testing part 80/20
X_train, X_test, y_train, y_test = train_test_split(normalised_data[0:-2, :], normalised_data[1:-1, :], test_size=0.2)

print("training data shape {}".format(X_train.shape))
print("testing data shape {}".format(X_test.shape))
print("training label shape {}".format(y_train.shape))
print("testing label shape {}".format(y_test.shape))


# build the NN
model = Sequential()
model.add(Dense(26, activation='relu', input_dim=13))
model.add(Dense(26, activation='relu', input_dim=26))

model.add(Dense(26, activation='relu', input_dim=26))    # accuracy: 0.3258   # accuracy on training data 0.3106

model.add(Dense(13, activation='relu', input_dim=26))

# model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])

"""
# optimizer adam
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) # accuracy: 0.3258
"""

# optimizer adagrad
#keras.optimizers.adagrad(learning_rate=0.1)
model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mse', 'mae', 'accuracy'])


# train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=500)
score = model.evaluate(X_test, y_test, batch_size=500)
print(score)
# Returns the loss value & metrics values for the model in test mode.
plt.plot(history.history['mse'])
plt.plot(history.history['mae'])
plt.plot(history.history['accuracy'])
plt.show()