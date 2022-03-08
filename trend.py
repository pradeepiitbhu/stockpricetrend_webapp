import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
start = '2010-01-01'
end = '2021-12-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data
st.subheader('Data from 2010-2021')
st.write(df.describe())

# visualizations
st.subheader("closing price vs time chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

# simple moving average
df["sma100"] = df.Close.rolling(100).mean()
df["sma200"] = df.Close.rolling(200).mean()

# visualizing Closing Price with simple moving average
st.subheader("closing price vs time chart with 100sma and 200sma")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price')
plt.plot(df.sma100, label='sma100')
plt.plot(df.sma200, label='sma200')
plt.legend()
st.pyplot(fig)

# splitting data into training and testing
data_training = pd.DataFrame(df.Close[0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df.Close[int(len(df)*0.7):])

# data scaling
scaled_training_data = scaler.fit_transform(data_training)
scaled_testing_data = scaler.fit_transform(data_testing)

# testing part
past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)
X_test = []
y_test = []
for i in range(100, len(input_data)):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# load_model
model = load_model('keras_model.h5')
# making prediction
y_predicted = model.predict(X_test)
y_test = y_test/scaler.scale_
y_predicted = y_predicted/scaler.scale_

st.subheader("actual Price and predicted Price ")
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Original Price')
plt.legend()
st.pyplot(fig)

st.subheader("actual Price and predicted Price ")
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Original Price')
plt.plot(y_predicted, label='Predicted Price')
plt.legend()
st.pyplot(fig)