# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import timedelta
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import mean_absolute_error
import keras
import json

print(tf.__version__)
print(np.__version__)
dts = pd.read_csv('solarpowergeneration.csv')
# dts.head(10)
dts1 = dts
dts1['index'] = range(1, len(dts1) + 1)
dts1.info()
dts.info()
result = dts1.to_json(orient="columns")
parsed = json.loads(result)
json.dumps(parsed, indent=10)

X = dts.iloc[:, :].values
y = dts.iloc[:, -2].values


print(X.shape, y.shape)


y = np.reshape(y, (-1,1))
X = y
print(X.shape)

# exit()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
# Maxmin = MinMaxScaler(feature_range=(0, 1))
y_strud_lstm = scaler.fit_transform(X)
# y_p =  Maxmin.fit_transform(X[:,-1].reshape(-1,1))

print("mmmm",y_strud_lstm.shape[0])

train_size = int(y_strud_lstm.shape[0] * 0.7)


print('kich thuoc train la',train_size)



test_size = y_strud_lstm.shape[0] - train_size
train  = y_strud_lstm[0:train_size,:]
test  = y_strud_lstm[train_size:y_strud_lstm.shape[0],:]
print('kich thuoc data' ,train.shape, test.shape)


import math 
nodes = 40
epochs = 400
verbose = 2 # 0=print no output, 1=most, 2=less, 3=least
look_back = 22
lstm_params = [nodes, epochs, verbose]
def create_dataset(dataset, look_back=look_back):
    print(dataset.shape[0], look_back)
    dataX, dataY = [], []
    for i in range(dataset.shape[0]-look_back-1):
        # print(i)
        a = dataset[i:(i+look_back)]
        # print(a)
        # print(dataset[:,i + look_back])
        # exit()
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


batch_size = 30



# creating the training and testing datasets
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)
print(X_train.shape,X_test.shape)


# print(X_train.shape, y_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0],1, X_train.shape[1]))
# y_train = np.reshape(y_train,(y_train.shape[0],1))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))
# y_test = np.reshape(y_test,(y_test.shape[0],1))
# X_train = X_train[...,  np.newaxis]
# X_test = X_test[...,  np.newaxis]


print(X_train.shape, y_train.shape)

# training the model
model = tf.keras.models.Sequential([
    LSTM(lstm_params[0], input_shape=(1,look_back),return_sequences=True),
    LSTM(units = 40, return_sequences = True),
    Dense(units = 1)
    ])

model.compile(loss='mean_squared_error', optimizer='adam',metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.fit(X_train, y_train, epochs=lstm_params[1], batch_size = batch_size, verbose=lstm_params[2])



print(model.summary())


trainPredict = model.predict(X_train, batch_size=batch_size)

print("trainPredict",trainPredict.shape,trainPredict[0])

print(y_train[0])

testPredict = model.predict(X_test, batch_size=batch_size)

model.reset_states()


# reshape the data for invert predicts 
trainPredict = np.reshape(trainPredict,(trainPredict.shape[0],trainPredict.shape[1]))
# y_train = np.reshape(y_train,(y_train.shape[0],y_train.shape[1]))
testPredict = np.reshape(testPredict,(testPredict.shape[0],testPredict.shape[1]))
# y_test = np.reshape(y_test,(y_test.shape[0],y_test.shape[1]))

print(trainPredict.shape, y_train.shape)


# invert predictions
y_train = scaler.inverse_transform(y_train)
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform(y_test)


print(trainPredict.shape, y_train.shape)

# calculate root mean squared error
trainScore = mean_squared_error(y_test, testPredict)
print('Train Score: %.2f RMSE' % (trainScore))
testScore = mean_absolute_error(y_test, testPredict)
print('Test Score: %.2f MAE' % (testScore))

# mse = mean_absolute_error(y_pred_orig, y_test_orig)

exit()


model.save("my_h5_model_lstm_h1_onestep.h5")


# joblib.dump(sc_y, "data_transformer.joblib")
reconstructed_model = keras.models.load_model("my_h5_model_lstm_h1_onestep.h5")

# 


run_model = tf.function(lambda x: model(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = 1
INPUT_SIZE = 22
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

# model directory.
MODEL_DIR = "keras_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()


print("model 1 finish")
# print(X_train[0,...].shape)
# print(X_train[0,:])

# exit()

# at = np.testing.assert_allclose(
#     model.predict(X_train[0,:,...].reshape(1,-1)), reconstructed_model.predict(X_train[0,:,...].reshape(1,-1))
# )

# import time
# atime = []
# for i in range(100):
#   start_time = time.time()
#   model.predict(X_train[0,:].reshape(1,-1))
#   at = time.time() - start_time
#   atime.append(at)


# atime = np.array(atime)
# print(atime)

# print(atime.mean())
# TensorFlow Lite model d



# Convert using dynamic range quantization 
converter_optimize = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter_optimize.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant_optimize = converter_optimize.convert()

print("model 2 finish")


# Convert using float fallback quantization

X_train = X_train.astype(np.float32)
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

tflite_model_quant_fallback_quantization = converter.convert()


interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_fallback_quantization)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)


print("model 3 finish")

# Convert using integer-only quantization


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant_integer_only_quantization = converter.convert()


interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_integer_only_quantization)
input_type = interpreter.get_input_details()[0]['shape']
print('input: ', interpreter.get_input_details()[0]['dtype'].shape)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

# Save the models as files


print("model 4 finish")

import pathlib

tflite_models_dir = pathlib.Path("C:/Users/NGOCNEO/Desktop/Solar-Power-Generation-Forecasting-main/Solar-Power-Generation-Forecasting-main/LSTM_onestep")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"tflite_model_defaults.tflite"
tflite_model_file.write_bytes(tflite_model)
# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"tflite_model_quant_fallback_quantization.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant_fallback_quantization)


# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"tflite_model_quant_optimize.tflite"
tflite_model_file.write_bytes(tflite_model_quant_optimize)
# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"tflite_model_quant_integer_only_quantization.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant_integer_only_quantization)

# run model

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(X_train[0].shape)

test_image = np.expand_dims(X_train[0], axis=0).astype(input_details["dtype"])

print(test_image.shape)
interpreter.set_tensor(input_details["index"], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details["index"])[0]

# predictions[i] = output.argmax()

print(output)
print(reconstructed_model.predict(X_train[0][np.newaxis,...]))