
# %% id="-kMcwx2S7jOR"
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import RMSprop, Adam, SGD
import keras
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math 
from sklearn.metrics import mean_absolute_error
import joblib


dts = pd.read_csv('solarpowergeneration.csv')
dts.head(10)

X = dts.iloc[:, :].values[:-1]
y = dts.iloc[:, -1].values[1:]
print(X.shape, y.shape)
y = np.reshape(y, (-1,1))
y.shape



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Train Shape: {} {} \nTest Shape: {} {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))



from sklearn.preprocessing import StandardScaler
# input scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# outcome scaling:
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)    
y_test = sc_y.transform(y_test)



# def create_spfnet(n_layers, n_activation, kernels):
#   model = tf.keras.models.Sequential()
#   for i, nodes in enumerate(n_layers):
#     if i==0:
#       model.add(Dense(nodes, kernel_initializer=kernels, activation=n_activation, input_dim=X_train.shape[1]))
#       #model.add(Dropout(0.3))
#     else:
#       model.add(Dense(nodes, activation=n_activation, kernel_initializer=kernels))
#       #model.add(Dropout(0.3))
  
#   model.add(Dense(1))
#   model.compile(loss='mse', 
#                 optimizer='adam',
#                 metrics=[tf.keras.metrics.RootMeanSquaredError()])
#   return model



# # %% id="4GmFjW2UePTI" tags=[]
# spfnet = create_spfnet([32, 64], 'relu', 'normal')
# spfnet.summary()

# # %% tags=[]
# from keras.utils.vis_utils import plot_model
# # plot_model(spfnet, to_file='spfnet_model.png', show_shapes=True, show_layer_names=True)

# # %% id="lY4tgg3jjiqF" executionInfo={"status": "ok", "timestamp": 1601102450392, "user_tz": -330, "elapsed": 21938, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="b9756462-30db-4b7c-fab5-a39b088b0e32" colab={"base_uri": "https://localhost:8080/", "height": 1000} tags=[]
# hist = spfnet.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),epochs=300, verbose=2)

# # %% id="f4co7KnVAdTH" executionInfo={"status": "ok", "timestamp": 1601102373259, "user_tz": -330, "elapsed": 2446, "user": {"displayName": "Anant Gupta", "photoUrl": "", "userId": "16039351497174952039"}} outputId="b37e337a-00df-4791-85af-760e67c5eda1" colab={"base_uri": "https://localhost:8080/", "height": 295}

# # plt.plot(hist.history['root_mean_squared_error'])
# # #plt.plot(hist.history['val_root_mean_squared_error'])
# # plt.title('Root Mean Squares Error')
# # plt.xlabel('Epochs')
# # plt.ylabel('error')
# # plt.show()

# # %% tags=[]
# spfnet.evaluate(X_train, y_train)

# # %%
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

# y_pred = spfnet.predict(X_test) # get model predictions (scaled inputs here)
# y_pred_orig = sc_y.inverse_transform(y_pred) # unscale the predictions
# y_test_orig = sc_y.inverse_transform(y_test) # unscale the true test outcomes

# RMSE_orig = mean_squared_error(y_pred_orig, y_test_orig, squared=False)
# print(RMSE_orig)

# # # %%
# # train_pred = spfnet.predict(X_train) # get model predictions (scaled inputs here)
# # train_pred_orig = sc_y.inverse_transform(train_pred) # unscale the predictions
# # y_train_orig = sc_y.inverse_transform(y_train) # unscale the true train outcomes

# mse = mean_absolute_error(y_pred_orig, y_test_orig)

# print(mse) 

# # import pandas as pd 
# save_numpy = np.concatenate((y_pred_orig,y_test_orig), axis=1)
# pd.DataFrame(save_numpy).to_csv("file_ann.csv")

# spfnet.save("my_h5_model.h5")
# sc_y

# joblib.dump(sc_y, "data_transformer.joblib")

reconstructed_model = keras.models.load_model("my_h5_model.h5")


# exit()

# at = np.testing.assert_allclose(
#     spfnet.predict(X_train[0,:].reshape(1,-1)), reconstructed_model.predict(X_train[0,:].reshape(1,-1))
# )


# print(at)

# TensorFlow Lite model defaults 

converter_defaults = tf.lite.TFLiteConverter.from_keras_model(reconstructed_model)

tflite_model_defaults = converter_defaults.convert()

# Convert using dynamic range quantization 
converter_optimize = tf.lite.TFLiteConverter.from_keras_model(reconstructed_model)
converter_optimize.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant_optimize = converter_optimize.convert()

# Convert using float fallback quantization

X_train = X_train.astype(np.float32)
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(reconstructed_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

tflite_model_quant_fallback_quantization = converter.convert()


interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_fallback_quantization)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)


# Convert using integer-only quantization


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(reconstructed_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant_integer_only_quantization = converter.convert()


interpreter = tf.lite.Interpreter(model_content=tflite_model_quant_integer_only_quantization)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

# Save the models as files

import pathlib

tflite_models_dir = pathlib.Path("C:/Users/NGOCNEO/Desktop/Solar-Power-Generation-Forecasting-main/Solar-Power-Generation-Forecasting-main")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"tflite_model_defaults.tflite"
tflite_model_file.write_bytes(tflite_model_defaults)
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

test_image = np.expand_dims(X_train[0], axis=0).astype(input_details["dtype"])
print(test_image.shape)
interpreter.set_tensor(input_details["index"], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details["index"])[0]

# predictions[i] = output.argmax()

print(output)
print(reconstructed_model.predict(X_train[0,:].reshape(1,-1)))




# Helper function to run inference on a TFLite model
# def run_tflite_model(tflite_file, test_image_indices):
#   global test_images

#   # Initialize the interpreter
#   interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
#   interpreter.allocate_tensors()

#   input_details = interpreter.get_input_details()[0]
#   output_details = interpreter.get_output_details()[0]

#   predictions = np.zeros((len(test_image_indices),), dtype=int)
#   for i, test_image_index in enumerate(test_image_indices):
#     test_image = test_images[test_image_index]
#     test_label = test_labels[test_image_index]

#     # Check if the input type is quantized, then rescale input data to uint8
#     if input_details['dtype'] == np.uint8:
#       input_scale, input_zero_point = input_details["quantization"]
#       test_image = test_image / input_scale + input_zero_point

#     test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
#     interpreter.set_tensor(input_details["index"], test_image)
#     interpreter.invoke()
#     output = interpreter.get_tensor(output_details["index"])[0]

#     predictions[i] = output.argmax()

#   return predictions





