
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



reconstructed_model = keras.models.load_model("my_h5_model.h5")




# Convert using float fallback quantization

X_train = X_train.astype(np.uint8)


# Save the models as files

import pathlib

tflite_models_dir = pathlib.Path("C:/Users/NGOCNEO/Desktop/Solar-Power-Generation-Forecasting-main/Solar-Power-Generation-Forecasting-main/ANN")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"tflite_model_defaults.tflite"
# tflite_model_file.write_bytes(tflite_model_defaults)
# Save the quantized model:
tflite_model_quant_fallback_quantization = tflite_models_dir/"tflite_model_quant_fallback_quantization.tflite"
# tflite_model_quant_file.write_bytes(tflite_model_quant_fallback_quantization)


# Save the unquantized/float model:
tflite_model_quant_optimize = tflite_models_dir/"tflite_model_quant_optimize.tflite"
# tflite_model_file.write_bytes(tflite_model_quant_optimize)
# Save the quantized model:
tflite_model_quant_integer_only_quantization = tflite_models_dir/"tflite_model_quant_integer_only_quantization.tflite"
# tflite_model_quant_file.write_bytes(tflite_model_quant_integer_only_quantization)

# run model

def run_tflite(tflite_model_file,predict_test):

	interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]

	test_image = np.expand_dims(X_train[0], axis=0).astype(input_details["dtype"])
	print(test_image.shape)
	interpreter.set_tensor(input_details["index"], test_image)
	interpreter.invoke()
	output = interpreter.get_tensor(output_details["index"])[0]
	return output

# predictions[i] = output.argmax()

# print(output)
# print(reconstructed_model.predict(X_train[0,:].reshape(1,-1)))
# X_test = X_test.astype(np.float32)


y_pred = []
# print(X_test[0])

for i in range(len(X_test)):

	predict_test = X_test[i]

	# print(predict_test)
	# exit()
	output = run_tflite(tflite_model_quant_integer_only_quantization,predict_test)
	y_pred.append(output)


y_pred = np.array(y_pred)

y_pred = y_pred.astype(np.float32)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y_pred_orig = sc_y.inverse_transform(y_pred) # unscale the predictions
y_test_orig = sc_y.inverse_transform(y_test) # unscale the true test outcomes

RMSE_orig = mean_squared_error(y_pred_orig, y_test_orig, squared=False)
print(RMSE_orig)

# # %%
# train_pred = spfnet.predict(X_train) # get model predictions (scaled inputs here)
# train_pred_orig = sc_y.inverse_transform(train_pred) # unscale the predictions
# y_train_orig = sc_y.inverse_transform(y_train) # unscale the true train outcomes

mse = mean_absolute_error(y_pred_orig, y_test_orig)

print(mse) 










