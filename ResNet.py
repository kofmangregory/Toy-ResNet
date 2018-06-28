import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

json_file = open('model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model/model_weights.h5")

img_path = 'images/five.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
probabilities = model.predict(x)
one_hot = np.zeros(probabilities.shape)
one_hot[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
print('Input image shape:', x.shape)
print("Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(one_hot)