from keras.applications.resnet50 import preprocess_input
import numpy as np

def preprocess_image(image, target_size=(224, 224)):
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array
