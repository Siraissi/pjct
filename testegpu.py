import tensorflow as tf
from tensorflow.keras import layers, models

# Verifica se a GPU da AMD está disponível para TensorFlow
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))