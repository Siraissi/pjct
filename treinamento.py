import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# Configuração de GPU com limitação de memória
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Limite de memória em MB
    except RuntimeError as e:
        print(e)

# Caminhos dos diretórios
input_dir = 'C:/Users/Siraissi/Documents/GitHub/pjct/annotation/Radiografias/Panoramicas'
annotations_dir = 'C:/Users/Siraissi/Documents/GitHub/pjct/annotation'

# Parâmetros do modelo
input_shape = (768, 1536, 3)  # Ajustado para (768, 1536, 3)
num_classes = 36  # Número de classes de anotações

# Função para carregar as anotações
def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Função para carregar e preparar os dados
def load_data(input_dir, annotations_dir):
    images = []
    labels = []

    for filename in os.listdir(annotations_dir):
        if filename.endswith('.json'):
            annotation_path = os.path.join(annotations_dir, filename)
            annotation = load_annotations(annotation_path)
            image_path = os.path.join(input_dir, annotation['imageName'])
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erro ao carregar a imagem: {image_path}")
                continue

            image = cv2.resize(image, (1536, 768))

            images.append(image)

            # Para simplificar, usaremos apenas a primeira anotação como o rótulo
            if annotation['annotation']['objects']:
                label = annotation['annotation']['objects'][0]['classTitle']
                label_index = {
                    "quadrante 1": 0,
                    "Quadrante 2": 1,
                    "Quadrante 3": 2,
                    "Quadrante 4": 3,
                    "11": 4,
                    "12": 5,
                    "13": 6,
                    "14": 7,
                    "15": 8,
                    "16": 9,
                    "17": 10,
                    "18": 11,
                    "21": 12,
                    "22": 13,
                    "23": 14,
                    "24": 15,
                    "25": 16,
                    "26": 17,
                    "27": 18,
                    "28": 19,
                    "31": 20,
                    "32": 21,
                    "33": 22,
                    "34": 23,
                    "35": 24,
                    "36": 25,
                    "37": 26,
                    "38": 27,
                    "41": 28,
                    "42": 29,
                    "43": 30,
                    "44": 31,
                    "45": 32,
                    "46": 33,
                    "47": 34,
                    "48": 35
                }.get(label, 0)
                labels.append(label_index)

    images = np.array(images, dtype='float32') / 255.0
    labels = to_categorical(labels, num_classes=num_classes)
    return images, labels

# Carregar e dividir os dados
images, labels = load_data(input_dir, annotations_dir)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Definir o gerador de dados
datagen = ImageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=8)  # Tamanho do lote reduzido para 8
val_gen = datagen.flow(x_val, y_val, batch_size=8)  # Tamanho do lote reduzido para 8

# Definir o modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(train_gen, epochs=10, validation_data=val_gen)

# Salvar o modelo treinado
model.save('modelo_radiografias2.h5')

print("Treinamento concluído e modelo salvo.")
