import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split

# Caminhos dos diretórios
input_dir = 'C:/Users/Siraissi/Documents/GitHub/pjct/annotation/Radiografias/Panoramicas/pan_cut'
annotations_dir = 'C:/Users/Siraissi/Documents/GitHub/pjct/annotation'

# Parâmetros do modelo
input_shape = (1536, 768, 3)  # Ajustado para (1536, 768, 3)
num_classes = 1  # Número de classes de anotações

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
            
            # Verificar se a estrutura do JSON é válida
            if 'objects' in annotation and len(annotation['objects']) > 0:
                object_info = annotation['objects'][0]
                class_title = object_info.get('classTitle', None)

                if class_title == "Panoramica":  # Exemplo de condição para classe específica
                    # Obter o nome do arquivo da imagem a partir de outra fonte nas anotações
                    # Aqui você deve ajustar conforme a estrutura real das suas anotações
                    # Por exemplo, se o nome do arquivo estiver em algum campo específico do JSON:
                    # image_filename = annotation.get('imageName', None)

                    # Supondo que você tenha o nome do arquivo como parte do nome do JSON
                    image_filename = filename.replace('.json', '.jpg')
                    image_path = os.path.join(input_dir, image_filename)

                    # Carregar e redimensionar a imagem
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Erro ao carregar a imagem: {image_path}")
                        continue

                    image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Redimensionamento compatível
                    images.append(image)

                    # Adicionar label correspondente
                    labels.append(0)  # Defina o índice da classe, se necessário
                else:
                    print(f"Classe não suportada: {class_title}")
            else:
                print(f"Estrutura inválida no arquivo JSON: {filename}")

    images = np.array(images, dtype='float32') / 255.0
    labels = to_categorical(labels, num_classes=num_classes)
    return images, labels

# Carregar e dividir os dados
images, labels = load_data(input_dir, annotations_dir)
print(f"Total de imagens carregadas: {len(images)}")
print(f"Total de labels carregados: {len(labels)}")

# Verificar o tamanho dos dados após o carregamento
if len(images) == 0 or len(labels) == 0:
    raise ValueError("Não foram carregadas imagens ou labels suficientes.")

# Dividir os dados em treino e validação
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Verificar o tamanho dos conjuntos de treino e validação
print(f"Tamanho de x_train: {len(x_train)}, Tamanho de x_val: {len(x_val)}")

# Usar tf.data.Dataset para carregamento de dados
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(124).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(124).prefetch(tf.data.experimental.AUTOTUNE)

# Definir o modelo
model = models.Sequential([
    keras.Input(shape=input_shape),  # Definindo a entrada explicitamente
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Configurar callbacks para liberar memória
callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.ModelCheckpoint(filepath='model_checkpoint.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

# Treinar o modelo
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)

# Salvar o modelo treinado
model.save('modelo_radiografias2.keras')

print("Treinamento concluído e modelo salvo.")
