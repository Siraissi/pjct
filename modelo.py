import tensorflow as tf
import cv2
import numpy as np

# Carregar o modelo salvo
model = tf.keras.models.load_model('modelo_radiografias2.keras')

# Função para preparar a imagem
def prepare_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None
    image = cv2.resize(image, (768, 384))  # Redimensionamento
    image = image.astype('float32') / 255.0  # Normalização
    image = np.expand_dims(image, axis=0)  # Adicionar uma dimensão extra para batch
    return image

# Exemplo de uso
image_path = 'C:/Users/Siraissi/Pictures/Unimagem-1.jpg'
prepared_image = prepare_image(image_path)

# Fazer previsão e verificar a distribuição das probabilidades
if prepared_image is not None:
    predictions = model.predict(prepared_image)
    print(f"Distribuição das probabilidades: {predictions}")
    predicted_class = np.argmax(predictions, axis=1)
    label_map = {
        0: "panoramica",
        1: "Q1",
        2: "Q2",
        3: "Q3",
        4: "Q4",
    }
    predicted_label = label_map.get(predicted_class[0], "Classe desconhecida")
    print(f"Rótulo previsto: {predicted_label}")
