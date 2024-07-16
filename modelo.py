import tensorflow as tf
import cv2
import numpy as np

# Carregar o modelo salvo
model = tf.keras.models.load_model('modelo_radiografias.h5')

# Função para preparar a imagem
def prepare_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return None
    image = cv2.resize(image, (1536, 768))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Adicionar uma dimensão extra para batch
    return image

# Exemplo de uso
image_path = 'C:/Users/Siraissi/Documents/GitHub/pjct/img_test/test (1).jpg'
prepared_image = prepare_image(image_path)

# Fazer previsão
if prepared_image is not None:
    predictions = model.predict(prepared_image)
    predicted_class = np.argmax(predictions, axis=1)
    label_map = {
        0: "quadrante 1",
        1: "Quadrante 2",
        2: "Quadrante 3",
        3: "Quadrante 4",
        4: "11",
        5: "12",
        6: "13",
        7: "14",
        8: "15",
        9: "16",
        10: "17",
        11: "18",
        12: "21",
        13: "22",
        14: "23",
        15: "24",
        16: "25",
        17: "26",
        18: "27",
        19: "28",
        20: "31",
        21: "32",
        22: "33",
        23: "34",
        24: "35",
        25: "36",
        26: "37",
        27: "38",
        28: "41",
        29: "42",
        30: "43",
        31: "44",
        32: "45",
        33: "46",
        34: "47",
        35: "48"
    }
    predicted_label = label_map.get(predicted_class[0], "Classe desconhecida")
    print(f"Rótulo previsto: {predicted_label}")
