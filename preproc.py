import cv2
import os

# Diretório de entrada e saída
input_dirs = ['C:/Users/Siraissi/Documents/GitHub/pjct/img_treino', 'C:/Users/Siraissi/Documents/GitHub/pjct/img_valid']
output_dirs = ['C:/Users/Siraissi/Documents/GitHub/pjct/img_treino/img_treino_cut', 'C:/Users/Siraissi/Documents/GitHub/pjct/img_valid/img_valid_cut']

# Verificar se as listas de diretórios têm o mesmo comprimento
if len(input_dirs) != len(output_dirs):
    raise ValueError("As listas de diretórios de entrada e saída devem ter o mesmo comprimento.")

# Função para cortar e redimensionar imagens
def preprocess_image(image_path, output_path, target_size=(1536, 768)):
    # Carregar a imagem
    image = cv2.imread(image_path)
    
    # Verificar se a imagem foi carregada corretamente
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return
    
    # Cortar a imagem (exemplo: cortando uma área central)
    height, width = image.shape[:2]
    margin_x = int(width * 0.1)  # 10% da largura
    margin_y = int(height * 0.1)  # 10% da altura
    start_x = margin_x
    start_y = margin_y
    end_x = width - margin_x
    end_y = height - margin_y
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    # Redimensionar a imagem
    resized_image = cv2.resize(cropped_image, target_size)
    
    # Salvar a imagem processada
    cv2.imwrite(output_path, resized_image)

# Processar todas as imagens nos diretórios de entrada
for input_dir, output_dir in zip(input_dirs, output_dirs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            preprocess_image(input_path, output_path)

print("Pré-processamento concluído.")