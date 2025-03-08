import cv2
import numpy as np
import matplotlib.pyplot as plt
import otsu
import watershed

def detectar_objetos(imagem_path, min_area=100, kernel_size=5):
    # 1. Carregar imagem e converter para escala de cinza
    img = cv2.imread(imagem_path)
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplicar Otsu para binarização automática
    _, binaria = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Pós-processamento morfológico (fechamento para remover ruídos)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    processada = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    
    # 4. Encontrar contornos dos objetos
    contornos, _ = cv2.findContours(processada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Filtrar e desenhar resultados
    resultado = img.copy()
    objetos = []
    
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Retângulo delimitador
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Armazenar dados do objeto
            objetos.append({
                'posicao': (x, y),
                'dimensoes': (w, h),
                'area': area
            })
            
            # Desenhar contorno e retângulo
            cv2.drawContours(resultado, [cnt], -1, (0, 255, 0), 2)
            cv2.rectangle(resultado, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return resultado, objetos

# Exemplo de uso
imagem_path = f"7.jpg"
watershed.all_watershed(imagem_path)
otsu.all_otsu("../resultados/watershed.jpg")
imagem_resultado, objetos_detectados = detectar_objetos("./resultados/otsu.jpg")

# Mostrar resultados
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread(f"./imagens/{imagem_path}"), cv2.COLOR_BGR2RGB))
plt.title('Imagem Original')
plt.subplot(122), plt.imshow(cv2.cvtColor(imagem_resultado, cv2.COLOR_BGR2RGB))
plt.title('Objetos Detectados')
plt.show()

print(f"Foram detectados {len(objetos_detectados)} objetos:")
for i, obj in enumerate(objetos_detectados, 1):
    print(f"Objeto {i}:")
    print(f" - Posição: {obj['posicao']}")
    print(f" - Dimensões: {obj['dimensoes']}")
    print(f" - Área: {obj['area']} pixels")