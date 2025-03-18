import cv2
import numpy as np
import matplotlib.pyplot as plt
import watershed
from skimage import io, color, filters, measure, morphology, img_as_ubyte
from skimage.draw import rectangle_perimeter

def detectar_objetos(imagem_path, min_area=100, kernel_size=5):
    # Carregar a imagem e converter para escala de cinza
    img = io.imread(imagem_path)
    if img.ndim == 3:
        cinza = color.rgb2gray(img)
    else:
        cinza = img

    # Aplicar Otsu para binarização automática (inverso igual ao THRESH_BINARY_INV)
    limiar = filters.threshold_otsu(cinza)
    binaria = cinza < limiar  # inverte para fundo preto e objeto branco

    # Fechamento morfológico para remover ruídos
    binaria = morphology.closing(binaria, morphology.square(kernel_size))

    # Rotular objetos conectados
    rotulado = measure.label(binaria)
    props = measure.regionprops(rotulado)

    # Filtrar por área e registrar objetos
    objetos = []
    resultado = img_as_ubyte(color.gray2rgb(cinza))  # Converte para desenhar colorido

    for prop in props:
        if prop.area > min_area:
            minr, minc, maxr, maxc = prop.bbox
            objetos.append({
                'posicao': (minc, minr),
                'dimensoes': (maxc - minc, maxr - minr),
                'area': prop.area
            })
            # Desenhar retângulo vermelho
            rr, cc = rectangle_perimeter(start=(minr, minc), end=(maxr-1, maxc-1), shape=resultado.shape)
            resultado[rr, cc] = [255, 0, 0]  # vermelho

    return resultado, objetos

# Exemplo de uso
imagem_path = f"./imagens_treinamento/7.jpg"
watershed.all_watershed("../imagens_treinamento/7.jpg")
imagem_resultado, objetos_detectados = detectar_objetos("./resultados/watershed.jpg")

# Mostrar resultados
plt.figure(figsize=(12, 6))
plt.subplot(121), 
plt.imshow(cv2.cvtColor(cv2.imread(f"{imagem_path}"), cv2.COLOR_BGR2RGB))
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