import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import watershed

def detectar_objetos(imagem_path, threshold, min_area=100, kernel_size=5):
    img = watershed.watershed_segmentation(imagem_path, threshold)

    # Carrega a imagem e converte para escala de cinza
    img = img.convert('L')
    img_array = np.array(img)

    # Aplica Otsu
    hist, bins = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
    total = img_array.size
    sum_total = np.dot(np.arange(256), hist)
    sumB, wB, wF, var_max, threshold = 0, 0, 0, 0, 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > var_max:
            var_max = var_between
            threshold = t

    # Binariza com o threshold de Otsu
    binaria = (img_array < threshold).astype(np.uint8)

    # Fechamento morfológico simples (dilatação seguida de erosão)
    def morfologia(feat):
        pad = kernel_size // 2
        padded = np.pad(feat, pad, mode='constant')
        dilatado = np.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                dilatado[i, j] = np.max(padded[i:i + kernel_size, j:j + kernel_size])
        padded = np.pad(dilatado, pad, mode='constant')
        fechado = np.zeros_like(feat)
        for i in range(feat.shape[0]):
            for j in range(feat.shape[1]):
                fechado[i, j] = np.min(padded[i:i + kernel_size, j:j + kernel_size])
        return fechado

    binaria = morfologia(binaria)

    # Implementação de labeling (marcadores)
    def flood_fill(label_img, x, y, label):
        stack = [(x, y)]
        coords = []
        while stack:
            cx, cy = stack.pop()
            if (0 <= cx < label_img.shape[1] and 0 <= cy < label_img.shape[0]
                and label_img[cy, cx] == 1):
                label_img[cy, cx] = label
                coords.append((cx, cy))
                stack.extend([
                    (cx + 1, cy), (cx - 1, cy),
                    (cx, cy + 1), (cx, cy - 1)
                ])
        return coords

    label_img = binaria.copy()
    current_label = 2
    objetos = []
    resultado = img.convert('RGB')
    draw = ImageDraw.Draw(resultado)

    for y in range(label_img.shape[0]):
        for x in range(label_img.shape[1]):
            if label_img[y, x] == 1:
                pixels = flood_fill(label_img, x, y, current_label)
                area = len(pixels)
                if area >= min_area:
                    xs, ys = zip(*pixels)
                    minx, maxx = min(xs), max(xs)
                    miny, maxy = min(ys), max(ys)
                    objetos.append({
                        'posicao': (minx, miny),
                        'dimensoes': (maxx - minx, maxy - miny),
                        'area': area
                    })
                    # Desenhar retângulo
                    draw.rectangle([minx, miny, maxx, maxy], outline=(255, 0, 0), width=2)
                current_label += 1

    return np.array(resultado), objetos, img
 
# Detecta objetos
def detectar_todos_objetos(imagem, threshold):
    imagem_resultado, objetos_detectados, grayscale_img = detectar_objetos(imagem, threshold)

    plt.figure(figsize=(18, 6))

    imagem = imagem.convert("RGB")

    # Imagem original
    plt.subplot(1, 4, 1)
    plt.imshow(imagem)
    plt.title('Imagem Original')
    plt.axis('off')

    # Grayscale
    plt.subplot(1, 4, 2)
    plt.imshow(grayscale_img, cmap='gray')
    plt.title('Watershed')
    plt.axis('off')

    # Objetos Detectados
    plt.subplot(1, 4, 3)
    plt.imshow(imagem_resultado)
    plt.title('Objetos Detectados')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print dos objetos detectados
    print(f"Foram detectados {len(objetos_detectados)} objetos:")
    for i, obj in enumerate(objetos_detectados, 1):
        print(f"Objeto {i}:")
        print(f" - Posição: {obj['posicao']}")
        print(f" - Dimensões: {obj['dimensoes']}")
        print(f" - Área: {obj['area']} pixels")

    return imagem_resultado, objetos_detectados, grayscale_img