# Suavização com Gaussiana:A imagem é suavizada com um filtro Gaussiano para reduzir ruídos. 
# Cálculo da Laplaciana: A segunda derivada da imagem suavizada é calculada, resultando na Laplaciana do Gaussiano (LoG).
# Identificação de zero-crossings: As bordas são detectadas localizando onde a Laplaciana cruza o valor zero.

# Computacionalmente mais barato.

# O critério de detecção de bordas (zero-crossing) pode falhar em algumas situações.
# Difícil controlar a espessura das bordas.

import numpy as np
from PIL import Image
import scipy.ndimage
import matplotlib.pyplot as plt

# Suavização com Gaussiana
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * (ax ** 2) / (sigma ** 2))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image_array, size=5, sigma=2.0):
    kernel = gaussian_kernel(size, sigma)
    return scipy.ndimage.convolve(image_array, kernel, mode='reflect')

# Cálculo da Laplaciana (LoG)
def apply_laplacian(image_array):
    laplacian_kernel = np.array([[1, 1, 1],
                                 [1, -8, 1],
                                 [1, 1, 1]])
    return scipy.ndimage.convolve(image_array, laplacian_kernel, mode='reflect')

# Identificação de zero-crossings
def detect_zero_crossings(log_image, threshold):
    rows, cols = log_image.shape
    edges = np.zeros_like(log_image, dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            patch = log_image[i-1:i+2, j-1:j+2]
            min_val, max_val = patch.min(), patch.max()
            if min_val < 0 and max_val > 0 and (max_val - min_val) > threshold:
                edges[i, j] = 255

    return edges

def marr_hildreth_edge_detection(image, size=7, sigma=2.5):
    # Convertendo imagem para array float
    image_gray = image.convert("L")
    image_array = np.array(image_gray, dtype=np.float32)

    # Aplicando filtro Gaussiano
    smoothed = apply_gaussian_filter(image_array, size, sigma)

    # Aplicando Laplaciano
    log_image = apply_laplacian(smoothed)

    # Normalização
    log_image = (log_image - log_image.min()) / (log_image.max() - log_image.min())
    log_image = log_image * 255 - 128

    # Detectando cruzamentos por zero
    edges = detect_zero_crossings(log_image, threshold=15.0)

    # Convertendo resultado para PIL Image
    result = Image.fromarray(edges)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title("Imagem Grayscale")
    axes[0].axis("off")

    axes[1].imshow(result, cmap="gray")
    axes[1].set_title("Bordas Marr-Hildreth")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return result

