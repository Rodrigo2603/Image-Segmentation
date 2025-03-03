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

def all_marr_hildreth(imagem):

    # Suavização com Gaussiana
    def gaussian_kernel(size, sigma):
        ax = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * (ax ** 2) / (sigma ** 2))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    def apply_gaussian_filter(image_array, size=5, sigma=2.0):
        kernel = gaussian_kernel(size, sigma)
        return scipy.ndimage.convolve(image_array, kernel, mode='reflect')

    # Cálculo da Laplaciana
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

    def marr_hildreth_edge_detection(image_path, size=7, sigma=2.5):
        image = Image.open(image_path).convert("L")
        image_array = np.array(image, dtype=np.float32)
        
        smoothed = apply_gaussian_filter(image_array, size, sigma)
        log_image = apply_laplacian(smoothed)

        # Normalização para evitar valores extremos
        log_image = (log_image - log_image.min()) / (log_image.max() - log_image.min())
        log_image = log_image * 255 - 128

        edges = detect_zero_crossings(log_image, threshold=15.0)
        
        result = Image.fromarray(edges)
        result.save("./resultados/marr_hildreth_edges.jpg")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Imagem Grayscale")
        axes[0].axis("off")

        axes[1].imshow(result, cmap="gray")
        axes[1].set_title("Bordas Marr-Hildreth")
        axes[1].axis("off")

        plt.show()

    image_path = f"./imagens/{imagem}"
    marr_hildreth_edge_detection(image_path)
