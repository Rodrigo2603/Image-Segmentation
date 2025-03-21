# Suavização com Gaussiana: Assim como no Marr-Hildreth, a imagem é suavizada para reduzir ruídos.
# Cálculo do gradiente da imagem: Nesse caso utilizando derivada de Sobel
# Supressão não-máxima: Apenas os pixels com maior intensidade ao longo da direção do gradiente são mantidos, tornando as bordas mais finas e precisas.
# Limiarização dupla: Define dois limiares (alto e baixo)->
# Bordas fortes (acima do limiar alto) são mantidas.
# Bordas fracas (entre os limiares) são mantidas apenas se estiverem conectadas a bordas fortes.

# Produz bordas finas e precisas.
# Menos sensível a ruído devido à suavização inicial.

# Mais complexo e computacionalmente mais caro que Marr-Hildreth.
# Sensível à escolha dos limiares.

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def gaussian_filter(image, size=5, sigma=1.0):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * 
                             np.exp(-((x-(size//2))**2 + (y-(size//2))**2) / (2*sigma**2)), 
                             (size, size))
    kernel /= np.sum(kernel)
    return image.filter(ImageFilter.Kernel((size, size), kernel.flatten(), scale=np.sum(kernel)))

def sobel_gradients(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gray_array = np.array(image.convert("L"), dtype=np.float32)
    gx, gy = np.zeros_like(gray_array), np.zeros_like(gray_array)

    for i in range(1, gray_array.shape[0] - 1):
        for j in range(1, gray_array.shape[1] - 1):
            gx[i, j] = np.sum(sobel_x * gray_array[i-1:i+2, j-1:j+2])
            gy[i, j] = np.sum(sobel_y * gray_array[i-1:i+2, j-1:j+2])

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    return magnitude, direction

def non_max_suppression(magnitude, direction):
    suppressed = np.zeros_like(magnitude)
    angle = direction * (180 / np.pi) % 180
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            q, r = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q, r = magnitude[i, j+1], magnitude[i, j-1]
            elif (22.5 <= angle[i, j] < 67.5):
                q, r = magnitude[i+1, j-1], magnitude[i-1, j+1]
            elif (67.5 <= angle[i, j] < 112.5):
                q, r = magnitude[i+1, j], magnitude[i-1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q, r = magnitude[i-1, j-1], magnitude[i+1, j+1]

            if magnitude[i, j] >= q and magnitude[i, j] >= r:
                suppressed[i, j] = magnitude[i, j]
    return suppressed

def double_threshold(image, low_threshold, high_threshold):
    strong, weak = 255, 75
    edges = np.zeros_like(image)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    edges[strong_i, strong_j] = strong
    edges[weak_i, weak_j] = weak
    return edges

def hysteresis(image):
    strong, weak = 255, 75
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] == weak:
                if ((image[i+1, j] == strong) or (image[i-1, j] == strong) or 
                    (image[i, j+1] == strong) or (image[i, j-1] == strong) or 
                    (image[i+1, j+1] == strong) or (image[i-1, j-1] == strong) or 
                    (image[i+1, j-1] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection(image, Tl=50, Th=150):
    image = image.convert("L")
    smoothed = gaussian_filter(image)
    magnitude, direction = sobel_gradients(smoothed)
    suppressed = non_max_suppression(magnitude, direction)
    thresholded = double_threshold(suppressed, Tl, Th)
    final_edges = hysteresis(thresholded)
    final_edges = final_edges.astype(np.uint8)
    result = Image.fromarray(final_edges)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(image.convert("RGB"))
    ax[0].set_title("Imagem Original")
    ax[0].axis("off")

    ax[1].imshow(result, cmap="gray")
    ax[1].set_title("Canny")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

    return result
