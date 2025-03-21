import matplotlib.pyplot as plt
from PIL import Image

# Converte a imagem PIL para matriz de escala de cinza
def pil_to_grayscale_matrix(img):
    img = img.convert("L")
    width, height = img.size
    pixels = list(img.getdata())
    grayscale = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return grayscale, width, height

# Aplica o filtro da media
def apply_box_filter(image, width, height, kernel_size):
    if kernel_size == 2:
        return apply_box_filter_2x2(image, width, height)
    
    offset = kernel_size // 2
    new_image = [[0] * width for _ in range(height)]

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            total = 0
            count = 0
            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    total += image[y + ky][x + kx]
                    count += 1
            new_image[y][x] = total // count
    return new_image

# Caso especial para kernel 2x2
def apply_box_filter_2x2(image, width, height):
    new_image = [[0] * width for _ in range(height)]
    for y in range(0, height - 1, 2):
        for x in range(0, width - 1, 2):
            total = image[y][x] + image[y][x + 1] + image[y + 1][x] + image[y + 1][x + 1]
            avg = total // 4
            new_image[y][x] = avg
            new_image[y][x + 1] = avg
            new_image[y + 1][x] = avg
            new_image[y + 1][x + 1] = avg
    return new_image

# Converte a matriz de volta para PIL
def matrix_to_pil(image_matrix):
    img = Image.new("L", (len(image_matrix[0]), len(image_matrix)))
    flat_pixels = [pixel for row in image_matrix for pixel in row]
    img.putdata(flat_pixels)
    return img

# Função para plotar a imagem original e filtradas
def plot_images(original_pil, filtered_pils):
    fig, axs = plt.subplots(1, len(filtered_pils) + 1, figsize=(15, 5))
    
    axs[0].imshow(original_pil.convert('L'), cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    sizes = [2, 3, 5, 7]
    for i, (filtered_img, size) in enumerate(zip(filtered_pils, sizes)):
        axs[i + 1].imshow(filtered_img, cmap='gray')
        axs[i + 1].set_title(f"Filtro {size}x{size}")
        axs[i + 1].axis('off')

    plt.show()

# Função principal
def box_filter(pil_image):
    grayscale, width, height = pil_to_grayscale_matrix(pil_image)

    resultados = []
    for size in [2, 3, 5, 7]:
        filtrada = apply_box_filter(grayscale, width, height, size)
        resultados.append(matrix_to_pil(filtrada))

    plot_images(pil_image, resultados)

    return resultados
