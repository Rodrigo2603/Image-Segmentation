import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Recebe a imagem PIL
def binarize_image(image, threshold=128):
    img_gray = image.convert('L')
    img_array = np.array(img_gray)
    binary_array = np.where(img_array > threshold, 255, 0)
    return binary_array.astype(np.uint8)

def freeman_chain_code(binary_image):
    def find_start_point(image):
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                if image[row, col] == 255:
                    return row, col
        return None

    # Direção do vizinho
    def get_neighbor(row, col, direction):
        directions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        dr, dc = directions[direction]
        return row + dr, col + dc

    # Encontra o próximo vizinho
    def find_next_boundary_pixel(image, current_row, current_col, previous_direction):
        search_order = [(previous_direction + i) % 8 for i in range(1, 9)]
        for direction in search_order:
            new_row, new_col = get_neighbor(current_row, current_col, direction)
            if 0 <= new_row < image.shape[0] and 0 <= new_col < image.shape[1]:
                if image[new_row, new_col] == 255:
                    return new_row, new_col, direction
        return None

    # Procura por objeto na imagem
    start_point = find_start_point(binary_image)
    if start_point is None:
        print("Não foi encontrado nenhum objeto na imagem.")
        return None, None

    chain_code = ""
    current_row, current_col = start_point
    previous_direction = 7
    first_pixel = True
    visited_pixels = set()
    boundary_pixels = [(current_row, current_col)]

    # Busca próximo vizinho até alcançar o primeiro vizinho
    while True:
        if first_pixel:
            next_pixel_info = find_next_boundary_pixel(binary_image, current_row, current_col, previous_direction)
            first_pixel = False
        else:
            next_pixel_info = find_next_boundary_pixel(binary_image, current_row, current_col, (previous_direction + 4) % 8)

        if next_pixel_info is None:
            break

        next_row, next_col, direction = next_pixel_info

        if (next_row, next_col) in visited_pixels:
            if (next_row, next_col) == start_point:
                break
            else:
                print("Loop detectado. Encerrando a iteração.")
                break

        visited_pixels.add((next_row, next_col))
        boundary_pixels.append((next_row, next_col))
        chain_code += str(direction)
        current_row, current_col = next_row, next_col
        previous_direction = direction

    return chain_code, boundary_pixels

def executar_freeman_chain_code(original_image):
    binary_image = binarize_image(original_image)

    if binary_image is not None:
        chain_code, boundary_pixels = freeman_chain_code(binary_image)
        if chain_code:
            print("Código de Freeman:", chain_code)

            original_image = original_image.convert('RGB')

            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Imagem Original")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(binary_image, cmap='gray')
            if boundary_pixels:
                rows, cols = zip(*boundary_pixels)
                plt.plot(cols, rows, 'b-', linewidth=2)
            plt.title("Fronteira Conectada")
            plt.axis('off')
            plt.show()

            return chain_code
        else:
            print("Não foi possível gerar o código da cadeia.")
            return None
    else:
        print("Erro na binarização da imagem.")
        return None
