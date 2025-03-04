import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def all_cadeia_freeman(imagem):

    # Ler e converter para grayscale
    def binarize_image(image_path, threshold=128):
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        binary_array = np.where(img_array > threshold, 255, 0)
        return binary_array.astype(np.uint8)

    def freeman_chain_code(binary_image):

        # Encontra o pixel de fronteira mais na esquerda e pra cima
        def find_start_point(image):
            for row in range(image.shape[0]):
                for col in range(image.shape[1]):
                    if image[row, col] == 255:
                        return row, col
            return None

        # Retorna a direção do vizinho
        def get_neighbor(row, col, direction):
            if direction == 0:  # Direita
                return row, col + 1
            elif direction == 1:  # Diagonal direita-baixo
                return row + 1, col + 1
            elif direction == 2:  # Baixo
                return row + 1, col
            elif direction == 3:  # Diagonal esquerda-baixo
                return row + 1, col - 1
            elif direction == 4:  # Esquerda
                return row, col - 1
            elif direction == 5:  # Diagonal esquerda-cima
                return row - 1, col - 1
            elif direction == 6:  # Cima
                return row - 1, col
            elif direction == 7:  # Diagonal direita-cima
                return row - 1, col + 1
            else:
                return None

        # Encontra o próximo vizinha na fronteira
        def find_next_boundary_pixel(image, current_row, current_col, previous_direction):
            search_order = [(previous_direction + i) % 8 for i in range(1, 9)]
            for direction in search_order:
                new_row, new_col = get_neighbor(current_row, current_col, direction)

                # Checa a fronteira
                if 0 <= new_row < image.shape[0] and 0 <= new_col < image.shape[1]:
                    if image[new_row, new_col] == 255:  # Pixel encontrado
                        return new_row, new_col, direction
            return None

        start_point = find_start_point(binary_image)
        if start_point is None:
            print("No object found in the image.")
            return None

        chain_code = ""
        current_row, current_col = start_point
        previous_direction = 7
        first_pixel = True
        visited_pixels = set() # Armazena pixels visitados
        boundary_pixels = [(current_row, current_col)] # Armazena pixels da fronteira

        while True:
            if first_pixel:
                next_pixel_info = find_next_boundary_pixel(binary_image, current_row, current_col, previous_direction)
                first_pixel = False
            else:
                next_pixel_info = find_next_boundary_pixel(binary_image, current_row, current_col, (previous_direction + 4) % 8)

            if next_pixel_info is None:
                break
            
            next_row, next_col, direction = next_pixel_info

            # Verifica se o pixel já foi visitado
            if (next_row, next_col) in visited_pixels:
                if (next_row, next_col) == start_point:
                    break  # Completa a busca
                else:
                    print("Loop detectado. Encerrando a iteração.")
                    break

            visited_pixels.add((next_row, next_col))
            boundary_pixels.append((next_row, next_col))

            chain_code += str(direction)
            current_row, current_col = next_row, next_col
            previous_direction = direction

        return chain_code, boundary_pixels

    image_path = f"./imagens/{imagem}"
    original_image = Image.open(image_path).convert('RGB')
    binary_image = binarize_image(image_path)
    
    if binary_image is not None:
        chain_code, boundary_pixels = freeman_chain_code(binary_image)
        if chain_code:
            print("Freeman Chain Code:", chain_code)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_image)
            plt.title("Imagem Grayscale")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(binary_image, cmap='gray')
            rows, cols = zip(*boundary_pixels)
            plt.plot(cols, rows, 'r-', linewidth=2)
            plt.title("Fronteira Conectada")
            plt.axis('off')
            plt.show()
        else:
            print("Could not generate chain code.")