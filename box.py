import matplotlib.pyplot as plt
from PIL import Image

def all_box(imagem):
    
    # Retorna a imagem em grayscale, sua altura e sua largura
    def read_image(filename):
        img = Image.open(filename).convert("L")
        width, height = img.size
        pixels = list(img.getdata())
        grayscale = [pixels[i * width:(i + 1) * width] for i in range(height)]
        return grayscale, width, height

    # Lógica para movimentação do kernel pela imagem realizando o processo de convolução 
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
    
    # Caso o kernel seja 2x2 ele não possui um ponto central, tendo um tratamento diferente
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
    
    # Cria uma imagem vazia em escala de cinza e preenche com os valores da imagem filtrada 
    def save_jpg(filename, image_matrix):
        img = Image.new("L", (len(image_matrix[0]), len(image_matrix)))
        flat_pixels = [pixel for row in image_matrix for pixel in row]
        img.putdata(flat_pixels)
        img.save(filename, "JPEG")

    # Plota todas os processamentos, um ao lado do outro
    def plot_images(original, filtered_2x2, filtered_3x3, filtered_5x5, filtered_7x7):
        fig, axs = plt.subplots(1, 5, figsize=(15, 5))
        axs[0].imshow(original, cmap='gray')
        axs[0].set_title("Grayscale")

        axs[1].imshow(filtered_2x2, cmap='gray')
        axs[1].set_title("Filtro 2x2")

        axs[2].imshow(filtered_3x3, cmap='gray')
        axs[2].set_title("Filtro 3x3")

        axs[3].imshow(filtered_5x5, cmap='gray')
        axs[3].set_title("Filtro 5x5")

        axs[4].imshow(filtered_7x7, cmap='gray')
        axs[4].set_title("Filtro 7x7")

        for ax in axs:
            ax.axis("off")

        plt.show()
    
    # Carrega a imagem
    image, width, height = read_image(f"./imagens/{imagem}")
    
    # Aplica filtros para todos os casos
    filtered_2x2 = apply_box_filter(image, width, height, 2)
    filtered_3x3 = apply_box_filter(image, width, height, 3)
    filtered_5x5 = apply_box_filter(image, width, height, 5)
    filtered_7x7 = apply_box_filter(image, width, height, 7)
    
    # Salva as imagens processadas
    save_jpg("./resultados/filtered_2x2.jpg", filtered_2x2)
    save_jpg("./resultados/filtered_3x3.jpg", filtered_3x3)
    save_jpg("./resultados/filtered_5x5.jpg", filtered_5x5)
    save_jpg("./resultados/filtered_7x7.jpg", filtered_7x7)
    
    # Exibe as imagens
    plot_images(image, filtered_2x2, filtered_3x3, filtered_5x5, filtered_7x7)
