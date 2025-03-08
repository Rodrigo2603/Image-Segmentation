from PIL import Image
import matplotlib.pyplot as plt

def all_otsu(imagem):
    def otsu_threshold(image):
        gray_image = image.convert("L")
        pixels = list(gray_image.getdata())
        width, height = gray_image.size
        total_pixels = width * height
        
        # Construir o histograma da imagem
        hist = [0] * 256
        for pixel in pixels:
            hist[pixel] += 1
        
        # Inicializando variáveis
        max_variance = 0
        best_threshold = 0
        sum_total = sum(i * hist[i] for i in range(256))
        sum_background = 0
        weight_background = 0
        
        # Itera sobre todos os possíveis valores de limiar no intervalo de 0 a 255
        for i in range(256):
            
            # Atualiza o peso da classe do fundo somando a frequência do histograma para o valor atual
            weight_background += hist[i]
            if weight_background == 0:
                continue
            
            # Calcula o peso da classe do primeiro plano (total de pixels menos os de fundo)
            weight_foreground = total_pixels - weight_background
            
            if weight_foreground == 0:
                break
            
            # Atualizamos a soma dos pixels do fundo
            sum_background += i * hist[i]
            
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            # Variância entre classes (separação ideal entre os grupos)
            variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
            
            # Se a variância entre classes for maior que a máxima encontrada, novo limiar ótimo
            if variance_between > max_variance:
                max_variance = variance_between
                best_threshold = i

        return best_threshold

    def segment_image(image, threshold):
        gray_image = image.convert("L")
        pixels = list(gray_image.getdata())
        
        # Aplicar a segmentação usando o limiar
        segmented_pixels = [255 if pixel > threshold else 0 for pixel in pixels]
        
        # Criar nova imagem segmentada
        segmented_image = Image.new("L", gray_image.size)
        segmented_image.putdata(segmented_pixels)
        
        return segmented_image

    # Carregar imagem e calcular limiar de Otsu
    image_path = f"./imagens/{imagem}"
    image = Image.open(image_path)
    threshold = otsu_threshold(image)
    print("Limiar ótimo de Otsu:", threshold)

    # Segmentar a imagem e salvar o resultado
    segmented_image = segment_image(image, threshold)
    segmented_image.save("./resultados/otsu.jpg")
    
    # Exibir imagens lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.convert("L"), cmap="gray")
    axes[0].set_title("Imagem Grayscale")
    axes[0].axis("off")

    axes[1].imshow(segmented_image, cmap="gray")
    axes[1].set_title("Segmentação Otsu")
    axes[1].axis("off")

    plt.show()
