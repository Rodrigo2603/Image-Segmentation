from PIL import Image

def all_contar_objetos(imagem):
    def load_and_binarize(image_path, threshold=128):
        img = Image.open(image_path).convert("L")  # Converte para escala de cinza
        img = img.point(lambda p: 255 if p > threshold else 0)  # Binariza (0 = fundo, 255 = objeto)
        binary_matrix = list(img.getdata())
        width, height = img.size
        binary_matrix = [binary_matrix[i * width:(i + 1) * width] for i in range(height)]
        return binary_matrix

    # Separa imagem do fundo
    def flood_fill(image, x, y, new_value, old_value):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < len(image) and 0 <= cy < len(image[0]) and image[cx][cy] == old_value:
                image[cx][cy] = new_value  # Marca o pixel como visitado
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])  # Adiciona vizinhos

    # Contar a quantidade de objetos
    def count_objects(binary_image):
        rows, cols = len(binary_image), len(binary_image[0])
        object_count = 0
        visited_marker = 128  # Valor diferente de 0 (fundo) e 255 (objeto)

        for i in range(rows):
            for j in range(cols):
                if binary_image[i][j] == 255:  # Encontrou um novo objeto
                    object_count += 1
                    flood_fill(binary_image, i, j, visited_marker, 255)  # Marca todo o objeto
        
        return object_count

    image_path = f"./imagens/{imagem}"

    binary_image = load_and_binarize(image_path)
    num_objects = count_objects(binary_image)

    print(f"Número de objetos encontrados: {num_objects}")
