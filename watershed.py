import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def all_watershed(imagem):
    def load_image(image_path):
        img = Image.open(image_path).convert("L")
        pixels = np.array(img, dtype=np.uint8)
        return pixels

    # Aplica binarização simples usando um limiar baseado na média
    def threshold_manual(image):
        threshold = 150
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary

    # Aplica uma transformada de distância simples
    def distance_transform(binary):
        dist = np.full(binary.shape, np.inf)
        
        # Inicializa os pontos do primeiro plano
        queue = []
        for y in range(binary.shape[0]):
            for x in range(binary.shape[1]):
                if binary[y, x] == 255:
                    dist[y, x] = 0
                    heapq.heappush(queue, (0, y, x))  # Usa uma fila de prioridade
        
        # Propagação de distância
        while queue:
            d, y, x = heapq.heappop(queue)
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:  # Movimentos 4-direções
                ny, nx = y + dy, x + dx
                if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                    new_d = d + 1
                    if new_d < dist[ny, nx]:  # Apenas atualiza se a nova distância for menor
                        dist[ny, nx] = new_d
                        heapq.heappush(queue, (new_d, ny, nx))
        
        return dist

    # Identifica componentes conectados manualmente
    def connected_components(binary):
        labels = np.zeros(binary.shape, dtype=np.int32)
        label_id = 1

        def flood_fill(y, x):
            stack = [(y, x)]
            while stack:
                cy, cx = stack.pop()
                if labels[cy, cx] == 0 and binary[cy, cx] == 255:
                    labels[cy, cx] = label_id
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4 direções
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                            stack.append((ny, nx))

        for y in range(binary.shape[0]):
            for x in range(binary.shape[1]):
                if binary[y, x] == 255 and labels[y, x] == 0:
                    flood_fill(y, x)
                    label_id += 1

        return labels

    def watershed_algorithm(image):
        binary = threshold_manual(image)
        dist = distance_transform(binary)
        markers = connected_components(binary)
        
        segmented = np.zeros_like(image, dtype=np.uint8)
        
        for label in np.unique(markers):
            if label == 0:
                continue
            segmented[markers == label] = 50 + (label * 50) % 200
        
        return image, segmented

    image_path = f"./imagens/{imagem}"
    image = load_image(image_path)
    original, segmented_image = watershed_algorithm(image)

    # Exibir as imagens lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Imagem Grayscale")
    axes[0].axis("off")

    axes[1].imshow(segmented_image, cmap="gray")
    axes[1].set_title("Watershed")
    axes[1].axis("off")

    plt.show()

    # Salvar a imagem segmentada
    img = Image.fromarray(segmented_image)
    img.save("./resultados/watershed.jpg")
