import heapq
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def threshold_manual(image_array, threshold):
    binary = np.where(image_array > threshold, 255, 0).astype(np.uint8)
    return binary

def distance_transform(binary):
    dist = np.full(binary.shape, np.inf)
    queue = []

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 255:
                dist[y, x] = 0
                heapq.heappush(queue, (0, y, x))

    while queue:
        d, y, x = heapq.heappop(queue)
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                new_d = d + 1
                if new_d < dist[ny, nx]:
                    dist[ny, nx] = new_d
                    heapq.heappush(queue, (new_d, ny, nx))
    return dist

def connected_components(binary):
    labels = np.zeros(binary.shape, dtype=np.int32)
    label_id = 1

    def flood_fill(y, x):
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            if labels[cy, cx] == 0 and binary[cy, cx] == 255:
                labels[cy, cx] = label_id
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                        stack.append((ny, nx))

    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 255 and labels[y, x] == 0:
                flood_fill(y, x)
                label_id += 1

    return labels

def watershed_segmentation(image, threshold):
    # Converte a PIL Image para numpy array
    image_array = np.array(image.convert("L"), dtype=np.uint8)

    binary = threshold_manual(image_array, threshold)
    markers = connected_components(binary)

    segmented = np.zeros_like(image_array, dtype=np.uint8)
    for label in np.unique(markers):
        if label == 0:
            continue
        segmented[markers == label] = 50 + (label * 50) % 200

    # Retorna imagem segmentada como PIL
    return Image.fromarray(segmented)

def plotar(image, threshold):

    image_array = np.array(image.convert("L"), dtype=np.uint8)
    segmented = watershed_segmentation(image, threshold)

    # Exibe resultados
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_array, cmap="gray")
    axes[0].set_title("Imagem Grayscale")
    axes[0].axis("off")

    axes[1].imshow(segmented, cmap="gray")
    axes[1].set_title("Watershed")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


