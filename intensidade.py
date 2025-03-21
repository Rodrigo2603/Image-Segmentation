from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def segment_image(image):
    img = image.convert("L")
    img_array = np.array(img)

    # Aplicar segmentação de intensidade
    segmented_array = np.select(
        [
            (img_array >= 0) & (img_array <= 50),
            (img_array >= 51) & (img_array <= 100),
            (img_array >= 101) & (img_array <= 150),
            (img_array >= 151) & (img_array <= 200),
            (img_array >= 201) & (img_array <= 255),
        ],
        [25, 75, 125, 175, 255]
    )

    # Converte o array segmentado de volta para PIL Image
    segmented_img = Image.fromarray(segmented_array.astype(np.uint8))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Imagem Grayscale")
    axes[0].axis("off")

    axes[1].imshow(segmented_img, cmap="gray")
    axes[1].set_title("Imagem Segmentada")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    return segmented_img
