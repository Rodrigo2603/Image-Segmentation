import cv2
import numpy as np

def all_contar_objetos(imagem):
    def contar_objetos(imagem_path):
        # Carregar a imagem
        imagem = cv2.imread(imagem_path)

        # Converter para HSV e extrair o canal de saturação (melhora a separação de cores)
        imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
        saturacao = imagem_hsv[:, :, 1]  

        # Aplicar suavização para reduzir ruído
        saturacao = cv2.GaussianBlur(saturacao, (5,5), 0)

        # Aplicar segmentação com Otsu
        _, binarizada = cv2.threshold(saturacao, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remover pequenos ruídos
        kernel = np.ones((3, 3), np.uint8)
        binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_OPEN, kernel, iterations=2)

        # Aplicar K-Means para agrupar regiões semelhantes
        Z = imagem.reshape((-1,3))
        Z = np.float32(Z)
        K = 3  # Número de clusters ajustável
        _, labels, centers = cv2.kmeans(Z, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        clustered = labels.reshape((imagem.shape[0], imagem.shape[1]))

        # Encontrar contornos
        contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar objetos muito pequenos
        objetos_filtrados = [c for c in contornos if cv2.contourArea(c) > 500]  

        return len(objetos_filtrados)

    # Exemplo de uso
    imagem_path = f"./imagens/{imagem}"  # Substituir pelo caminho da imagem
    quantidade = contar_objetos(imagem_path)
    print(f"Quantidade de objetos detectados: {quantidade}")
