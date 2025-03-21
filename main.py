import os
from PIL import Image
import box 
import cadeia_freeman 
import canny 
import contar_objetos 
import intensidade
import marr_hildreth 
import otsu 
import watershed

nomes = [
    "Filtro Box ",
    "Cadeia de Freeman",
    "Canny",
    "Contar Objetos",
    "Modificar Intensidade",
    "Marr-Hildreth",
    "Otsu",
    "Watershed"
]

def listar_imagens(diretorio):
    imagens = [f for f in os.listdir(diretorio) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    return imagens

def exibir_menu_algoritmos(nomes_algoritmos):
    print("\nEscolha o algoritmo a ser aplicado:")
    for i, nome in enumerate(nomes_algoritmos, start=1):
        print(f"{i} - {nome}")
    while True:
        try:
            escolha = int(input("Digite o número do algoritmo: "))
            if 1 <= escolha <= len(nomes_algoritmos):
                return escolha
            else:
                print("Escolha inválida. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite um número.")

def main():
    imagens_dir = './imagens'
    resultados_dir = './resultados'
    os.makedirs(resultados_dir, exist_ok=True)

    imagens = listar_imagens(imagens_dir)

    if not imagens:
        print("Nenhuma imagem encontrada na pasta ./imagens")
        return

    print("Imagens disponíveis:")
    for idx, nome_img in enumerate(imagens, start=1):
        print(f"{idx} - {nome_img}")

    img_idx = int(input("Escolha a imagem pelo número: ")) - 1
    if img_idx < 0 or img_idx >= len(imagens):
        print("Escolha inválida!")
        return

    imagem_escolhida = imagens[img_idx]
    print(imagem_escolhida)
    caminho_imagem = os.path.join(imagens_dir, imagem_escolhida)
    imagem = Image.open(caminho_imagem)

    algoritmo = exibir_menu_algoritmos(nomes)

    if algoritmo == 1:
        resultados = box.box_filter(imagem)
        for i, img in enumerate(resultados, start=1):
            img.save(f"{resultados_dir}/box_filter_{i}.jpg")
    elif algoritmo == 2:
        cadeia_freeman.executar_freeman_chain_code(imagem)
    elif algoritmo == 3:
        resultado = canny.canny_edge_detection(imagem)
        resultado.save(f"{resultados_dir}/canny.jpg")
    elif algoritmo == 4: 
        if imagem_escolhida in ('0.jpg', '1.jpg', '4.jpg', '6.jpg', '7.jpg'):
            resultado = contar_objetos.detectar_todos_objetos(imagem, 195)
        elif imagem_escolhida in ('3.png', '5.jpg'):
            resultado = contar_objetos.detectar_todos_objetos(imagem, 115)
        else:
            resultado = contar_objetos.detectar_todos_objetos(imagem, 105)
    elif algoritmo == 5:
        resultado = intensidade.segment_image(imagem)
        resultado.save(f"{resultados_dir}/intensidade_modificada.jpg")
    elif algoritmo == 6:
        resultado = marr_hildreth.marr_hildreth_edge_detection(imagem)
        resultado.save(f"{resultados_dir}/marr_hildreth.jpg")
    elif algoritmo == 7:
        resultado = otsu.otsu_segmentation(imagem)
        resultado.save(f"{resultados_dir}/otsu.jpg")
    elif algoritmo == 8:
        resultado = watershed.plotar(imagem, 100)
    else:
        print("Algoritmo inválido!")
        return

if __name__ == "__main__":
    main()
