import os
import box
import cadeia_freeman
import canny
import contar_objetos
import intensidade
import marr_hildreth
import otsu
import watershed

def listar_imagens(diretorio="imagens"):
    imagens = [f for f in os.listdir(diretorio) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
    return imagens

def main():
    metodos = {
        "1": ("Box", box.all_box),
        "2": ("Cadeia Freeman", cadeia_freeman.all_cadeia_freeman),
        "3": ("Canny", canny.all_canny),
        "4": ("Contar Objetos", contar_objetos.all_contar_objetos),
        "5": ("Intensidade", intensidade.all_intensidade),
        "6": ("Marr-Hildreth", marr_hildreth.all_marr_hildreth),
        "7": ("Otsu", otsu.all_otsu),
        "8": ("Watershed", watershed.all_watershed),
    }

    print("Escolha um método para aplicar:")
    for key, (nome, _) in metodos.items():
        print(f"{key} - {nome}")

    escolha_metodo = input("Digite o número do método desejado: ")

    if escolha_metodo not in metodos:
        print("Opção inválida. Tente novamente.")
        return

    imagens = listar_imagens()
    if not imagens:
        print("Nenhuma imagem encontrada no diretório 'imagens/'.")
        return

    print("\nEscolha uma imagem:")
    for idx, img in enumerate(imagens, 1):
        print(f"{idx} - {img}")

    escolha_img = input("Digite o número da imagem desejada: ")

    if not escolha_img.isdigit() or int(escolha_img) not in range(1, len(imagens) + 1):
        print("Opção inválida. Tente novamente.")
        return

    imagem_escolhida = f"{imagens[int(escolha_img) - 1]}"
    funcao = metodos[escolha_metodo][1]
    funcao(imagem_escolhida) 

if __name__ == '__main__':
    main()
