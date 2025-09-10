# CLIP-NAO

Sistema de visão computacional integrado ao robô humanoide NAO, utilizando o modelo CLIP (Contrastive Language-Image Pre-Training) para reconhecimento de objetos em tempo real com classificação zero-shot e síntese de voz.

## Repositório

[https://github.com/vitor-souza-ime/clipnao](https://github.com/vitor-souza-ime/clipnao)

## Descrição

Este projeto permite que o robô NAO capture imagens com sua câmera, utilize o modelo CLIP para reconhecer objetos sem necessidade de treinamento adicional (zero-shot) e descreva os objetos em voz alta utilizando síntese de fala. O sistema também fornece um display interativo em tempo real para visualização da câmera e das legendas geradas.

## Funcionalidades

- Captura de imagens da câmera do robô NAO.
- Reconhecimento de objetos em tempo real com CLIP.
- Classificação zero-shot (não requer treinamento específico para novas categorias).
- Síntese de voz para descrever objetos identificados.
- Display interativo com imagens e legendas.
- Armazenamento de imagens e informações de cada iteração.

## Requisitos

- Python 3.8 ou superior
- NAOqi SDK
- PyTorch
- CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- Pillow
- matplotlib
- numpy
- qi (NAOqi Python SDK)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/vitor-souza-ime/clipnao.git
cd clipnao
````

2. Instale as dependências:

```bash
pip install torch torchvision matplotlib pillow numpy
pip install git+https://github.com/openai/CLIP.git
```

3. Certifique-se de ter o NAOqi SDK instalado e configurado.

## Uso

1. Conecte seu robô NAO à mesma rede.
2. Atualize o IP do robô no código (`connect_to_nao`).
3. Execute o script principal:

```bash
python main.py
```

4. Para interromper o sistema, pressione `Ctrl+C`.

## Estrutura do Projeto

```
clipnao/
│
├── main.py            # Script principal
├── README.md          # Este arquivo
└── requirements.txt   # Dependências (opcional)
```

## Resultados

Durante a execução, o sistema:

* Exibe as imagens capturadas pelo NAO.
* Mostra a legenda (caption) gerada pelo CLIP.
* Fala a legenda em voz alta.
* Salva imagens e arquivos de texto com informações de cada iteração em uma pasta timestampada.

## Referências

* Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). **Learning Transferable Visual Models From Natural Language Supervision**. [CLIP paper](https://arxiv.org/abs/2103.00020)

## Licença



Se você quiser, posso também criar um `requirements.txt` pronto para este projeto, com todas as bibliotecas necessárias para rodar o `main.py` no seu ambiente local. Quer que eu faça?
```
