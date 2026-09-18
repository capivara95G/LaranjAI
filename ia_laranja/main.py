import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
from collections import Counter
import threading
import time


# =========================
# CARREGAR MODELO
# =========================

modelo = tf.saved_model.load("modelo/model.savedmodel")
assinatura = modelo.signatures["serving_default"]

with open("modelo/labels.txt", "r", encoding="utf-8") as arquivo:
    classes = [linha.strip() for linha in arquivo.readlines()]


# =========================
# VARIÁVEIS DA WEBCAM
# =========================

camera = None
camera_ativa = False

ultimo_resultado = "Aguardando..."
ultima_confianca = 0

historico = []


# =========================
# ANÁLISE DA IMAGEM
# =========================

def analisar_array(frame):

    imagem = cv2.resize(
        frame,
        (224, 224)
    )

    imagem = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2RGB
    )

    imagem = imagem.astype(
        np.float32
    ) / 255.0

    imagem = np.expand_dims(
        imagem,
        axis=0
    )

    resultado = assinatura(
        tf.constant(imagem)
    )

    previsoes = list(
        resultado.values()
    )[0].numpy()[0]

    indice = np.argmax(
        previsoes
    )

    return (
        classes[indice],
        previsoes[indice] * 100
    )


# =========================
# ESCOLHER IMAGEM
# =========================

def escolher_imagem():

    caminho = filedialog.askopenfilename(
        title="Escolha uma imagem",
        filetypes=[
            ("Imagens", "*.jpg *.jpeg *.png")
        ]
    )

    if not caminho:
        return

    imagem = Image.open(
        caminho
    ).convert("RGB")

    imagem_original = np.array(
        imagem
    )

    frame = cv2.cvtColor(
        imagem_original,
        cv2.COLOR_RGB2BGR
    )

    classe, confianca = analisar_array(
        frame
    )

    resultado_texto.config(
        text=(
            f"Resultado: {classe}\n"
            f"Confiança: {confianca:.2f}%"
        )
    )

    imagem.thumbnail(
        (500, 400)
    )

    imagem_tk = ImageTk.PhotoImage(
        imagem
    )

    imagem_label.config(
        image=imagem_tk
    )

    imagem_label.image = imagem_tk


# =========================
# ANÁLISE DA WEBCAM
# =========================

def analisar_webcam():

    global ultimo_resultado
    global ultima_confianca
    global historico

    while camera_ativa:

        if camera is None:
            break

        sucesso, frame = camera.read()

        if not sucesso:
            continue

        # =========================
        # RETÂNGULO
        # =========================

        altura, largura = frame.shape[:2]

        largura_retangulo = 500
        altura_retangulo = 280

        x1 = (
            largura - largura_retangulo
        ) // 2

        y1 = (
            altura - altura_retangulo
        ) // 2

        x2 = x1 + largura_retangulo
        y2 = y1 + altura_retangulo

        # =========================
        # ÁREA DENTRO DO RETÂNGULO
        # =========================

        area = frame[
            y1:y2,
            x1:x2
        ]

        # =========================
        # ANALISAR
        # =========================

        classe, confianca = analisar_array(
            area
        )

        historico.append(
            classe
        )

        if len(historico) > 5:
            historico.pop(0)

        classe_estavel = Counter(
            historico
        ).most_common(1)[0][0]

        ultimo_resultado = classe_estavel
        ultima_confianca = confianca

        time.sleep(1)


# =========================
# MOSTRAR WEBCAM
# =========================

def atualizar_camera():

    if not camera_ativa or camera is None:
        return

    sucesso, frame = camera.read()

    if sucesso:

        # =========================
        # RETÂNGULO DE ENQUADRAMENTO
        # =========================

        altura, largura = frame.shape[:2]

        largura_retangulo = 500
        altura_retangulo = 280

        x1 = (
            largura - largura_retangulo
        ) // 2

        y1 = (
            altura - altura_retangulo
        ) // 2

        x2 = x1 + largura_retangulo
        y2 = y1 + altura_retangulo

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        # =========================
        # MOSTRAR CÂMERA
        # =========================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagem = Image.fromarray(
            frame_rgb
        )

        imagem.thumbnail(
            (500, 400)
        )

        imagem_tk = ImageTk.PhotoImage(
            imagem
        )

        imagem_label.config(
            image=imagem_tk
        )

        imagem_label.image = imagem_tk

        # =========================
        # RESULTADO
        # =========================

        resultado_texto.config(
            text=(
                f"Webcam: {ultimo_resultado}\n"
                f"Confiança: {ultima_confianca:.2f}%"
            )
        )

    janela.after(
        30,
        atualizar_camera
    )


# =========================
# INICIAR WEBCAM
# =========================

def iniciar_camera():

    global camera
    global camera_ativa
    global historico

    if camera_ativa:
        return

    historico = []

    camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    if not camera.isOpened():

        resultado_texto.config(
            text="Não foi possível abrir a webcam."
        )

        return

    camera_ativa = True

    thread = threading.Thread(
        target=analisar_webcam,
        daemon=True
    )

    thread.start()

    atualizar_camera()


# =========================
# FECHAR
# =========================

def fechar():

    global camera_ativa
    global camera

    camera_ativa = False

    if camera is not None:
        camera.release()

    janela.destroy()


# =========================
# INTERFACE
# =========================

janela = tk.Tk()

janela.title(
    "IA Laranja 🍊"
)

janela.geometry(
    "650x700"
)


# =========================
# TÍTULO
# =========================

titulo = tk.Label(
    janela,
    text="IA Laranja 🍊",
    font=("Arial", 24, "bold")
)

titulo.pack(
    pady=20
)


# =========================
# BOTÃO IMAGEM
# =========================

botao_imagem = tk.Button(
    janela,
    text="🖼️ Escolher imagem",
    command=escolher_imagem,
    font=("Arial", 14)
)

botao_imagem.pack(
    pady=8
)


# =========================
# BOTÃO WEBCAM
# =========================

botao_camera = tk.Button(
    janela,
    text="📷 Usar webcam",
    command=iniciar_camera,
    font=("Arial", 14)
)

botao_camera.pack(
    pady=8
)


# =========================
# ÁREA DA IMAGEM
# =========================

imagem_label = tk.Label(
    janela
)

imagem_label.pack(
    pady=20
)


# =========================
# RESULTADO
# =========================

resultado_texto = tk.Label(
    janela,
    text="Escolha uma imagem ou use a webcam",
    font=("Arial", 16)
)

resultado_texto.pack(
    pady=20
)


# =========================
# FECHAR
# =========================

janela.protocol(
    "WM_DELETE_WINDOW",
    fechar
)


janela.mainloop()