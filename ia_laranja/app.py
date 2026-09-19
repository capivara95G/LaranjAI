from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64

app = Flask(__name__)

# ==========================
# CARREGAR MODELO
# ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "modelo",
    "model.savedmodel"
)

LABELS_PATH = os.path.join(
    BASE_DIR,
    "modelo",
    "labels.txt"
)

print("Carregando modelo...")

modelo = tf.saved_model.load(MODEL_PATH)
assinatura = modelo.signatures["serving_default"]

print("Modelo carregado!")

with open(LABELS_PATH, "r", encoding="utf-8") as arquivo:
    classes = []

    for linha in arquivo:
        linha = linha.strip()

        if " " in linha:
            classes.append(linha.split(" ", 1)[1])
        else:
            classes.append(linha)

# ==========================
# ANÁLISE
# ==========================

def analisar(imagem):

    imagem = imagem.convert("RGB")
    imagem = imagem.resize((224, 224))

    imagem = np.array(imagem).astype(np.float32)
    imagem = imagem / 255.0

    imagem = np.expand_dims(imagem, axis=0)

    resultado = assinatura(tf.constant(imagem))

    previsoes = list(resultado.values())[0].numpy()[0]

    indice = np.argmax(previsoes)

    return (
        classes[indice],
        float(previsoes[indice] * 100)
    )

# ==========================
# HTML
# ==========================

HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>LaranjAI 🍊</title>

<style>

body{
background:#111;
color:white;
font-family:Arial;
text-align:center;
padding:20px;
}

.container{
max-width:700px;
margin:auto;
background:#222;
padding:30px;
border-radius:15px;
}

button{
background:#ff7a00;
color:white;
border:none;
padding:12px 20px;
border-radius:8px;
cursor:pointer;
}

button:hover{
background:#ff8e26;
}

.resultado{
margin-top:20px;
padding:20px;
background:#333;
border-radius:10px;
}

img{
max-width:100%;
border-radius:10px;
margin-top:15px;
}

</style>

</head>
<body>

<div class="container">

<h1>🍊 LaranjAI</h1>

<p>Escolha uma imagem para análise.</p>

/analisar_imagem

<input type="file" name="imagem" accept="image/*" required>

<br><br>

<button type="submit">
Analisar Imagem
</button>

</form>

{% if resultado_imagem %}

<div class="resultado">

<h3>Resultado</h3>

<p>{{ resultado_imagem }}</p>

<p>Confiança: {{ confianca_imagem }}%</p>

</div>

{% endif %}

</div>

</body>
</html>
"""

# ==========================
# HOME
# ==========================

@app.route("/")
def inicio():

    return render_template_string(
        HTML,
        resultado_imagem=None,
        confianca_imagem=None
    )

# ==========================
# ANALISAR IMAGEM
# ==========================

@app.route("/analisar_imagem", methods=["POST"])
def analisar_imagem():

    try:

        arquivo = request.files.get("imagem")

        if not arquivo:

            return render_template_string(
                HTML,
                resultado_imagem="Nenhuma imagem enviada.",
                confianca_imagem=None
            )

        imagem = Image.open(
            io.BytesIO(arquivo.read())
        )

        resultado, confianca = analisar(imagem)

        return render_template_string(
            HTML,
            resultado_imagem=resultado,
            confianca_imagem=f"{confianca:.2f}"
        )

    except Exception as erro:

        print(f"ERRO: {erro}")

        return render_template_string(
            HTML,
            resultado_imagem="Erro ao processar imagem.",
            confianca_imagem=None
        )

# ==========================
# API CAMERA
# ==========================

@app.route("/analisar_camera", methods=["POST"])
def analisar_camera():

    try:

        dados = request.json

        if not dados:
            return jsonify({
                "erro": "Dados inválidos."
            })

        imagem_base64 = dados["imagem"]

        imagem_base64 = imagem_base64.split(",")[1]

        bytes_imagem = base64.b64decode(
            imagem_base64
        )

        imagem = Image.open(
            io.BytesIO(bytes_imagem)
        )

        resultado, confianca = analisar(imagem)

        return jsonify({
            "resultado": resultado,
            "confianca": round(confianca, 2)
        })

    except Exception as erro:

        print(f"ERRO CAMERA: {erro}")

        return jsonify({
            "erro": "Erro ao analisar imagem."
        })

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
