from flask import Flask, request, render_template_string, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# MODELO
modelo = tf.saved_model.load("modelo/model.savedmodel")
assinatura = modelo.signatures["serving_default"]

with open("modelo/labels.txt", "r", encoding="utf-8") as arquivo:
    classes = [linha.strip() for linha in arquivo.readlines()]


def analisar(imagem):
    imagem = imagem.convert("RGB")
    imagem = imagem.resize((224, 224))

    imagem = np.array(imagem).astype(np.float32) / 255.0
    imagem = np.expand_dims(imagem, axis=0)

    resultado = assinatura(tf.constant(imagem))
    previsoes = list(resultado.values())[0].numpy()[0]

    indice = np.argmax(previsoes)

    return classes[indice], previsoes[indice] * 100


HTML = """
<!DOCTYPE html>
<html lang="pt-BR">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>LaranjAI 🍊</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #111;
    color: white;
    text-align: center;
}

header {
    background: #ff7a00;
    padding: 25px;
    font-size: 30px;
    font-weight: bold;
}

.container {
    width: 90%;
    max-width: 700px;
    margin: 30px auto;
    padding: 25px;
    background: #1d1d1d;
    border-radius: 20px;
}

button {
    background: #ff7a00;
    color: white;
    border: none;
    padding: 13px 22px;
    margin: 8px;
    border-radius: 10px;
    font-size: 16px;
    cursor: pointer;
}
label.botao {
    display: inline-block;
    background: #ff7a00;
    color: white;
    border: none;
    padding: 13px 22px;
    margin: 8px;
    border-radius: 10px;
    font-size: 16px;
    cursor: pointer;
}

label.botao:hover {
    background: #ff912e;
}
button:hover {
    background: #ff912e;
}

input {
    margin: 15px;
    max-width: 100%;
}

.camera-container {
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: 20px auto;
    overflow: hidden;
    border-radius: 15px;
}

#video {
    display: none;
    width: 100%;
    border-radius: 15px;
}

#enquadramento {
    display: none;
    position: absolute;
    left: 50%;
    top: 50%;
    width: 75%;
    height: 55%;
    transform: translate(-50%, -50%);
    border: 4px solid white;
    border-radius: 8px;
    pointer-events: none;
}

#preview {
    display: none;
    max-width: 100%;
    max-height: 400px;
    margin: 15px auto;
    border-radius: 15px;
}

#canvas {
    display: none;
}

.resultado {
    margin-top: 25px;
    padding: 22px;
    background: #292929;
    border-radius: 15px;
    font-size: 21px;
}

.status {
    margin-top: 10px;
    color: #ccc;
    font-size: 14px;
}

hr {
    margin: 30px 0;
    border: none;
    border-top: 1px solid #444;
}

</style>
</head>

<body>

<header>
🍊 LaranjAI
</header>

<div class="container">

<h2>Identificação de doenças em laranjas</h2>

<p>Aponte a laranja para a câmera ou envie uma imagem.</p>

<h3>📷 Webcam</h3>

<button onclick="abrirCamera()">Abrir webcam</button>

<button onclick="fecharCamera()">❌ Fechar webcam</button>

<div class="camera-container">

<video id="video" autoplay playsinline></video>

<div id="enquadramento"></div>

</div>

<canvas id="canvas"></canvas>

<div class="status" id="status">
Webcam desligada
</div>

<div class="resultado">

<strong>Resultado:</strong>

<br>

<span id="resultadoCamera">Aguardando...</span>

<br><br>

<strong>Confiança:</strong>

<br>

<span id="confiancaCamera">--</span>

</div>

<hr>

<h3>🖼️ Analisar uma imagem</h3>

<form method="POST" action="/analisar_imagem" enctype="multipart/form-data">


   <label for="arquivoImagem" class="botao">
    📁 Escolher imagem
</label>

<input
    type="file"
    id="arquivoImagem"
    name="imagem"
    accept="image/*"
    onchange="mostrarImagem(event)"
    required
    style="display: none;"
>

<button type="button" onclick="removerImagem()">
    ❌ Remover imagem
</button>

<br>

<img id="preview">

<br>

<button type="submit">
🔍 Analisar imagem
</button>

</form>

{% if resultado_imagem %}

<div class="resultado" id="resultadoImagem">

<strong>Resultado:</strong>

<br>

{{ resultado_imagem }}

<br><br>

<strong>Confiança:</strong>

<br>

{{ confianca_imagem }}%

</div>

{% endif %}

</div>

<script>

let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let enquadramento = document.getElementById("enquadramento");
let statusTexto = document.getElementById("status");
let resultadoTexto = document.getElementById("resultadoCamera");
let confiancaTexto = document.getElementById("confiancaCamera");

let cameraLigada = false;
let analisando = false;


async function abrirCamera() {

    if (cameraLigada) {
        return;
    }

    try {

        let stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            },
            audio: false
        });

        video.srcObject = stream;

        video.style.display = "block";
        enquadramento.style.display = "block";

        cameraLigada = true;

        statusTexto.innerText =
            "Webcam ligada — coloque a laranja dentro do enquadramento.";

        video.onloadedmetadata = function() {
            analisarAutomaticamente();
        };

    } catch (erro) {

        console.error(erro);

        statusTexto.innerText =
            "Não foi possível acessar a webcam.";

        alert("Permita o acesso à câmera no navegador.");
    }
}

function fecharCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    video.style.display = "none";
    enquadramento.style.display = "none";

    cameraLigada = false;
    analisando = false;

    statusTexto.innerText = "Webcam desligada";
    resultadoTexto.innerText = "Aguardando...";
    confiancaTexto.innerText = "--";
}
async function analisarAutomaticamente() {

    if (!cameraLigada) {
        return;
    }

    if (video.videoWidth === 0 || video.videoHeight === 0) {

        setTimeout(analisarAutomaticamente, 1000);

        return;
    }

    if (!analisando) {

        analisando = true;

        try {

            let largura = video.videoWidth;
            let altura = video.videoHeight;

            let larguraCorte = largura * 0.75;
            let alturaCorte = altura * 0.55;

            let x = (largura - larguraCorte) / 2;
            let y = (altura - alturaCorte) / 2;

            canvas.width = larguraCorte;
            canvas.height = alturaCorte;

            let contexto = canvas.getContext("2d");

            contexto.drawImage(
                video,
                x,
                y,
                larguraCorte,
                alturaCorte,
                0,
                0,
                larguraCorte,
                alturaCorte
            );

            let imagem = canvas.toDataURL(
                "image/jpeg",
                0.85
            );

            let resposta = await fetch(
                "/analisar_camera",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        imagem: imagem
                    })
                }
            );

            let dados = await resposta.json();

            if (dados.erro) {

                statusTexto.innerText = dados.erro;

            } else {

                resultadoTexto.innerText =
                    dados.resultado;

                confiancaTexto.innerText =
                    dados.confianca + "%";

                statusTexto.innerText =
                    "Analisando automaticamente...";
            }

        } catch (erro) {

            console.error(erro);

        } finally {

            analisando = false;
        }
    }

    setTimeout(analisarAutomaticamente, 1000);
}


function mostrarImagem(event) {

    let arquivo = event.target.files[0];

    if (!arquivo) {
        return;
    }

    let preview = document.getElementById("preview");

    preview.src = URL.createObjectURL(arquivo);
    preview.style.display = "block";
}

function removerImagem() {
    document.getElementById("arquivoImagem").value = "";
    document.getElementById("preview").src = "";
    document.getElementById("preview").style.display = "none";

    let resultado = document.getElementById("resultadoImagem");

    if (resultado) {
        resultado.style.display = "none";
    }
}

</script>

</body>
</html>
"""


@app.route("/", methods=["GET"])
def inicio():

    return render_template_string(
        HTML,
        resultado_imagem=None,
        confianca_imagem=None
    )


@app.route("/analisar_imagem", methods=["POST"])
def analisar_imagem():

    try:
        arquivo = request.files.get("imagem")

        if not arquivo:
            return render_template_string(
                HTML,
                resultado_imagem="Nenhuma imagem foi enviada.",
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
        print("ERRO AO ANALISAR IMAGEM:", erro)

        return render_template_string(
            HTML,
            resultado_imagem="Erro ao analisar a imagem.",
            confianca_imagem=None
        )
def analisar_imagem():

    resultado = None
    confianca = None

    arquivo = request.files.get("imagem")

    if arquivo:

        imagem = Image.open(
            io.BytesIO(arquivo.read())
        )

        resultado, confianca = analisar(imagem)

    return render_template_string(
        HTML,
        resultado_imagem=resultado,
        confianca_imagem=(
            f"{confianca:.2f}"
            if confianca is not None
            else None
        )
    )


@app.route("/analisar_camera", methods=["POST"])
def analisar_camera():

    try:

        dados = request.json["imagem"]

        dados = dados.split(",")[1]

        imagem = Image.open(
            io.BytesIO(
                base64.b64decode(dados)
            )
        )

        resultado, confianca = analisar(imagem)

        return jsonify({
            "resultado": resultado,
            "confianca": f"{confianca:.2f}"
        })

    except Exception as erro:

        print(erro)

        return jsonify({
            "erro": "Erro ao analisar a imagem."
        })


if __name__ == "__main__":
    app.run(debug=True)
