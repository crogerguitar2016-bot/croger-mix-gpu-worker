import base64
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import runpod
import torch


def baixar_audio(url, destino):

    with requests.get(
        url,
        stream=True,
        timeout=120
    ) as resposta:

        resposta.raise_for_status()

        with open(destino, "wb") as arq:

            for bloco in resposta.iter_content(
                chunk_size=1024 * 1024
            ):

                if bloco:
                    arq.write(bloco)


def salvar_base64(valor, destino):

    dados = base64.b64decode(
        valor
    )

    Path(destino).write_bytes(
        dados
    )


def sha256_arquivo(caminho):

    sha = hashlib.sha256()

    with open(caminho, "rb") as arq:

        while True:

            bloco = arq.read(
                1024 * 1024
            )

            if not bloco:
                break

            sha.update(bloco)

    return sha.hexdigest()


def encontrar_stems(saida):

    nomes = {
        "voz": "vocals.mp3",
        "bateria": "drums.mp3",
        "baixo": "bass.mp3",
        "outros": "other.mp3",
    }

    resultado = {}

    for chave, nome in nomes.items():

        encontrados = list(
            Path(saida).rglob(nome)
        )

        if not encontrados:
            raise RuntimeError(
                f"Stem não encontrado: {nome}"
            )

        resultado[chave] = (
            encontrados[0]
        )

    return resultado


def handler(event):

    inicio = time.time()

    entrada = (
        event.get("input")
        or {}
    )

    modo = str(
        entrada.get(
            "modo",
            "rapido"
        )
    ).strip().lower()

    if modo not in (
        "rapido",
        "qualidade",
    ):
        modo = "rapido"

    audio_url = entrada.get(
        "audio_url"
    )

    audio_base64 = entrada.get(
        "audio_base64"
    )

    filename = str(
        entrada.get(
            "filename",
            "audio.mp3"
        )
    )

    if not audio_url and not audio_base64:

        return {
            "error": (
                "Informe audio_url "
                "ou audio_base64."
            )
        }

    if not torch.cuda.is_available():

        return {
            "error": (
                "CUDA não disponível "
                "neste worker."
            )
        }

    trabalho = Path(
        tempfile.mkdtemp(
            prefix="croger_gpu_"
        )
    )

    try:

        entrada_path = (
            trabalho
            /
            Path(filename).name
        )

        if audio_url:

            baixar_audio(
                audio_url,
                entrada_path
            )

        else:

            salvar_base64(
                audio_base64,
                entrada_path
            )

        sha256 = sha256_arquivo(
            entrada_path
        )

        saida = (
            trabalho
            /
            "saida"
        )

        saida.mkdir(
            parents=True,
            exist_ok=True
        )

        comando = [
            "python",
            "-m",
            "demucs",
            "-n",
            "htdemucs",
            "--device",
            "cuda",
        ]

        if modo == "rapido":

            comando.extend(
                [
                    "--overlap",
                    "0.10",
                ]
            )

        comando.extend(
            [
                "--mp3",
                "--mp3-bitrate",
                "192",
                "--out",
                str(saida),
                str(entrada_path),
            ]
        )

        processo = subprocess.run(
            comando,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if processo.returncode != 0:

            return {
                "error": (
                    "Demucs falhou."
                ),
                "returncode": (
                    processo.returncode
                ),
                "log": (
                    processo.stdout[-8000:]
                )
            }

        stems = encontrar_stems(
            saida
        )

        resposta_stems = {}

        for chave, caminho in stems.items():

            dados = caminho.read_bytes()

            resposta_stems[chave] = {
                "filename": (
                    caminho.name
                ),
                "bytes": (
                    len(dados)
                ),
                "base64": (
                    base64.b64encode(
                        dados
                    )
                    .decode("ascii")
                )
            }

        duracao = (
            time.time()
            -
            inicio
        )

        return {
            "ok": True,
            "backend": "gpu",
            "modo": modo,
            "sha256": sha256,
            "gpu": (
                torch.cuda.get_device_name(
                    0
                )
            ),
            "torch": (
                torch.__version__
            ),
            "cuda": (
                torch.version.cuda
            ),
            "tempo_segundos": (
                round(
                    duracao,
                    2
                )
            ),
            "stems": (
                resposta_stems
            )
        }

    except Exception as exc:

        return {
            "error": str(exc)
        }

    finally:

        shutil.rmtree(
            trabalho,
            ignore_errors=True
        )


if __name__ == "__main__":

    runpod.serverless.start(
        {
            "handler": handler
        }
    )
