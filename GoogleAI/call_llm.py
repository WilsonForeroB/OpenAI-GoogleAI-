import json
import time
from typing import AsyncGenerator
import anyio
import google.generativeai as genai
from variables import API_KEY_GEMINI

# Configurar Gemini
genai.configure(api_key=API_KEY_GEMINI)


async def obtener_respuesta_stream_gemini(
    prompt: str,
    lista_variables: dict
) -> AsyncGenerator[str, None]:
    """
    Genera respuesta de Gemini en streaming y la devuelve en formato SSE.
    """
    tiempo_inicio = time.time()
    respuesta_completa = ""
    ultimo_chunk = None

    # Mensaje de inicio
    yield json.dumps({"type": "inicio", "content": "Comienza la generación"})

    model = genai.GenerativeModel("gemini-2.5-flash")  # <-- modelo válido según list_models()

    # Generador síncrono de Gemini
    stream = model.generate_content(
        prompt,
        stream=True,
        generation_config={
            "temperature": lista_variables.get("MODELO_TEMPERATURA", 0.7),
            "top_p": lista_variables.get("top_p", 1.0),
            "max_output_tokens": lista_variables.get("MODELO_TOKENS", 1024),
        },
    )

    # Consumimos el stream en un hilo separado para no bloquear FastAPI
    def _consume_stream():
        for chunk in stream:
            yield chunk

    async def _async_stream():
        for chunk in _consume_stream():
            yield chunk

    async for chunk in _async_stream():
        ultimo_chunk = chunk
        if chunk.text:
            respuesta_completa += chunk.text
            yield json.dumps({"type": "chunk", "content": chunk.text})

    # Mensaje de fin
    yield json.dumps({"type": "fin", "content": respuesta_completa})

    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio

    # Log con métricas
    resultado_json = {
        "id": getattr(ultimo_chunk, "id", None),
        "object": "gemini.completion",
        "created": int(tiempo_fin),
        "model": "gemini-2.5-flash",
        "usage": {
            "prompt_tokens": None,  # Gemini aún no expone tokens aquí
            "completion_tokens": None,
            "total_tokens": None,
            "tiempo_inicio": tiempo_inicio,
            "tiempo_fin": tiempo_fin,
            "tiempo_total": tiempo_total,
        },
    }

    yield json.dumps({"type": "log", "content": resultado_json}, indent=4)
    yield "data: [DONE]"
