import json
import time
import tiktoken
from typing import AsyncGenerator
from openai import AsyncOpenAI
from variables import API_KEY_OPENAI, MODELO_CHATGPT

encoder = tiktoken.encoding_for_model("gpt-4o")
client = AsyncOpenAI(api_key=API_KEY_OPENAI)


async def obtener_respuesta_stream_openai(
    prompt_openai: list[dict],
    lista_variables: dict
) -> AsyncGenerator[str, None]:
    """
    Genera la respuesta del modelo OpenAI en streaming.
    Cada fragmento se devuelve como un JSON string:
      - {type: "inicio", content: "..."}
      - {type: "chunk", content: "..."}
      - {type: "fin", content: "..."}
      - {type: "log", content: {...}}
    Se finaliza con `data: [DONE]` para compatibilidad.
    """
    tiempo_inicio = time.time()
    respuesta_completa = ""
    ultimo_chunk = None

    # Emitimos mensaje de inicio
    yield json.dumps({"type": "inicio", "content": "Comienza la generación"})

    # Configuración de la request con streaming
    response = await client.chat.completions.create(
        model=MODELO_CHATGPT,
        messages=prompt_openai,
        max_tokens=lista_variables.get("MODELO_TOKENS", 1024),
        top_p=lista_variables.get("top_p", 1.0),
        temperature=lista_variables.get("MODELO_TEMPERATURA", 0.7),
        stream=True,
    )

    # Iteramos sobre los fragmentos
    async for chunk in response:
        ultimo_chunk = chunk  # guardamos por si queremos metadata
        if chunk.choices and chunk.choices[0].delta.content:
            content_piece = chunk.choices[0].delta.content
            respuesta_completa += content_piece
            yield json.dumps({"type": "chunk", "content": content_piece})

    # Emitimos mensaje de fin con todo el texto acumulado
    yield json.dumps({"type": "fin", "content": respuesta_completa})

    # Calculamos métricas
    tiempo_fin = time.time()
    tiempo_total = tiempo_fin - tiempo_inicio

    prompt_text = "\n".join([msg["content"] for msg in prompt_openai])
    tokens_prom = len(encoder.encode(prompt_text))
    tokens_respuesta = len(encoder.encode(respuesta_completa))

    # Emitimos log final
    resultado_json = {
        "id": getattr(ultimo_chunk, "id", None),
        "object": "chat.completion",
        "created": getattr(ultimo_chunk, "created", None),
        "model": MODELO_CHATGPT,
        "usage": {
            "prompt_tokens": tokens_prom,
            "completion_tokens": tokens_respuesta,
            "total_tokens": tokens_prom + tokens_respuesta,
            "tiempo_inicio": tiempo_inicio,
            "tiempo_fin": tiempo_fin,
            "tiempo_total": tiempo_total,
        },
    }

    yield json.dumps({"type": "log", "content": resultado_json}, indent=4)

    # Cerramos al estilo OpenAI
    yield "data: [DONE]"
