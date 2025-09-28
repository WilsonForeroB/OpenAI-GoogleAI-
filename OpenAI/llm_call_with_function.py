# OpenAI/llm_call.py
import json
import time
from typing import AsyncGenerator, Dict, Any, List
import tiktoken
from openai import AsyncOpenAI
from variables import API_KEY_OPENAI, MODELO_CHATGPT

# Si usas otra codificación, ajusta aquí
encoder = tiktoken.encoding_for_model("gpt-4o")
client = AsyncOpenAI(api_key=API_KEY_OPENAI)


def _extrae_vars(vars_in: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza variables de generación con defaults."""
    return {
        "model": vars_in.get("model", MODELO_CHATGPT),
        "max_tokens": vars_in.get("MODELO_TOKENS", 1024),
        "temperature": vars_in.get("MODELO_TEMPERATURA", 0.7),
        "top_p": vars_in.get("top_p", 1.0),
    }

# ====== Function calling (streaming + variables + 2ª ronda) ======

# Función Python a exponer
def obtener_hora_ciudad(ciudad: str) -> str:
    """Ejemplo simple: devuelve hora local de una zona horaria (IANA)."""
    # Para evitar dependencias en producción puedes usar zoneinfo (Py>=3.9).
    # Aquí uso zoneinfo estándar.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        now = datetime.now(ZoneInfo(ciudad))
        return now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return f"No se pudo obtener la hora para '{ciudad}'. Usa formato IANA, p.ej. 'Europe/Madrid'."


async def obtener_respuesta_con_funcion(
    prompt_openai: List[Dict[str, Any]],
    lista_variables: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Streaming con 'tools' (function calling) + variables.
    - 1ª ronda (stream): el modelo puede pedir tool_calls.
    - Ejecutamos la función Python.
    - 2ª ronda (stream): devolvemos el resultado como mensaje 'tool' y el modelo redacta la respuesta final.
    Emite: inicio, chunk, function_call, function_result, fin, log, [DONE]
    """
    cfg = _extrae_vars(lista_variables)

    # Definición de la tool (función)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "obtener_hora_ciudad",
                "description": "Devuelve la hora actual en una ciudad (zona horaria IANA).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ciudad": {
                            "type": "string",
                            "description": "Ej.: 'Europe/Madrid', 'America/Bogota'"
                        }
                    },
                    "required": ["ciudad"],
                },
            },
        }
    ]

    # Estado
    respuesta_completa = ""
    tiempo_inicio = time.time()

    # Emitimos inicio
    yield json.dumps({"type": "inicio", "content": "Comienza la generación con función"})

    # ===== RONDA 1: el modelo decide si llamar a la función =====
    resp1 = await client.chat.completions.create(
        model=cfg["model"],
        messages=prompt_openai,
        tools=tools,
        tool_choice="auto",
        max_tokens=cfg["max_tokens"],
        top_p=cfg["top_p"],
        temperature=cfg["temperature"],
        stream=True,
    )

    # Acumular tool_calls en streaming (pueden venir fragmentados)
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}

    async for chunk in resp1:
        for choice in chunk.choices:
            delta = choice.delta

            # Texto normal de la 1ª ronda
            if delta and delta.content:
                respuesta_completa += delta.content
                yield json.dumps({"type": "chunk", "content": delta.content})

            # tool_calls en streaming (name/arguments llegan por partes)
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = getattr(tc, "index", 0) or 0
                    acc = tool_calls_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    # id parcial
                    if getattr(tc, "id", None):
                        acc["id"] = tc.id
                    # función parcial
                    if tc.function:
                        if getattr(tc.function, "name", None):
                            acc["name"] = tc.function.name
                        if getattr(tc.function, "arguments", None):
                            acc["arguments"] += tc.function.arguments

            # Si la razón de fin es "tool_calls", cerró la petición de función
            if getattr(choice, "finish_reason", None) == "tool_calls":
                # Emitimos el/los function_call completo(s)
                for acc in tool_calls_acc.values():
                    yield json.dumps({
                        "type": "function_call",
                        "name": acc.get("name"),
                        "arguments": acc.get("arguments", "")
                    })

    # Si no hubo tool_calls, cerramos flujo aquí
    if not tool_calls_acc:
        # fin + log + [DONE]
        yield json.dumps({"type": "fin", "content": respuesta_completa})
        tiempo_fin = time.time()
        prompt_text = "\n".join([m.get("content", "") for m in prompt_openai])
        tokens_prom = len(encoder.encode(prompt_text))
        tokens_resp = len(encoder.encode(respuesta_completa))
        yield json.dumps({
            "type": "log",
            "content": {
                "model": cfg["model"],
                "usage": {
                    "prompt_tokens": tokens_prom,
                    "completion_tokens": tokens_resp,
                    "total_tokens": tokens_prom + tokens_resp,
                    "tiempo_inicio": tiempo_inicio,
                    "tiempo_fin": tiempo_fin,
                    "tiempo_total": tiempo_fin - tiempo_inicio,
                }
            }
        }, indent=4)
        yield "[DONE]"
        return

    # ===== Ejecutar funciones y montar mensajes para la 2ª ronda =====
    assistant_tool_calls_msg = {
        "role": "assistant",
        "tool_calls": []
    }
    tool_messages: List[Dict[str, Any]] = []

    for acc in tool_calls_acc.values():
        name = acc.get("name") or ""
        args_json = acc.get("arguments", "") or "{}"
        tool_id = acc.get("id") or "tool_call_0"

        # Ejecutar función Python
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            args = {}

        if name == "obtener_hora_ciudad":
            result = obtener_hora_ciudad(args.get("ciudad", ""))
        else:
            result = f"Función '{name}' no implementada en el servidor."

        # Emitimos al cliente el resultado de la función
        yield json.dumps({"type": "function_result", "name": name, "content": result})

        # Construimos mensajes para la 2ª ronda
        assistant_tool_calls_msg["tool_calls"].append({
            "id": tool_id,
            "type": "function",
            "function": {"name": name, "arguments": args_json}
        })
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result
        })

    # ===== RONDA 2: devolvemos el resultado como 'tool' y que el modelo redacte =====
    messages_round2 = prompt_openai + [assistant_tool_calls_msg] + tool_messages
    resp2 = await client.chat.completions.create(
        model=cfg["model"],
        messages=messages_round2,
        max_tokens=cfg["max_tokens"],
        top_p=cfg["top_p"],
        temperature=cfg["temperature"],
        stream=True,
    )

    ultimo_chunk = None
    async for chunk in resp2:
        ultimo_chunk = chunk
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            respuesta_completa += piece
            yield json.dumps({"type": "chunk", "content": piece})

    # fin + log + [DONE]
    yield json.dumps({"type": "fin", "content": respuesta_completa})

    tiempo_fin = time.time()
    prompt_text = "\n".join([m.get("content", "") for m in prompt_openai])
    tokens_prom = len(encoder.encode(prompt_text))
    tokens_resp = len(encoder.encode(respuesta_completa))

    resultado_json = {
        "id": getattr(ultimo_chunk, "id", None),
        "object": "chat.completion",
        "created": getattr(ultimo_chunk, "created", None),
        "model": cfg["model"],
        "usage": {
            "prompt_tokens": tokens_prom,
            "completion_tokens": tokens_resp,
            "total_tokens": tokens_prom + tokens_resp,
            "tiempo_inicio": tiempo_inicio,
            "tiempo_fin": tiempo_fin,
            "tiempo_total": tiempo_fin - tiempo_inicio,
        },
    }
    yield json.dumps({"type": "log", "content": resultado_json}, indent=4)
    yield "[DONE]"
