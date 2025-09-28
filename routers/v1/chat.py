import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from GoogleAI.call_llm import obtener_respuesta_stream_gemini
from OpenAI.llm_call import obtener_respuesta_stream_openai
from OpenAI.llm_call_with_function import obtener_respuesta_con_funcion

router = APIRouter(
    prefix="/api/v1/messages",
    tags=["Messages"]
)

@router.post("/stream_gemini")
async def chat_stream_gemini(request: Request):
    """
    Endpoint SSE para streaming de Gemini.
    """
    body = await request.json()
    prompt_gemini = body.get("messages", [])
    lista_variables = body.get("variables", {})

    # Si recibimos estilo OpenAI [{"role": "user", "content": "..."}]
    if isinstance(prompt_gemini, list):
        prompt_gemini = "\n".join([f"{m['role']}: {m['content']}" for m in prompt_gemini])

    async def event_generator() -> AsyncGenerator[str, None]:
        async for fragmento in obtener_respuesta_stream_gemini(prompt_gemini, lista_variables):
            yield f"data: {fragmento}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@router.post("/stream_openai")
async def chat_stream(request: Request):
    """
    Endpoint de streaming de chat completions (Server-Sent Events).
    """
    body = await request.json()
    prompt_openai = body.get("messages", [])
    lista_variables = body.get("variables", {})

    async def event_generator() -> AsyncGenerator[str, None]:
        async for fragmento in obtener_respuesta_stream_openai(prompt_openai, lista_variables):
            # SSE: cada fragmento viene como "data: ..."
            yield f"data: {fragmento}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stream_with_function_openai")
async def chat_stream_function(request: Request):
    """
    Endpoint de streaming de chat completions (Server-Sent Events).
    """
    body = await request.json()
    prompt_openai = body.get("messages", [])
    lista_variables = body.get("variables", {})

    async def event_generator() -> AsyncGenerator[str, None]:
        async for fragmento in obtener_respuesta_con_funcion(prompt_openai, lista_variables):
            # SSE: cada fragmento viene como "data: ..."
            yield f"data: {fragmento}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

