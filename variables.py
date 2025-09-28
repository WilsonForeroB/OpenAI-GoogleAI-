from dotenv import load_dotenv
import os
import json

archivo_configuracion = ".env" 
load_dotenv(archivo_configuracion)

API_KEY_OPENAI=os.getenv('API_KEY_OPENAI')
MODELO_CHATGPT=os.getenv('MODELO_CHATGPT')
URL_CHATGPT_COMPLETIONS=os.getenv('URL_CHATGPT_COMPLETIONS')

API_KEY_GEMINI=os.getenv('API_KEY_GEMINI')
MODELO_GEMINI=os.getenv('MODELO_GEMINI')


