import json
from openai import OpenAI, AssistantEventHandler
from dotenv import load_dotenv
import requests
from typing_extensions import override
import os
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.models.message_model import Message
import base64

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads", "images")
# Cria o diretório de uploads, caso não exista
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_image_locally(image_path: str) -> str:
    """Salva uma imagem localmente no diretório de uploads."""
    filename = os.path.basename(image_path)
    destination_path = os.path.join(UPLOAD_DIR, filename)
    with open(image_path, "rb") as src:
        with open(destination_path, "wb") as dst:
            dst.write(src.read())
    return destination_path


def process_user_image(image_path: str, prompt: str = None):
    """
    Processa uma imagem enviada pelo usuário, gera uma descrição e embedding,
    e salva no banco de dados.
    """
    try:
        # Salva a imagem localmente no diretório de uploads
        saved_image_path = save_image_locally(image_path)

        # Codifica a imagem em base64
        with open(saved_image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        # Chave da API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("A chave da API OpenAI não foi encontrada.")

        # Prompt padrão
        if not prompt:
            prompt = """
            Descreva detalhadamente o conteúdo desta imagem. 
            Para refeições, inclua detalhes sobre macronutrientes e calorias.
            """

        # Cabeçalhos e payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        }

        # Envia a requisição para a API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        # Processa a resposta
        response_data = response.json()
        if "choices" not in response_data:
            raise ValueError(f"Erro na resposta da API: {response_data}")

        description = response_data["choices"][0]["message"]["content"]

        # Gera o embedding da imagem
        embedding = generate_text_embedding(description)

        # Salva no banco de dados
        save_message(
            sender="assistant",
            content=description,
            image_path=saved_image_path,
            image_embedding=embedding,
        )

        return description

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return "Erro ao processar a imagem."


def generate_text_embedding(text: str):
    """
    Gera o embedding de um texto usando a API da OpenAI.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )

        embedding = response.data[0].embedding
        return embedding

    except Exception as e:
        print(f"Erro ao gerar embedding do texto: {e}")
        return None

def save_message(sender: str, content: str = None, image_path: str = None, image_embedding: dict = None):
    """
    Salva uma mensagem no banco de dados.
    """
    db: Session = SessionLocal()
    try:
        message = Message(
            sender=sender,
            content=content,
            image_path=image_path,
            image_embedding=image_embedding,
        )
        db.add(message)
        db.commit()
    except Exception as e:
        print(f"Erro ao salvar mensagem no banco: {e}")
    finally:
        db.close()


def create_assistant():
    """Cria um Assistant configurado."""
    assistant = client.beta.assistants.create(
        name="Chat Assistant",
        instructions="""
        Você é um assistente que identifica mensagens que se referem a imagens
        ou estão relacionadas ao conteúdo de imagens.
        """,
        model="gpt-4o-mini",
    )
    print(f"Assistant criado com ID: {assistant.id}")
    return assistant


def create_thread():
    """Cria um Thread para gerenciar a conversa."""
    thread = client.beta.threads.create()
    print(f"Thread criada com ID: {thread.id}")
    return thread


def add_message_to_thread(thread_id, role, content):
    """Adiciona uma mensagem ao Thread."""
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role=role,
        content=content,
    )
    print(f"Mensagem adicionada ao Thread com ID: {message.id}")
    return message


def run_thread(thread_id, assistant_id, instructions):
    """Executa o Thread com o Assistant e processa a resposta via streaming."""
    class EventHandler(AssistantEventHandler):
        def __init__(self):
            super().__init__()
            self.response = ""
            self.has_initial_text = False

        @override
        def on_text_created(self, text):
            """Captura o texto inicial gerado pelo Assistant."""
            if not self.has_initial_text:
                self.response += text.value
                self.has_initial_text = True
                print(f"\nAssistant > {text.value}")

        @override
        def on_text_delta(self, delta, snapshot):
            """Captura incrementos no texto gerado."""
            if delta.value:
                self.response += delta.value
                print(delta.value, end="", flush=True)

    # Inicializar o EventHandler
    event_handler = EventHandler()

    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()

    save_message(sender="assistant", content=event_handler.response)

    return event_handler.response
