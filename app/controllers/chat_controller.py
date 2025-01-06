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
import numpy as np


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


def get_last_user_message():
    """Recupera a última mensagem enviada pelo usuário."""
    db: Session = SessionLocal()
    try:
        last_message = (
            db.query(Message)
            .filter(Message.sender == "user")
            .order_by(Message.timestamp.desc())
            .first()
        )
        return last_message.content if last_message else None
    except Exception as e:
        print(f"Erro ao buscar a última mensagem do usuário: {e}")
        return None
    finally:
        db.close()


def cosine_similarity(vec1, vec2):
    """Calcula a similaridade de cosseno entre dois vetores."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def get_all_embeddings():
    """Busca todos os embeddings salvos no banco."""
    db: Session = SessionLocal()
    try:
        embeddings = db.query(Message).filter(Message.image_embedding.isnot(None)).all()
        return [
            {"embedding": record.image_embedding, "message": record.content}
            for record in embeddings
        ]
    except Exception as e:
        print(f"Erro ao buscar embeddings do banco: {e}")
        return []
    finally:
        db.close()


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
            Você é uma IA de reconhecimento de imagem.
            Descreva detalhadamente o conteúdo desta imagem.
            Retire o máximo de descrição e detalhes da imagem e escolha palavras chaves para a imagem.
            Deve existir características visuais, tags da imagem e detalhes específicos da imagem.
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

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        response_data = response.json()
        if "choices" not in response_data:
            raise ValueError(f"Erro na resposta da API: {response_data}")

        description = response_data["choices"][0]["message"]["content"]

        embedding = generate_text_embedding(description)

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


def generate_text_embedding(text, model="text-embedding-3-small"):
    """
    Gera um embedding para o texto usando o modelo da OpenAI.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=model
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
            self.has_initial_text = False  # Flag para verificar se o texto inicial foi processado

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

    event_handler = EventHandler()

    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions,
        event_handler=event_handler,
    ) as stream:
        stream.until_done()

    save_message(sender="assistant", content=event_handler.response)


    if event_handler.response.lower().startswith("sim"):
        user_last_message = get_last_user_message()
        print(f'funcinou')
        if user_last_message:
            user_embedding = generate_text_embedding(user_last_message)
            print('entrou')
            if user_embedding:
                embeddings = get_all_embeddings()
                print('buscando embeddings')
                similar_message = find_similar_embedding(user_embedding, embeddings)

                if similar_message:
                    print(f"Mensagem semelhante encontrada: {similar_message['message']}")
                else:
                    print("Nenhuma mensagem semelhante encontrada.")
            else:
                print("Erro ao gerar embedding do texto do usuário.")
        else:
            print("Nenhuma mensagem do usuário encontrada.")

    return event_handler.response




def find_similar_embedding(user_embedding, embeddings, threshold=0.7):
    """
    Encontra embeddings semelhantes com base no embedding do usuário.
    """
    try:
        if user_embedding is None:
            print("O embedding do usuário está vazio.")
            return None

        valid_embeddings = [
            {"embedding": emb["embedding"], "message": emb["message"]}
            for emb in embeddings
            if emb["embedding"] is not None
        ]

        if not valid_embeddings:
            print("Nenhum embedding válido encontrado no banco.")
            return None

        results = []
        for emb in valid_embeddings:
            similarity = cosine_similarity(user_embedding, emb["embedding"])
            if similarity >= threshold:
                results.append({"similarity": similarity, "message": emb["message"]})

        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[0] if results else None
    except Exception as e:
        print(f"Erro ao buscar embedding semelhante: {e}")
        return None


def fetch_last_user_message():
    """Recupera a última mensagem enviada pelo usuário."""
    db: Session = SessionLocal()
    try:
        last_message = db.query(Message).filter(Message.sender == "user").order_by(Message.timestamp.desc()).first()
        return last_message.content if last_message else None
    finally:
        db.close()
