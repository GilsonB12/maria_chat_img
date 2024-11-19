import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.orm import Session
from app.models.message_model import Message
from app.models.database import SessionLocal

# Carregar variáveis de ambiente do .env
load_dotenv()

# Configuração do cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def save_message(sender: str, content: str):
    """Salva uma mensagem no banco de dados."""
    db: Session = SessionLocal()
    try:
        message = Message(sender=sender, content=content)
        db.add(message)
        db.commit()
    except Exception as e:
        print(f"Erro ao salvar mensagem no banco: {e}")
    finally:
        db.close()

def get_chat_context(limit: int = 10) -> str:
    """Recupera o contexto das últimas mensagens como texto concatenado."""
    db: Session = SessionLocal()
    try:
        messages = db.query(Message).order_by(Message.id.desc()).limit(limit).all()
        messages.reverse()  # Reverter para ordem cronológica
        return "\n".join([f"{msg.sender.capitalize()}: {msg.content}" for msg in messages])
    except Exception as e:
        print(f"Erro ao recuperar contexto do banco: {e}")
        return ""
    finally:
        db.close()

def get_gpt_response(user_message: str) -> str:
    """Obtém a resposta do GPT com base no contexto como texto concatenado."""
    # Salvar mensagem do usuário no banco
    save_message("user", user_message)

    # Criar o contexto concatenado como texto
    context = get_chat_context()
    prompt = f"{context}\nUser: {user_message}\nAssistant:"
    print(prompt)
    try:
        # Chamar a API OpenAI com o prompt como texto
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Altere para outro modelo se necessário
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extrair a resposta
        gpt_reply = completion.choices[0].message.content

        # Salvar resposta do GPT no banco
        save_message("assistant", gpt_reply)
        return gpt_reply
    except Exception as e:
        print(f"Erro ao processar a solicitação para a OpenAI: {e}")
        return "Houve um problema ao obter a resposta. Tente novamente mais tarde."
