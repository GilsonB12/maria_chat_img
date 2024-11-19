import tkinter as tk
from tkinter import scrolledtext
from app.controllers.chat_controller import get_gpt_response
from app.models.database import initialize_database

def send_message(user_input, chat_display):
    """Envia a mensagem do usuário e exibe a resposta do GPT."""
    user_message = user_input.get("1.0", tk.END).strip()
    if user_message:
        chat_display.insert(tk.END, f"Você: {user_message}\n")
        user_input.delete("1.0", tk.END)

        # Obter resposta do GPT
        gpt_reply = get_gpt_response(user_message)
        chat_display.insert(tk.END, f"GPT: {gpt_reply}\n")

def start_chat_app():
    """Inicia a aplicação e a interface Tkinter."""
    initialize_database()

    root = tk.Tk()
    root.title("Chat com GPT")

    # Tela de exibição do chat
    chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=20, width=50)
    chat_display.pack(pady=10)
    chat_display.config(state=tk.NORMAL)

    # Entrada de texto
    user_input = tk.Text(root, height=2, width=40)
    user_input.pack(pady=5)

    # Botão de enviar
    send_button = tk.Button(root, text="Enviar", command=lambda: send_message(user_input, chat_display))
    send_button.pack()

    # Rodar a interface
    root.mainloop()
