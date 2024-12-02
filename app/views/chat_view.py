from tkinter import Tk, Text, Entry, Button, filedialog
from app.controllers.chat_controller import (
    create_assistant,
    create_thread,
    add_message_to_thread,
    process_user_image,
    run_thread,
    save_message,
    generate_text_embedding,
)

def start_chat_ui():
    """Inicia a interface gráfica do chat."""
    assistant = create_assistant()
    thread = create_thread()

    root = Tk()
    root.title("Chat Assistant")

    chat_display = Text(root, height=20, width=80, state="disabled")
    chat_display.pack()

    user_input = Entry(root, width=70)
    user_input.pack(side="left", padx=(10, 0))

    def send_message():
        """Envia uma mensagem de texto para o Assistant."""
        user_message = user_input.get().strip()
        if user_message:
            chat_display.config(state="normal")
            chat_display.insert("end", f"Você: {user_message}\n")
            chat_display.config(state="disabled")

            save_message(sender="user", content=user_message)

            assistant_reply = run_thread(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions="Responda à mensagem do usuário de forma clara e concisa."
            )

            chat_display.config(state="normal")
            chat_display.insert("end", f"Assistant: {assistant_reply}\n")
            chat_display.config(state="disabled")
            user_input.delete(0, "end")

    def send_image():
        """Seleciona uma imagem e a envia para processamento."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            chat_display.config(state="normal")
            chat_display.insert("end", f"Você enviou uma imagem: {file_path}\n")
            chat_display.config(state="disabled")

            # Processar a imagem
            assistant_reply = process_user_image(file_path)

            chat_display.config(state="normal")
            chat_display.insert("end", f"Assistant: {assistant_reply}\n")
            chat_display.config(state="disabled")


    def generate_image_embedding_ui():
        """Seleciona uma imagem e gera o embedding."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            embedding = generate_text_embedding(file_path)
            if embedding:
                chat_display.config(state="normal")
                chat_display.insert("end", f"Embedding gerado para a imagem: {file_path}\n{embedding}\n")
                chat_display.config(state="disabled")
            else:
                chat_display.config(state="normal")
                chat_display.insert("end", "Erro ao gerar o embedding da imagem.\n")
                chat_display.config(state="disabled")

    send_button = Button(root, text="Enviar Mensagem", command=send_message)
    send_button.pack(side="left", padx=(10, 0))

    image_button = Button(root, text="Enviar Imagem", command=send_image)
    image_button.pack(side="left", padx=(10, 0))

    embedding_button = Button(root, text="Gerar Embedding de Imagem", command=generate_image_embedding_ui)
    embedding_button.pack(side="left", padx=(10, 0))

    root.mainloop()
