from app.models.database import initialize_database

if __name__ == "__main__":
    print("Criando tabelas no banco de dados...")
    initialize_database()
    print("Tabelas criadas com sucesso!")
