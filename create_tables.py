# create_tables.py
from app.models.database import initialize_database

print("Inicializando o banco de dados e criando tabelas...")
initialize_database()
print("Tabelas criadas ou atualizadas com sucesso!")
