from ..utils import VectorDatabaseManager

teste_db = VectorDatabaseManager()
(
    teste_db
    .initialize_db(persist_directory='./vectorstore_test', collection_name='test_collection')
#    .store_documents(directory='./artigos', uploader='Yna')
)

print(teste_db.query("Acoplador Optico"))
print(teste_db.query("Acoplador Optico", type='Yna'))
print(teste_db.query("Acoplador Optico", type='Yna', source='Obtencao-de-portas-logicas-com-acoplador-de-fibra-optica.pdf'))
