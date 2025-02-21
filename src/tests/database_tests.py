from ..utils import VectorDatabaseManager

def main():
    teste_db = VectorDatabaseManager()
    (
        teste_db
        .initialize_db(persist_directory='./vectorstore_test', collection_name='nova_collection')
    )

    teste_db.store_documents(directory_list=['./artigos'], uploader='Yna')

    # Uncomment and modify the following lines as needed for testing
    query = teste_db.query('Processamento de dados ópticos')
    # results = teste_db.search_doc_by_meta(source='Obtencao-de-portas-logicas-com-acoplador-de-fibra-optica.pdf', uploader='Yna')
    # results2 = teste_db.search_doc_by_meta(uploader='Yna', metadata_only=True)
    # results3 = teste_db.search_doc_by_meta(uploader='Yna', metadata_only=False)
    print(query)
    # print(results)
    # print(results2)
    # print(results3)

if __name__ == "__main__":
    main()

