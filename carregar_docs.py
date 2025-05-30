from src.utils import VectorDatabaseManager
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    articles_db = VectorDatabaseManager()
    articles_db.initialize_db(
        persist_directory="./article_vectorstore",
        collection_name="flow_test_collection"
    )
    articles_db.store_documents(directory_list=['./drafts'], doc_type='draft')
    articles_db.store_documents(directory_list=['./artigos'], doc_type='reference')
    articles_db.store_documents(directory_list=['./relatorio'], doc_type='report')

    print(articles_db.search_doc_by_meta(source='Resultado1.pdf', doc_type='draft')) 