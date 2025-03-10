from src.utils import VectorDatabaseManager
from src.tools import QueryArticlesTool, FetchMetadataTool, FetchArticlesTool

db = VectorDatabaseManager()
db.initialize_db(
    persist_directory="./article_vectorstore",
    collection_name="flow_test_collection"
)

print(QueryArticlesTool()._run(search_query="Acopladores Ópticos"))
print(FetchMetadataTool()._run(source='Resultado1.pdf', type='draft'))
print(FetchArticlesTool()._run(source='Resultado1.pdf', chunk_id=0))

