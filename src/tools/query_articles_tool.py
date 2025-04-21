from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any, Literal
import json


class QueryArticlesToolInput(BaseModel):
    search_query: str = Field(
        ..., description=(
            "Query to search through out the database's articles for. "
            "The tool will search for snippets of text the better allign with the query."
        )
    )

    doc_type: Optional[List[Literal['draft', 'reference', 'report']]] = Field(
        None, description=(
            "A list of Document's types to filter the search for."
            "Each value can be either 'draft', 'reference' or 'report'."
            "Use to limit the search for documentes of a specific type."
            "It is a optional parameter."
        )
    )

    source: Optional[List[str]] = Field(
        None, description=(
            "A list of Document's filename to filter the search for."
            "Should be used only if trying to query a specific document or documents."
            "In normal operation it should not be specified, as a broader search is"
            "generally be better. "
            "It is case sensitive. It is a optional parameter."
        )
    )

    top_k: Optional[int] = Field(
        3, description=(
            "The number of text chunks that have high similarity to the search query to return."
            "It is a optional parameter."
        )
    )


class QueryArticlesTool(BaseTool):
    name: str = "QueryArticlesTool"
    description: str = (
        "This tool searches the articles database for snippets of text that are most similar to the given search query,"
        "returning a json string with the chunks that better allign with the query term, presiting the text content and metadata."
        "for identification. Supports filtering by type of document (doc_type parameter) and/or source (name of the article file)."
    )

    args_schema: Type[BaseModel] = QueryArticlesToolInput
    _vectordatabase: VectorDatabaseManager = PrivateAttr()

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._vectordatabase = VectorDatabaseManager()  # Must be already initialized before use of the tool

    def _run(
        self,
        search_query: str,
        doc_type: Optional[Literal['draft', 'reference', 'report']] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = 10,

    ) -> List[Dict[str, Any]]:        
        return json.dumps(
            self._vectordatabase.query(
                query=search_query,
                doc_type=doc_type,
                source=source,
                top_k=top_k
            ),
            indent=2
        )

if __name__ == "__main__":
    tool = QueryArticlesTool()
    print(tool._run(search_query="emulacao de portas logicas em cristais fotonicos", doc_type=['reference'], top_k=5))
