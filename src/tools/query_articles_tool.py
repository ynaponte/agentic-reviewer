from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any, Literal
import json


class QueryArticlesToolInput(BaseModel):
    search_query: str = Field(
        ..., description="Query to search through out the database's articles for."
    )

    doc_type: Optional[Literal['draft', 'reference']] = Field(
        None, description=(
            "Type of document to be searched for. Can be draft or reference"
            "Use to limit the search for documentes of a specific type."
            "Can be either 'draft' or 'reference'. It is a optional parameter."
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
        "This tool is used when a broad search through the database is needed."
        "It searches the articles database for chunks that are most similar to the given search query,"
        "returning a json string with a list of results that each has the chunk's content and it's metadata"
        "for identification and contextualization."
        "Supports filtering by uploader (who uploaded the article to the database) and/or source (name of the article file)."
    )

    args_schema: Type[BaseModel] = QueryArticlesToolInput
    _vectordatabase: VectorDatabaseManager = PrivateAttr()

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._vectordatabase = VectorDatabaseManager()  # Must be already initialized before use of the tool

    def _run(
        self,
        search_query: str,
        doc_type: Optional[Literal['draft', 'reference']] = None,
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
