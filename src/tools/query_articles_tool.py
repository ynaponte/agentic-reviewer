from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any


class QueryArticlesToolInput(BaseModel):
    search_query: str = Field(
        ..., description="Query to search through out the database's articles for."
    )

    uploader: Optional[str] = Field(
        None, description=(
            "Name of a user of the database to filter the search fordocument uploader."
        )
    )

    source: Optional[str] = Field(
        None, description="Document's filename to filter the search for."
    )

    top_k: Optional[int] = Field(
        10, description=(
            "The number of text chunks that have high similarity to the search query to return."
        )
    )


class QueryArticlesTool(BaseTool):
    name: str = "Search the articles database"
    description: str = (
        "Searches the articles database for chunks that are most similar to the given search query."
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
        uploader: Optional[str] = None,
        source: Optional[str] = None,
        top_k: Optional[int] = 10,

    ) -> List[Dict[str, Any]]:        
        return self._vectordatabase.query(
            query=search_query,
            uploader=uploader,
            source=source,
            top_k=top_k
        )
