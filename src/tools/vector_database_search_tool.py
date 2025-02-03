from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict


class VectorDatabaseToolInput(BaseModel):
    search_query: str = Field(
        ..., description="The query to search in the articles database for."
    )
    uploader: Optional[str] = Field(
        None, description="The name of the document uploader to search for."
    )
    top_k: Optional[int] = Field(
        10, description="The number of top results to return."
    )


class VectorDatabaseTool(BaseTool):
    name: str = 'Search the database for articles'
    description: str = (
        'Search the database for articles'
    )
    args_schema: Type[BaseModel] = VectorDatabaseToolInput
    _vectordatabase: VectorDatabaseManager = PrivateAttr()

    def __init__(self, manager: VectorDatabaseManager,**kwargs):
        super().__init__(**kwargs)
        self._vectordatabase = manager

    def _run(
        self,
        search_query: str,
        uploader: Optional[str] = None,
        top_k: Optional[int] = 10,

    ) -> List[Dict]:        
        return self._vectordatabase.query(
            query=search_query,
            uploader=uploader,
            top_k=top_k
        )
