from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any


class FetchArticlesToolInput(BaseModel):
    doc_id: Optional[str] = Field(
        None, description="Document ID of the article to fetch"
    )
    
    source: Optional[str] = Field(
        None, description="The of the article's filename to fetch"
    )

    uploader: Optional[str] = Field(
        None, description=(
            "Name of a user of the database to fetch all articles he or she uploaded."
        )
    )

    chunk_id: Optional[int] = Field(
        None, "A integer corresponding to the id of the chunk to fetch from the document."
    )


class FetchArticlesTool(BaseTool):
    name: str = "Fetch entire articles from the database"
    description: str = (
        "Fetches all articles' chunks from the database that mach the given source and/or uploader. "
        "Can fetch by document id, source, uploader, document id and uploader or source and uploader."
        "Can also a fetch a specific chunk, if a chunk_id is provided. If not, all chunks are returned."
    )

    args_schema: Type[BaseModel] = FetchArticlesToolInput
    _vectordatabase: VectorDatabaseManager = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectordatabase = VectorDatabaseManager()  # Must be already initialized before use of the tool

    def _run(
        self,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        uploader: Optional[str] = None,
        chunk_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self._vectordatabase.search_doc_by_meta(
            doc_id=doc_id, 
            source=source, 
            uploader=uploader,
            metadata_only=False,
            chunk_id=chunk_id
        )
    