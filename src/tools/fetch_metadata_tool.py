from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any


class FetchMetadataToolInput(BaseModel):
    doc_id: Optional[str] = Field(
        None, description="Document ID to fetch information (metadata) from."
    )
    
    source: Optional[str] = Field(
        None, description="Article's filename to fetch information (metadata) from."
    )

    uploader: Optional[str] = Field(
        None, description=(
            "Name of a user that has uploaded articles to the database. "
            "Used to fetch information (metadata) from all articles uploaded by this user."
        )
    )


class FetchMetadataTool(BaseTool):
    name: str = "Fetch information about articles in the database"
    description: str = (
        "Fetches information about articles in the database, also known as metadata. "
        "This information includes: document id(doc_id), file name(source), "
        "the total of chunks that the article is divided by (total_chunks), "
        "the total of pages that the article contains(total_pages) and finally "
        "who uploaded the article to the database(uploader)."
    )

    args_schema = Type[BaseModel] = FetchMetadataToolInput
    _vectorstore: VectorDatabaseManager = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = VectorDatabaseManager()

    def _run(
        self,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        uploader: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._vectorstore.search_doc_by_meta(
            doc_id=doc_id, 
            source=source, 
            uploader=uploader,
            metadata_only=True
        )