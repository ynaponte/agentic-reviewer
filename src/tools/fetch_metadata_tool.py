from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any, Literal
import json


class FetchMetadataToolInput(BaseModel):
    source: Optional[str] = Field(
        None, description=(
            "Article's filename to fetch information (metadata) from."
        )
    )

    doc_type: Optional[Literal['draft', 'reference']] = Field(
        None, description=(
            "Type of document to fetch information (metadata) from. "
            "Can be used either to fetch information (metadata) from all articles"
            "of such type (if all other parameters are not specified) or to narrow"
            "the search to only documents of such type."
            "Can be either 'draft' or 'reference'."
            "If can't find any documents with requested inputs, the tool will give the message 'Artigo não encontrado. "
            "Dados da busca:' followed by the inputs provided to the tool"
        )
    )


class FetchMetadataTool(BaseTool):
    name: str = "FetchMetadataTool"
    description: str = (
        "Fetches information about articles in the database, also known as metadata. "
        "This information includes: document id(doc_id), file name(source), "
        "the total of chunks that the article is divided by (total_chunks), "
        "the total of pages that the article contains(total_pages) and finally "
        "what is the type of the document(draft or reference)."
        "This tool does not return any content at all, only information."
    )

    args_schema: Type[BaseModel] = FetchMetadataToolInput
    _vectorstore: VectorDatabaseManager = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectorstore = VectorDatabaseManager()

    def _run(
        self,
        source: Optional[str] = None,
        doc_type: Optional[Literal['draft', 'reference']] = None,
    ) -> List[Dict[str, Any]]:
        return json.dumps(
            self._vectorstore.search_doc_by_meta(
                source=source, 
                doc_type=doc_type,
                metadata_only=True
            )
        )