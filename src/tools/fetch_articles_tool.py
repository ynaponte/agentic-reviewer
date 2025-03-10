from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any, Literal
import json


class FetchArticlesToolInput(BaseModel):    
    source: Optional[str] = Field(
        None, description=(
            "The of the article's filename to fetch"
            "It is case sensitive."
        )
    )

    type: Optional[Literal['draft', 'reference']] = Field(
        None, description=(
            "Type of document to be searched for. Can be draft or reference"
            "Can be used either to fetch all articles of such type"
            "(if all other parameters are not specified) or to narrow"
            "the search to only documents of such type."
        )
    )

    chunk_id: Optional[int] = Field(
        None, description=(
            "A integer corresponding to the id of the chunk to fetch from the document."
        )
    )


class FetchArticlesTool(BaseTool):
    name: str = "FetchArticlesTool"
    description: str = (
        "This tool returns the content of an article as a json string containing as keys the document's"
        "name and as value a list of it's chunks as dictionaries that have information about their"
        "content and metadata(for more contextualization)."
        "It is able to fetch all of an articles' chunks from the database, that match the given source and type"
        "(if specified) if the 'chunk_id' parameter is omitted, otherwise, it fetches data of the one chunk with the"
        "specified 'chunk_id'"
        "This tool must be used when the content of a specific source is need, may that be all of it or chunk by chunk."
    )

    args_schema: Type[BaseModel] = FetchArticlesToolInput
    _vectordatabase: VectorDatabaseManager = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectordatabase = VectorDatabaseManager()  # Must be already initialized before use of the tool

    def _run(
        self,
        source: Optional[str] = None,
        type: Optional[str] = None,
        chunk_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return json.dumps(
            self._vectordatabase.search_doc_by_meta(
                source=source, 
                type=type,
                metadata_only=False,
                chunk_id=chunk_id
            ),
            indent=4
        )
    