from crewai.tools import BaseTool
from ..utils import VectorDatabaseManager
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any, Literal
import json


class FetchArticlesToolInput(BaseModel):    
    source: str = Field(
        ..., description=(
            "The document's filename to search in the database and fetch its content."
            "It is case sensitive."
        )
    )

    doc_type: Literal['draft', 'reference'] = Field(
        ..., description=(
            "Type of document to be searched for. Can be draft or reference. "
            "Can be used to fetch all articles of the passed type, "
            "narrowing down the search to only documents of such type."
        )
    )

    chunk_id: List[int] = Field(
        ..., description=(
            "A list of integers, where each corresponds to the id of a chunk. "
            "The tool will fetch all of the document's chunks that have a "
            "corresponding id to those on the list."
        )
    )


class FetchArticlesTool(BaseTool):
    name: str = "FetchArticlesTool"
    description: str = (
        "This tool returns the content of a document, sliced in chunks, with addition to metadata such as the total of chunks and total of "
        "pages the document was sliced into and has, respectivally. This information is return as a JSON type string, where the document "
        "is identified by it's name (for example, 'Document1.pdf') at the root of the JSON object.Within this JSON object, the document's "
        "content is segmented into chunks labeled sequentially (such as 'Chunk 1', 'Chunk 2', etc.), and each chunk contains a 'content' "
        "field with the corresponding portion of text. Additionally, the JSON includes a metadata section at the end that specifies the "
        "total number of chunks and the total number of pages of the document. The chunks returned will vary accordingly to the requested "
        "ids and may be based on any list of integers, as long as the corresponding chunks exist."
        "If can't find anything with the requested inputs, a warning will be triggered. "
        "All inputs must be passed for proper functioning."
    )

    args_schema: Type[BaseModel] = FetchArticlesToolInput
    _vectordatabase: VectorDatabaseManager = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vectordatabase = VectorDatabaseManager()  # Must be already initialized before use of the tool

    def _run(
        self,
        source: str,
        doc_type: str,
        chunk_id: List[int]
    ) -> str:
        try:
            search_result = self._vectordatabase.search_doc_by_meta(
                source=source, 
                type=doc_type,
                metadata_only=False,
                chunk_id=chunk_id
            )
            if search_result == []:
                return json.dumps(
                    {"[WARNING]": "No document found. Please, ensure inputs are correct"}
                )
            return json.dumps(
                search_result,
                indent=4
            )
        except AttributeError:
            return json.dumps(
                {"[ERROR]": "Incorrect usage of the tool. All inputs must be passed."}
            )
    