from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Union
from .base_types import Tabela, Figura, Equacao, Codigo


class TechLmntItem(BaseModel):
    doc_ref: str = Field(
        description="Nome do documento em qual o elemento foi estraído"
    )
    lmnt_type: Literal['tabela', 'figura', 'equação', 'código'] = Field(
        description="Tipo do elemento extraído"
    )
    description: str = Field(
        description="Descrição objetiva sobre o que o elemento representa ou como é usado"
    )
    content: Union[Tabela, Figura, Equacao, Codigo] = Field(
        description=(
            "Copia do elemento encontrado, seja ele Tabela, Figura, Equacao, Codigo, "
            "com todos os atributos e conteúdo"
        )
    )


class ElementsList(BaseModel):
    elements_list: List[Union[None, TechLmntItem]] = Field(
        description="""
        Lista contendo os elementos técnicos encontrados no texto, sendo cada elemento um item da lista.
        Se não forem encontrados elementos, o item será None.
        """
    )