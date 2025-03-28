from pydantic import BaseModel, Field
from typing import List, Optional
from .base_types import Tabela, Figura, Equacao, Codigo


class ElementsExtraction(BaseModel):
    document: str = Field(
        description="Nome do documento processado"
    )
    batch_identifier: int = Field(
        description="Numero que identifica a batch "
    )
    resultados: List[str] = []
    metricas: List[str] = []
    tabelas: List[Tabela] = []
    figuras: List[Figura] = []
    equacoes: List[Equacao] = []
    codigos: List[Codigo] = []