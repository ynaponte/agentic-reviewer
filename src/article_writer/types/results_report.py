from pydantic import BaseModel
from typing import List, Optional
from .base_types import Tabela, Figura, Equacao, Codigo


class ElementsExtraction(BaseModel):
    resultados: List[str] = []
    metricas: List[str] = []
    tabelas: List[Tabela] = []
    figuras: List[Figura] = []
    equacoes: List[Equacao] = []
    codigos: List[Codigo] = []