from pydantic import BaseModel
from typing import List, Optional


class Tabela(BaseModel):
    html: str
    titulo: Optional[str] = None
    description: Optional[str] = None

class Figura(BaseModel):
    nome: Optional[str] = None
    descricao: Optional[str] = None

class Equacao(BaseModel):
    latex: str

class Codigo(BaseModel):
    codigo: str
    language: Optional[str] = None
