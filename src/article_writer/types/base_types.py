from pydantic import BaseModel, Field
from typing import List, Optional


class Tabela(BaseModel):
    titulo: str = Field(
        description="Título da tabela extraída"
    )
    html: str = Field(
        description="Código HTML que copie a estrutura e dados da tabela extraída",
    )

class Figura(BaseModel):
    titulo: str = Field(description="Título da figura")
    descricao: str = Field(description="Descrição da inferida da figura")

class Equacao(BaseModel):
    latex: str = Field(description="Código LaTeX que copie a equação extraída")

class Codigo(BaseModel):
    language: str = Field(description="Línguagem identificada na qual o código foi escrtio")
    codigo: str = Field(description="Transcrição do código encontrado no texto")
    
