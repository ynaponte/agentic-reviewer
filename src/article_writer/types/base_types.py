from pydantic import BaseModel, Field
from typing import List, Optional


class Tabela(BaseModel):
    table: str = Field(description="Transcrição das linhas e colunas da tabelas e seus valores")    

class Figura(BaseModel):
    descricao: str = Field(description="Descrição da inferida da figura")

class Equacao(BaseModel):
    latex: str = Field(description="Código LaTeX que copie a equação extraída")

class Codigo(BaseModel):
    codigo: str = Field(description="Transcrição do código encontrado no texto")

class Resultados(BaseModel):
    resultado: str = Field(description="Transcrição dos resultados numéricos encontrados")
    
