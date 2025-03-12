from pydantic import BaseModel, Field
from typing import List, Dict


class Arguments(BaseModel):
    pontos_principais: List[str] = Field(
        description="Lista de argumentos centrais e relevantes do documento"
    )
    evidencias: List[str] = Field(
        description="Exemplos, dados ou citações que sustentam os argumentos, com análise de consistência"
    )
    validade: str = Field(
        description="Avaliação da coerência e robustez geral dos argumentos"
    )

class Structure(BaseModel):
    organizacao: str = Field(
        description="Análise da estrutura do documento (ex: introdução, desenvolvimento, conclusão)"
    )
    coerencia: str = Field(
        description="Avaliação da fluidez e lógica na disposição dos tópicos"
    )

class AnaliseCritica(BaseModel):
    arguments: Arguments = Field(description="Os pontos principais, as evidências que os sustentam e uma avaliação da validade dos argumentos.")
    structure: Structure = Field(description="Aborda a organização interna do documento e a coerência entre suas partes.")


class ChunkReport(BaseModel):
    doc_name: str = Field(description="Nome do documento base")
    chunk_id: List[int] = Field(description="Lista dos índices das chunks analisadas")
    contexto: str = Field(description="Descrição do tema e do contexto em que o documento foi produzido")
    resumo: str = Field(description="Resumo breve das ideias e pontos principais do documento, com suas próprias palavras")
    analise_critica : AnaliseCritica = Field(description="Seção que agrupa as principais dimensões da leitura crítica")
    text: str
    summary: str
    keywords: list[str]
