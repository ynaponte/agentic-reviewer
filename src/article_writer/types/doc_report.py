from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class AnalysisPoint(BaseModel):
    point_name: str = Field(
        ...,
        description="Nome do ponto analisado, por exemplo, 'Coerência dos dados' ou 'Clareza das discussões'."
    )
    point_analysis: List[str] = Field(
        ...,
        description="Lista de observações e análises referentes ao ponto analisado."
    )

class Equation(BaseModel):
    equation_text: str = Field(
        ...,
        description="Representação textual da equação, preferencialmente em formato LaTeX ou similar."
    )
    description: str = Field(
        None,
        description="Descrição ou interpretação da equação e seu papel na análise."
    )

class Constant(BaseModel):
    name: str = Field(
        ...,
        description="Símbolo ou nome da constante ou parâmetro experimental (ex.: ε, n₁, n₂, a, λ)."
    )
    value: str = Field(
        ...,
        description="Valor da constante, incluindo unidades se aplicável."
    )
    description: Optional[str] = Field(
        None,
        description="Breve descrição do significado ou função da constante no contexto do experimento."
    )


class TableData(BaseModel):
    title: str = Field(
        ...,
        description="Título ou legenda da tabela (ex.: 'Tabela verdade para a primeira porta OR')."
    )
    headers: List[str] = Field(
        ...,
        description="Lista dos cabeçalhos das colunas da tabela."
    )
    rows: List[List[str]] = Field(
        ...,
        description="Linhas da tabela, onde cada linha é uma lista de valores em formato textual."
    )
    description: Optional[str] = Field(
        None,
        description="Observações ou comentários sobre a tabela."
    )


class ResultItem(BaseModel):
    context: str = Field(
        ...,
        description="Contexto do resultado extraído do texto (ex.: evolução do fitness ou contraste obtido)."
    )
    metrics: Optional[Dict[str, str]] = Field(
        None,
        description="Métricas associadas ao resultado, como valores numéricos importantes."
    )


class SectionAnalysis(BaseModel):
    section_name: str = Field(
        ...,
        description="Nome da seção analisada, por exemplo, 'Resultados', 'Discussão' ou sub-seções como 'Primeira porta OR'."
    )
    analysis_points: List[AnalysisPoint] = Field(
        ...,
        description="Lista de pontos de análise críticos referentes à seção."
    )
    equations: Optional[List[Equation]] = Field(
        None,
        description="Lista de equações extraídas ou mencionadas na seção."
    )
    constants: Optional[List[Constant]] = Field(
        None,
        description="Lista de constantes identificadas na seção."
    )
    tables: Optional[List[TableData]] = Field(
        None,
        description="Lista de tabelas apresentadas ou referenciadas na seção."
    )
    extracted_results: Optional[List[ResultItem]] = Field(
        None,
        description="Resultados ou achados extraídos da análise da seção."
    )


class ChunkReport(BaseModel):
    doc_name: str = Field(description="Nome do documento base")
    chunk_id: List[int] = Field(description="Lista dos índices das chunks analisadas")
    resumo: str = Field(description="Resumo breve das ideias e pontos principais dos chunks analisados")
    sections_analysis: List[SectionAnalysis] = Field(
        ...,
        description="Analise das seções do presentes nas chunks."
    )
    overall_discussion: Optional[List[str]] = Field(
        None,
        description="Comentários gerais e análise crítica global do conteúdo."
    )
