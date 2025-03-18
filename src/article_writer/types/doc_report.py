from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class MathObject(BaseModel):
    math_object: str = Field(description="Equação, expressão ou símbolo matemático formatado em LaTeX")


class TableObject(BaseModel):
    html:str = Field(
        description=(
            "Tabela extraida do texto, convertida corretamente para HTML completo e funcional. "
            "O formato esperado deve conter todas as tags necessarias, incluindo <table>, <thead>, <tbody> e <tr>. "
            "Caso algum elemento de cabeçalho faltante, deve-se atribuir algo generico."
        )
    )


class GraphObject(BaseModel):
    name:str = Field(description="Nome da imagem")
    description:str = Field(description="Descrição da imagem")


class CodeSnipet(BaseModel):
    language: str = Field(description="Linguagem do código")
    code: str = Field(description="Trecho de código")


class ContentReport(BaseModel):
    synthesis: str = Field(description=(
            "Um resumo crítico do conteúdo **em markdown**, destacando ideias, argumentos e conclusões principais"
        )
    )
    conclusion: str = Field(description=(
            "Síntese final dos principais achados do texto analisado, destacando como os elementos técnicos, "
            "métodos e resultados se conectam aos objetivos do documento. Deve indicar a relevância dos dados extraídos "
            "e possíveis implicações das conclusões apresentadas."
    ))
    key_points: str = Field(description=(
        "Pontos chaves, insights e resultados, estruturados **em markdown**, com seções organizadas, "
        "escritos como bullet points"
    ))
    math_expressions: Optional[List[MathObject]] = Field(
        description=(
            "Uma lista com as expressoes matematicas encontradas no texto, formatadas em LaTeX. "
            "Cada elemento da lista é uma equacao ou expressao matematica."
        ),
        default_factory=list
    )
    code_snipets: Optional[List[CodeSnipet]] = Field(
        description=(
            "Uma lista com os snippets de codigo, onde cada elemento da lista é um snippet completo."
        ),
        default_factory=list
    )
    tables: Optional[List[TableObject]] = Field(
        description=(
            "Uma lista dos trechos que estão em conformidade com o padrão regex para tabelas, copiados do texto, formatados"
            "como tabelas em HTML."
        ),
        default_factory=list
    )
    graphs_and_images: Optional[List[GraphObject]] = Field(
        description=(
            "Uma lista com a descrição de imagens e gráficos que foram encontrados no texto"
        ),
        default_factory=list
    )
