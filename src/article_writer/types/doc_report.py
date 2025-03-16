from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class MathObject(BaseModel):
    math_object: str = Field(description="Equação matemática formatada em LaTeX")
    description: str = ""


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
    analysis: str = Field(description=(
            "Relatorio detalhado e extenso, discursivo e descritivo, sobre o conteúdo analisado. "
            "Deve conter explicações completas e estruturadas, amarrando elementos argumentativos "
            "não textuais, com o argumento do conteúdo."
        )
    )
    key_points: str = Field(description="Key-points and insights from the documents argumentation")
    math_expressions: Optional[List[MathObject]] = Field(
        description=(
            "Uma lista com as expressoes matematicas encontradas no texto, formatadas em LaTeX. "
            "Cada elemento da lista é uma equacao matematica."
        ),
        default_factory=list
    )
    code_snipets: Optional[List[CodeSnipet]] = Field(
        description=(
            "Uma lista com os snippets de codigo"
        )
    )
    tables: Optional[List[TableObject]] = Field(
        description=(
            "Uma lista com as tabelas encontradas e copiadas do texto. Cada elemento da lista é uma "
            "tabela distinta, convertida para formato HTML, com uma descrição inclusa."
        ),
        default_factory=list
    )
    graphs_and_images: Optional[List[GraphObject]] = Field(
        description=(
            "Uma lista com a descrição de imagens e gráficos que foram encontrados no texto"
        ),
        default_factory=list
    )
