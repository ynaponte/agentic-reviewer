from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Optional, List, Dict, Any, Literal
import re


class TableSearchToolInputs(BaseModel):
    texto: str = Field(..., description="String contendo o texto a ser analisado.")


class TableSearchTool(BaseTool):
    name: str = "TableSearchTool"
    description: str = "Busca no texto informado por blocos que seguem o padrão de tabela."
    args_schema: Type[BaseModel] = TableSearchToolInputs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _run(
        self,
        texto: str,
    ) -> str:
        """
        Busca por blocos de texto que seguem o padrão de tabela.
        O padrão identifica trechos que iniciam com "Tabela <número> -- <título>" e
        se estendem até a ocorrência da palavra "Fonte:" ou até o fim do texto.

        :param texto: String contendo o texto a ser analisado.
        :return: Lista de strings, cada uma contendo o conteúdo identificado como tabela.
        """
        # Expressão regular que utiliza \u2013 para representar o en dash.
        # Explicação:
        #   (Tabela\s+\d+\s+\u2013\s+.*?\n   => Captura a linha de título da tabela.
        #   (?:.*\n)+?                     => Captura as linhas subsequentes de forma não gananciosa.
        #   (?=Fonte:|$)                   => Para a captura ao encontrar "Fonte:" ou o fim do texto.
        pattern = r"(Tabela\s+\d+\s+\u2013\s+.*?\n(?:.*\n)+?)(?=Fonte:|$)"
        
        # Busca todas as ocorrências considerando múltiplas linhas e o padrão Unicode.
        tabelas = re.findall(pattern, texto, flags=re.UNICODE | re.DOTALL)
        return tabelas

if __name__ == "__main__":
    texto_exemplo = (
        "Tabela 1 \u2013 Tabela verdade para a primeira porta\n"
        "OR\n\n"
        "I0 I1 I2 I3 I4 Target _|Out|_ CR(Out)\n\n"
        "0 0 0 0 0 0 0,918661 -0,736887\n"
        "0 0 0 0 1 1 1,121670 0,997304\n"
        "0 0 0 1 0 1 1,125088 1,023734\n"
        "0 0 0 1 1 1 1,247111 1,918108\n"
        "\nFonte: autoria própria (2023).\n"
    )

    resultado = TableSearchTool()._run(texto_exemplo)
    for i, tabela in enumerate(resultado, start=1):
        print(f"Tabela {i} encontrada:\n{tabela}\n")
