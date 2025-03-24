from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from .base_types import Tabela, Figura, Equacao, Codigo


class Resultado(BaseModel):
    resultado: str
    evidencias_de_suporte: Optional[str] = None
    condicoes_contexto: Optional[str] = None

class Discussao(BaseModel):
    ponto_de_discussao: str
    relacao_com_os_resultados: Optional[str] = None
    possiveis_implicacoes_significado: Optional[str] = None
    conexao_com_outros_trabalhos: Optional[str] = None   

class PontoChave(BaseModel):
    ponto: str

class Insight(BaseModel):
    insight: str

class Inconsistencia(BaseModel):
    lotes: List[int]
    descricao: str
    nota: Optional[str] = None

class TechnicalData(BaseModel):
    documento_draft: str
    tema: str
    resultados_apresentados: List[Resultado] = []
    discussao_interpretacao: List[Discussao] = []
    elementos_visuais_complementares: Optional[Dict[str, List[Tabela | Figura | Equacao | Codigo]]] = {
        "tabelas": [],
        "figuras": [],
        "equacoes": [],
        "codigos": []
    }
    pontos_chave_para_integracao_outline: List[PontoChave] = []
    insights_conexoes_entre_lotes: List[Insight] = []
    sintese_geral_dos_achados: Optional[str] = None
    inconsistencias_divergencias_identificadas: List[Inconsistencia] = []
