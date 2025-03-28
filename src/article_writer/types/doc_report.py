from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from .base_types import Tabela, Figura, Equacao, Codigo


class CriticalAnalysis(BaseModel):
    document: str = Field(
        description="Nome do documento processado"
    )
    batch_identifier: int = Field(
        description="Numero que identifica a batch "
    )
    resumo_critico: str = Field(
        description="Transcrição da analise crítica performada sob o conteúdo analisado"
    )
    pontos_chave_para_integracao_outline: List[PontoChave] = []
    insights_conexoes_entre_lotes: List[Insight] = []
    sintese_geral_dos_achados: Optional[str] = None
    inconsistencias_divergencias_identificadas: List[Inconsistencia] = []
