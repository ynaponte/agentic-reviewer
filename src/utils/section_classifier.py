import re
from typing import Dict

class SectionClassifier:
    def __init__(self):
        # A memória inicia como "Pré-textual"
        self.last_section = "Pré-textual"
        # Padrão para cabeçalho numerado: **<número>.** **<nome_da_seção>**
        self.numbered_pattern = re.compile(r'\*\*\s*(\d+\.)\s*\*\*\s*\*\*(.*?)\*\*', re.IGNORECASE)
        # Padrão para cabeçalho do Abstract: **Abstract: ...**
        self.abstract_pattern = re.compile(r'\*\*\s*Abstract\s*:\s*(.*?)\*\*', re.IGNORECASE)

    def classify_chunk(self, chunk: Dict) -> Dict:
        """
        Recebe um dicionário com a chave "text" (texto da chunk) e
        insere uma nova chave "sections" com uma lista de seções às quais
        o texto pertence, considerando a memória da última seção encontrada.
        """
        text = chunk.get("text", "")
        sections_in_chunk = []
        
        # 1. Procura cabeçalho do Abstract
        abs_match = self.abstract_pattern.search(text)
        if abs_match:
            # Se houver texto antes do abstract, então esse trecho pertence à seção corrente.
            if abs_match.start() > 0:
                sections_in_chunk.append(self.last_section)
            # Atualiza a memória para "Abstract" e adiciona à lista.
            self.last_section = "Abstract"
            sections_in_chunk.append("Abstract")
        
        # 2. Procura cabeçalhos numerados
        numbered_matches = list(self.numbered_pattern.finditer(text))
        for match in numbered_matches:
            # Se o cabeçalho não está no início e nenhuma seção foi adicionada ainda,
            # isso indica que há texto anterior que pertence à seção corrente.
            if match.start() > 0 and not sections_in_chunk:
                sections_in_chunk.append(self.last_section)
            # Extrai o nome da nova seção (remove espaços extras)
            header_title = match.group(2).strip()
            sections_in_chunk.append(header_title)
            # Atualiza a memória para a nova seção
            self.last_section = header_title
        
        # Se nenhum cabeçalho foi encontrado no chunk, assume-se que todo o texto pertence à última seção conhecida.
        if not sections_in_chunk:
            sections_in_chunk.append(self.last_section)
        
        # Insere a lista de seções no dicionário e retorna-o.
        chunk["sections"] = sections_in_chunk
        return chunk
