import re
from langchain.schema import Document

class SectionClassifier:
    def __init__(self):
        # Memória inicia como "Pré-textual"
        self.last_section = "pre-text"
        # Padrão para cabeçalho numerado em negrito: **1.** **INTRODUCTION**
        self.numbered_pattern = re.compile(
            r'\*\*\s*(\d+\.)\s*\*\*\s*\*\*(.*?)\*\*',
            re.IGNORECASE | re.DOTALL
        )
        # Padrão para cabeçalho do Abstract, em negrito: **Abstract: ...**
        self.abstract_pattern = re.compile(
            r'\*\*\s*Abstract\s*:\s*(.*?)\*\*',
            re.IGNORECASE | re.DOTALL
        )
        # Novo padrão para cabeçalhos em markdown: linhas iniciadas por "##" ou "###"
        # Exemplo: "## 1 Introduction" ou "### 1.1 Details"
        self.markdown_header_pattern = re.compile(
            r'^(#{2,})\s*(\d+(?:\.\d+)*)\s+(.*)',
            re.IGNORECASE | re.MULTILINE
        )

    def classify_document(self, doc: Document) -> Document:
        """
        Recebe um objeto Document (com page_content e metadata) e
        insere em metadata uma nova chave "sections" contendo uma lista
        das seções a que aquele chunk (página) pertence, utilizando memória
        da última seção identificada.
        """
        text = doc.page_content
        sections_in_doc = []
        
        # 1. Procura cabeçalho do Abstract (padrão em negrito)
        abs_match = self.abstract_pattern.search(text)
        if abs_match:
            # Se houver texto antes do Abstract, esse trecho pertence à seção corrente
            if abs_match.start() > 0:
                sections_in_doc.append(self.last_section)
            self.last_section = "Abstract"
            sections_in_doc.append("Abstract")
        
        # 2. Procura cabeçalhos numerados usando ambos os padrões:
        #    a) Cabeçalhos em negrito (padrão antigo)
        #    b) Cabeçalhos em markdown (padrão novo)
        numbered_matches = list(self.numbered_pattern.finditer(text))
        markdown_matches = list(self.markdown_header_pattern.finditer(text))
        all_matches = numbered_matches + markdown_matches
        # Ordena as ocorrências pelo índice de início no texto
        all_matches.sort(key=lambda m: m.start())
        
        for match in all_matches:
            # Se o cabeçalho não está no início e nenhum cabeçalho foi adicionado ainda,
            # adiciona a seção corrente
            if match.start() > 0 and not sections_in_doc:
                sections_in_doc.append(self.last_section)
            # Extrai o título do cabeçalho dependendo do padrão encontrado
            if match.re == self.numbered_pattern:
                header_title = match.group(2).strip()
            elif match.re == self.markdown_header_pattern:
                header_title = match.group(3).strip()
            else:
                header_title = ""
            sections_in_doc.append(header_title)
            # Atualiza a memória para a nova seção
            self.last_section = header_title
        
        # Se nenhum cabeçalho foi encontrado, o chunk herda a última seção conhecida.
        if not sections_in_doc:
            sections_in_doc.append(self.last_section)
        
        # Insere a lista de seções nos metadados do Document
        doc.metadata["sections"] = "; ".join(sections_in_doc)
        return doc
    
    @property
    def reset(self):
        """Reseta o parser para o estado inicial."""
        self.last_section = 'pre-text'

# Exemplo de uso:
if __name__ == "__main__":
    # Exemplo de conteúdo de Document com diferentes padrões de cabeçalho
    textos = [
        "Research Article Applied Optics 1\n\nConteúdo pré-textual sem cabeçalho.",
        "7 **Abstract: Digital multiplexers/demultiplexers (MUX/DEMUXes) are essential for computing,**\n\nDados do abstract...",
        "Continuação do abstract sem novo cabeçalho.",
        "16 **1.** **Introduction**\n\nAll-optical data processing is one of the most promising solutions.",
        "## 2 Theoretical Model\n\nDetalhes sobre o modelo teórico...",
        "Chunk com texto antes e depois do cabeçalho:\nTexto da seção anterior.\n\n### 3.1 Conclusion\n\nConsiderações finais."
    ]
    
    # Cria objetos Document simulados
    docs = [Document(page_content=t, metadata={}) for t in textos]
    
    classifier = SectionClassifier()
    
    for idx, doc in enumerate(docs, start=1):
        classified_doc = classifier.classify_document(doc)
        print(f"----- Documento {idx} -----")
        print("Seções:", classified_doc.metadata.get("sections"))
        print("Conteúdo:")
        print(classified_doc.page_content)
        print()
