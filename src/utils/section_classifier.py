import re
from langchain.schema import Document

class SectionClassifier:
    def __init__(self):
        # Memória inicia como "Pré-textual"
        self.last_section = "pre-text"
        # Padrão para cabeçalho numerado em negrito: **1.** **INTRODUCTION**
        self.numbered_pattern = re.compile(
            r'\*\*\s*([\d\.]+)\s*\*\*\s*\*\*(.*?)\*\*',
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
            r'^(#{1,})\s*([\d\.]+)\s+(.*)',
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
        "Journal of Artificial Intelligence Research\nVolume 32, Issue 4\n\n**Abstract: This paper presents a novel approach to automatic text classification using transformer models. We demonstrate superior performance on benchmark datasets while reducing computational requirements.**\n\nIn this study, we address the challenge of efficient text classification in resource-constrained environments.",
        "## 1 Introduction\n\nNatural Language Processing (NLP) has seen tremendous advances in recent years, largely due to the development of transformer-based architectures. These models have revolutionized how machines understand and process human language, enabling unprecedented performance on a variety of tasks including text classification, sentiment analysis, and information extraction.\n\nDespite these advances, deploying state-of-the-art models in production environments remains challenging due to their substantial computational requirements. This is particularly problematic in edge computing scenarios where processing power and memory are limited.",
        "### 1.1 Related Work\n\nPrevious approaches to efficient text classification can be broadly categorized into three groups:\n\n1. Model compression techniques\n2. Knowledge distillation methods\n3. Architectural innovations",
        "**2.** **Methodology**\n\nOur approach combines aspects of these three categories to create a more efficient classification system. We start with a pre-trained transformer model and apply a novel pruning technique that selectively removes attention heads with minimal impact on performance.\n\n#### 2.1.1 Model Architecture\n\nThe base architecture consists of a 6-layer transformer encoder with 8 attention heads per layer. We propose a systematic method for identifying redundant attention patterns.\n\nEXPERIMENTAL SETUP\n\nAll experiments were conducted using the GLUE benchmark dataset. Models were trained on a single NVIDIA A100 GPU with 40GB of memory.",
        "## 3 Results and Discussion\n\nTable 1 presents the main experimental results. Our pruned model achieves 98.7% of the full model's accuracy while reducing inference time by 43% and memory usage by 37%.",
        "### 3.1 Ablation Studies\n\nTo understand the contribution of each component, we conducted a series of ablation studies. Removing the adaptive pruning threshold reduced performance by 2.3 percentage points.",
        "IV. Limitations\n\nWhile our approach shows promise, several limitations should be noted:\n\n1. Performance degradation on long documents\n2. Domain-specific fine-tuning requirements",
        "**4.1** **Future Work**\n\nThere are several promising directions for future research:\n\n__Extending Our Approach__\n\nWe plan to explore whether our techniques can be applied to other transformer architectures.",
        "# 5 Conclusion\n\nIn this paper, we presented an efficient approach to text classification that maintains performance while significantly reducing computational requirements. Our method enables deployment of near-state-of-the-art models in resource-constrained environments.",
        "Acknowledgments\n\nThis research was supported by grants from the National Science Foundation (Grant No. AI-2023-456).\n\nReferences\n\nSmith, J., & Johnson, P. (2023). Efficient Transformers: A Survey.\nZhang et al. (2022). Knowledge Distillation Techniques for NLP."
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
