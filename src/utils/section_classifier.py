import re
from langchain.schema import Document

class SectionClassifier:
    def __init__(self):
        # Memória inicia como "Pré-textual"
        self.last_section = "pre-text"
        self.pattern = None  # Padrão a ser identificado a partir da introdução
        # Padrão para cabeçalhos numerados em negrito: **1.** **INTRODUCTION**
        self.numbered_pattern = re.compile(
            r'\*\*\s*([\d\.]+)\s*\*\*\s*\*\*(.*?)\*\*',
            re.IGNORECASE | re.DOTALL
        )
        # Padrão para cabeçalhos em markdown: linhas iniciadas por "##" ou "###"
        self.markdown_header_pattern = re.compile(
            r'^(#{1,})\s*([\d\.]+)\s+(.*)',
            re.IGNORECASE | re.MULTILINE
        )
        # Novo padrão para cabeçalhos com números romanos: Ex: I. Introdução, IV. Resultados
        self.roman_pattern = re.compile(
            r'^(?P<roman>[IVXLCDM]+)\.\s+(.*)',
            re.IGNORECASE | re.MULTILINE
        )

    def identify_pattern(self, text):
        """Encontra a introdução e define o padrão de cabeçalhos."""
        all_matches = (list(self.numbered_pattern.finditer(text)) +
                       list(self.markdown_header_pattern.finditer(text)) +
                       list(self.roman_pattern.finditer(text)))
        if all_matches:
            first_header = all_matches[0]
            if first_header.re == self.numbered_pattern:
                self.pattern = self.numbered_pattern
            elif first_header.re == self.markdown_header_pattern:
                self.pattern = self.markdown_header_pattern
            elif first_header.re == self.roman_pattern:
                self.pattern = self.roman_pattern

    def classify_document(self, doc: Document) -> Document:
        text = doc.page_content
        sections_in_doc = []
        
        # Identificar padrão na introdução, se ainda não foi definido
        if self.pattern is None:
            self.identify_pattern(text)
        
        if self.pattern:
            all_matches = list(self.pattern.finditer(text))
            all_matches.sort(key=lambda m: m.start())
            
            for match in all_matches:
                if match.start() > 0 and not sections_in_doc:
                    sections_in_doc.append(self.last_section)
                
                if self.pattern == self.numbered_pattern:
                    header_title = match.group(2).strip()
                elif self.pattern == self.markdown_header_pattern:
                    header_title = match.group(3).strip()
                elif self.pattern == self.roman_pattern:
                    header_title = match.group(2).strip()
                else:
                    header_title = ""
                
                sections_in_doc.append(header_title)
                self.last_section = header_title
        
        if not sections_in_doc:
            sections_in_doc.append(self.last_section)
        
        doc.metadata["sections"] = "; ".join(sections_in_doc)
        return doc
    
    @property
    def reset(self):
        """Reseta o parser para o estado inicial."""
        self.last_section = "pre-text"
        self.pattern = None

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
