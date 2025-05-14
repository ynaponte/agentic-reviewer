from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from .crews.doc_chunk_review_crew import ChunkReviewCrew
from .crews.report_writer_crew.report_writer_crew import ReportWriterCrew
from .crews.outline_crew.outline_crew import OutlineCrew
from .crews.res_and_disc_research_crew.rdresearch_crew import RDResearchCrew
from .crews.final_editing_crew.final_editing_crew import FinalEditingCrew
from pydantic import BaseModel  
from ..tools import QueryArticlesTool
from typing import Dict, List
from ..utils import VectorDatabaseManager
# from .types.doc_report import AnaliseCriticaResultadosDiscussao
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
import asyncio
import itertools
import json

class ChunkReview(BaseModel):
    """A class representing a chunk review."""
    critical_analysis: List[str]
    key_points_detailing: List[str]
    methodology_analysis: List[str]
    techenical_elements: List[str]


class ArticleWriterState(BaseModel):
    drafts_documents: List[str] = ['Resultado1.pdf']
    full_analysis: str = ""
    technical_elements: str = ""
    results_discussion_outline: dict = {}
    conclusion_outline: str = ""
    methodology_outline: str = ""
    discussion: str = ""
    conclusion: str = ""
    # chunk_reports: List[AnaliseCriticaResultadosDiscussao] = []


class ArticleWriterFlow(Flow[ArticleWriterState]):

    articles_db = VectorDatabaseManager()

    @start()
    def start_flow(self):
        # Inicializa a base de dados para uso das ferramentas
        self.articles_db.initialize_db(
            persist_directory="article_vectorstore",
            collection_name="flow_test_collection"
        )
        results_discussion_outline = """
{
  "section_name": "Resultados e Discussão",
  "discution_topics": [
    {
      "topic": "APRESENTAR o propósito do capítulo: relatar os resultados obtidos com a modelagem da porta lógica OR de cinco entradas utilizando algoritmos genéticos e discutir suas implicações."
    },
    {
      "topic": "ANTECIPAR os aspectos que serão explorados: eficiência do algoritmo, desempenho óptico da porta OR, robustez estrutural e limitações identificadas."  
    }
  ],
  "subsections": [
    {
      "section_title": "Resultados",
      "discussion_topics": [
        {
          "topic": "DESCREVER a eficiência do algoritmo genético na convergência das soluções ao longo das gerações.",
          "visual_elements_to_contextualize": [
            {
              "name": "Figura 13",
              "description": "Evolução do fitness para a primeira porta."
            },
            {
              "name": "Figura 15",
              "description": "Evolução do fitness para a segunda porta."
            }
          ],
          "numerical_results_to_include": [
            "Saturação observada por volta da 30ª e 50ª gerações."
          ]
        },
        {
          "topic": "ANALISAR os resultados das tabelas verdade para validar a funcionalidade lógica da porta OR com cinco entradas.",
          "visual_elements_to_contextualize": [
            {
              "name": "Tabela 1",
              "description": "Contrastes da primeira porta."
            },
            {
              "name": "Tabela 2",
              "description": "Contrastes da segunda porta."
            }
          ],
          "numerical_results_to_include": [
            "CR(Out) > 0,30 em 100\u0025 dos casos simulados."
          ]
        }
      ]
    },
    {
      "section_title": "Discussão",
      "discussion_topics": [
        {
          "topic": "RELACIONAR os resultados obtidos com a literatura anterior, destacando a viabilidade da solução proposta e sugerindo necessidade de comparação quantitativa."
        },
        {
          "topic": "ANALISAR as implicações teóricas do uso de algoritmos genéticos na modelagem de circuitos ópticos multientrada."
        },
        {
          "topic": "IDENTIFICAR as principais limitações da pesquisa, como: Ausência de validação experimental; Violação parcial de critérios geométricos; Ausência de análise estatística entre múltiplas execuções."
        },
        {
          "topic": "DESTACAR as contribuições do estudo, incluindo: Preenchimento de lacuna na literatura sobre portas OR quíntuplas em fibra óptica; Redução da complexidade estrutural em circuitos lógicos ópticos; Proposta replicável com base em código-fonte claro e condições bem definidas.",
          "visual_elements_to_contextualize": [
            {
              "name": "Figuras 13, 15, 16",
              "description": "Diagramas que demonstram a eficiência e o desempenho da porta OR."
            },
            {
              "name": "Código-fonte 12",
              "description": "O código utilizado para a modelagem da porta lógica OR de cinco entradas."
            }
          ]
        }
      ] 
    }
  ]
}        
        """

    @listen(start_flow)
    def generate_res_and_disc_outlines(self):
      return
      draft_report = self.articles_db.search_doc_by_meta(
          source='Relatorio.pdf', metadata_only=False)
      outcrew_output = OutlineCrew().crew().kickoff(
          inputs={
              "report": draft_report['Relatorio.pdf']['text_content'],
          }
      )
      self.state.results_discussion_outline = outcrew_output

    @listen(generate_res_and_disc_outlines)
    def dev_final_editing(self):
      section_and_topics = {
         'Results': [
            {
              'topic': 'Genetic Algorithm for Optimizing Optical Logic Gates', 
              'text': 'Para garantir que a resposta seja abrangente e atenda a todas as necessidades, aqui está um resumo detalhado e coerente do processo e dos resultados:\n\n#### Algoritmo Genético para Otimizar Portas Lógicas Ópticas\n\n**Objetivo:** Projeto de portas lógicas ópticas usando algoritmos genéticos, assegurando confinamento forte para estados desejados e fraco para estados indesejados.\n\n**Processo:*\n1. **Inicialização:**\n   - Gere uma população inicial de designs com parâmetros aleatórios como comprimento e largura das onduladas, formas e posições.\n2. **Evolução:**\n   - Utilize ferramentas de simulação (ex.: FDTD) para avaliar cada design baseando-se em:\n     - Força e fraqueza do confinamento óptico para diferentes estados lógicos.\n     - Compactação (área mínima).\n     - Robustez sob variações na entrada ou tolerâncias de fabricação.\n3. **Seleção:**\n   - Implemente mecanismos de seleção como seleção por torneio para escolher os melhores designs da população atual.\n4. **Crossover:**\n   - Aplicar técnicas de crossover como ponto único, multi-ponto e uniforme para combinar designs selecionados e criar designs descendentes.\n5. **Mutação:**\n   - Introduza mudanças aleatórias pequenas nos parâmetros das onduladas em cada design descendente para manter a diversidade na população.\n6. **Iteração:**\n   - Repita os passos de seleção, crossover, mutação e avaliação até que critérios de convergência sejam atingidos (ex.: um número fixo de gerações ou uma valor ótimo de fitness).\n\n**Exemplo: Porta Lógica Óptica AND**\n1. **Resultados da Otimização:**\n   - **Força:** Confinamento óptico forte para '
            },
            {
              'topic': 'Accuracy Comparison Between Proposed Five-Input OR Gate and Conventional Four-Input OR Gates', 
              'text': 'Neste estudo, realizamos uma comparação entre a precisão de um circuito lógico proposto para uma porta OR com cinco entradas (OR5) e as portas OR convencionais com quatro entradas (OR4). A precisão desses circuitos foi avaliada através da análise das suas respectivas funções booleanas, performance em termos de potência de saída (`Pout`) e distribuições magnéticas. As referências 1, 2 e 3 forneceram dados cruciais para esse estudo.\n\nPrimeiramente, as referências 1 e 3 apresentaram tabelas verdadeiras para portas lógicas básicas (AND, OR, NOT), que foram usadas como base para a comparação entre os circuitos OR5 e OR4. A tabela verdade da porta OR4, conforme apresentada na referência 1, mostra que ela produz uma saída `Pout` em função das suas quatro entradas. No entanto, as referências 3 tem valores diferentes para a `Pout`, o que sugere que os circuitos podem ter diferentes implementações ou materiais utilizados.\n\nPara a porta OR5 proposta, não encontramos dados diretos na literatura. Portanto, a comparação foi realizada com base nas características conhecidas das portas OR4 e no entendimento dos princípios básicos da lógica booleana. A função booleana de uma porta OR5 pode ser descrita como segue: se qualquer uma ou todas as entradas forem `1`, então a saída é `1`. Em contraste, uma porta OR4 produz um `1` se pelo menos uma das suas quatro entradas for `1`.\n\nA performance desses circuitos em termos de potência foi comparada utilizando as informações fornecidas pelas referências 1 e 3. A referência 1 apresenta valores de `Pout` para portas OR, mas não especificamente para a porta OR5. A referência 3, embora tenha dados diferentes para a `Pout`, fornece uma base para comparação.\n\nA distribuição magnética foi analisada utilizando os dados fornecidos na referência 2, que apresenta detalhes sobre as distribuições magnéticas de portas OR, AND e NOT. Essas informações são cruciais para entender a eficiência energética desses circuitos lógicos.\n\nA comparação entre as portas OR5 proposta e OR4 convencional revelou algumas diferenças significativas em termos de precisão. As portas OR5 tendem a ser mais precisas, pois permitem o processamento de mais entradas simultaneamente sem comprometer a saída correta. No entanto, elas podem enfrentar desafios adicionais em termos de implementação e otimização.\n\nEm conclusão, embora os dados diretos sobre a porta OR5 não estejam disponíveis na literatura consultada, podemos concluir que ela oferece uma precisão superior à porta OR4 convencional. As portas OR5 são mais flexíveis e podem processar um maior número de entradas sem perda significativa de precisão, mas requerem atenção especial para a otimização da sua implementação.\n\nEsses resultados têm implicações importantes na área de circuitos lógicos avançados e podem ser utilizados como base para futuros estudos e desenvolvimentos nessa área.'
            },
            {
              'topic': 'Silicon Photonic Device Design', 
              'text': 'O design de dispositivos fotônicos em silício é um campo importante de pesquisa que tem como objetivo otimizar a performance desses componentes ópticos utilizando técnicas avançadas. Para abordar esta área, uma representação detalhada do processo de design pode ser incluída na forma de um diagrama esquemático. Este diagrama deve ilustrar as principais etapas e componentes envolvidos no design do dispositivo, desde a concepção inicial até o teste final. Além disso, uma fluxograma do Algoritmo de Busca Binária Direta (DBS) pode ser inserido para demonstrar como esse algoritmo é utilizado em otimização topológica para otimizar a performance do dispositivo.\n\nO uso do DBS com otimização topológica oferece várias vantagens, incluindo a capacidade de encontrar soluções ótimas ou quase-ótimas de forma eficiente. O DBS pode ser aplicado em uma ampla gama de problemas de design, desde a otimização da disposição dos componentes até o ajuste das propriedades ópticas do dispositivo. A eficácia e as vantagens desse método podem ser exploradas na discussão, destacando como ele melhora significativamente o processo de design comparado a abordagens tradicionais.'
            }
          ]
      }

      subs_outlines = [
        {
          'subsection_name': 'Results', 
          'subsection_flow': 'Throughput Performance of Optical Gate -> Accuracy Comparison Between Proposed Five-Input OR Gate and Conventional Four-Input OR Gates -> Signal Transmission Efficiency Analysis', 
          'discussion_topics': [
            {
              'rhetorical_purpose': 'Present finding', 
              'topic': 'Throughput Performance of Optical Gate', 
              'visual_elements': [], 
              'numerical_results': [], 
              'narrative_guidance': 'Begin by presenting the throughput performance of the five-input OR optical logic gate, emphasizing its accuracy rate. Mention that it achieved 95% with statistical significance at p < 0.01.'
            }, 
            {
              'rhetorical_purpose': 'Present finding', 
              'topic': 'Accuracy Comparison Between Proposed Five-Input OR Gate and Conventional Four-Input OR Gates', 
              'visual_elements': [
                {
                  'identifier': 'accuracy_comparison_graph', 
                  'name': 'Accuracy Comparison Between Proposed Five-Input OR Gate and Conventional Four-Input OR Gates', 
                  'description': 'Graph comparing the accuracy rates of both gate types under various input conditions, with error bars indicating standard deviation.', 
                  'source': 'Created for this study.', 
                  'role_in_topic': 'Illustrates the superior performance of the proposed five-input OR optical logic gate.'
                }
              ], 
              'numerical_results': [], 
              'narrative_guidance': 'Use Figure 1 to visually demonstrate the accuracy comparison. Highlight that the new design outperforms conventional four-input OR gates with statistical significance.'
            }, 
            {
              'rhetorical_purpose': 'Present finding', 
              'topic': 'Signal Transmission Efficiency Analysis', 
              'visual_elements': [
                {
                  'identifier': 'signal_transmission_efficiency_bar_chart', 
                  'name': 'Signal Transmission Efficiency Analysis', 
                  'description': 'Bar chart showing signal transmission efficiency percentages across different experimental conditions, with a clear distinction between conventional and new designs.', 
                  'source': 'Created for this study.', 
                  'role_in_topic': 'Illustrates the enhanced signal transmission efficiency of the five-input OR gate compared to traditional four-input gates.'
                }
              ], 
              'numerical_results': [
                {
                  'verbatim_value': '17%', 
                  'context_description': 'In terms of signal transmission efficiency, the new design outperformed conventional four-input OR gates by 17%.', 
                  'associated_visual': 'signal_transmission_efficiency_bar_chart'
                }
              ], 
              'narrative_guidance': 'Present the findings on signal transmission efficiency using Figure 2. Describe how the five-input OR gate outperforms traditional designs in terms of signal transmission, highlighting a 17% improvement.'
            }
          ]
        }, 
      ]
      final_edit_outputs = []
      for subsection_title, discussion_topics in section_and_topics.items():
        subsec_outline = next(
          (outline for outline in subs_outlines if outline['subsection_name'] == subsection_title), 
          None
        )
        if subsec_outline is None:
          raise ValueError(f"Subsection outline not found for {subsection_title}")

        final_edit_output = FinalEditingCrew().crew().kickoff(
          inputs={
            "subsection_title": subsection_title,
            "topics_text_json": {'topics': discussion_topics},
            "subsection_outline": subsec_outline
          }
        )
        final_edit_outputs.append(final_edit_output)
      final_edit_outputs = [final_edit_output.json_dict for final_edit_output in final_edit_outputs]
      
      print(final_edit_outputs)

    
    async def res_and_disc_chapter_generation(self):
      # Função para chamada assincrona da crew de escrita dos tópicos
      async def acall_write_topic_crew(
        section_title, 
        topic, 
        visual_elements_to_contextualize, 
        numerical_results_to_include,
        rhetorical_purpose,
        narrative_guidance,
        subsection_flow
      ):
        reasearch_crew_output = await RDResearchCrew().crew().kickoff_async(
          inputs={
            "section_title": section_title,
            "discussion_topic": topic,
            "visual_elements_to_contextualize": visual_elements_to_contextualize,
            "numerical_results_to_include": numerical_results_to_include,
            "rhetorical_purpose": rhetorical_purpose,
            "narrative_guidance": narrative_guidance,
            "subsection_flow": subsection_flow
          }
        )
        return reasearch_crew_output
      
      async def acall_final_editing_crew(
        subsection_title: str,
        topics_text_json: str,
        subsection_outline: str
      ):
        final_edit_output = await FinalEditingCrew().crew().kickoff_async(
          inputs={
            "subsection_title": subsection_title,
            "topics_text_json": topics_text_json,
            "subsection_outline": subsection_outline
          }
        )
        return final_edit_output

      #outline = json.loads(self.state.results_discussion_outline)
      subs_outlines = self.state.results_discussion_outline
      visual_elements_to_retrive = {}
      section_and_topics = {}
      for subsection in subs_outlines:
        async_tasks_to_exec = []
        for disc_topic in subsection['discussion_topics']:
          visual_elements_to_contextualize = "\n".join([
            (
              f"- name: {element['identifier']} {element['name']}; "
              f"description: {element['description']}; "
              f"role_in_topic: {element['role_in_topic']}"
            )
            for element in disc_topic.get('visual_elements', [])
          ])
          if subsection['subsection_name'] not in visual_elements_to_retrive:
            visual_elements_to_retrive[subsection['subsection_name']] = ""
      
          visual_elements_to_retrive[subsection['subsection_name']] += visual_elements_to_contextualize + "\n"
          numerical_results_to_include = "\n".join([
            (
               f'- value: {result['verbatim_value']}; '
               f'context: {result["context_description"]}; '
               f'associated visual: {result["associated_visual"]}'
            ) 
            for result in disc_topic.get('numerical_results', [])
          ])
          async_tasks_to_exec.append(
            asyncio.create_task(
              acall_write_topic_crew(
                  subsection['subsection_name'], 
                  disc_topic['topic'], 
                  visual_elements_to_contextualize, 
                  numerical_results_to_include,
                  disc_topic['rhetorical_purpose'],
                  disc_topic['narrative_guidance'],
                  subsection['subsection_flow']
              )
            )
          )
        topic_w_outputs = await asyncio.gather(*async_tasks_to_exec)
        section_and_topics[subsection['subsection_name']] = [
          topic_w_output.json_dict for topic_w_output in topic_w_outputs
        ]
      async_tasks_to_exec = []
      for subsection_title, discussion_topics in section_and_topics.items():
        subsec_outline = next(
          (outline for outline in subs_outlines if outline['subsection_name'] == subsection_title), 
          None
        )
        if subsec_outline is None:
          raise ValueError(f"Subsection outline not found for {subsection_title}")
        async_tasks_to_exec.append(
          asyncio.create_task(
            acall_final_editing_crew(
              subsection_title=subsection_title,
              topics_text_json=json.dumps({'topics': discussion_topics}, indent=2),
              subsection_outline=json.dumps(subsec_outline, indent=2)
            )
          )
        )
      final_edit_outputs = await asyncio.gather(*async_tasks_to_exec)
      final_edit_outputs = [final_edit_output.json_dict for final_edit_output in final_edit_outputs]
      
      print(final_edit_outputs)


def kickoff():
    article_flow = ArticleWriterFlow()
    article_flow.kickoff()


def plot():
    article_flow = ArticleWriterFlow()
    article_flow.plot()


if __name__ == "__main__":
    kickoff()
