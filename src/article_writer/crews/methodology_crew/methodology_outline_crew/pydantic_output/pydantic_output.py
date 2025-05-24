from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class RhetoricalPurpose(str, Enum):
    DESCRIBE_RESEARCH_DESIGN = "Describe research design"
    DESCRIBE_EXPERIMENTAL_SETUP = "Describe experimental setup"
    DESCRIBE_MATERIALS_EQUIPMENT = "Describe materials and equipment"
    DESCRIBE_PROCEDURE_STEP = "Describe procedure step"
    DESCRIBE_DATA_COLLECTION = "Describe data collection"
    DESCRIBE_DATA_ANALYSIS = "Describe data analysis"
    DESCRIBE_VALIDATION_CALIBRATION = "Describe validation or calibration"
    DISCUSS_METHODOLOGICAL_LIMITATIONS = "Discuss methodological limitations"
    DISCUSS_METHODOLOGICAL_IMPLICATIONS = "Discuss methodological implications"
    PROVIDE_CONTEXTUAL_BACKGROUND = "Provide contextual background"
    EXPLAIN_RATIONALE_METHOD_CHOICE = "Explain rationale for method choice"

class SectionTitle(str, Enum):
    METHODOLOGY = "Methodology"

class DiscussionTopic(BaseModel):
    topic: str = Field(description="Concise title or description of the methodological topic (e.g., 'Preparação de Amostras', 'Análise Estatística').")
    rhetorical_purpose: RhetoricalPurpose = Field(
        description="Rhetorical purpose of the methodological topic (e.g., 'Describe research design', 'Detail experimental procedure')."
    )
    topic_description: str = Field(
        description="Brief summary of the topics's focus, key parameters, or specific details relevant for replication."
    )
    narrative_guidance: str = Field(
        description="Clear and actionable guidance for the Writer Agent on how to elaborate on this method, emphasizing clarity, detail, and replicability."
    )

class MethodologySubSection(BaseModel):
    subsection_name: str = Field(
        description="A concise title for the subsection, being either 'Introduction' or a method's name"
    )
    subsection_description: str = Field(
        description="A brief summary outlining the key elements that will be covered within this subsection."
    )
    discussion_topics: List[DiscussionTopic] = Field(
        description="List of topics to be discussed in the subsection."
    )


class MethodologySectionOutline(BaseModel):
    section_name: SectionTitle = Field(
        description="Name of the section, set to 'Methodology'."
    )
    subsections: List[MethodologySubSection] = Field(
        description="List of the subsections of the main section."
    )
