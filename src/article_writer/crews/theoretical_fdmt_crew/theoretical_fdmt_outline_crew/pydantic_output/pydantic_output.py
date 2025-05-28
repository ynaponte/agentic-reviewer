from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class RhetoricalPurpose(str, Enum):
  DEFINE_KEY_CONCEPT = "Define key concept/term"
  EXPLAIN_CORE_THEORY = "Explain core theory/model"
  REVIEW_FOUNDATIONAL_LITERATURE = "Review foundational literature on a specific aspect"
  DISCUSS_HISTORICAL_CONTEXT = "Discuss historical context of a theory/concept"
  COMPARE_AND_CONTRAST = "Compare and contrast related theories/concepts"
  IDENTIFY_THEORETICAL_GAP = "Identify theoretical gap or debate"
  JUSTIFY_FRAMEWORK = "Justify the chosen theoretical framework/lens for the study"
  LINK_THEORY_TO_OBJECTIVES = "Link theory to research objectives/questions"
  OUTLINE_SCOPE_AND_RELEVANCE = "Outline theoretical scope and relevance"
  INTRODUCE_SCHOOL_OF_THOUGHT = "Introduce a school of thought"

class SectionTitle(str, Enum):
  THEORETICAL_FUNDAMENTATION = "Theoretical Fundamentarion"

class DiscussionTopic(BaseModel):
  topic_title: str = Field(description="Concise title or description of the methodological topic (e.g., 'Preparação de Amostras', 'Análise Estatística').")
  rhetorical_purpose: RhetoricalPurpose = Field(
      description="Rhetorical purpose of the methodological topic (e.g., 'Describe research design', 'Detail experimental procedure')."
  )
  topic_description: str = Field(
      description="Brief summary of the topics's focus, key parameters, or specific details relevant for replication."
  )
  narrative_guidance: str = Field(
      description="Clear and actionable guidance for the Writer Agent on how to elaborate on this method, emphasizing clarity, detail, and replicability."
  )

class TheoreticalFdmtSubSection(BaseModel):
  subsection_name: str = Field(
      description="A concise title for the subsection, being either 'Introduction' or a method's name"
  )
  subsection_description: str = Field(
      description="A brief summary outlining the key elements that will be covered within this subsection."
  )
  discussion_topics: List[DiscussionTopic] = Field(
      default_factory=list,
      description="List of topics to be discussed in the subsection."
  )


class TheoreticalFdmtSectionOutline(BaseModel):
  section_name: SectionTitle = Field(
      description="Name of the section, set to 'Methodology'."
  )
  subsections: List[TheoreticalFdmtSubSection] = Field(
      description="List of the subsections of the main section."
  )
