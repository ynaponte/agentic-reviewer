from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum

class RhetoricalPurpose(str, Enum):
    RESTATE_PROBLEM = "restate_problem"
    PRESENT_FINDING = "present_finding"
    SYNTHESIZE_FINDINGS = "synthesize_findings"
    COMPARE_LITERATURE = "compare_literature"
    DISCUSS_IMPLICATIONS = "discuss_implications"
    ADDRESS_LIMITATIONS = "address_limitations"
    PROPOSE_FUTURE_RESEARCH = "propose_future_research"
    PROVIDE_PERSPECTIVE = "provide_perspective"
    FINAL_REMARK = "final_remark"

class ResearchStatus(str, Enum):
    SUCCESSFUL = "successful"
    INSUFFICIENT_RESULTS = "insufficient_results"
    NOT_RESEARCHED = "not_researched"

class NumericalResult(BaseModel):
    verbatim_value: str = Field(description="Exact numerical result to be included verbatim (e.g., 'p < 0.05')")
    context_description: str = Field(description="Explanation of the result's significance or context")
    research_scope: Optional[str] = Field(
        default=None,
        description="Guidance for research query (e.g., 'Search for significance tests in [context]')"
    )
    associated_visual: Optional[str] = Field(
        default=None,
        description="Identifier of the associated visual (e.g., 'Figure 1')"
    )
    research_status: ResearchStatus = Field(
        default=ResearchStatus.NOT_RESEARCHED,
        description="Status of research for this result"
    )

class VisualElement(BaseModel):
    identifier: str = Field(description="Unique identifier (e.g., 'Figure 1', 'Table 2')")
    name: str = Field(description="Short name or title of the visual")
    description: str = Field(description="Detailed explanation of the visual's content and context")
    source: Optional[str] = Field(default=None, description="Source (e.g., 'Author, 2023')")
    role_in_topic: str = Field(
        description="Role of the visual in the topic (e.g., 'summarizes data', 'illustrates trend')"
    )
    research_status: ResearchStatus = Field(
        default=ResearchStatus.NOT_RESEARCHED,
        description="Status of research for this visual"
    )

class DiscussionTopic(BaseModel):
    rhetorical_purpose: RhetoricalPurpose = Field(
        description="Rhetorical purpose of the topic (e.g., 'present_finding' for Results)"
    )
    topic: str = Field(description="Description of the topic to be discussed")
    research_keywords: List[str] = Field(
        description="Keywords for querying the topic(e.g., '[variable] trends')",
        min_items=1
    )

    visual_elements: List[VisualElement] = Field(
        default=[],
        description="Visual elements relevant to the topic"
    )
    numerical_results: List[NumericalResult] = Field(
        default=[],
        description="Numerical results relevant to the topic"
    )
    narrative_guidance: Optional[str] = Field(
        default=None,
        description="Guidance for paragraph structure (e.g., 'First paragraph: introduce [finding]')"
    )
    research_status: ResearchStatus = Field(
        default=ResearchStatus.NOT_RESEARCHED,
        description="Status of research for this topic"
    )

class RDSubSectionOutline(BaseModel):
    subsection_name: str = Field(description="Name of the subsection (e.g., 'Results', 'Discussion', 'Conclusion')")
    subsection_flow: Optional[str] = Field(
        default=None,
        description="Guidance on how topics connect (e.g., 'Progress from findings to implications')"
    )
    discussion_topics: List[DiscussionTopic] = Field(
        description="Structured outline of the subsection as a list of topics"
    )
    numerical_results: Optional[List[NumericalResult]] = Field(
        default=[],
        description="Numerical results applicable across topics, if any"
    )

class ConclusionSectionOutline(RDSubSectionOutline):
    word_count_guidance: str = Field(
        default="200-300 words per topic",
        description="Guidance for topic text length, reflecting Conclusion's concise nature"
    )

    @field_validator('discussion_topics')
    @classmethod
    def enforce_conclusion_structure(cls, v):
        if len(v) < 5 or len(v) > 7:
            raise ValueError("Conclusion outline must have 5-7 discussion topics")
        required_purposes = [
            RhetoricalPurpose.RESTATE_PROBLEM,
            RhetoricalPurpose.SYNTHESIZE_FINDINGS,
            RhetoricalPurpose.DISCUSS_IMPLICATIONS,
            RhetoricalPurpose.PROPOSE_FUTURE_RESEARCH,
            RhetoricalPurpose.FINAL_REMARK
        ]
        purposes = [topic.rhetorical_purpose for topic in v]
        for purpose in required_purposes:
            if purpose not in purposes:
                raise ValueError(f"Conclusion outline missing required rhetorical purpose: {purpose}")
        return v