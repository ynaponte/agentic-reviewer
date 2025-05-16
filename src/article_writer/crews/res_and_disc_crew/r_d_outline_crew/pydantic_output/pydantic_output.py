from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class RhetoricalPurpose(str, Enum):
    RESTATE_PROBLEM = "Restate problem"
    PRESENT_FINDING = "Present finding"
    SYNTHESIZE_FINDINGS = "Synthesize findings"
    COMPARE_LITERATURE = "Compare literature"
    DISCUSS_IMPLICATIONS = "Discuss implications"
    ADDRESS_LIMITATIONS = "Address limitations"
    PROPOSE_FUTURE_RESEARCH = "Propose future research"
    PROVIDE_PERSPECTIVE = "Provide perspective"
    FINAL_REMARK = "Final remark"

class SubSectionTitle(str, Enum):
    RESULTS = "Results"
    DISCUSSION = "Discussion"

class NumericalResult(BaseModel):
    verbatim_value: str = Field(description="Exact numerical result to be included verbatim (e.g., 'p < 0.05')")
    role_in_topic: str = Field(description="Explanation of the result's significance or context")
    associated_visual: str = Field(
        default="None",
        description="Identifier of the associated visual (e.g., 'Figure 1')"
    )

class VisualElement(BaseModel):
    name: str = Field(description="Full  name or title of the visual, including identifier")
    role_in_topic: str = Field(
        description="Role of the visual in the topic (e.g., 'summarizes data', 'illustrates trend')"
    )

class DiscussionTopic(BaseModel):
    topic: str = Field(description="Description of the topic to be discussed")
    rhetorical_purpose: RhetoricalPurpose = Field(
        description="Rhetorical purpose of the topic (e.g., 'present_finding' for Results)"
    )
    topic_description: str = Field(description="summary of the topic's focus and key points.")
    visual_elements: List[VisualElement] = Field(
        default=[],
        description="Visual elements relevant to the topic"
    )
    numerical_results: List[NumericalResult] = Field(
        default=[],
        description="Numerical results relevant to the topic"
    )
    narrative_guidance: str = Field(
        description="Guidance for paragraph structure (e.g., 'First paragraph: introduce [finding]')"
    )

class RDSubSectionOutline(BaseModel):
    subsection_name: SubSectionTitle = Field(
        description="Name of the subsection (e.g., 'Results', 'Discussion'), indicating the subsection's purpose"
    )
    subsection_flow: str = Field(
        description="Guidance on how topics connect (e.g., 'Progress from findings to implications')"
    )
    discussion_topics: List[DiscussionTopic] = Field(
        description="Structured outline of the subsection as a list of topics"
    )

class ConclusionSectionOutline(RDSubSectionOutline):
    word_count_guidance: str = Field(
        default="200-300 words per topic",
        description="Guidance for topic text length, reflecting Conclusion's concise nature"
    )
