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
    CONCLUSION = "Conclusion"

class DiscussionTopic(BaseModel):
    topic: str = Field(description="Description of the topic to be discussed")
    rhetorical_purpose: RhetoricalPurpose = Field(
        description="Rhetorical purpose of the topic (e.g., 'present_finding' for Results)"
    )
    topic_description: str = Field(description="summary of the topic's focus and key points.")
    narrative_guidance: str = Field(
        description="Guidance for paragraph structure (e.g., 'First paragraph: introduce [finding]')"
    )

class ConclusionSectionOutline(BaseModel):
    subsection_name: SubSectionTitle = Field(
        description="Name of the subsection (e.g., 'Results', 'Discussion'), indicating the subsection's purpose"
    )
    discussion_topics: List[DiscussionTopic] = Field(
        description="Structured outline of the subsection as a list of topics"
    )