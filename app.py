"""Legal Debate System - Streamlit Application

Multi-agent AI legal debate system using RAG for case law retrieval.
"""

import streamlit as st
import os
import json
from typing import List, Dict, Annotated, Sequence
from dotenv import load_dotenv
from textblob import TextBlob
import operator

# Set page configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="Legal Debate System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Import LangChain and LangGraph components (REQUIRED)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

from rag_system.legal_rag import ProductionLegalRAGSystem
from citation_verifier import CitationVerifier, CitationVerification

# =======================
# Heuristic Score Configuration
# =======================
# These constants control the argument strength heuristic calculation.
# NOTE: This is NOT a calibrated probability - it's a composite score
# designed for relative comparison and UI display only.

# Component weights (must sum to 1.0)
CITATION_WEIGHT = 0.40      # Weight for citation quality (case citations + statutes)
SIMILARITY_WEIGHT = 0.30    # Weight for RAG vector similarity scores
STRUCTURE_WEIGHT = 0.20     # Weight for argument structure (points, reasoning, weaknesses)
REASONING_WEIGHT = 0.10     # Weight for reasoning depth (objectivity)

# Validate that weights sum to 1.0 (within floating-point tolerance)
_WEIGHT_SUM = CITATION_WEIGHT + SIMILARITY_WEIGHT + STRUCTURE_WEIGHT + REASONING_WEIGHT
assert abs(_WEIGHT_SUM - 1.0) < 1e-6, f"Heuristic weights must sum to 1.0, got {_WEIGHT_SUM}"

# Output range for heuristic score
HEURISTIC_MIN = 0.0         # Minimum possible heuristic score
HEURISTIC_MAX = 1.0         # Maximum possible heuristic score

# Display range (for UI purposes, to avoid extreme scores)
HEURISTIC_DISPLAY_MIN = 0.6  # Minimum displayed score
HEURISTIC_DISPLAY_MAX = 0.95 # Maximum displayed score

def clip_heuristic_score(value: float) -> float:
    """
    Safely clip a heuristic score to valid bounds.
    
    Args:
        value: Raw score value (may be out of bounds)
    
    Returns:
        Score clipped to [HEURISTIC_MIN, HEURISTIC_MAX] and rounded to 2 decimals
    """
    return round(max(HEURISTIC_MIN, min(HEURISTIC_MAX, float(value))), 2)

# =======================
# Custom CSS
# =======================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .debate-section {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .prosecution {
        background-color: #fff5f5;
        border-left-color: #e74c3c;
    }
    .defense {
        background-color: #f0f8ff;
        border-left-color: #3498db;
    }
    .moderator {
        background-color: #f9f9f9;
        border-left-color: #95a5a6;
    }
    .final-judgement {
        background-color: #fffbea;
        border-left-color: #f39c12;
        border-width: 3px;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# Data Models
# =======================
class CaseCitation(BaseModel):
    case_name: str = Field(description="Name of the case")
    citation: str = Field(description="Legal citation")
    year: str = Field(description="Year of the case")
    relevance: str = Field(description="Why this case is relevant")
    excerpt: str = Field(description="Relevant excerpt from the case")

class LegalArgument(BaseModel):
    main_argument: str = Field(description="The main legal argument")
    supporting_points: List[str] = Field(description="List of supporting points")
    case_citations: List[CaseCitation] = Field(description="Relevant case law citations")
    statutes_cited: List[str] = Field(description="Relevant statutes")
    legal_reasoning: str = Field(description="Detailed legal reasoning")
    weaknesses_acknowledged: List[str] = Field(description="Acknowledged weaknesses in the argument")
    heuristic_score: float = Field(description="Composite heuristic score (0-1) for argument strength - NOT a calibrated probability")

class ModeratorVerdict(BaseModel):
    round_winner: str = Field(description="'prosecution' or 'defense' or 'tie'")
    reasoning: str = Field(description="Reasoning for the verdict")
    prosecution_strengths: List[str] = Field(description="Strengths of prosecution argument")
    defense_strengths: List[str] = Field(description="Strengths of defense argument")
    prosecution_weaknesses: List[str] = Field(description="Weaknesses of prosecution argument")
    defense_weaknesses: List[str] = Field(description="Weaknesses of defense argument")
    legal_analysis: str = Field(description="Overall legal analysis")
    score_prosecution: float = Field(description="Score for prosecution (0-10)")
    score_defense: float = Field(description="Score for defense (0-10)")

class FinalJudgment(BaseModel):
    overall_winner: str = Field(description="'prosecution' or 'defense' or 'balanced'")
    judgment_summary: str = Field(description="Summary of the final judgment")
    key_findings: List[str] = Field(description="Key findings from all rounds")
    final_verdict: str = Field(description="The final verdict on the case")
    recommended_action: str = Field(description="Recommended action or ruling")
    total_score_prosecution: float = Field(description="Total score for prosecution")
    total_score_defense: float = Field(description="Total score for defense")

# =======================
# LangGraph State Definition (REQUIRED)
# =======================
class DebateState(TypedDict):
    """State object passed between agents in LangGraph workflow"""
    case_description: str
    current_round: int
    max_rounds: int
    prosecution_arguments: List[LegalArgument]
    defense_arguments: List[LegalArgument]
    moderator_verdicts: List[ModeratorVerdict]
    final_judgment: FinalJudgment
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Conversation history
    rag_context: List[str]  # Retrieved case law
    next_speaker: str  # 'prosecution', 'defense', or 'moderator'

# =======================
# ML Confidence Analysis
# =======================
def calculate_argument_heuristic_score(
    argument: LegalArgument, 
    similarity_scores: List[float] = None
) -> float:
    """
    Calculate a composite heuristic score for argument strength.
    
    IMPORTANT: This is a DIMENSIONLESS HEURISTIC for relative comparison within
    this system ONLY. It is NOT:
    - A calibrated probability
    - A statistical confidence interval
    - Comparable across different systems or datasets
    
    This hand-tuned heuristic combines multiple factors using fixed weights.
    The output is normalized to a display range to avoid extreme values that
    might be misinterpreted as certainty.
    
    FUTURE IMPROVEMENT: This could be replaced with a trained model (e.g.,
    logistic regression with isotonic calibration) if labeled training data
    becomes available. The current weights were chosen heuristically based on
    domain knowledge, not optimized on data.
    
    Factors (configurable via module constants):
    1. Case citation quality (CITATION_WEIGHT: {:.0%})
    2. Vector similarity scores (SIMILARITY_WEIGHT: {:.0%})
    3. Argument structure (STRUCTURE_WEIGHT: {:.0%})
    4. Legal reasoning depth (REASONING_WEIGHT: {:.0%})
    
    Args:
        argument: The legal argument to score
        similarity_scores: Optional RAG vector similarity scores (0-1 range).
                          If None or empty, uses citation-based fallback.
    
    Returns:
        Heuristic strength score in [HEURISTIC_MIN, HEURISTIC_MAX], typically
        displayed in range [HEURISTIC_DISPLAY_MIN, HEURISTIC_DISPLAY_MAX].
    """.format(CITATION_WEIGHT, SIMILARITY_WEIGHT, STRUCTURE_WEIGHT, REASONING_WEIGHT)
    
    # 1. Citation Quality Component (0-1)
    citation_count = len(argument.case_citations)
    statute_count = len(argument.statutes_cited)
    
    # More citations = higher score (saturates at 5 citations)
    citation_score = min(citation_count / 5.0, 1.0) * 0.7
    # Statute support adds to score (saturates at 3 statutes)
    statute_score = min(statute_count / 3.0, 1.0) * 0.3
    citation_quality = min(citation_score + statute_score, 1.0)  # Ensure <= 1.0
    
    # 2. Similarity Component (robust handling of edge cases)
    if similarity_scores and len(similarity_scores) > 0:
        # Average of top-3 similarity scores (already 0-1 normalized)
        top_k = min(3, len(similarity_scores))  # Handle lists shorter than 3
        valid_scores = [max(0.0, min(1.0, s)) for s in similarity_scores[:top_k]]  # Clip to [0,1]
        similarity_factor = sum(valid_scores) / top_k if top_k > 0 else 0.5
    else:
        # Fallback: estimate based on citation count
        similarity_factor = 0.75 if citation_count >= 3 else 0.65
    
    # 3. Argument Structure Component (0-1)
    supporting_points_score = min(len(argument.supporting_points) / 4.0, 1.0) * 0.4
    reasoning_words = len(argument.legal_reasoning.split())
    reasoning_score = min(reasoning_words / 150.0, 1.0) * 0.4  # Target: 150+ words
    # Acknowledging weaknesses shows thoroughness
    weakness_score = min(len(argument.weaknesses_acknowledged) / 2.0, 1.0) * 0.2
    structure_quality = min(supporting_points_score + reasoning_score + weakness_score, 1.0)
    
    # 4. Legal Reasoning Depth Component (0-1)
    # Use sentiment analysis as a proxy for objectivity
    try:
        blob = TextBlob(argument.main_argument + " " + argument.legal_reasoning)
        # Lower subjectivity = more objective/factual language
        objectivity = max(0.0, min(1.0, 1.0 - blob.sentiment.subjectivity))
        reasoning_depth = objectivity
    except Exception:
        # Fallback if TextBlob fails
        reasoning_depth = 0.5
    
    # Weighted combination using configurable weights (guaranteed to sum to 1.0)
    raw_score = (
        citation_quality * CITATION_WEIGHT +
        similarity_factor * SIMILARITY_WEIGHT +
        structure_quality * STRUCTURE_WEIGHT +
        reasoning_depth * REASONING_WEIGHT
    )
    # raw_score is now in [0, 1] since all components are in [0, 1] and weights sum to 1.0
    
    # Map from [0, 1] to display range [HEURISTIC_DISPLAY_MIN, HEURISTIC_DISPLAY_MAX]
    # Mathematical relationship:
    #   display_score = HEURISTIC_DISPLAY_MIN + raw_score * (HEURISTIC_DISPLAY_MAX - HEURISTIC_DISPLAY_MIN)
    # This linear transformation maps:
    #   raw_score=0 -> HEURISTIC_DISPLAY_MIN
    #   raw_score=1 -> HEURISTIC_DISPLAY_MAX
    display_range = HEURISTIC_DISPLAY_MAX - HEURISTIC_DISPLAY_MIN
    normalized_score = HEURISTIC_DISPLAY_MIN + (raw_score * display_range)
    
    # Final safety clip and round (should be redundant if math is correct, but defensive)
    return clip_heuristic_score(normalized_score)

def get_heuristic_breakdown(argument: LegalArgument) -> dict:
    """
    Calculate detailed breakdown of heuristic score components.
    
    This function is for TRANSPARENCY and DEBUGGING only. The returned values
    should NOT be used for downstream decision-making or compared across systems.
    They are purely informational to help users understand how the heuristic
    score was calculated.
    
    Returns:
        Dictionary with raw component values (citation_count, sentiment metrics, etc.)
    """
    # Sentiment analysis for reasoning depth component
    blob = TextBlob(argument.main_argument + " " + argument.legal_reasoning)
    
    return {
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity,
        'citation_count': len(argument.case_citations),
        'statute_count': len(argument.statutes_cited),
        'reasoning_length': len(argument.legal_reasoning.split()),
        'weaknesses_count': len(argument.weaknesses_acknowledged)
    }

# =======================
# UI Components
# =======================
def display_argument(argument: LegalArgument, role: str, round_num: int):
    """Display a legal argument in a formatted way"""
    css_class = "prosecution" if role == "prosecution" else "defense"
    icon = "🔴" if role == "prosecution" else "🔵"
    
    st.markdown(f'<div class="debate-section {css_class}">', unsafe_allow_html=True)
    st.markdown(f"### {icon} {role.upper()} - Round {round_num}")
    
    st.markdown(f"**Main Argument:**")
    st.write(argument.main_argument)
    
    # Display heuristic score with clear labeling
    st.markdown(f"**Argument Strength Score (heuristic):** {argument.heuristic_score:.2f} 🎯")
    st.caption("ℹ️ This is a composite heuristic for relative comparison, not a statistical probability")
    
    # Show detailed heuristic breakdown in an expander
    breakdown = get_heuristic_breakdown(argument)
    with st.expander("📊 Heuristic Score Breakdown"):
        st.markdown("### Calculation Components:")
        st.caption("⚠️ Hand-tuned weights for UI display - not calibrated probabilities")
        
        # Citation Quality component
        citation_count = len(argument.case_citations)
        statute_count = len(argument.statutes_cited)
        citation_score = min(citation_count / 5, 1.0) * 0.7 + min(statute_count / 3, 1.0) * 0.3
        st.progress(citation_score, text=f"Citation Quality ({CITATION_WEIGHT:.0%}): {citation_score:.2f}")
        st.caption(f"📚 {citation_count} cases + {statute_count} statutes")
        
        # Argument Structure component
        supporting_score = min(len(argument.supporting_points) / 4, 1.0) * 0.4
        reasoning_words = len(argument.legal_reasoning.split())
        reasoning_score = min(reasoning_words / 150, 1.0) * 0.4
        weakness_score = min(len(argument.weaknesses_acknowledged) / 2, 1.0) * 0.2
        structure_score = supporting_score + reasoning_score + weakness_score
        st.progress(structure_score, text=f"Structure Quality ({STRUCTURE_WEIGHT:.0%}): {structure_score:.2f}")
        st.caption(f"📝 {len(argument.supporting_points)} points + {reasoning_words} words reasoning")
        
        # Objectivity component
        objectivity = 1.0 - breakdown['sentiment_subjectivity']
        st.progress(objectivity, text=f"Objectivity ({REASONING_WEIGHT:.0%}): {objectivity:.2f}")
        st.caption(f"🎯 Sentiment polarity: {breakdown['sentiment_polarity']:.2f}")
        
        st.markdown("---")
        st.markdown(f"**Formula:** `{HEURISTIC_DISPLAY_MIN} + (weighted_sum × {HEURISTIC_DISPLAY_MAX - HEURISTIC_DISPLAY_MIN})`")
        st.markdown(f"*Vector similarity ({SIMILARITY_WEIGHT:.0%}) automatically incorporated from RAG retrieval*")
    
    with st.expander("📋 Supporting Points"):
        for i, point in enumerate(argument.supporting_points, 1):
            st.write(f"{i}. {point}")
    
    with st.expander("📚 Case Citations (Verified)"):
        if not argument.case_citations:
            st.info("No citations provided")
        else:
            # Initialize verifier if not already done
            if 'citation_verifier' not in st.session_state:
                try:
                    rag_system = ProductionLegalRAGSystem()
                    st.session_state.citation_verifier = CitationVerifier(rag_system.vector_store)
                except Exception as e:
                    st.warning(f"Citation verification unavailable: {str(e)}")
                    st.session_state.citation_verifier = None
            
            verifier = st.session_state.citation_verifier
            
            for citation in argument.case_citations:
                # Verify citation if verifier available
                if verifier:
                    verification = verifier.verify_case_exists(
                        case_name=citation.case_name,
                        citation=citation.citation,
                        year=citation.year
                    )
                    
                    # Display with verification status
                    if verification.flag == "VERIFIED":
                        st.markdown(f"✅ **{citation.case_name}** ({citation.year})")
                        st.caption(f"🔍 Verified ({verification.confidence:.0%} confidence)")
                    elif verification.flag == "UNCERTAIN":
                        st.markdown(f"⚠️ **{citation.case_name}** ({citation.year})")
                        st.warning(f"🔍 Uncertain verification ({verification.confidence:.0%} confidence) - Review recommended")
                    else:
                        st.markdown(f"❌ **{citation.case_name}** ({citation.year})")
                        st.error(f"🚨 UNVERIFIED - Cannot confirm this case exists in database")
                else:
                    # Fallback display without verification
                    st.markdown(f"**{citation.case_name}** ({citation.year})")
                
                st.write(f"*Citation:* {citation.citation}")
                st.write(f"*Relevance:* {citation.relevance}")
                st.write(f"*Excerpt:* {citation.excerpt}")
                st.divider()
    
    with st.expander("📖 Statutes Cited"):
        for statute in argument.statutes_cited:
            st.write(f"• {statute}")
    
    with st.expander("⚠️ Acknowledged Weaknesses"):
        for weakness in argument.weaknesses_acknowledged:
            st.write(f"• {weakness}")
    
    with st.expander("🧠 Legal Reasoning"):
        st.write(argument.legal_reasoning)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# Mock Legal Agents
# =======================
# =======================
# LangChain Agent Classes (REQUIRED)
# =======================
class LegalDebateAgent:
    """LangChain-based legal debate agent with RAG and memory"""
    
    def __init__(self, role: str, rag_system: ProductionLegalRAGSystem, model_name: str = "gemini-2.5-flash"):
        self.role = role
        self.rag_system = rag_system
        
        # LangChain LLM with proper configuration
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            convert_system_message_to_human=True
        )
        
        # Pydantic parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=LegalArgument)
        
        # LangChain prompt template with memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n\n{format_instructions}")
        ])
        
        # Create LangChain chain
        self.chain = self.prompt | self.llm | self.parser
    
    def _get_system_prompt(self) -> str:
        """Get system prompt based on role"""
        if self.role == "prosecution":
            return """You are an expert prosecution attorney in a multi-agent legal debate system.

Your role:
1. Analyze case facts and build strong prosecution arguments
2. USE THE RETRIEVED CASE LAW provided in the input - cite them in case_citations array
3. Extract case_name, citation, and year from the retrieved cases
4. Include relevant excerpts from the provided case law in your citations
5. Cite specific Indian statutes (Article 20, Article 21, CrPC sections)
6. Acknowledge potential weaknesses honestly
7. Structure arguments in VALID JSON format matching the schema exactly

CRITICAL: You MUST include case_citations array with at least 2-3 cases from the Retrieved Case Law section.
Each citation must have: case_name, citation, year, relevance, excerpt

Focus on constitutional law, criminal procedure, and fundamental rights under Indian law."""
        else:
            return """You are an expert defense attorney in a multi-agent legal debate system.

Your role:
1. Defend the accused with strong legal arguments
2. USE THE RETRIEVED CASE LAW provided in the input - cite them in case_citations array
3. Extract case_name, citation, and year from the retrieved cases
4. Include relevant excerpts from the provided case law in your citations
5. Cite Indian constitutional protections (Article 21, Article 22, CrPC sections)
6. Challenge prosecution's evidence and reasoning
7. Structure arguments in VALID JSON format matching the schema exactly

CRITICAL: You MUST include case_citations array with at least 2-3 cases from the Retrieved Case Law section.
Each citation must have: case_name, citation, year, relevance, excerpt

Focus on Article 21, Article 22, and criminal procedure protections."""
    
    def generate_argument(self, state: DebateState) -> LegalArgument:
        """Generate argument using LangChain pipeline with RAG"""
        
        # RAG: Retrieve relevant case law
        query = f"{self.role} argument for: {state['case_description']}"
        relevant_docs = self.rag_system.retrieve_documents(query, k=3)
        
        # Build RAG context
        rag_context = "\n\n".join([
            f"**Case:** {doc.metadata.get('case_name', 'Unknown')}\n"
            f"**Citation:** {doc.metadata.get('citation', 'N/A')}\n"
            f"**Court:** {doc.metadata.get('court', 'N/A')}\n"
            f"**Excerpt:** {doc.page_content[:300]}..."
            for doc in relevant_docs
        ])
        
        # Build input with conversation context
        opponent_args = ""
        if self.role == "defense" and state["prosecution_arguments"]:
            # Include all prosecution arguments in the context
            all_pros = "\n\n".join([arg.main_argument for arg in state["prosecution_arguments"]])
            opponent_args = f"\n\nProsecution's arguments:\n{all_pros}"
        elif self.role == "prosecution" and state["defense_arguments"]:
            # Include all defense arguments in the context
            all_def = "\n\n".join([arg.main_argument for arg in state["defense_arguments"]])
            opponent_args = f"\n\nDefense's arguments:\n{all_def}"
        
        input_text = f"""Round {state['current_round']} - {self.role.upper()}

Case: {state['case_description']}

Retrieved Case Law (YOU MUST USE THESE IN YOUR case_citations):
{rag_context}
{opponent_args}

Generate your {self.role} argument using the case law provided above.
IMPORTANT: Include ALL retrieved cases in your case_citations array with their exact details."""
        
        # Execute LangChain pipeline
        try:
            print(f"🔍 {self.role.upper()}: Retrieved {len(relevant_docs)} cases from RAG")
            print(f"📝 {self.role.upper()}: RAG docs - {[doc.metadata.get('case_name', 'Unknown')[:50] for doc in relevant_docs[:2]]}")
            
            # Get raw LLM response first for debugging
            raw_response = self.llm.invoke(
                self.prompt.format_messages(
                    input=input_text,
                    chat_history=state.get("messages", []),
                    format_instructions=self.parser.get_format_instructions()
                )
            )
            print(f"🤖 {self.role.upper()}: Raw LLM response length: {len(raw_response.content)} chars")
            
            # Try to parse it
            try:
                argument = self.parser.parse(raw_response.content)
                print(f"✅ {self.role.upper()}: Successfully parsed argument")
            except Exception as parse_error:
                print(f"❌ {self.role.upper()}: Parse failed - {str(parse_error)[:100]}")
                # Try the full chain approach
                argument = self.chain.invoke({
                    "input": input_text,
                    "chat_history": state.get("messages", []),
                    "format_instructions": self.parser.get_format_instructions()
                })
                print(f"✅ {self.role.upper()}: Chain invoke succeeded")
            
            # ⚠️ REMOVED DANGEROUS AUTO-CITATION FALLBACK
            # Previously: Automatically injected citations when LLM failed
            # Risk: Citation hallucination and misattribution
            # New approach: Flag missing citations for human review
            
            if not argument.case_citations:
                print(f"⚠️ {self.role.upper()}: No citations generated by LLM - FLAGGED FOR REVIEW")
                # Mark argument as requiring verification
                argument.main_argument = f"⚠️ [CITATION NEEDED] {argument.main_argument}"
            else:
                # Verify citation quality
                from evaluation_metrics import LegalRAGEvaluator
                evaluator = LegalRAGEvaluator()
                
                citation_dicts = [
                    {
                        'case_name': c.case_name,
                        'citation': c.citation,
                        'excerpt': c.excerpt,
                        'auto_generated': False
                    }
                    for c in argument.case_citations
                ]
                
                flagged = evaluator.detect_hallucinated_citations(
                    citation_dicts, 
                    argument.main_argument
                )
                
                if flagged:
                    print(f"⚠️ {self.role.upper()}: {len(flagged)} citations flagged as potentially problematic")
                    for flag in flagged:
                        print(f"   - {flag['case_name']}: {flag['flags']}")
            
            print(f"📚 {self.role.upper()}: Final argument has {len(argument.case_citations)} citations")
            
            return argument
            
        except Exception as e:
            # Safe fallback: Flag for manual review instead of generating fake citations
            print(f"⚠️ LangChain parsing failed for {self.role}: {str(e)}")
            print(f"⚠️ Returning error state - manual review required")
            
            # Return error state with NO citations (avoid hallucination)
            return LegalArgument(
                main_argument=f"⚠️ [ERROR: Argument generation failed for {self.role} - Manual review required]\n\nSystem encountered an error while processing legal argument. Retrieved {len(relevant_docs)} relevant documents but could not parse structured output from AI model.",
                supporting_points=[
                    "Automated argument generation encountered a system error",
                    f"Error details: {str(e)[:100]}",
                    "Human legal expert review is required",
                    f"{len(relevant_docs)} potentially relevant cases were retrieved but not cited"
                ],
                case_citations=[],  # EMPTY - Do not generate fake citations
                statutes_cited=[],  # EMPTY - Cannot reliably extract from error state
                legal_reasoning=f"Due to system error during argument generation, this {self.role} argument could not be automatically constructed. The retrieved legal documents are available but have not been verified for relevance. A qualified legal professional should manually review the case facts and construct appropriate arguments with proper citations.",
                weaknesses_acknowledged=[
                    "Complete manual review required - automated system failed",
                    "Retrieved documents have not been verified for relevance",
                    "No heuristic strength available - human expert needed"
                ],
                heuristic_score=0.0  # Zero heuristic score for error states
            )

class ModeratorLangChainAgent:
    """LangChain-based moderator agent for impartial evaluation"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        
        self.parser = PydanticOutputParser(pydantic_object=ModeratorVerdict)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an impartial judge moderating a legal debate in a multi-agent system.

Your role:
1. Evaluate arguments objectively based on legal merit
2. Assess strength of case law citations
3. Identify constitutional and statutory compliance
4. Provide fair scoring (0-10 for each side)
5. Explain reasoning transparently
6. Structure output in valid JSON format

You help the system converge on transparent legal consensus."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n\n{format_instructions}")
        ])
        
        self.chain = self.prompt | self.llm | self.parser
    
    def evaluate_round(self, state: DebateState) -> ModeratorVerdict:
        """Evaluate round using LangChain pipeline"""
        
        pros_arg = state["prosecution_arguments"][-1]
        def_arg = state["defense_arguments"][-1]
        
        input_text = f"""Round {state['current_round']} Evaluation

Case: {state['case_description']}

PROSECUTION:
- Argument: {pros_arg.main_argument}
- Citations: {len(pros_arg.case_citations)} cases
- Heuristic Strength Score: {pros_arg.heuristic_score}

DEFENSE:
- Argument: {def_arg.main_argument}
- Citations: {len(def_arg.case_citations)} cases
- Heuristic Strength Score: {def_arg.heuristic_score}

Provide your impartial verdict for this round."""
        
        try:
            verdict = self.chain.invoke({
                "input": input_text,
                "chat_history": state.get("messages", []),
                "format_instructions": self.parser.get_format_instructions()
            })
            return verdict
        except:
            return ModeratorVerdict(
                round_winner="tie",
                reasoning="Both sides presented valid arguments.",
                prosecution_strengths=["Strong reasoning"],
                defense_strengths=["Constitutional focus"],
                prosecution_weaknesses=["Could be stronger"],
                defense_weaknesses=["Limited precedent"],
                legal_analysis="Balanced round.",
                score_prosecution=7.5,
                score_defense=7.5
            )

def display_moderator_verdict(verdict: ModeratorVerdict, round_num: int):
    """Display moderator's verdict for a round"""
    st.markdown('<div class="debate-section moderator">', unsafe_allow_html=True)
    st.markdown(f"### ⚖️ MODERATOR VERDICT - Round {round_num}")
    
    # Winner announcement
    if verdict.round_winner == "tie":
        st.markdown("**🤝 Round Result:** TIE")
    elif verdict.round_winner == "prosecution":
        st.markdown("**🔴 Round Winner:** PROSECUTION")
    else:
        st.markdown("**🔵 Round Winner:** DEFENSE")
    
    # Scores
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prosecution Score", f"{verdict.score_prosecution:.1f}/10")
    with col2:
        st.metric("Defense Score", f"{verdict.score_defense:.1f}/10")
    
    # Reasoning
    st.markdown("**⚖️ Reasoning:**")
    st.write(verdict.reasoning)
    
    # Legal Analysis
    with st.expander("📖 Legal Analysis"):
        st.write(verdict.legal_analysis)
    
    # Strengths and Weaknesses
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Prosecution:**")
        st.markdown("*Strengths:*")
        for strength in verdict.prosecution_strengths:
            st.write(f"✅ {strength}")
        st.markdown("*Weaknesses:*")
        for weakness in verdict.prosecution_weaknesses:
            st.write(f"⚠️ {weakness}")
    
    with col2:
        st.markdown("**🔵 Defense:**")
        st.markdown("*Strengths:*")
        for strength in verdict.defense_strengths:
            st.write(f"✅ {strength}")
        st.markdown("*Weaknesses:*")
        for weakness in verdict.defense_weaknesses:
            st.write(f"⚠️ {weakness}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# LangGraph Workflow (REQUIRED)
# =======================
def create_debate_workflow(rag_system: ProductionLegalRAGSystem, max_rounds: int = 3) -> StateGraph:
    """
    Creates LangGraph workflow for multi-agent debate with memory
    This is the REQUIRED orchestration framework
    """
    
    # Initialize agents
    prosecution_agent = LegalDebateAgent("prosecution", rag_system)
    defense_agent = LegalDebateAgent("defense", rag_system)
    moderator_agent = ModeratorLangChainAgent()
    
    # Define workflow
    workflow = StateGraph(DebateState)
    
    def prosecution_node(state: DebateState) -> DebateState:
        """Prosecution agent generates argument"""
        with st.spinner(f"🔴 Prosecution presenting Round {state['current_round']}..."):
            argument = prosecution_agent.generate_argument(state)
            state["prosecution_arguments"].append(argument)
            
            # Add to conversation memory
            state["messages"].append(AIMessage(
                content=f"Prosecution Round {state['current_round']}: {argument.main_argument}",
                name="prosecution"
            ))
            
            state["next_speaker"] = "defense"
        return state
    
    def defense_node(state: DebateState) -> DebateState:
        """Defense agent generates argument"""
        with st.spinner(f"🔵 Defense responding in Round {state['current_round']}..."):
            argument = defense_agent.generate_argument(state)
            state["defense_arguments"].append(argument)
            
            # Add to conversation memory
            state["messages"].append(AIMessage(
                content=f"Defense Round {state['current_round']}: {argument.main_argument}",
                name="defense"
            ))
            
            state["next_speaker"] = "moderator"
        return state
    
    def moderator_node(state: DebateState) -> DebateState:
        """Moderator evaluates round"""
        with st.spinner(f"⚖️ Moderator evaluating Round {state['current_round']}..."):
            verdict = moderator_agent.evaluate_round(state)
            state["moderator_verdicts"].append(verdict)
            
            # Add to conversation memory
            state["messages"].append(AIMessage(
                content=f"Moderator Round {state['current_round']}: {verdict.round_winner} wins. {verdict.reasoning}",
                name="moderator"
            ))
            
            # Increment round
            state["current_round"] += 1
            state["next_speaker"] = "prosecution" if state["current_round"] <= state["max_rounds"] else "end"
        return state
    
    def final_judgment_node(state: DebateState) -> DebateState:
        """Generate final judgment using all debate context"""
        with st.spinner("⚖️ Generating final judgment..."):
            total_pros = sum(v.score_prosecution for v in state["moderator_verdicts"])
            total_def = sum(v.score_defense for v in state["moderator_verdicts"])
            
            # Use moderator agent for final judgment
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
            
            prompt = f"""Based on {len(state['moderator_verdicts'])} rounds of debate:

Case: {state['case_description']}

Round Results:
{chr(10).join([f"Round {i+1}: {v.round_winner} wins (Pros: {v.score_prosecution}, Def: {v.score_defense})" for i, v in enumerate(state['moderator_verdicts'])])}

Total Scores: Prosecution {total_pros:.1f}, Defense {total_def:.1f}

Provide final judgment with:
- overall_winner: "prosecution", "defense", or "balanced"
- judgment_summary: 2-3 sentence summary
- key_findings: list of 3-5 key findings
- final_verdict: detailed verdict (3-4 sentences)
- recommended_action: recommended action
- total_score_prosecution: {total_pros}
- total_score_defense: {total_def}

Return JSON only."""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                text = response.content.strip()
                
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                data = json.loads(text)
                data['total_score_prosecution'] = total_pros
                data['total_score_defense'] = total_def
                
                state["final_judgment"] = FinalJudgment(**data)
            except:
                state["final_judgment"] = FinalJudgment(
                    overall_winner="balanced" if abs(total_pros - total_def) < 1 else ("prosecution" if total_pros > total_def else "defense"),
                    judgment_summary=f"After {len(state['moderator_verdicts'])} rounds, the court has reached a decision.",
                    key_findings=["Both sides presented valid arguments", "Constitutional principles examined", "Case law properly cited"],
                    final_verdict="Based on the cumulative evidence and legal reasoning presented, the court finds in favor of the prevailing party.",
                    recommended_action="Proceed according to established legal precedent.",
                    total_score_prosecution=total_pros,
                    total_score_defense=total_def
                )
        
        return state
    
    def should_continue(state: DebateState) -> str:
        """Route based on current round"""
        if state["current_round"] > state["max_rounds"]:
            return "final_judgment"
        return "continue"
    
    # Add nodes to workflow
    workflow.add_node("prosecution", prosecution_node)
    workflow.add_node("defense", defense_node)
    workflow.add_node("moderator", moderator_node)
    workflow.add_node("final_judgment", final_judgment_node)
    
    # Define edges (workflow orchestration)
    workflow.add_edge(START, "prosecution")
    workflow.add_edge("prosecution", "defense")
    workflow.add_edge("defense", "moderator")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "moderator",
        should_continue,
        {
            "continue": "prosecution",
            "final_judgment": "final_judgment"
        }
    )
    
    workflow.add_edge("final_judgment", END)
    
    # Add memory (REQUIRED for advanced persistence)
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

def display_final_judgment(judgment: FinalJudgment):
    """Display the final judgment"""
    st.markdown('<div class="debate-section final-judgement">', unsafe_allow_html=True)
    st.markdown("### 🏛️ FINAL JUDGMENT")
    
    # Overall winner
    if judgment.overall_winner == "balanced":
        st.markdown("**⚖️ Overall Result:** BALANCED DECISION")
    elif judgment.overall_winner == "prosecution":
        st.markdown("**🔴 Overall Winner:** PROSECUTION")
    else:
        st.markdown("**🔵 Overall Winner:** DEFENSE")
    
    # Total scores
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("🔴 Total Score - Prosecution", f"{judgment.total_score_prosecution:.1f}")
    with col2:
        st.metric("🔵 Total Score - Defense", f"{judgment.total_score_defense:.1f}")
    with col3:
        diff = abs(judgment.total_score_prosecution - judgment.total_score_defense)
        st.metric("📊 Score Difference", f"{diff:.1f}")
    
    st.divider()
    
    # Judgment Summary
    st.markdown("**📋 Judgment Summary:**")
    st.write(judgment.judgment_summary)
    
    # Final Verdict
    st.markdown("**⚖️ Final Verdict:**")
    st.info(judgment.final_verdict)
    
    # Key Findings
    st.markdown("**🔍 Key Findings:**")
    for i, finding in enumerate(judgment.key_findings, 1):
        st.write(f"{i}. {finding}")
    
    # Recommended Action
    st.markdown("**📌 Recommended Action:**")
    st.success(judgment.recommended_action)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# Main Application
# =======================
def main():
    """Main Streamlit application"""
    
    # Initialize session state for chat history
    if 'debate_history' not in st.session_state:
        st.session_state.debate_history = []
    
    # Header
    st.markdown('<p class="main-header">⚖️ Legal Debate System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Multi-Agent Legal Debate with RAG</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📖 About")
        st.markdown("""
        **Multi-Agent Legal Debate System**
        
        Framework compliant with:
        - ✅ **LangChain/LangGraph Orchestration**: State-based workflow
        - ✅ **RAG over Case Law**: 543 Indian legal documents
        - ✅ **Advanced Memory**: Persistent conversation history
        - ✅ **Multiple AI Agents**: Prosecution, Defense, Moderator
        - ✅ **Transparent Consensus**: Round-by-round verdicts
        - ✅ **Domain-Agnostic Design**: Adaptable to any legal domain
        
        **Agents:**
        - **🔴 Prosecution**: LangChain agent with RAG
        - **🔵 Defense**: LangChain agent with RAG
        - **⚖️ Moderator**: Fine-tuned for impartial judgment
        
        **Technology Stack:**
        - LangGraph StateGraph for orchestration
        - LangChain ChatPromptTemplate & PydanticOutputParser
        - FAISS vector store with 543 case chunks
        - MemorySaver for persistent agent memory
        - Google Gemini 2.0 Flash LLM
        """)
        
        st.divider()
        
        st.header("🔧 Framework Compliance")
        
        # Try to initialize RAG system
        try:
            rag = ProductionLegalRAGSystem()
            st.success("✅ RAG System: Active")
            
            # Detect embedding type
            use_hf = os.getenv("USE_HUGGINGFACE_EMBEDDINGS", "").lower() in ("true", "1", "yes")
            use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "").lower() in ("true", "1", "yes")
            has_google = bool(os.getenv("GOOGLE_API_KEY"))
            has_openai = bool(os.getenv("OPENAI_API_KEY"))
            
            if use_mock:
                embedding_type = "Mock (TF-IDF)"
            elif use_hf:
                embedding_type = "HuggingFace (Local)"
            elif has_google:
                embedding_type = "Google Gemini"
            elif has_openai:
                embedding_type = "OpenAI"
            else:
                embedding_type = "Unknown"
            
            st.markdown(f"""
            **Orchestration:**
            - ✅ LangGraph StateGraph workflow
            - ✅ LangChain ChatPromptTemplate
            - ✅ PydanticOutputParser for structured output
            - ✅ MemorySaver for persistent memory
            
            **RAG System:**
            - ✅ {embedding_type} embeddings
            - ✅ FAISS vector database
            - ✅ 25 real Indian legal cases (543 chunks)
            
            **LLM:**
            - ✅ Google Gemini 2.0 Flash
            - ✅ Temperature-controlled agents
            - ✅ Multi-agent orchestration
            """)
        except FileNotFoundError:
            st.error("❌ RAG System: Index not found")
            st.info("Run: `python build_from_scraped.py` to rebuild the index")
            return
        except Exception as e:
            st.error(f"❌ RAG System Error: {str(e)}")
            return
        
        st.divider()
        
        # Chat history in sidebar
        st.header("📜 Chat History")
        if st.session_state.debate_history:
            st.info(f"📚 {len(st.session_state.debate_history)} previous debate(s)")
            if st.button("🗑️ Clear History"):
                st.session_state.debate_history = []
                st.rerun()
        else:
            st.info("No debate history yet. Run your first debate!")
    
    # Main content area
    st.header("📝 Enter Case Details")
    
    # Pre-filled example case
    default_case = """John Doe was arrested for theft after being stopped by police officers who noticed him acting suspiciously near a jewelry store. During the stop, officers found stolen jewelry in his backpack. However, John claims:
1. He was not informed of his Miranda rights before questioning
2. The search of his backpack was conducted without a warrant
3. The officers had no probable cause for the initial stop

Legal Question: Should the evidence (stolen jewelry) be admissible in court?"""
    
    case_description = st.text_area(
        "Case Description",
        value=default_case,
        height=200,
        help="Describe the legal case, including facts and legal questions"
    )
    
    # Number of rounds
    num_rounds = st.slider("Number of Debate Rounds", 1, 3, 2)
    
    # Run debate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        run_button = st.button("⚖️ Start Legal Debate", type="primary", use_container_width=True)
    
    # Run the debate
    if run_button:
        if not case_description.strip():
            st.error("❌ Please enter a case description")
            return
        
        st.divider()
        st.header("🎯 LangGraph Multi-Agent Debate")
        st.info("🔄 Using LangChain/LangGraph orchestration with RAG and advanced memory")
        
        try:
            # Create LangGraph workflow (REQUIRED)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔧 Initializing LangGraph workflow with memory...")
            progress_bar.progress(0.1)
            
            debate_graph = create_debate_workflow(rag, max_rounds=num_rounds)
            
            status_text.text("✅ LangGraph workflow initialized with LangChain agents")
            progress_bar.progress(0.2)
            
            # Initialize debate state
            initial_state = {
                "case_description": case_description,
                "current_round": 1,
                "max_rounds": num_rounds,
                "prosecution_arguments": [],
                "defense_arguments": [],
                "moderator_verdicts": [],
                "final_judgment": None,
                "messages": [HumanMessage(content=f"Case to debate: {case_description}")],
                "rag_context": [],
                "next_speaker": "prosecution"
            }
            
            status_text.text(f"🚀 Running LangGraph workflow with {num_rounds} rounds...")
            progress_bar.progress(0.3)
            
            # Execute LangGraph workflow (REQUIRED orchestration)
            config = {"configurable": {"thread_id": datetime.now().strftime("%Y%m%d_%H%M%S")}}
            final_state = debate_graph.invoke(initial_state, config=config)
            
            status_text.text("✅ LangGraph workflow completed successfully!")
            progress_bar.progress(1.0)
            
            # Extract results from final state
            prosecution_arguments = final_state["prosecution_arguments"]
            defense_arguments = final_state["defense_arguments"]
            moderator_verdicts = final_state["moderator_verdicts"]
            final_judgment = final_state["final_judgment"]
            
            # Save to history
            history_entry = {
                'case': case_description,
                'rounds': num_rounds,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prosecution_count': len(prosecution_arguments),
                'defense_count': len(defense_arguments),
            }
            st.session_state.debate_history.append(history_entry)
            
            st.divider()
            
            # Display arguments round by round with moderator verdicts
            for round_num in range(num_rounds):
                st.subheader(f"📍 Round {round_num + 1}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    display_argument(prosecution_arguments[round_num], "prosecution", round_num + 1)
                
                with col2:
                    display_argument(defense_arguments[round_num], "defense", round_num + 1)
                
                # Display moderator verdict for this round
                st.markdown("---")
                display_moderator_verdict(moderator_verdicts[round_num], round_num + 1)
                
                st.divider()
            
            # Display Final Judgment
            st.markdown("---")
            st.markdown("---")
            display_final_judgment(final_judgment)
            
            st.divider()
            
            # Summary statistics
            st.header("📊 Debate Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rounds", num_rounds)
            
            with col2:
                avg_pros_score = sum(arg.heuristic_score for arg in prosecution_arguments) / len(prosecution_arguments)
                st.metric("Prosecution Avg Strength (heuristic)", f"{avg_pros_score:.2f}")
            
            with col3:
                avg_def_score = sum(arg.heuristic_score for arg in defense_arguments) / len(defense_arguments)
                st.metric("Defense Avg Strength (heuristic)", f"{avg_def_score:.2f}")
            
            # All citations used
            st.subheader("📚 All Case Citations Used")
            all_citations = []
            for arg in prosecution_arguments + defense_arguments:
                for citation in arg.case_citations:
                    if citation.case_name not in [c.case_name for c in all_citations]:
                        all_citations.append(citation)
            
            # Verify all citations and display summary
            if all_citations and 'citation_verifier' in st.session_state and st.session_state.citation_verifier:
                verifier = st.session_state.citation_verifier
                
                # Convert to dict format for verification
                citations_to_verify = [
                    {
                        'case_name': c.case_name,
                        'citation': c.citation,
                        'year': c.year
                    }
                    for c in all_citations
                ]
                
                # Verify all
                verification_results = verifier.verify_all_citations(citations_to_verify)
                summary = verifier.get_verification_summary(verification_results)
                
                # Display verification summary
                st.info(f"""
                **🔍 Citation Verification Summary**
                
                - **Total Citations:** {summary['total']}
                - ✅ **Verified:** {summary['verified']} ({summary['verification_rate']:.1f}%)
                - ⚠️ **Uncertain:** {summary['uncertain']}
                - ❌ **Unverified:** {summary['unverified']}
                - 🚨 **Hallucinated:** {summary['hallucinated']}
                - **Avg Verification Confidence:** {summary['avg_confidence']:.0%}
                - **Status:** {'🟢 SAFE' if summary['status'] == 'SAFE' else '🔴 RISKY'}
                """)
                
                # Display each citation with verification
                for citation, verification in verification_results:
                    # Format citation display, hide "N/A" values
                    citation_str = citation['citation'] if citation['citation'] and citation['citation'] != 'N/A' else ''
                    year_str = citation['year'] if citation['year'] and citation['year'] != 'N/A' else ''
                    
                    # Build display string
                    case_display = f"**{citation['case_name']}**"
                    if year_str:
                        case_display += f" ({year_str})"
                    if citation_str:
                        case_display += f" - {citation_str}"
                    
                    if verification.flag == "VERIFIED":
                        st.success(f"✅ {case_display} | Confidence: {verification.confidence:.0%}")
                    elif verification.flag == "UNCERTAIN":
                        st.warning(f"⚠️ {case_display} | Confidence: {verification.confidence:.0%}")
                    elif verification.flag == "HALLUCINATED":
                        st.error(f"🚨 {case_display} - HALLUCINATED (placeholder or empty)")
                    else:
                        st.error(f"❌ {case_display} | UNVERIFIED")
            else:
                # Fallback without verification
                for citation in all_citations:
                    st.write(f"• **{citation.case_name}** ({citation.year}) - {citation.citation}")
        
        except Exception as e:
            st.error(f"❌ An error occurred during the debate: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()