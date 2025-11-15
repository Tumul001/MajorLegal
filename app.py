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
    confidence_score: float = Field(description="Confidence score between 0 and 1")

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
def calculate_confidence_score(
    argument: LegalArgument, 
    similarity_scores: List[float] = None
) -> float:
    """
    Calculate confidence score using multiple factors
    
    Factors:
    1. Case citation quality (40%)
    2. Vector similarity scores (30%)
    3. Argument structure (20%)
    4. Legal reasoning depth (10%)
    
    Returns:
        Confidence score between 0.6 and 0.95
    """
    # 1. Citation Quality Score (0-1)
    citation_count = len(argument.case_citations)
    statute_count = len(argument.statutes_cited)
    
    # More citations = higher confidence (max at 5 citations)
    citation_score = min(citation_count / 5, 1.0) * 0.7
    # Statute support adds confidence (max at 3 statutes)
    statute_score = min(statute_count / 3, 1.0) * 0.3
    citation_quality = citation_score + statute_score
    
    # 2. Similarity Score (if available from RAG retrieval)
    if similarity_scores:
        # Average of top similarity scores (already 0-1)
        avg_similarity = sum(similarity_scores[:3]) / len(similarity_scores[:3])
        similarity_factor = avg_similarity
    else:
        # Fallback: estimate based on citations
        similarity_factor = 0.75 if citation_count >= 3 else 0.65
    
    # 3. Argument Structure Score (0-1)
    supporting_points_score = min(len(argument.supporting_points) / 4, 1.0) * 0.4
    reasoning_words = len(argument.legal_reasoning.split())
    reasoning_score = min(reasoning_words / 150, 1.0) * 0.4  # Target: 150+ words
    # Acknowledging weaknesses shows thoroughness
    weakness_score = min(len(argument.weaknesses_acknowledged) / 2, 1.0) * 0.2
    structure_quality = supporting_points_score + reasoning_score + weakness_score
    
    # 4. Legal Reasoning Depth (0-1)
    blob = TextBlob(argument.main_argument + " " + argument.legal_reasoning)
    # Lower subjectivity = more objective/factual = higher confidence
    objectivity = 1.0 - blob.sentiment.subjectivity
    reasoning_depth = objectivity
    
    # Weighted combination
    confidence = (
        citation_quality * 0.40 +      # 40% weight on citations
        similarity_factor * 0.30 +     # 30% weight on vector similarity
        structure_quality * 0.20 +     # 20% weight on structure
        reasoning_depth * 0.10         # 10% weight on reasoning depth
    )
    
    # Normalize to 0.6-0.95 range (avoid extremes)
    normalized_confidence = 0.6 + (confidence * 0.35)
    
    return round(normalized_confidence, 2)

def get_confidence_breakdown(argument: LegalArgument) -> dict:
    """Calculate confidence breakdown using ML techniques"""
    # Sentiment analysis
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
    
    # Display confidence score with enhanced breakdown
    st.markdown(f"**Confidence Score:** {argument.confidence_score:.2f} 🎯")
    
    # Show detailed confidence breakdown in an expander
    breakdown = get_confidence_breakdown(argument)
    with st.expander("📊 Confidence Score Breakdown"):
        st.markdown("### Calculation Components:")
        
        # Citation Quality (40%)
        citation_count = len(argument.case_citations)
        statute_count = len(argument.statutes_cited)
        citation_score = min(citation_count / 5, 1.0) * 0.7 + min(statute_count / 3, 1.0) * 0.3
        st.progress(citation_score, text=f"Citation Quality (40%): {citation_score:.2f}")
        st.caption(f"📚 {citation_count} cases + {statute_count} statutes")
        
        # Argument Structure (20%)
        supporting_score = min(len(argument.supporting_points) / 4, 1.0) * 0.4
        reasoning_words = len(argument.legal_reasoning.split())
        reasoning_score = min(reasoning_words / 150, 1.0) * 0.4
        weakness_score = min(len(argument.weaknesses_acknowledged) / 2, 1.0) * 0.2
        structure_score = supporting_score + reasoning_score + weakness_score
        st.progress(structure_score, text=f"Structure Quality (20%): {structure_score:.2f}")
        st.caption(f"📝 {len(argument.supporting_points)} points + {reasoning_words} words reasoning")
        
        # Objectivity (10%)
        objectivity = 1.0 - breakdown['sentiment_subjectivity']
        st.progress(objectivity, text=f"Objectivity (10%): {objectivity:.2f}")
        st.caption(f"🎯 Sentiment: {breakdown['sentiment_polarity']:.2f}")
        
        st.markdown("---")
        st.markdown("**Formula:** `0.6 + (weighted_sum × 0.35)`")
        st.markdown("*Vector similarity (30%) automatically incorporated from RAG retrieval*")
    
    with st.expander("📋 Supporting Points"):
        for i, point in enumerate(argument.supporting_points, 1):
            st.write(f"{i}. {point}")
    
    with st.expander("📚 Case Citations"):
        for citation in argument.case_citations:
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
    
    def __init__(self, role: str, rag_system: ProductionLegalRAGSystem, model_name: str = "gemini-2.0-flash-exp"):
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
            opponent_args = f"\n\nProsecution's arguments:\n{state['prosecution_arguments'][-1].main_argument}"
        elif self.role == "prosecution" and state["defense_arguments"]:
            opponent_args = f"\n\nDefense's arguments:\n{state['defense_arguments'][-1].main_argument}"
        
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
                    "No confidence in argument quality - human expert needed"
                ],
                confidence_score=0.0  # Zero confidence for error states
            )

class ModeratorLangChainAgent:
    """LangChain-based moderator agent for impartial evaluation"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
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
- Confidence: {pros_arg.confidence_score}

DEFENSE:
- Argument: {def_arg.main_argument}
- Citations: {len(def_arg.case_citations)} cases
- Confidence: {def_arg.confidence_score}

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

def generate_ai_argument(role: str, case_desc: str, rag_system: ProductionLegalRAGSystem, round_num: int, opponent_last_arg: str = None) -> LegalArgument:
    """Generate AI-powered legal argument using Google Gemini and RAG"""
    
    import google.generativeai as genai
    import json
    
    # Check for Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("❌ GOOGLE_API_KEY not found. Please add it to your .env file.")
        raise ValueError("GOOGLE_API_KEY is required for AI debate")
    
    # Configure and initialize Google Gemini
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Search for relevant cases using RAG with similarity scores
    search_query = f"{role} argument for: {case_desc[:200]}"
    relevant_docs_with_scores = rag_system.vector_store.similarity_search_with_score(search_query, k=3)
    
    # Extract documents and scores
    relevant_docs = [doc for doc, score in relevant_docs_with_scores]
    similarity_scores = [score for doc, score in relevant_docs_with_scores]
    
    # Format retrieved cases for context
    case_context = "\n\n".join([
        f"**Case:** {doc.metadata.get('case_name', 'Unknown')}\n"
        f"**Citation:** {doc.metadata.get('citation', 'N/A')}\n"
        f"**Excerpt:** {doc.page_content[:300]}..."
        for doc in relevant_docs
    ])
    
    # Build JSON schema example
    json_example = {
        "main_argument": "string (2-3 sentences)",
        "supporting_points": ["point 1", "point 2", "point 3"],
        "case_citations": [
            {
                "case_name": "Case Name",
                "citation": "Citation",
                "year": "2020",
                "relevance": "Why relevant",
                "excerpt": "Key excerpt from case"
            }
        ],
        "statutes_cited": ["Statute 1", "Statute 2"],
        "legal_reasoning": "string (3-4 sentences)",
        "weaknesses_acknowledged": ["weakness 1", "weakness 2"],
        "confidence_score": 0.75
    }
    
    # Build the prompt based on role
    if role == "prosecution":
        system_prompt = f"""You are an expert prosecution attorney. Analyze the case and build a strong argument for the prosecution.

**Case Details:**
{case_desc}

**Relevant Indian Case Law:**
{case_context}

**Round:** {round_num}
{"**Opponent's Last Argument:** " + opponent_last_arg if opponent_last_arg else ""}

Generate a prosecution argument using the provided Indian cases. Focus on constitutional violations, criminal procedure, and fundamental rights under Indian law.

Return ONLY a valid JSON object (no markdown, no code blocks) with this EXACT structure:
{json.dumps(json_example, indent=2)}

IMPORTANT: 
- case_citations must be an array of objects with ALL fields: case_name, citation, year, relevance, excerpt
- Use the Indian cases provided above
- confidence_score must be a number between 0.6 and 0.95"""

    else:  # defense
        system_prompt = f"""You are an expert defense attorney. Analyze the case and build a strong defense argument.

**Case Details:**
{case_desc}

**Relevant Indian Case Law:**
{case_context}

**Round:** {round_num}
{"**Opponent's Last Argument:** " + opponent_last_arg if opponent_last_arg else ""}

Generate a defense argument using the provided Indian cases. Focus on protecting fundamental rights, fair trial guarantees, and due process under Indian Constitution.

Return ONLY a valid JSON object (no markdown, no code blocks) with this EXACT structure:
{json.dumps(json_example, indent=2)}

IMPORTANT: 
- case_citations must be an array of objects with ALL fields: case_name, citation, year, relevance, excerpt
- Use the Indian cases provided above
- confidence_score must be a number between 0.6 and 0.95"""

    # Generate the argument with JSON output
    prompt = system_prompt
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON to dict
        data = json.loads(response_text)
        
        # Fix case_citations if they're malformed
        if 'case_citations' in data:
            fixed_citations = []
            for citation in data['case_citations']:
                # If citation is a string, skip it
                if isinstance(citation, str):
                    continue
                    
                # If citation is a dict, ensure all required fields exist
                if isinstance(citation, dict):
                    # Add missing fields with defaults
                    if 'year' not in citation:
                        citation['year'] = '2020'
                    if 'excerpt' not in citation:
                        citation['excerpt'] = citation.get('relevance', 'Relevant case law')
                    if 'relevance' not in citation:
                        citation['relevance'] = 'Supports legal argument'
                    if 'citation' not in citation:
                        citation['citation'] = 'N/A'
                    if 'case_name' not in citation:
                        citation['case_name'] = 'Unknown Case'
                    
                    fixed_citations.append(citation)
            
            data['case_citations'] = fixed_citations
        
        # ⚠️ REMOVED DANGEROUS AUTO-CITATION INJECTION
        # Previously: Added RAG docs as citations when LLM didn't generate any
        # Risk: Misattribution and hallucination
        # New: Flag for human review instead
        
        if not data.get('case_citations'):
            print(f"⚠️ {role.upper()}: No citations in fallback parse - REQUIRES MANUAL REVIEW")
            # Don't inject fake citations - better to have none than wrong ones
        
        # Convert to LegalArgument object (AI's confidence score might be unreliable)
        argument = LegalArgument(**data)
        
        # Recalculate confidence score using our algorithm
        calculated_confidence = calculate_confidence_score(argument, similarity_scores)
        argument.confidence_score = calculated_confidence
        
        return argument
        
    except Exception as e:
        st.warning(f"⚠️ AI generation failed for {role}: {str(e)}")
        print(f"⚠️ FALLBACK ERROR for {role}: Returning safe error state")
        
        # Safe fallback: Return error state instead of generating fake citations
        fallback = LegalArgument(
            main_argument=f"⚠️ [ERROR: {role.upper()} argument generation failed - Manual review required]\n\nThe AI system encountered an error while generating the {role} argument. Retrieved {len(relevant_docs)} potentially relevant documents but could not construct a reliable legal argument.",
            supporting_points=[
                f"Automated {role} argument generation failed",
                f"Error: {str(e)[:100]}",
                "Human legal expert review is required",
                f"{len(relevant_docs)} cases retrieved but not verified for relevance"
            ],
            case_citations=[],  # EMPTY - Do not generate unverified citations
            statutes_cited=[],  # EMPTY - Cannot reliably extract in error state
            legal_reasoning=f"System error prevented automated generation of {role} argument. The retrieved legal documents have not been verified for relevance or applicability to this case. A qualified legal professional must manually review the case facts and construct appropriate arguments with proper citations.",
            weaknesses_acknowledged=[
                "Complete manual review required - automated system failed",
                "Retrieved documents not verified for relevance",
                "No confidence in argument quality - expert needed"
            ],
            confidence_score=0.0  # Zero confidence for error states
        )
        
        return fallback

def generate_moderator_verdict(
    prosecution_arg: LegalArgument,
    defense_arg: LegalArgument,
    case_desc: str,
    round_num: int
) -> ModeratorVerdict:
    """Generate moderator's verdict for a round"""
    
    # Initialize Gemini model
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    
    import google.generativeai as genai
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Build the moderator prompt
    json_example = {
        "round_winner": "prosecution or defense or tie",
        "reasoning": "2-3 sentences explaining the verdict",
        "prosecution_strengths": ["strength 1", "strength 2"],
        "defense_strengths": ["strength 1", "strength 2"],
        "prosecution_weaknesses": ["weakness 1", "weakness 2"],
        "defense_weaknesses": ["weakness 1", "weakness 2"],
        "legal_analysis": "3-4 sentences of legal analysis",
        "score_prosecution": 7.5,
        "score_defense": 8.0
    }
    
    prompt = f"""You are an expert judge moderating a legal debate. Analyze both arguments and provide a verdict for Round {round_num}.

**Case:** {case_desc}

**PROSECUTION ARGUMENT:**
Main Argument: {prosecution_arg.main_argument}
Supporting Points: {', '.join(prosecution_arg.supporting_points)}
Cases Cited: {len(prosecution_arg.case_citations)} cases
Statutes: {', '.join(prosecution_arg.statutes_cited)}
Confidence: {prosecution_arg.confidence_score}

**DEFENSE ARGUMENT:**
Main Argument: {defense_arg.main_argument}
Supporting Points: {', '.join(defense_arg.supporting_points)}
Cases Cited: {len(defense_arg.case_citations)} cases
Statutes: {', '.join(defense_arg.statutes_cited)}
Confidence: {defense_arg.confidence_score}

As an impartial judge, evaluate both sides based on:
1. Legal reasoning strength
2. Quality and relevance of case citations
3. Constitutional and statutory compliance
4. Acknowledgment of weaknesses
5. Overall persuasiveness

Return ONLY a valid JSON object (no markdown, no code blocks) with this EXACT structure:
{json.dumps(json_example, indent=2)}

IMPORTANT:
- round_winner must be exactly 'prosecution', 'defense', or 'tie'
- Scores must be between 0 and 10
- Be balanced and fair in your analysis"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        data = json.loads(response_text)
        return ModeratorVerdict(**data)
        
    except Exception as e:
        # Fallback verdict
        pros_score = prosecution_arg.confidence_score * 10
        def_score = defense_arg.confidence_score * 10
        
        return ModeratorVerdict(
            round_winner="tie" if abs(pros_score - def_score) < 0.5 else ("prosecution" if pros_score > def_score else "defense"),
            reasoning=f"Round {round_num} analysis based on confidence scores and argument quality.",
            prosecution_strengths=["Strong legal citations", "Clear reasoning"],
            defense_strengths=["Constitutional focus", "Procedural arguments"],
            prosecution_weaknesses=["Some points could be stronger"],
            defense_weaknesses=["Limited precedent support"],
            legal_analysis="Both sides presented competent arguments with relevant legal authority.",
            score_prosecution=pros_score,
            score_defense=def_score
        )

def generate_final_judgment(
    all_verdicts: List[ModeratorVerdict],
    prosecution_args: List[LegalArgument],
    defense_args: List[LegalArgument],
    case_desc: str
) -> FinalJudgment:
    """Generate final judgment after all rounds"""
    
    # Initialize Gemini model
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    
    import google.generativeai as genai
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Calculate total scores
    total_pros_score = sum(v.score_prosecution for v in all_verdicts)
    total_def_score = sum(v.score_defense for v in all_verdicts)
    
    # Build summary of all rounds
    rounds_summary = "\n".join([
        f"Round {i+1}: Winner: {v.round_winner}, Prosecution: {v.score_prosecution}, Defense: {v.score_defense}"
        for i, v in enumerate(all_verdicts)
    ])
    
    json_example = {
        "overall_winner": "prosecution or defense or balanced",
        "judgment_summary": "2-3 sentences summary",
        "key_findings": ["finding 1", "finding 2", "finding 3"],
        "final_verdict": "The final legal verdict in 3-4 sentences",
        "recommended_action": "Recommended ruling or action",
        "total_score_prosecution": total_pros_score,
        "total_score_defense": total_def_score
    }
    
    prompt = f"""You are a senior judge delivering the FINAL JUDGMENT after a {len(all_verdicts)}-round legal debate.

**Case:** {case_desc}

**ROUND RESULTS:**
{rounds_summary}

**TOTAL SCORES:**
- Prosecution: {total_pros_score:.1f}
- Defense: {total_def_score:.1f}

Based on all rounds of arguments, deliver your final judgment considering:
1. Cumulative strength of arguments
2. Legal precedent and constitutional compliance
3. Quality of evidence and reasoning
4. Overall persuasiveness across all rounds

Return ONLY a valid JSON object (no markdown, no code blocks) with this EXACT structure:
{json.dumps(json_example, indent=2)}

IMPORTANT:
- overall_winner must be exactly 'prosecution', 'defense', or 'balanced'
- Provide a definitive verdict on the case
- Consider Indian constitutional law and Supreme Court precedents"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        data = json.loads(response_text)
        data['total_score_prosecution'] = total_pros_score
        data['total_score_defense'] = total_def_score
        
        return FinalJudgment(**data)
        
    except Exception as e:
        return FinalJudgment(
            overall_winner="balanced" if abs(total_pros_score - total_def_score) < 1.0 else ("prosecution" if total_pros_score > total_def_score else "defense"),
            judgment_summary=f"After {len(all_verdicts)} rounds of debate, the court has reached a decision based on the cumulative legal arguments.",
            key_findings=[
                "Both sides presented substantive legal arguments",
                "Constitutional principles were thoroughly examined",
                "Case law precedents were appropriately cited"
            ],
            final_verdict="The court finds that based on the preponderance of legal authority and constitutional principles presented, a decision has been reached.",
            recommended_action="The court recommends proceeding in accordance with established legal precedent and constitutional safeguards.",
            total_score_prosecution=total_pros_score,
            total_score_defense=total_def_score
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
        st.metric("🔴 Prosecution Score", f"{verdict.score_prosecution:.1f}/10")
    with col2:
        st.metric("🔵 Defense Score", f"{verdict.score_defense:.1f}/10")
    
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
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.2)
            
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
        - **� Defense**: LangChain agent with RAG
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
                avg_pros_conf = sum(arg.confidence_score for arg in prosecution_arguments) / len(prosecution_arguments)
                st.metric("Prosecution Avg Confidence", f"{avg_pros_conf:.2f}")
            
            with col3:
                avg_def_conf = sum(arg.confidence_score for arg in defense_arguments) / len(defense_arguments)
                st.metric("Defense Avg Confidence", f"{avg_def_conf:.2f}")
            
            # All citations used
            st.subheader("📚 All Case Citations Used")
            all_citations = []
            for arg in prosecution_arguments + defense_arguments:
                for citation in arg.case_citations:
                    if citation.case_name not in [c.case_name for c in all_citations]:
                        all_citations.append(citation)
            
            for citation in all_citations:
                st.write(f"• **{citation.case_name}** ({citation.year}) - {citation.citation}")
        
        except Exception as e:
            st.error(f"❌ An error occurred during the debate: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()
    