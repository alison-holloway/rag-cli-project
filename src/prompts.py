"""Prompt templates for RAG CLI.

Provides structured prompts for different RAG scenarios.
"""

from dataclasses import dataclass
from string import Template


@dataclass
class PromptTemplate:
    """A reusable prompt template."""

    name: str
    system_template: str
    user_template: str
    description: str = ""

    def format_system(self, **kwargs) -> str:
        """Format the system prompt with provided variables."""
        return Template(self.system_template).safe_substitute(**kwargs)

    def format_user(self, **kwargs) -> str:
        """Format the user prompt with provided variables."""
        return Template(self.user_template).safe_substitute(**kwargs)

    def format(self, **kwargs) -> tuple[str, str]:
        """Format both prompts and return (system, user) tuple."""
        return self.format_system(**kwargs), self.format_user(**kwargs)


# =============================================================================
# RAG Prompt Templates
# =============================================================================

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions \
based on the provided context.

Rules you MUST follow strictly:
- Answer concisely and directly in a professional, instructional tone.
- Use numbered steps for installation or multi-step procedures.
- Always include exact commands in inline code blocks using backticks (e.g. `sudo dnf install ...`).
- If the context contains relevant information from specific chunks, cite them briefly at the start or end like: "From the Oracle Linux documentation:" or list sources at the end.
- If multiple chunks are relevant, synthesize them into one clear answer without mentioning chunk numbers unless necessary.
- If the question is not answerable from the context, reply only: "I don't have sufficient information in the provided context to answer this."
- Do NOT add extra explanations, warnings, opinions, or fluff.
- Do NOT say things like "According to Chunk X" unless explicitly helpful — prefer clean, user-friendly citations.
- Keep responses short: aim for 3-8 sentences maximum unless more detail is required."""

RAG_USER_PROMPT = """Context:
$context

---

Question: $question

Please answer the question based only on the provided context above."""


RAG_TEMPLATE = PromptTemplate(
    name="rag_default",
    system_template=RAG_SYSTEM_PROMPT,
    user_template=RAG_USER_PROMPT,
    description="Default RAG prompt for question answering",
)


# =============================================================================
# Specialized Templates
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a helpful assistant that summarizes documents.

Instructions:
- Provide a clear, concise summary of the provided content
- Highlight the key points and main ideas
- Organize the summary in a logical structure
- Keep the summary focused and avoid unnecessary details"""

SUMMARY_USER_PROMPT = """Content to summarize:
$context

---

Please provide a summary of the above content."""

SUMMARY_TEMPLATE = PromptTemplate(
    name="summary",
    system_template=SUMMARY_SYSTEM_PROMPT,
    user_template=SUMMARY_USER_PROMPT,
    description="Template for summarizing document content",
)


CHAT_SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.
You answer questions based on the provided context from the knowledge base.

Instructions:
- Use the context to answer questions accurately
- If you don't have enough information, say so
- Be conversational but accurate
- Cite sources when relevant"""

CHAT_USER_PROMPT = """Context from knowledge base:
$context

---

User: $question"""

CHAT_TEMPLATE = PromptTemplate(
    name="chat",
    system_template=CHAT_SYSTEM_PROMPT,
    user_template=CHAT_USER_PROMPT,
    description="Template for conversational RAG chat",
)


EXTRACT_SYSTEM_PROMPT = """You are an assistant that extracts specific \
information from documents.

Instructions:
- Extract only the information requested
- Be precise and factual
- If the information is not found, state that clearly
- Format the output in a clear, structured way"""

EXTRACT_USER_PROMPT = """Context:
$context

---

Extract the following information: $question"""

EXTRACT_TEMPLATE = PromptTemplate(
    name="extract",
    system_template=EXTRACT_SYSTEM_PROMPT,
    user_template=EXTRACT_USER_PROMPT,
    description="Template for extracting specific information",
)


# =============================================================================
# No-Context Templates (when retrieval returns no results)
# =============================================================================

NO_CONTEXT_SYSTEM_PROMPT = """You are a helpful assistant.

The user asked a question but no relevant documents were found in the knowledge base.
Please let the user know that you couldn't find relevant information and suggest
they might want to add relevant documents or rephrase their question."""

NO_CONTEXT_USER_PROMPT = """The user asked: $question

No relevant documents were found in the knowledge base.
Please respond appropriately."""

NO_CONTEXT_TEMPLATE = PromptTemplate(
    name="no_context",
    system_template=NO_CONTEXT_SYSTEM_PROMPT,
    user_template=NO_CONTEXT_USER_PROMPT,
    description="Template when no relevant context is found",
)


# =============================================================================
# Template Registry
# =============================================================================

TEMPLATES = {
    "rag_default": RAG_TEMPLATE,
    "summary": SUMMARY_TEMPLATE,
    "chat": CHAT_TEMPLATE,
    "extract": EXTRACT_TEMPLATE,
    "no_context": NO_CONTEXT_TEMPLATE,
}


def get_template(name: str = "rag_default") -> PromptTemplate:
    """Get a prompt template by name.

    Args:
        name: Template name.

    Returns:
        PromptTemplate instance.

    Raises:
        ValueError: If template name is unknown.
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template: {name}. Available: {available}")
    return TEMPLATES[name]


def list_templates() -> list[str]:
    """List available template names."""
    return list(TEMPLATES.keys())


def format_rag_prompt(
    context: str,
    question: str,
    template_name: str = "rag_default",
) -> tuple[str, str]:
    """Format a RAG prompt with context and question.

    Convenience function for the common RAG use case.

    Args:
        context: Retrieved context from documents.
        question: User's question.
        template_name: Template to use.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    template = get_template(template_name)
    return template.format(context=context, question=question)
