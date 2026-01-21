"""Tests for prompts module."""

import pytest

from src.prompts import (
    CHAT_TEMPLATE,
    EXTRACT_TEMPLATE,
    NO_CONTEXT_TEMPLATE,
    RAG_TEMPLATE,
    SUMMARY_TEMPLATE,
    PromptTemplate,
    format_rag_prompt,
    get_template,
    list_templates,
)


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_prompt_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            name="test",
            system_template="System: $role",
            user_template="User: $question",
            description="Test template",
        )

        assert template.name == "test"
        assert template.description == "Test template"

    def test_format_system(self):
        """Test formatting system prompt."""
        template = PromptTemplate(
            name="test",
            system_template="You are a $role assistant.",
            user_template="",
        )

        result = template.format_system(role="helpful")
        assert result == "You are a helpful assistant."

    def test_format_user(self):
        """Test formatting user prompt."""
        template = PromptTemplate(
            name="test",
            system_template="",
            user_template="Question: $question",
        )

        result = template.format_user(question="What is AI?")
        assert result == "Question: What is AI?"

    def test_format_both(self):
        """Test formatting both prompts."""
        template = PromptTemplate(
            name="test",
            system_template="System: $mode",
            user_template="User: $message",
        )

        system, user = template.format(mode="chat", message="Hello")
        assert system == "System: chat"
        assert user == "User: Hello"

    def test_safe_substitute_missing_var(self):
        """Test that missing variables are left as-is."""
        template = PromptTemplate(
            name="test",
            system_template="Hello $name, your $missing is ready.",
            user_template="",
        )

        # safe_substitute leaves missing vars unchanged
        result = template.format_system(name="Alice")
        assert "$missing" in result
        assert "Alice" in result

    def test_format_with_special_characters(self):
        """Test formatting with special characters in values."""
        template = PromptTemplate(
            name="test",
            system_template="",
            user_template="Code: $code",
        )

        code = "def foo():\n    return $bar"
        result = template.format_user(code=code)
        assert "def foo():" in result


class TestBuiltInTemplates:
    """Tests for built-in templates."""

    def test_rag_template_exists(self):
        """Test RAG template is defined."""
        assert RAG_TEMPLATE.name == "rag_default"
        assert "$context" in RAG_TEMPLATE.user_template
        assert "$question" in RAG_TEMPLATE.user_template

    def test_rag_template_format(self):
        """Test formatting RAG template."""
        system, user = RAG_TEMPLATE.format(
            context="Document content here",
            question="What is this about?",
        )

        assert "Document content here" in user
        assert "What is this about?" in user
        assert "context" in system.lower() or "answer" in system.lower()

    def test_summary_template_exists(self):
        """Test summary template is defined."""
        assert SUMMARY_TEMPLATE.name == "summary"
        assert "$context" in SUMMARY_TEMPLATE.user_template

    def test_summary_template_format(self):
        """Test formatting summary template."""
        system, user = SUMMARY_TEMPLATE.format(
            context="Long document content...",
        )

        assert "Long document content..." in user
        assert "summary" in system.lower() or "summarize" in system.lower()

    def test_chat_template_exists(self):
        """Test chat template is defined."""
        assert CHAT_TEMPLATE.name == "chat"
        assert "$context" in CHAT_TEMPLATE.user_template
        assert "$question" in CHAT_TEMPLATE.user_template

    def test_extract_template_exists(self):
        """Test extract template is defined."""
        assert EXTRACT_TEMPLATE.name == "extract"
        assert "$context" in EXTRACT_TEMPLATE.user_template
        assert "$question" in EXTRACT_TEMPLATE.user_template

    def test_no_context_template_exists(self):
        """Test no-context template is defined."""
        assert NO_CONTEXT_TEMPLATE.name == "no_context"
        assert "$question" in NO_CONTEXT_TEMPLATE.user_template

    def test_no_context_template_format(self):
        """Test formatting no-context template."""
        system, user = NO_CONTEXT_TEMPLATE.format(
            question="What is machine learning?",
        )

        assert "What is machine learning?" in user
        # System should mention no documents found
        assert "no" in system.lower() or "not" in system.lower()


class TestGetTemplate:
    """Tests for get_template function."""

    def test_get_rag_default(self):
        """Test getting default RAG template."""
        template = get_template("rag_default")
        assert template.name == "rag_default"

    def test_get_template_default_param(self):
        """Test get_template with default parameter."""
        template = get_template()
        assert template.name == "rag_default"

    def test_get_summary_template(self):
        """Test getting summary template."""
        template = get_template("summary")
        assert template.name == "summary"

    def test_get_chat_template(self):
        """Test getting chat template."""
        template = get_template("chat")
        assert template.name == "chat"

    def test_get_extract_template(self):
        """Test getting extract template."""
        template = get_template("extract")
        assert template.name == "extract"

    def test_get_no_context_template(self):
        """Test getting no-context template."""
        template = get_template("no_context")
        assert template.name == "no_context"

    def test_get_unknown_template_raises(self):
        """Test that unknown template raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_template("nonexistent")

        assert "Unknown template" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
        # Should list available templates
        assert "rag_default" in str(exc_info.value)


class TestListTemplates:
    """Tests for list_templates function."""

    def test_list_templates_returns_list(self):
        """Test that list_templates returns a list."""
        templates = list_templates()
        assert isinstance(templates, list)

    def test_list_templates_contains_all(self):
        """Test that list_templates contains all built-in templates."""
        templates = list_templates()

        assert "rag_default" in templates
        assert "summary" in templates
        assert "chat" in templates
        assert "extract" in templates
        assert "no_context" in templates

    def test_list_templates_count(self):
        """Test the number of built-in templates."""
        templates = list_templates()
        assert len(templates) >= 5  # At least 5 built-in templates


class TestFormatRagPrompt:
    """Tests for format_rag_prompt convenience function."""

    def test_format_rag_prompt_basic(self):
        """Test basic RAG prompt formatting."""
        system, user = format_rag_prompt(
            context="Document about AI.",
            question="What is AI?",
        )

        assert "Document about AI." in user
        assert "What is AI?" in user

    def test_format_rag_prompt_default_template(self):
        """Test format_rag_prompt uses default template."""
        system, user = format_rag_prompt(
            context="Test context",
            question="Test question",
        )

        # Should match RAG_TEMPLATE output
        expected_system, expected_user = RAG_TEMPLATE.format(
            context="Test context",
            question="Test question",
        )
        assert system == expected_system
        assert user == expected_user

    def test_format_rag_prompt_custom_template(self):
        """Test format_rag_prompt with custom template."""
        system, user = format_rag_prompt(
            context="Test context",
            question="Test question",
            template_name="chat",
        )

        # Should use chat template
        expected_system, expected_user = CHAT_TEMPLATE.format(
            context="Test context",
            question="Test question",
        )
        assert system == expected_system
        assert user == expected_user

    def test_format_rag_prompt_invalid_template(self):
        """Test format_rag_prompt with invalid template."""
        with pytest.raises(ValueError):
            format_rag_prompt(
                context="Test",
                question="Test",
                template_name="invalid",
            )

    def test_format_rag_prompt_empty_context(self):
        """Test format_rag_prompt with empty context."""
        system, user = format_rag_prompt(
            context="",
            question="What is AI?",
        )

        assert "What is AI?" in user
        # Context should be empty in output
        assert "Context:" in user or "context" in user.lower()

    def test_format_rag_prompt_multiline_context(self):
        """Test format_rag_prompt with multiline context."""
        context = """Line 1
Line 2
Line 3"""
        system, user = format_rag_prompt(
            context=context,
            question="What are the lines?",
        )

        assert "Line 1" in user
        assert "Line 2" in user
        assert "Line 3" in user
