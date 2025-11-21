"""
Unit tests for AnthropicProvider.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.generation.providers.base import (
    LLMConfig,
    ChatMessage,
    MessageRole,
    GenerationError,
    AuthenticationError,
    RateLimitError,
    ContextLengthExceededError
)

# Test if Anthropic is available
try:
    from src.generation.providers.anthropic import AnthropicProvider, ANTHROPIC_AVAILABLE
except ImportError:
    ANTHROPIC_AVAILABLE = False
    AnthropicProvider = None


@pytest.fixture
def config():
    """Create a test configuration."""
    return LLMConfig(
        model_name="claude-sonnet-4-5-20250929",
        temperature=0.7,
        max_tokens=2048,
        api_key="sk-ant-test-key"
    )


@pytest.fixture
def provider(config):
    """Create an AnthropicProvider instance."""
    if not ANTHROPIC_AVAILABLE:
        pytest.skip("Anthropic library not installed")
    return AnthropicProvider(config)


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestInitialization:
    """Tests for provider initialization."""

    def test_initialization(self, config):
        """Test basic initialization."""
        provider = AnthropicProvider(config)

        assert provider.config.model_name == "claude-sonnet-4-5-20250929"
        assert provider.config.temperature == 0.7
        assert provider.config.max_tokens == 2048
        assert provider.config.api_key == "sk-ant-test-key"
        assert provider._client is None

    def test_missing_model_name(self):
        """Test initialization without model name."""
        config = LLMConfig(
            model_name="",
            api_key="sk-ant-test"
        )

        with pytest.raises(ValueError, match="model_name is required"):
            AnthropicProvider(config)

    def test_missing_api_key(self):
        """Test initialization without API key."""
        config = LLMConfig(
            model_name="claude-sonnet-4-5-20250929",
            api_key=None
        )

        with pytest.raises(ValueError, match="api_key is required"):
            AnthropicProvider(config)

    def test_invalid_temperature(self):
        """Test initialization with invalid temperature."""
        config = LLMConfig(
            model_name="claude-sonnet-4-5-20250929",
            temperature=1.5,  # Invalid
            api_key="sk-ant-test"
        )

        with pytest.raises(ValueError, match="temperature must be between"):
            AnthropicProvider(config)

    def test_invalid_max_tokens(self):
        """Test initialization with invalid max_tokens."""
        config = LLMConfig(
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=0,
            api_key="sk-ant-test"
        )

        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AnthropicProvider(config)


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestMessageFormatting:
    """Tests for message formatting."""

    def test_format_single_user_message(self, provider):
        """Test formatting a single user message."""
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]

        system_msg, api_messages = provider._format_messages(messages)

        assert system_msg is None
        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "Hello"

    def test_format_system_message(self, provider):
        """Test formatting system message separately."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]

        system_msg, api_messages = provider._format_messages(messages)

        assert system_msg == "You are helpful"
        assert len(api_messages) == 1
        assert api_messages[0]["role"] == "user"

    def test_format_multiple_system_messages(self, provider):
        """Test concatenating multiple system messages."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            ChatMessage(role=MessageRole.SYSTEM, content="You are concise"),
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]

        system_msg, api_messages = provider._format_messages(messages)

        assert "You are helpful" in system_msg
        assert "You are concise" in system_msg
        assert len(api_messages) == 1

    def test_format_conversation(self, provider):
        """Test formatting a multi-turn conversation."""
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
            ChatMessage(role=MessageRole.USER, content="How are you?")
        ]

        system_msg, api_messages = provider._format_messages(messages)

        assert system_msg == "You are helpful"
        assert len(api_messages) == 3
        assert api_messages[0]["role"] == "user"
        assert api_messages[1]["role"] == "assistant"
        assert api_messages[2]["role"] == "user"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestGenerate:
    """Tests for text generation."""

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_generate_simple(self, mock_anthropic_class, provider):
        """Test simple text generation."""
        # Mock client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Hello! I'm Claude.")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)
        mock_response.id = "msg_123"

        mock_client.messages.create.return_value = mock_response

        # Test generation
        result = provider.generate("Hello")

        assert result.text == "Hello! I'm Claude."
        assert result.model == "claude-sonnet-4-5-20250929"
        assert result.finish_reason == "end_turn"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_generate_with_kwargs(self, mock_anthropic_class, provider):
        """Test generation with override parameters."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=5, output_tokens=10)
        mock_response.id = "msg_123"

        mock_client.messages.create.return_value = mock_response

        # Test with overrides
        result = provider.generate("Test", temperature=0.5, max_tokens=1000)

        # Check that overrides were used
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 1000

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_generate_auth_error(self, mock_anthropic_class, provider):
        """Test generation with authentication error."""
        from anthropic import AuthenticationError as AnthropicAuthError

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create properly formatted exception
        mock_response = Mock(status_code=401)
        error = AnthropicAuthError(
            message="Invalid API key",
            response=mock_response,
            body=None
        )
        mock_client.messages.create.side_effect = error

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            provider.generate("Hello")

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_generate_rate_limit_error(self, mock_anthropic_class, provider):
        """Test generation with rate limit error."""
        from anthropic import RateLimitError as AnthropicRateLimitError

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create properly formatted exception
        mock_response = Mock(status_code=429)
        error = AnthropicRateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body=None
        )
        mock_client.messages.create.side_effect = error

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            provider.generate("Hello")


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestChat:
    """Tests for chat generation."""

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_chat_with_messages(self, mock_anthropic_class, provider):
        """Test chat with message list."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="I'm doing well!")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=15, output_tokens=5)
        mock_response.id = "msg_123"

        mock_client.messages.create.return_value = mock_response

        messages = [
            ChatMessage(role=MessageRole.USER, content="How are you?")
        ]

        result = provider.chat(messages)

        assert result.text == "I'm doing well!"
        assert result.total_tokens == 20

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_chat_with_system_message(self, mock_anthropic_class, provider):
        """Test chat with system message."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.id = "msg_123"

        mock_client.messages.create.return_value = mock_response

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are helpful"),
            ChatMessage(role=MessageRole.USER, content="Hello")
        ]

        result = provider.chat(messages)

        # Verify system message was included
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["system"] == "You are helpful"


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestGenerateWithContext:
    """Tests for RAG-style generation."""

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_generate_with_context(self, mock_anthropic_class, provider):
        """Test generation with context (RAG pattern)."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Based on the context...")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=50, output_tokens=30)
        mock_response.id = "msg_123"

        mock_client.messages.create.return_value = mock_response

        result = provider.generate_with_context(
            query="How do I use buttons?",
            context="Button component documentation...",
            system_message="You are a design system expert."
        )

        assert "Based on the context" in result.text

        # Verify context was included in the user message
        call_args = mock_client.messages.create.call_args[1]
        user_message = call_args["messages"][0]["content"]
        assert "Button component documentation" in user_message
        assert "How do I use buttons?" in user_message
        assert call_args["system"] == "You are a design system expert."


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestStreaming:
    """Tests for streaming generation."""

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_generate_stream(self, mock_anthropic_class, provider):
        """Test streaming generation."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock stream context manager
        mock_stream = Mock()
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=False)
        mock_stream.text_stream = iter(["Hello", " ", "world", "!"])

        mock_client.messages.stream.return_value = mock_stream

        # Collect streamed tokens
        tokens = list(provider.generate_stream("Test"))

        assert tokens == ["Hello", " ", "world", "!"]
        assert mock_client.messages.stream.called

    def test_supports_streaming(self, provider):
        """Test streaming support flag."""
        assert provider.supports_streaming() is True


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestUtilityMethods:
    """Tests for utility methods."""

    def test_count_tokens(self, provider):
        """Test token counting estimate."""
        text = "This is a test message"
        count = provider.count_tokens(text)

        # Should be roughly 1/4 of character count
        assert count > 0
        assert count < len(text)

    def test_get_model_info(self, provider):
        """Test getting model info."""
        info = provider.get_model_info()

        assert info["model_name"] == "claude-sonnet-4-5-20250929"
        assert info["provider"] == "Anthropic"
        assert info["context_window"] == 200000
        assert info["supports_streaming"] is True
        assert info["supports_function_calling"] is True

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_cleanup(self, mock_anthropic_class, provider):
        """Test cleanup."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        provider._initialize_client()
        assert provider._client is not None

        provider.cleanup()
        assert provider._client is None


@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestContextManager:
    """Tests for context manager interface."""

    @patch('src.generation.providers.anthropic.Anthropic')
    def test_context_manager(self, mock_anthropic_class, config):
        """Test using provider as context manager."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.model = "claude-sonnet-4-5-20250929"
        mock_response.stop_reason = "end_turn"
        mock_response.usage = Mock(input_tokens=5, output_tokens=5)
        mock_response.id = "msg_123"

        mock_client.messages.create.return_value = mock_response

        with AnthropicProvider(config) as provider:
            assert provider._client is not None
            result = provider.generate("Test")
            assert result.text == "Response"

        # Client should be cleaned up after context
        assert provider._client is None


# Integration test marker
@pytest.mark.integration
@pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="Anthropic library not installed")
class TestAnthropicIntegration:
    """
    Integration tests with real Anthropic API.

    These tests are skipped by default. Run with:
    pytest tests/test_anthropic_provider.py -m integration

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def test_real_api_generation(self):
        """Test with real Anthropic API."""
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        config = LLMConfig(
            model_name="claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=100,
            api_key=api_key
        )

        with AnthropicProvider(config) as provider:
            result = provider.generate("What is 2+2? Answer in one word.")

            assert result.text
            assert result.prompt_tokens > 0
            assert result.completion_tokens > 0
            assert result.model
