"""
AI Service Interface
"""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import List, Optional
from models import ChatRequest, ChatResponse, ChatbotMessage
from llm_configuration import LLMConfiguration, AzureOpenAIConnectionInfo


class AIService(ABC):
    """
    Abstract base class for AI service integration.
    This class defines the interface for chat completion services.
    It should be implemented by specific AI service providers."""

    def __init__(
        self, connection_info: AzureOpenAIConnectionInfo, model_config: LLMConfiguration
    ):
        if not connection_info:
            raise ValueError("connection_info cannot be None")
        if not model_config:
            raise ValueError("model_config cannot be None")

        self.connection_info = connection_info
        self.model_config = model_config

    @abstractmethod
    def get_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Get chat completion from the AI service.
        """

    def _update_messages(self, request: ChatRequest) -> list:
        messages = self._get_system_messages()

        # Add conversation history
        for message in request.history:
            messages.append({"role": message.role, "content": message.content})

        # Add current user message
        messages.append({"role": "user", "content": request.message})
        return messages

    def _update_history(
        self, response_content: Optional[str], request: ChatRequest
    ) -> List[ChatbotMessage]:
        """
        Update the conversation history with the new message.
        """

        # Update history
        history = request.history.copy()
        history.append(
            ChatbotMessage(
                role="user",
                content=request.message,
                timestamp=format_datetime(datetime.now(tz=timezone.utc)),
            )
        )
        history.append(
            ChatbotMessage(
                role="assistant",
                content=response_content,
                timestamp=format_datetime(datetime.now(tz=timezone.utc)),
            )
        )
        return history

    def _get_system_messages(self):
        return [
            {"role": "system", "content": msg.content}
            for msg in self.model_config.messages
            if msg.role.lower() == "system"
        ]
