from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import format_datetime
from models import ChatRequest, ChatResponse, ChatbotMessage


class AIService(ABC):

    def __init__(self, connection_info: dict, model_config: dict):
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
        pass

    def _update_history(
        self, response_content: str, request: ChatRequest
    ) -> ChatResponse:
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
