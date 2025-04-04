"""
Azure OpenAI Service wrapper for chat completion.
"""

import logging
from datetime import datetime, timezone
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from models import ChatRequest, ChatResponse, ChatbotMessage
from llm_configuration import AzureOpenAIConnectionInfo, LLMConfiguration


logger = logging.getLogger(__name__)


class AzureOpenAIService:
    """
    Azure OpenAI Service wrapper for chat completion.
    """

    def __init__(
        self, connection_info: AzureOpenAIConnectionInfo, model_config: LLMConfiguration
    ):
        if not connection_info:
            raise ValueError("connection_info cannot be None")
        if not model_config:
            raise ValueError("model_config cannot be None")

        self.model_config = model_config

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

        self.client = AzureOpenAI(
            api_version=connection_info.api_version,
            azure_endpoint=connection_info.endpoint,
            azure_ad_token_provider=token_provider,
        )

    def get_chat_completion(self, request: ChatRequest) -> ChatResponse:
        """
        Get chat completion from Azure OpenAI service.
        """
        messages = self._get_system_messages()

        # Add conversation history
        for message in request.history:
            messages.append({"role": message.role, "content": message.content})

        # Add current user message
        messages.append({"role": "user", "content": request.message})

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=self.model_config.max_completion_tokens,
            temperature=self.model_config.temperature,
            top_p=1.0,
            model=self.model_config.model,
        )

        response_content = response.choices[0].message.content

        # Update history
        history = request.history.copy()
        history.append(
            ChatbotMessage(
                role="user",
                content=request.message,
                timestamp=datetime.now(tz=timezone.utc),
            )
        )
        history.append(
            ChatbotMessage(
                role="assistant",
                content=response_content,
                timestamp=datetime.now(tz=timezone.utc),
            )
        )

        return ChatResponse(message=response_content, history=history)

    def _get_system_messages(self):
        return [
            {"role": "system", "content": msg.content}
            for msg in self.model_config.messages
            if msg.role.lower() == "system"
        ]
