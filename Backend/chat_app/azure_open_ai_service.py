"""
Azure OpenAI Service wrapper for chat completion.
"""

import logging
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from .models import ChatRequest, ChatResponse
from .llm_configuration import AzureOpenAIConnectionInfo, LLMConfiguration
from .ai_service import AIService


logger = logging.getLogger(__name__)


class AzureOpenAIService(AIService):
    """
    Azure OpenAI Service wrapper for chat completion.
    """

    def __init__(
        self, connection_info: AzureOpenAIConnectionInfo, model_config: LLMConfiguration
    ):
        super().__init__(connection_info, model_config)

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
        response = self.client.chat.completions.create(
            messages=self._update_messages(request),
            max_tokens=self.model_config.max_completion_tokens,
            temperature=self.model_config.temperature,
            top_p=1.0,
            model=self.model_config.model,
        )

        response_content = response.choices[0].message.content

        return ChatResponse(
            message=response_content,
            history=self._update_history(response_content, request),
        )
