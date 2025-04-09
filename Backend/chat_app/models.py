"""
This module defines the data models used in the chat application.
"""

from datetime import datetime
from email.utils import format_datetime
from typing import List, Optional
from pydantic import BaseModel


class ChatbotMessage(BaseModel):
    """Represents a message in the chat history."""

    role: str
    content: Optional[str] = None
    timestamp: str = format_datetime(datetime.now())


class ChatRequest(BaseModel):
    """Represents a chat request."""

    message: str
    history: List[ChatbotMessage] = []


class ChatResponse(BaseModel):
    """Represents a chat response."""

    message: Optional[str] = None
    history: List[ChatbotMessage] = []
