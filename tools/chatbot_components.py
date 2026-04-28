from typing import TYPE_CHECKING, List, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from chatbot import LangChainChatBot


class ChatbotResponse(BaseModel):
    reply: str = Field(
        ...,
        description=(
            "Arabic reply shown to the customer. Keep it very short unless using the approved marketing template. "
            "If transfer=false, do not mention transfer, الموظف المختص, or follow-up by a human."
        ),
    )
    transfer: bool = Field(
        ...,
        description=(
            "Use transfer=false for factual company questions about who MLG is, what MLG offers, service categories, contact details, address, website, location, or working hours. "
            "Whether the chat should be transferred to a human. Legal/court/prosecution/case/procedure questions "
            "are in scope and should transfer=true, not be rejected as out of scope. Questions about النيابة العامة, "
            "court cases, waiving a case, hearings, lawsuits, judgments, execution, contracts, companies, or licenses are in scope. "
            "Code/programming/software requests such as writing Python code are outside MLG legal scope and should transfer=false."
        ),
    )
    transfer_to: Literal["mohamed musa", "eslam ghaleb", "wogoud"] | None = Field(
        default=None,
        description=(
            "Human owner for the transfer. Use 'mohamed musa' for accounting/payments/fees. "
            "Use 'wogoud' for any new customer, or not-proven-existing customer, asking about marketing, ads, offers, prices, products, services, "
            "new legal services, contracting, company formation/opening, company amendments, licenses, home/commercial licenses, activity registration, "
            "or putting a license under another person's name. Use 'eslam ghaleb' only for clearly existing client follow-up/support, "
            "or as the final fallback when no higher-priority owner applies. Use null when transfer is false."
        ),
    )


class ChatMemory:
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))


class AgentExecutor:
    def __init__(self, chatbot: "LangChainChatBot"):
        self.chatbot = chatbot

    def invoke(self, payload: dict) -> dict:
        message = payload.get("input", "")
        return {"output": self.chatbot.generate_reply(message)}
