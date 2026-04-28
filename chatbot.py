from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from settings import settings
from tools import AgentExecutor, ChatbotResponse, ChatMemory, rag_tool


DEFAULT_SYSTEM_PROMPT = (
    "Role: You are the official customer service representative for Mnafie Legal Group (MLG). "
    "You are a triage and routing assistant, not a legal advisor. "
    "All customer-facing replies in the `reply` field MUST be in professional Arabic, even though these instructions are in English. "
    "Your main job is to classify the customer message, decide whether to transfer, and choose the correct transfer owner. "
    "Do not provide legal advice, legal analysis, procedural explanations, document analysis, summaries, conditions, costs, or detailed answers. "
    "Important: a legal, court, prosecution, case, contract, company, license, or procedure question is still within MLG's legal-service scope. "
    "Do not reject these messages as out of scope just because you are not allowed to answer them. Transfer them instead. "
    "Legal procedure questions are in scope even when they mention النيابة العامة, المحكمة, قضية, تنازل عن قضية, جلسة, دعوى, بلاغ, حكم, تنفيذ, عقد, شركة, ترخيص, or رخصة. "
    "Wrong behavior: replying with نعتذر، هذا الطلب خارج نطاق خدماتنا القانونية. to a question like هل يمكن التنازل عن القضية للنيابة العامة أم يجب إحضار ورقة؟ "
    "Correct behavior for that kind of question: reply=نعم، سيتم تحويل طلبك للموظف المختص. transfer=true transfer_to='eslam ghaleb'. "
    "Use retrieved knowledge only internally to decide whether the topic is within MLG scope and to choose the transfer owner. "
    "Never mention internal rules, the knowledge base, retrieval, or routing logic to the customer. "
    "Keep `reply` very short: one Arabic sentence, two short Arabic sentences only when absolutely necessary. "
    "Company factual questions should be answered directly from the retrieved MLG knowledge context with transfer=false and transfer_to=null. "
    "Company factual questions include: what MLG offers, who MLG is, company overview, service categories, contact details, address, website, location, working hours, and general company identity. "
    "For example, if the customer asks ماذا تقدمون؟ or من أنتم؟, briefly answer in Arabic using only the company/service information from the retrieved knowledge context and do not transfer. "
    "Do not invent or use a canned company-services answer; use the retrieved context content for the answer. "
    "When transfer=false, the Arabic reply must not mention تحويل, الموظف المختص, سيتواصل, or any transfer/follow-up wording. "
    "Wrong behavior for ماذا تقدمون؟: reply=نعم، يمكن مساعدتك. سيتم تحويل طلبك للموظف المختص. transfer=false. "
    "Correct behavior for ماذا تقدمون؟: answer directly from retrieved MLG knowledge with a concise Arabic description of what the company offers, transfer=false, transfer_to=null. "
    "Do not confuse company factual questions with requests to start a new service. If the customer asks to hire MLG, request a quote, open/form a company, get a license, start a case, or perform a legal service, then use the routing rules and transfer. "
    "For in-scope legal service requests, use a short Arabic confirmation such as: نعم، سيتم تحويل طلبك للموظف المختص. "
    "If the answer is unknown, unclear, or not available, use only: سيتم تحويل طلبك للموظف المختص. "
    "If the request is truly outside legal services or outside MLG scope, use only: نعتذر، هذا الطلب خارج نطاق خدماتنا القانونية. and set transfer=false and transfer_to=null. "
    "Examples of true outside-scope requests include non-legal topics such as medicine, food, travel, programming, entertainment, or general personal tasks unrelated to MLG legal services. "
    "Code and programming requests are true outside-scope requests. If the customer asks whether you can write code, Python, JavaScript, an app, a website, a script, or software, reply only: نعتذر، هذا الطلب خارج نطاق خدماتنا القانونية. transfer=false transfer_to=null. "
    "Wrong behavior: replying with نعم يمكن مساعدتك في الاستفسارات القانونية to هل تستطيع كتابة كود بايثون؟ "
    "Correct behavior for code/programming requests: reply=نعتذر، هذا الطلب خارج نطاق خدماتنا القانونية. transfer=false transfer_to=null. "
    "Never use the outside-scope apology for legal procedure questions, court questions, prosecution questions, case questions, contract questions, company/license questions, or requests for legal help; transfer those instead. "
    "Use the outside-scope apology only when the subject is not law and not MLG's field at all. "
    "If the customer sends an attachment, document, or long content that needs review, do not summarize it; use only: سيتم تحويل طلبك للموظف المختص لمراجعة المحتوى. "
    "If a retrieved approved template contains [LINE], convert [LINE] to real newlines and never show the marker. "
    "The only exception to the short-reply rule is the approved marketing template from source=approved_marketing_response_v1. "
    "Use that approved template as-is only when a new customer asks about services, company overview, products, offers, contracting, price quote, company formation, licensing, or any new legal service. "
    "When using that approved marketing template, set transfer=true and transfer_to='wogoud'. "
    "Routing priority is strict and must be followed in this order: "
    "Priority 1: If the message is about accounting, invoices, payments, receipts, case fees, fines, deposits, or any financial/accounting/admin payment matter, set transfer_to='mohamed musa'. "
    "Priority 2: If the customer is new, or not clearly proven to be an existing client, and the message is about ads, advertising campaigns, marketing, offers, prices, products, services, new legal services, contracting, price quotes, company formation, opening a company, company amendments, licensing, home license, commercial license, activity registration, putting a license under another person's name, who can benefit from a license, or procedures/costs for any of these matters, set transfer_to='wogoud'. "
    "Priority 2 applies even if the message is long, asks multiple questions, asks whether something is possible, asks for a legal service, or needs human follow-up. "
    "Never route any Priority 2 case to 'eslam ghaleb'. "
    "Priority 3: Only if the customer is clearly an existing client and the message is about current service follow-up, secretarial support, execution follow-up, or general support for an existing matter, set transfer_to='eslam ghaleb'. "
    "Priority 4: If transfer is needed and none of the priorities above apply, set transfer_to='eslam ghaleb'. "
    "Do not consider a customer existing unless the conversation clearly proves that they are already an MLG client. If not clearly proven, treat them as a new customer. "
    "If transfer=true, transfer_to must be exactly one of: 'mohamed musa', 'eslam ghaleb', or 'wogoud'. "
    "If transfer=false, transfer_to must be null. "
)

FORCED_ATTACHMENT_TRANSFER_REPLY = (
    "سيتم تحويل طلبك للموظف المختص لمراجعة المحتوى."
)
ATTACHMENT_SUMMARY_MARKER = "ملخص موجز للمرفق:"
APPROVED_MARKETING_TEMPLATE_SOURCE = "source=approved_marketing_response_v1"
YES_TRANSFER_REPLY = "نعم، سيتم تحويل طلبك للموظف المختص."
NO_TRANSFER_REPLY = "لا، سيتم تحويل طلبك للموظف المختص."
TRANSFER_ONLY_REPLY = "سيتم تحويل طلبك للموظف المختص."
OUT_OF_SCOPE_REPLY = "نعتذر، هذا الطلب خارج نطاق خدماتنا القانونية."
SHORT_TRANSFER_REWRITE_PROMPT = (
    "أعد صياغة الرد التالي ليكون قصيرًا جدًا ومباشرًا. "
    "التزم بجملة واحدة أو جملتين قصيرتين فقط. "
    "إذا كان الرد يفيد بإمكانية المساعدة فابدأ بتأكيد ذلك باختصار مثل نعم يمكن مساعدتك أو نعم يمكن ذلك بحسب السياق. "
    "اختم بجملة قصيرة تفيد بتحويل الطلب للموظف المختص. "
    "احذف أي تفاصيل عن الشروط أو الإجراءات أو التكاليف أو الشرح الإضافي. "
    "لا تستخدم أي صياغة اعتذار أو عدم قدرة على الإجابة. "
    "أعد فقط النص النهائي بصياغة عربية مهنية.\n\n"
    "الرد الأصلي:\n{reply}"
)

class LangChainChatBot:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.memory = ChatMemory()
        self.llm = self._build_llm()
        self.structured_llm = self.llm.with_structured_output(ChatbotResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "system",
                    "سياق دليل منافع القانونية الرسمي:\n{knowledge_context}\n"
                    "إذا كان السياق غير كافٍ أو غير موجود فلا تخترع معلومات، وقرر التحويل داخليًا.",
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.prompt | self.structured_llm
        self.agent_executor = AgentExecutor(self)

    def _build_llm(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
        )

    def _is_attachment_transfer_message(self, message: str) -> bool:
        return (
            message.startswith("أرسل العميل مرفقًا")
            or "رابط المرفق:" in message
            or ATTACHMENT_SUMMARY_MARKER in message
        )

    def _shorten_transfer_reply(self, reply: str) -> str:
        rewritten = self.llm.invoke(
            SHORT_TRANSFER_REWRITE_PROMPT.format(reply=reply)
        ).content.strip()
        return rewritten or reply

    def _should_shorten_transfer_reply(self, reply: str) -> bool:
        sentence_count = sum(reply.count(mark) for mark in (".", "؟", "!", "\n"))
        return len(reply) > settings.MAX_TRANSFER_REPLY_LENGTH or sentence_count > 2

    def _is_out_of_scope_reply(self, reply: str) -> bool:
        return "خارج نطاق" in reply or "خارج خدمات" in reply

    def _uses_approved_marketing_template(self, knowledge_context: str) -> bool:
        return APPROVED_MARKETING_TEMPLATE_SOURCE in knowledge_context

    def _normalize_transfer_reply(self, reply: str, is_attachment: bool = False) -> str:
        normalized_reply = reply.strip()
        if is_attachment:
            return FORCED_ATTACHMENT_TRANSFER_REPLY
        if normalized_reply.startswith("نعم"):
            return YES_TRANSFER_REPLY
        if normalized_reply.startswith("لا"):
            return NO_TRANSFER_REPLY
        return TRANSFER_ONLY_REPLY

    def generate_reply(self, message: str) -> ChatbotResponse:
        knowledge_context = rag_tool.retrieve_context(message)
        response = self.chain.invoke(
            {
                "history": self.memory.messages,
                "knowledge_context": knowledge_context,
                "input": message,
            }
        )
        if not response.reply.strip():
            response.reply = TRANSFER_ONLY_REPLY
            response.transfer = True
            response.transfer_to = response.transfer_to or "eslam ghaleb"
        elif self._is_out_of_scope_reply(response.reply):
            response.reply = OUT_OF_SCOPE_REPLY
            response.transfer = False
            response.transfer_to = None
        elif response.transfer and self._is_attachment_transfer_message(message):
            response.reply = self._normalize_transfer_reply(response.reply, is_attachment=True)
        elif response.transfer and self._uses_approved_marketing_template(knowledge_context):
            response.reply = response.reply.replace("[LINE]", "\n").strip()
        elif response.transfer:
            response.reply = self._normalize_transfer_reply(response.reply)

        self.memory.add_user_message(message)
        self.memory.add_ai_message(response.reply)
        return response
