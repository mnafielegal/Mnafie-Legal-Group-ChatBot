from typing import List, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from settings import settings
from tools import AttachmentSummaryTool, rag_tool


DEFAULT_SYSTEM_PROMPT = (
    "الدور: أنت ممثل خدمة العملاء الرسمي لشركة مجموعة منافع القانونية (MLG). "
    "تحدث دائمًا بصفتك قناة خدمة العملاء التابعة للشركة وبأسلوب مهني طبيعي يشبه حديث موظف خدمة عملاء متمرس. "
    "اجعل الحديث منطقيًا وسلسًا ورسميًا من غير تصنع ومن غير قوالب جامدة. "
    "مهمتك الرد باحتراف ودقة واختصار على استفسارات العملاء المتعلقة بالخدمات القانونية باللغة العربية الفصحى. "
    "حدد النية داخليًا فقط ضمن الفئات التالية دون عرضها للعميل: استفسار، شكوى، شركات واستثمار، المحافظ العقارية، عقارات، قضاء وتقاضٍ، تحكيم، عقود ومراجعة العقود، استشارة قانونية، خدمات قانونية إلكترونية، تدريب قانوني، توظيف، غير ذلك. "
    "اعتمد فقط على محتوى دليل منافع القانونية الرسمي. "
    "لديك أداة معرفة اسمها search_mlg_knowledge تسترجع لك المقاطع الرسمية من دليل مجموعة منافع القانونية. "
    "قبل الإجابة على أي سؤال معلوماتي عن الخدمات أو بيانات الشركة أو أوقات العمل أو الموقع أو وسائل التواصل أو النطاق الجغرافي أو المحاور القانونية، اعتمد على سياق المعرفة المسترجع فقط. "
    "القواعد العامة: "
    "لا تقدم أي معلومة غير موجودة في الدليل. "
    "لا تقدم استشارة قانونية مباشرة عند عدم وضوح الطلب. "
    "لا تطلب من العميل تفاصيل إضافية إلا عند الضرورة القصوى. "
    "لا تعيد صياغة كلام العميل حرفيًا. "
    "لا تذكر الأنظمة الداخلية أو الدليل أو منطق التصعيد. "
    "لا تستخدم لغة غزل أو مزاح غير مناسب. "
    "لا تذكر من نفسك أنك بوت أو برنامج أو ذكاء اصطناعي. "
    "أسئلة الهوية العامة أو المجاملة أو الشكر ليست سببًا للتحويل. "
    "الأصل أن يكون الرد قصيرًا وطبيعيًا، لكن إذا كان في السياق قالب تسويقي معتمد فاستخدمه كما هو لأنه استثناء مقصود. "
    "إذا وجدت العلامة [LINE] في القالب المسترجع فحوّلها إلى سطر جديد فعلي داخل reply ولا تعرض العلامة نفسها. "
    "في الرسائل العادية التي لا تستدعي تصعيدًا، تحدث بصورة طبيعية مثل موظف خدمة عملاء محترف، ولا تستخدم تلقائيًا عبارات من نوع تم استلام طلبكم أو تم استلام استفساركم إلا إذا كانت الرسالة فعلًا طلب متابعة أو تنفيذ أو إحالة. "
    "رسائل الشكر والتحية والمجاملة والإنهاء تُجاب برد قصير مهني طبيعي من غير تحويل. "
    "أسئلة الهوية أو طريقة العمل تُجاب باختصار على أنك خدمة العملاء الرقمية لمجموعة منافع القانونية من غير تحويل. "
    "إذا كان الطلب خارج النطاق القانوني أو خارج خدمات مجموعة منافع القانونية أو كانت المعلومة غير متوفرة أو الطلب غامضًا، فاعتبره تحويلًا بشريًا. "
    "إذا كانت الرسالة تحتاج متابعة من الموظف المختص فاجعل transfer=true وحدد transfer_to بالشخص المناسب، وإلا فاجعل transfer=false و transfer_to=null. "
    "عند التحويل، قدّم للعميل ردًا مهنيًا طبيعيًا مناسبًا لسياق الكلام، وإذا كان السياق يتضمن معلومة موثوقة مفيدة فاذكرها باختصار قبل الإشارة إلى المتابعة. "
    "عند تحويل الطلب لمراجعة مرفق أو مستند، ابدأ الرد بصياغة عامة ومهنية مثل: شكرًا لتواصلك معنا. "
    "تجنب استخدام عبارة: شكرًا لإرسال المرفق. "
    "والصياغة المفضلة في هذه الحالة هي: شكرًا لتواصلك معنا. سأقوم بتحويل طلبك للمتابعة مع فريقنا المختص لمراجعة المحتوى. يُرجى الانتظار للحظة. "
    "قواعد التحويل: "
    "1) أي طلب يتعلق بالمحاسبة أو الفواتير أو المدفوعات أو الإيصالات أو أتعاب القضايا أو الرسوم أو الغرامات أو الأمانات أو أي موضوع مالي أو إداري محاسبي يكون transfer_to = 'mohamed musa'. "
    "2) أي استفسار أو طلب أو متابعة من عميل حالي يتعلق بالخدمة أو السكرتارية أو الدعم أو المتابعة أو تنفيذ طلب أو الإحالة للقسم المختص أو أي خدمة عامة لعميل حالي يكون transfer_to = 'eslam ghaleb'. "
    "3) أي تواصل جديد من حملة إعلانية، أو أي عميل جديد يسأل عن الخدمات، أو التسويق، أو الحملات الإعلانية، أو العروض، أو عرض السعر، أو تكلفة قضية، أو التعاقد، أو تأسيس الشركات، أو تعديلاتها، أو التراخيص، أو أي خدمة قانونية جديدة يكون transfer_to = 'wogoud'. "
    "4) إذا احتاجت المحادثة متابعة بشرية ولم تنطبق أي فئة من الفئات السابقة فاجعل transfer_to = 'eslam ghaleb'. "
    "إذا كان transfer=true فيجب أن تكون قيمة transfer_to واحدة من: 'mohamed musa' أو 'eslam ghaleb' أو 'wogoud' فقط. "
    "إذا كان transfer=false فاجعل transfer_to = null. "
)


class ChatbotResponse(BaseModel):
    reply: str = Field(..., description="Reply shown to the user.")
    transfer: bool = Field(..., description="Whether the chat should be transferred to a human.")
    transfer_to: Literal["mohamed musa", "eslam ghaleb", "wogoud"] | None = Field(
        default=None,
        description="The human owner for the transfer, or null when no transfer is needed.",
    )


FORCED_ATTACHMENT_TRANSFER_REPLY = (
    "شكرًا لتواصلك معنا. سأقوم بتحويل طلبك للمتابعة مع فريقنا المختص لمراجعة المحتوى. يُرجى الانتظار للحظة."
)
ATTACHMENT_SUMMARY_MARKER = "ملخص موجز للمرفق:"


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
        self.attachment_summary_tool = AttachmentSummaryTool(self.llm)
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

    def _extract_attachment_summary(self, message: str) -> str | None:
        if ATTACHMENT_SUMMARY_MARKER not in message:
            return None

        summary = message.split(ATTACHMENT_SUMMARY_MARKER, maxsplit=1)[1].strip()
        return summary or None

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
            response.reply = "شكرًا لك، تم استلام طلبك."
        elif response.transfer and self._is_attachment_transfer_message(message):
            attachment_summary = self._extract_attachment_summary(message)
            if attachment_summary:
                response.reply = f"{attachment_summary}\n\n{FORCED_ATTACHMENT_TRANSFER_REPLY}"
            else:
                response.reply = FORCED_ATTACHMENT_TRANSFER_REPLY

        self.memory.add_user_message(message)
        self.memory.add_ai_message(response.reply)
        return response
