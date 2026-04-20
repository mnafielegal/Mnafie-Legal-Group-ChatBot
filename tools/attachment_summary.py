from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class AttachmentSummaryTool:
    def __init__(self, llm):
        self.llm = llm
        self.attachment_summary_chain = self._build_attachment_summary_chain()

    def _build_attachment_summary_chain(self):
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "أنت مساعد داخلي لفريق خدمة العملاء في مجموعة منافع القانونية. "
                    "سيصلك نص مستخرج من ملف أرسله العميل. "
                    "اكتب ملخصًا عربيًا قصيرًا جدًا من جملتين إلى ثلاث جمل يوضح موضوع الملف والطلب الظاهر فيه. "
                    "احرص على الحفاظ صراحةً على أي كلمات أو عبارات تؤثر على قرار التحويل البشري، مثل: عميل جديد، استشارة قانونية، طلب خدمة قانونية جديدة، تأسيس شركة، تعديل شركة، ترخيص، عرض سعر، تكلفة قضية، تسويق، حملة إعلانية، إنستغرام، تعاقد. "
                    "إذا ظهر في النص ما يدل على واحدة من هذه الحالات فاذكرها بوضوح داخل الملخص ولا تستبدلها بعبارات عامة. "
                    "لا تقدّم استشارة قانونية، ولا تذكر أنك نموذج أو أنك لم تقرأ الملف كاملًا، ولا تضف مقدمات أو تعدادًا.",
                ),
                (
                    "human",
                    "نوع المرفق: {attachment_type}\n"
                    "رابط المرفق: {attachment_url}\n"
                    "النص المستخرج من المرفق:\n{attachment_text}",
                ),
            ]
        )
        return summary_prompt | self.llm | StrOutputParser()

    def summarize_attachment_text(
        self,
        attachment_text: str,
        attachment_type: str | None = None,
        attachment_url: str | None = None,
    ) -> str:
        summary = self.attachment_summary_chain.invoke(
            {
                "attachment_type": attachment_type or "unknown",
                "attachment_url": attachment_url or "",
                "attachment_text": attachment_text,
            }
        )
        return summary.strip()
