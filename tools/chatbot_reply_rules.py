from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from settings import settings


FORCED_ATTACHMENT_TRANSFER_REPLY = (
    "سيتم تحويل طلبك للموظف المختص لمراجعة المحتوى."
)
ATTACHMENT_SUMMARY_MARKER = "ملخص موجز للمرفق:"
APPROVED_MARKETING_TEMPLATE_SOURCE = "source=approved_marketing_response_v1"
YES_TRANSFER_REPLY = "نعم، سيتم تحويل طلبك للموظف المختص."
NO_TRANSFER_REPLY = "لا، سيتم تحويل طلبك للموظف المختص."
TRANSFER_ONLY_REPLY = "سيتم تحويل طلبك للموظف المختص."
OUT_OF_SCOPE_REPLY = "نعتذر، هذا الطلب خارج نطاق خدماتنا القانونية."
NOT_UNDERSTOOD_REPLY = "لم نفهم طلبك بوضوح. يرجى توضيح طلبك."
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
REPEATED_REPLY_REWRITE_PROMPT = (
    "أعد صياغة reply1 بصياغة عربية مهنية قصيرة بحيث يكون مختلفًا عن reply2 "
    "لكن لا تغيّر المعنى ولا تضف معلومات جديدة. "
    "أعد النص النهائي فقط.\n\n"
    "reply1:\n{reply1}\n\n"
    "reply2:\n{reply2}"
)


def is_attachment_transfer_message(message: str) -> bool:
    return (
        message.startswith("أرسل العميل مرفقًا")
        or "رابط المرفق:" in message
        or ATTACHMENT_SUMMARY_MARKER in message
    )


def should_shorten_transfer_reply(reply: str) -> bool:
    sentence_count = sum(reply.count(mark) for mark in (".", "؟", "!", "\n"))
    return len(reply) > settings.MAX_TRANSFER_REPLY_LENGTH or sentence_count > 2


def is_out_of_scope_reply(reply: str) -> bool:
    return "خارج نطاق" in reply or "خارج خدمات" in reply


def is_not_understood_reply(reply: str) -> bool:
    return "لم نفهم" in reply or "غير مفهوم" in reply or "توضيح طلبك" in reply


def is_readable_text(message: str) -> bool:
    words = message.split()
    if len(words) < 2:
        return False
    return any(char.isalpha() for char in message)


def is_contextual_follow_up(message: str, has_history: bool) -> bool:
    if not has_history:
        return False

    stripped = message.strip()
    if not stripped:
        return False

    if is_readable_text(stripped):
        return True

    has_alpha = any(char.isalpha() for char in stripped)
    has_digit = any(char.isdigit() for char in stripped)
    is_punctuation_only = all(not char.isalnum() and not char.isspace() for char in stripped)
    has_punctuation = any(not char.isalnum() and not char.isspace() for char in stripped)
    is_short_text_follow_up = (
        has_alpha
        and not has_digit
        and has_punctuation
        and len(stripped.split()) <= 3
    )

    return is_punctuation_only or is_short_text_follow_up


def is_call_transfer_request(message: str) -> bool:
    normalized = message.strip().lower()
    if not normalized:
        return False

    call_phrases = (
        "اتصل علي",
        "اتصلوا علي",
        "اتصل بي",
        "اتصلوا بي",
        "تواصل معي",
        "تواصلوا معي",
        "كلمني",
        "كلموني",
        "ابغى احد يتصل",
        "ابي احد يتصل",
        "call me",
        "contact me",
    )
    return any(phrase in normalized for phrase in call_phrases)


def looks_like_customer_conversation(message: str) -> bool:
    return is_readable_text(message) and len(message.split()) >= 8


def uses_approved_marketing_template(knowledge_context: str) -> bool:
    return APPROVED_MARKETING_TEMPLATE_SOURCE in knowledge_context


def normalize_transfer_reply(reply: str, is_attachment: bool = False) -> str:
    normalized_reply = reply.strip()
    if is_attachment:
        return FORCED_ATTACHMENT_TRANSFER_REPLY
    if normalized_reply.startswith("نعم"):
        return YES_TRANSFER_REPLY
    if normalized_reply.startswith("لا"):
        return NO_TRANSFER_REPLY
    return TRANSFER_ONLY_REPLY


def shorten_transfer_reply(llm, reply: str) -> str:
    rewritten = llm.invoke(
        SHORT_TRANSFER_REWRITE_PROMPT.format(reply=reply)
    ).content.strip()
    return rewritten or reply


def rewrite_repeated_reply(llm, reply: str, previous_reply: str) -> str:
    rewritten = llm.invoke(
        REPEATED_REPLY_REWRITE_PROMPT.format(
            reply1=reply,
            reply2=previous_reply,
        )
    ).content.strip()
    return rewritten or reply


def build_out_of_scope_recheck_chain(system_prompt: str, structured_llm):
    recheck_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "system",
                "Recheck the previous classification because it produced an outside-scope apology. "
                "MLG scope includes customer service, staff interactions, prior replies, ongoing communication, legal services, admin/accounting matters, "
                "support, follow-up, documents, cases, contracts, companies, licenses, payments, invoices, fees, and any human-routable MLG matter. "
                "If the readable message could reasonably be part of an MLG customer conversation and is not clearly an unrelated personal/general request, "
                "correct it to transfer=true and choose the owner using the routing priorities. "
                "Only keep the outside-scope apology when the readable customer request is clearly unrelated to MLG."
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "Customer message:\n{input}\n\nPrevious reply:\n{previous_reply}"
            ),
        ]
    )
    return recheck_prompt | structured_llm
