from core.chat_request import ChatRequest


def build_effective_message(chat_request: ChatRequest) -> str:
    message = chat_request.message.strip()
    if message:
        return message

    return build_attachment_fallback_message(chat_request)


def build_attachment_fallback_message(chat_request: ChatRequest) -> str:
    attachment_parts = []
    if chat_request.attachment_type:
        attachment_parts.append(f"نوع المرفق: {chat_request.attachment_type}")
    if chat_request.attachment_url:
        attachment_parts.append(f"رابط المرفق: {chat_request.attachment_url}")

    if not attachment_parts:
        return ""

    return "أرسل العميل مرفقًا يحتاج إلى مراجعة بشرية. " + " ".join(attachment_parts)
