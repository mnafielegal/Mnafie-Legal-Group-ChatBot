from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request as URLRequest
from urllib.request import urlopen

from pypdf import PdfReader

from core.chat_request import ChatRequest

MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_ATTACHMENT_TEXT_CHARS = 12000
TEXT_FILE_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".rtf",
    ".log",
}


def build_effective_message(
    chat_request: ChatRequest,
    attachment_summary_tool,
) -> str:
    message = chat_request.message.strip()
    if message:
        return message

    return build_attachment_summary_message(chat_request, attachment_summary_tool) or build_attachment_fallback_message(
        chat_request
    )


def is_supported_text_attachment(chat_request: ChatRequest) -> bool:
    attachment_type = (chat_request.attachment_type or "").strip().lower()
    attachment_url = (chat_request.attachment_url or "").strip()
    extension = Path(urlparse(attachment_url).path).suffix.lower()

    if attachment_type in {"image", "img", "audio", "voice", "video"}:
        return False
    if attachment_type in {"file", "document", "pdf", "text"}:
        return True
    if extension == ".pdf" or extension in TEXT_FILE_EXTENSIONS:
        return True

    return False


def fetch_attachment_bytes(attachment_url: str) -> tuple[bytes, str]:
    parsed_url = urlparse(attachment_url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("Only http and https attachment URLs are supported.")

    request = URLRequest(
        attachment_url,
        headers={
            "User-Agent": "MLG-Chatbot/1.0",
            "Accept": "*/*",
        },
    )
    with urlopen(request, timeout=15) as response:
        content_type = response.headers.get("Content-Type", "")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_ATTACHMENT_BYTES:
            raise ValueError("Attachment is too large to process.")

        payload = response.read(MAX_ATTACHMENT_BYTES + 1)
        if len(payload) > MAX_ATTACHMENT_BYTES:
            raise ValueError("Attachment is too large to process.")

        return payload, content_type


def extract_text_from_attachment_bytes(
    payload: bytes,
    attachment_url: str,
    content_type: str,
) -> str:
    extension = Path(urlparse(attachment_url).path).suffix.lower()
    normalized_content_type = content_type.lower()

    if "pdf" in normalized_content_type or extension == ".pdf":
        reader = PdfReader(BytesIO(payload))
        pages = [_normalize_extracted_text(page.extract_text() or "") for page in reader.pages]
        return "\n".join(page for page in pages if page).strip()

    if extension not in TEXT_FILE_EXTENSIONS and "text" not in normalized_content_type:
        if "json" not in normalized_content_type and "xml" not in normalized_content_type:
            raise ValueError("Unsupported attachment type. Only readable text files and PDFs are supported.")

    for encoding in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            return _normalize_extracted_text(payload.decode(encoding))
        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode attachment text content.")


def build_attachment_summary_message(
    chat_request: ChatRequest,
    attachment_summary_tool,
) -> str | None:
    attachment_url = (chat_request.attachment_url or "").strip()
    if not attachment_url or not is_supported_text_attachment(chat_request):
        return None

    try:
        payload, content_type = fetch_attachment_bytes(attachment_url)
        extracted_text = extract_text_from_attachment_bytes(payload, attachment_url, content_type)
        if not extracted_text:
            return None

        summary = attachment_summary_tool.summarize_attachment_text(
            attachment_text=extracted_text,
            attachment_type=chat_request.attachment_type,
            attachment_url=attachment_url,
        )
        if not summary:
            return None

        return (
            "أرسل العميل مرفقًا نصيًا أو ملف PDF بدون نص مرفق. "
            f"ملخص موجز للمرفق: {summary}"
        )
    except (ValueError, HTTPError, URLError):
        return None
    except Exception:
        return None

    return None


def build_attachment_fallback_message(chat_request: ChatRequest) -> str:
    attachment_parts = []
    if chat_request.attachment_type:
        attachment_parts.append(f"نوع المرفق: {chat_request.attachment_type}")
    if chat_request.attachment_url:
        attachment_parts.append(f"رابط المرفق: {chat_request.attachment_url}")

    if not attachment_parts:
        return ""

    return "أرسل العميل مرفقًا يحتاج إلى مراجعة بشرية. " + " ".join(attachment_parts)


def _normalize_extracted_text(text: str) -> str:
    return " ".join(text.split())[:MAX_ATTACHMENT_TEXT_CHARS].strip()
