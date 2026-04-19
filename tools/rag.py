import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

from database import db
from database.models import KnowledgeChunk as KnowledgeChunkModel
from settings import settings

PDF_FILES = [
    "Mnafie legal profile .pdf",
    "mlg-ai-training.pdf",
]

MANUAL_KNOWLEDGE_SOURCES = [
    {
        "title": "MLG Knowledge Base",
        "source": "manual_knowledge",
        "content": (
            "نبذة عن مجموعة منافع القانونية (MLG) شركة محاماة قانونية متخصصة، مقرها الشرق بمدينة الكويت. "
            "يقدم منظومة متكاملة من الخدمات للقطاعين الخاص والعام، موزعة على محاور: "
            "المحور الأول: الشركات والاستثمار. "
            "تأسيس الشركات وتعديلاتها وفق قانون الشركات رقم 1 لسنة 2016. "
            "إعداد النظام الأساسي للشركات المقفلة. "
            "الاندماج والاستحواذ M&A. "
            "أسواق المال والأوراق المالية وفق قانون رقم 7 لسنة 2010. "
            "الاستثمار الأجنبي والتراخيص وفق قانون رقم 116 لسنة 2013 وإصلاحات 2024. "
            "المشروعات الصغيرة والمتوسطة والصندوق الوطني. "
            "إعداد السياسات الداخلية والحوكمة. "
            "المحور الثاني: العقارات والمحافظ العقارية. "
            "إدارة المحافظ العقارية للمؤسسات شبه الحكومية والخاصة. "
            "هيكلة صناديق الاستثمار العقاري REITs. "
            "فرز وتجنيب العقارات المشاعة وتوزيع التركات. "
            "الحجوزات على الشركات والأسهم والحصص. "
            "التصفية القضائية للشركات مع مصف قضائي معتمد. "
            "المحور الثالث: القضاء والتقاضي. "
            "التحكيم التجاري ومراكز دولية مع محكمين معتمدين. "
            "قضايا العمال وفق قانون رقم 6 لسنة 2010. "
            "قضايا الأسرة والأحوال الشخصية. "
            "القانون المدني وفق مرسوم رقم 67 لسنة 1980. "
            "إجراءات التنفيذ المدني. "
            "القضايا الإدارية والمناقصات الحكومية. "
            "القضايا الجنائية. "
            "نزاعات قطاع الطيران. "
            "صياغة العقود ومراجعتها. "
            "المحور الرابع: الاستشارات والخدمات الرقمية. "
            "آراء قانونية مكتوبة موثقة بالمراجع التشريعية. "
            "الخدمات القانونية الإلكترونية المعتمدة والتوثيق الرقمي. "
            "برامج التدريب والتأهيل القانوني. "
            "المحور الخامس: الخدمات الناشئة. "
            "الامتثال ومكافحة غسيل الأموال AML/CFT. "
            "ضريبة الشركات متعددة الجنسيات MNE وفق OECD/BEPS 2024. "
            "التكنولوجيا المالية FinTech وتراخيص CBK. "
            "مراقبة الاندماجات وقانون المنافسة. "
            "الاستدامة والحوكمة البيئية ESG. "
            "مشاريع البنية التحتية والشراكة العامة الخاصة PPP. "
            "إعادة الهيكلة المالية والإفلاس. "
            "الملكية الفكرية والتعاملات الرقمية. "
            "نطاق الخدمة الجغرافي: الكويت. "
            "معلومات التواصل: الهاتف 2220 9856 965+. "
            "العنوان: الكويت الشرق قطعة 5 شارع الشهداء برج الداون تاون الدور 5 مقابل بنك الوطني الكويتي. "
            "الموقع على الخريطة: https://maps.app.goo.gl/PiGZEskWNNkrCQft9. "
            "أوقات العمل: الأحد إلى الخميس من 9 صباحاً إلى 5 مساءً. "
            "الموقع الإلكتروني: mnafielaw.com. "
            "رابط التواصل الكامل: https://linko.page/rg3udkcmifss. "
            "متى يرسل رابط linko.page للعميل؟ "
            "يرسل الرابط تلقائياً عندما يطلب العميل طريقة التواصل أو بيانات الشركة. "
            "يرسل عندما يسأل عن الموقع أو رقم الهاتف أو وسائل التواصل. "
            "ويرسل في نهاية أي محادثة يحول فيها الطلب إلى المختص. "
            "نص الرد الموحد عند إرسال الرابط: تجد جميع بيانات التواصل الخاصة بنا في الرابط التالي: https://linko.page/rg3udkcmifss."
        ),
    },
    {
        "title": "MLG Approved Marketing Response Template",
        "source": "approved_marketing_response_v1",
        "content": (
            "هذا قالب تسويقي معتمد يستخدم عندما يكون العميل جديدًا ويسأل عن الخدمات أو نبذة الشركة "
            "أو المنتجات أو العروض أو التعاقد أو عرض السعر أو تأسيس الشركات أو أي خدمة قانونية جديدة. "
            "استخدم القالب التالي كما هو مع الحفاظ على ترتيب الأسطر: "
            "[LINE] أهلاً وسهلاً بكم "
            "[LINE] [LINE] يسعدنا تواصلكم مع مجموعة منافع القانونية - MLG "
            "[LINE] [LINE] نقدم الحماية القانونية الشاملة لجميع الكيانات "
            "[LINE] [LINE] للأفراد والشركات والمستثمرين . "
            "[LINE] [LINE] بكفاءة واحترافية عالية . "
            "[LINE] [LINE] متخصصون في: "
            "[LINE] [LINE] • الشركات التجارية والاستثمار "
            "[LINE] [LINE] • العقارات والمحافظ العقارية "
            "[LINE] [LINE] • التحصيل الودي والتنفيذ "
            "[LINE] [LINE] • صياغة العقود ومراجعتها "
            "[LINE] [LINE] • التقاضي والتحكيم "
            "[LINE] [LINE] • خدمات الشركات الناشئة "
            "[LINE] [LINE] سيتواصل معكم أحد مستشارينا "
            "[LINE] [LINE] لمناقشة طلبكم بالتفصيل إن شاء الله"
        ),
    },
]

EMBEDDING_MODEL = settings.OPENAI_EMBEDDING_MODEL
VECTOR_DIMENSION = settings.OPENAI_EMBEDDING_DIMENSIONS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_VECTOR_SCORE = 0.2
KEYWORD_SCORE_WEIGHT = 0.08
TOP_K_MATCHES = 3


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _make_chunks(text: str) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, 0)
    return chunks


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text).lower()
    return [token for token in re.findall(r"[\w\u0600-\u06FF]+", normalized) if len(token) > 1]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass
class KnowledgeChunk:
    title: str
    content: str
    source: str
    page: int
    chunk_index: int
    embedding: List[float] = field(default_factory=list)


class MLGKnowledgeRAGTool:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.embedding_client = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            dimensions=VECTOR_DIMENSION,
        )
        self.chunks: List[KnowledgeChunk] = []

    def _pdf_paths(self) -> List[Path]:
        return [self.project_root / file_name for file_name in PDF_FILES]

    def _chunks_from_text(self, title: str, source: str, text: str, page: int) -> List[KnowledgeChunk]:
        chunk_texts = _make_chunks(_normalize_text(text))
        return [
            KnowledgeChunk(
                title=title,
                content=chunk_text,
                source=source,
                page=page,
                chunk_index=chunk_index,
            )
            for chunk_index, chunk_text in enumerate(chunk_texts)
        ]

    def _build_seed_chunks(self) -> List[KnowledgeChunk]:
        chunks: List[KnowledgeChunk] = []

        for pdf_path in self._pdf_paths():
            if not pdf_path.exists():
                continue

            reader = PdfReader(str(pdf_path))
            for page_index, page in enumerate(reader.pages, start=1):
                page_text = _normalize_text(page.extract_text() or "")
                if not page_text:
                    continue
                chunks.extend(
                    self._chunks_from_text(
                        title=f"{pdf_path.stem} - page {page_index}",
                        source=pdf_path.name,
                        text=page_text,
                        page=page_index,
                    )
                )

        for source in MANUAL_KNOWLEDGE_SOURCES:
            chunks.extend(
                self._chunks_from_text(
                    title=source["title"],
                    source=source["source"],
                    text=source["content"],
                    page=1,
                )
            )

        return chunks

    def _load_chunks_from_db(self) -> List[KnowledgeChunk]:
        with db.session_scope() as session:
            records = (
                session.query(KnowledgeChunkModel)
                .order_by(
                    KnowledgeChunkModel.source.asc(),
                    KnowledgeChunkModel.page.asc(),
                    KnowledgeChunkModel.chunk_index.asc(),
                )
                .all()
            )
            return [
                KnowledgeChunk(
                    title=record.title,
                    content=record.content,
                    source=record.source,
                    page=record.page,
                    chunk_index=record.chunk_index,
                    embedding=json.loads(record.embedding_json),
                )
                for record in records
            ]

    def _persist_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        if not chunks:
            return

        payloads = [
            KnowledgeChunkModel(
                source=chunk.source,
                title=chunk.title,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding_json=json.dumps(chunk.embedding),
            )
            for chunk in chunks
        ]

        with db.session_scope() as session:
            session.add_all(payloads)

    def _sync_missing_sources(self) -> None:
        existing_chunks = self._load_chunks_from_db()
        existing_sources = {chunk.source for chunk in existing_chunks}

        source_chunks = self._build_seed_chunks()
        missing_chunks = [chunk for chunk in source_chunks if chunk.source not in existing_sources]

        if missing_chunks:
            embeddings = self.embedding_client.embed_documents(
                [f"{chunk.title}\n{chunk.content}" for chunk in missing_chunks]
            )
            for chunk, embedding in zip(missing_chunks, embeddings):
                chunk.embedding = embedding

            self._persist_chunks(missing_chunks)

        self.chunks = self._load_chunks_from_db()
        if not self.chunks:
            raise ValueError("Knowledge base is empty. Seed the database with knowledge chunks first.")

    def ensure_ready(self) -> None:
        if self.chunks:
            return
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

        self._sync_missing_sources()

    def _rank_chunks(self, query: str, k: int = TOP_K_MATCHES) -> List[dict]:
        self.ensure_ready()

        query_terms = set(_tokenize(query))
        query_embedding = self.embedding_client.embed_query(query)

        ranked = []
        for chunk in self.chunks:
            chunk_terms = set(_tokenize(f"{chunk.title} {chunk.content}"))
            keyword_score = float(len(query_terms & chunk_terms))
            vector_score = _cosine_similarity(query_embedding, chunk.embedding)
            combined_score = vector_score + (keyword_score * KEYWORD_SCORE_WEIGHT)

            if keyword_score <= 0 and vector_score < MIN_VECTOR_SCORE:
                continue

            ranked.append(
                {
                    "title": chunk.title,
                    "content": chunk.content,
                    "source": chunk.source,
                    "page": chunk.page,
                    "keyword_score": keyword_score,
                    "vector_score": round(vector_score, 4),
                    "combined_score": round(combined_score, 4),
                }
            )

        ranked.sort(key=lambda item: item["combined_score"], reverse=True)
        return ranked[:k]

    def retrieve_context(self, query: str) -> str:
        matches = self._rank_chunks(query)
        if not matches:
            return "NO_RELEVANT_CONTEXT"

        return "\n\n".join(
            f"[{match['title']} | source={match['source']} | page={match['page']}] {match['content']}"
            for match in matches
        )


rag_tool = MLGKnowledgeRAGTool()


@tool("search_mlg_knowledge")
def search_mlg_knowledge(query: str) -> str:
    """
    Search the official MLG knowledge base extracted from persisted knowledge chunks.
    Use this for factual answers about company services, contact information, office hours, legal areas, and official offerings.
    """
    return rag_tool.retrieve_context(query)
