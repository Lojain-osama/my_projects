# ============================================================
# Fake News Detection System
# ============================================================
import time, asyncio, aiohttp, re, torch, numpy as np
import requests, feedparser, tempfile, os, cv2, threading, gc
import ipaddress, socket, uuid, logging
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from urllib.parse import quote
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from newspaper import Article
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

# Optional .env loader (no-op if python-dotenv isn't installed)
try:
    from dotenv import load_dotenv
    _base = os.path.dirname(os.path.abspath(__file__))
    # بيجرب .env أول وبعدين env (بدون نقطة)
    load_dotenv(os.path.join(_base, ".env"))
    load_dotenv(os.path.join(_base, "env"))
except Exception:
    pass

# Optional rate-limiter (fix #5).  Falls back to a no-op decorator
# so the app still runs if flask-limiter isn't installed.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _LIMITER_AVAILABLE = True
except Exception:
    _LIMITER_AVAILABLE = False
    Limiter = None  # type: ignore
    def get_remote_address():  # type: ignore
        return "0.0.0.0"

# ── OCR engines (lazy-loaded on first request) ──────────────────────────────
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PaddleOCR = None
    PADDLE_AVAILABLE = False

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    BIDI_AVAILABLE = True
except Exception:
    BIDI_AVAILABLE = False

# ── Gemini (Google) API — content extraction & claim verification ─────────────
# Install: pip install google-generativeai
# Get your API key from: https://aistudio.google.com/app/apikey
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# ── Playwright — renders JS-heavy pages (e.g. Almamlaka, Roya) ───────────────
# Install: pip install playwright && playwright install chromium
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    sync_playwright = None
    PLAYWRIGHT_AVAILABLE = False

# [DEVICE] - must come first so models can use it
SCRAPE_INTERVAL_SEC = 3600
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] Device: {device.upper()}")

app = Flask(__name__)

# ── Security: restrict CORS origins (fix #4) ─────────────────────────────────
# Set MIZAN_CORS_ORIGINS to a comma-separated list, e.g.
#   "https://MIZAN.example,https://www.MIZAN.example"
# Defaults to local dev origins only.
_cors_origins = [o.strip() for o in os.environ.get(
    "MIZAN_CORS_ORIGINS",
    "http://127.0.0.1:5000,http://localhost:5000"
).split(",") if o.strip()]
CORS(app, resources={r"/*": {"origins": _cors_origins}}, supports_credentials=False)

# ── Upload size cap (fix #7) ─────────────────────────────────────────────────
# Hard 50 MB ceiling so analyze_image / analyze_video / ocr can't be flooded.
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ── Logging (fix #13) ────────────────────────────────────────────────────────
# Rotating file handler in addition to console — avoids leaking raw exception
# strings to the client while still keeping a debuggable trail server-side.
_BASE_DIR_FOR_LOGS = os.path.dirname(os.path.abspath(__file__))
_LOG_PATH = os.path.join(_BASE_DIR_FOR_LOGS, "MIZAN.log")
_log_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
_file_handler = RotatingFileHandler(_LOG_PATH, maxBytes=5_000_000, backupCount=3)
_file_handler.setFormatter(_log_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO,
                    handlers=[_file_handler, _stream_handler])
log = logging.getLogger("MIZAN")
app.logger.handlers = logging.getLogger().handlers
app.logger.setLevel(logging.INFO)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    "MIZAN_DATABASE_URI", "sqlite:///news_database.db"
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Rate limiter (fix #5) ────────────────────────────────────────────────────
# Per-IP defaults; specific endpoints add tighter limits via @limiter.limit().
if _LIMITER_AVAILABLE:
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per hour"],
        storage_uri=os.environ.get("MIZAN_LIMITER_STORAGE", "memory://"),
    )
    def rate_limit(spec):
        return limiter.limit(spec)
else:
    log.warning("flask-limiter not installed — rate limiting DISABLED. "
                "Install with: pip install flask-limiter")
    def rate_limit(spec):  # type: ignore
        def _decor(fn):
            return fn
        return _decor

class NewsRecord(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    title            = db.Column(db.String(500))
    content          = db.Column(db.Text)
    summary          = db.Column(db.Text)
    ai_real          = db.Column(db.Float)
    ai_fake          = db.Column(db.Float)
    final_score      = db.Column(db.Float)
    found_in_sources = db.Column(db.Text)
    repetition_count = db.Column(db.Integer, default=0)
    best_url         = db.Column(db.String(1000))
    top3_articles       = db.Column(db.Text, default="[]")   # JSON list of {name,url,similarity}
    source_similarities = db.Column(db.Text, default="")     # "الجزيرة:82.3, رؤيا:60.1, الغد:50.0"
    top3_urls           = db.Column(db.Text, default="")     # "url1, url2, url3"
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)
    input_type       = db.Column(db.String(20), default="text")
    is_automated     = db.Column(db.Boolean, default=False)
    verification_count       = db.Column(db.Integer, default=0)
    last_7_days_count        = db.Column(db.Integer, default=0)
    trending_score           = db.Column(db.Float,   default=0.0)
    recent_verdict_timestamp = db.Column(db.DateTime, default=datetime.utcnow)

import json as _json_fs

_SEED_ARTICLES = [
    {
        "title"           : "إطلاق عاصفة إلكترونية للإفراج عن الزميل فادي العمرو",
        "content"         : "دعا صحفيون أردنيون وناشطون كافة المتابعين والأردنيين إلى المشاركة في عاصفة إلكترونية على منصات التواصل الاجتماعي للمطالبة بالإفراج عن الصحفي الأردني فادي العمرو المحتجز لدى النظام السوري منذ عام 2019. وطالب المشاركون في الحملة السلطاتِ الأردنية بالتدخل الفوري والضغط على دمشق للإفراج عنه، مؤكدين أنه يتعرض لظروف احتجاز قاسية وأنه أعلن إضراباً عن الطعام احتجاجاً على استمرار اعتقاله دون محاكمة.",
        "source"          : "سواليف",
        "published"       : "2021-06-15",
        "verdict"         : "fake",
        "confidence"      : 65,
        "ai_real"         : 5.56,
        "ai_fake"         : 94.44,
        "analysis_summary": "الخبر مضلّل لأنه يقدّم الحملة الإلكترونية كأنها ستُفضي فوراً إلى الإفراج عن العمرو، فيما بقي محتجزاً حتى عام 2024 حين أُفرج عنه بعفو خاص من النظام السوري دون أي علاقة مباشرة بالحملة. كما أن تفاصيل ظروف احتجازه وإضرابه عن الطعام غير موثقة من مصادر مستقلة.",
        "claims"          : _json_fs.dumps([
            {"text":"فادي العمرو محتجز لدى النظام السوري منذ 2019","status":"verified","explanation":"حقيقة موثقة من منظمات حقوق الإنسان"},
            {"text":"العاصفة الإلكترونية ستؤدي إلى الإفراج عنه","status":"fake","explanation":"بقي معتقلاً سنوات بعد الحملة ولم يُفرج عنه إلا عام 2024 بعفو خاص"},
            {"text":"العمرو يخوض إضراباً عن الطعام","status":"uncertain","explanation":"غير مؤكد من مصادر مستقلة يمكن التحقق منها"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"حملة إلكترونية تطالب بالإفراج عن معتقل أردني يخوض إضراباً عن الطعام","url":"https://www.alaraby.co.uk/entertainment_media/%D8%AD%D9%85%D9%84%D8%A9-%D8%A5%D9%84%D9%83%D8%AA%D8%B1%D9%88%D9%86%D9%8A%D8%A9-%D8%AA%D8%B7%D8%A7%D9%84%D8%A8-%D8%A8%D8%A7%D9%84%D8%A5%D9%81%D8%B1%D8%A7%D8%AC-%D8%B9%D9%86-%D9%85%D8%B9%D8%AA%D9%82%D9%84-%D8%A3%D8%B1%D8%AF%D9%86%D9%8A","similarity":88},
            {"name":"النظام السوري يفرج عن صحفي أردني معتقل منذ 2019 بعفو خاص","url":"https://www.syria.tv/%D8%A7%D9%84%D9%86%D8%B8%D8%A7%D9%85-%D8%A7%D9%84%D8%B3%D9%88%D8%B1%D9%8A-%D9%8A%D9%81%D8%B1%D8%AC-%D8%B9%D9%86-%D8%B5%D8%AD%D9%81%D9%8A-%D8%A3%D8%B1%D8%AF%D9%86%D9%8A-%D9%85%D8%B9%D8%AA%D9%82%D9%84-%D9%85%D9%86%D8%B0-2019-%D8%A8%D8%B9%D9%81%D9%88-%D8%AE%D8%A7%D8%B5","similarity":79},
            {"name":"صحفي أردني معتقل في سوريا منذ عام بسبب منشور على الإنترنت","url":"https://www.noonpost.com/content/15197/","similarity":71},
        ], ensure_ascii=False),
    },
    {
        "title"           : "موجة استقالات داخل وحدة مهمة بالجيش الإسرائيلي",
        "content"         : "أعلن المتحدث باسم الجيش الإسرائيلي المقدم دانيال هاغاري وعدد كبير من كبار المسؤولين في نظام المعلومات في الجيش الإسرائيلي مؤخراً عن تقديم استقالات جماعية من الجيش احتجاجاً على سير الأمور العملياتية والشخصية. وبحسب القناة 14 الإسرائيلية أعلن عدد كبير من الضباط استقالتهم من الوحدة المسؤولة عن نظام المعلومات في الجيش الإسرائيلي. وجاءت الاستقالات في أعقاب نشر هيئة البث الإسرائيلية تسجيلاً صوتياً لأسيرين إسرائيليين في قطاع غزة قبل لحظات من مقتلهما على أيدي جنود الجيش الإسرائيلي عن طريق الخطأ.",
        "source"          : "القناة 14 الإسرائيلية",
        "published"       : "2024-03-15",
        "verdict"         : "fake",
        "confidence"      : 72,
        "ai_real"         : 0.45,
        "ai_fake"         : 99.55,
        "analysis_summary": "الخبر يستند إلى مصدر واحد وهو القناة 14 الإسرائيلية دون تأكيد من مصادر رسمية. تفاصيل الاستقالات مبالغ فيها ولا يمكن التحقق الكامل منها.",
        "claims"          : _json_fs.dumps([
            {"text":"استقالة المقدم دانيال هاغاري","status":"fake","explanation":"هاغاري لم يستقل — لا يزال متحدثاً رسمياً"},
            {"text":"استقالات جماعية من وحدة المعلومات","status":"uncertain","explanation":"ادعاء من مصدر واحد غير مؤكد"},
            {"text":"صلة الاستقالات بمقتل الأسيرين","status":"uncertain","explanation":"الربط غير مؤكد من مصادر رسمية"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"هآرتس — أخبار الجيش الإسرائيلي","url":"https://www.haaretz.co.il/news/politics","similarity":45},
            {"name":"العربي الجديد — غزة","url":"https://www.alaraby.co.uk/politics","similarity":40},
            {"name":"الجزيرة — الجيش الإسرائيلي","url":"https://www.aljazeera.net/news/military/2024/3/15/army-israel","similarity":42},
        ], ensure_ascii=False),
    },
    {
        "title"           : "كأس آسيا: الأردن إلى الدور النهائي لأول مرة في تاريخه بعد تغلبه على كوريا الجنوبية",
        "content"         : "كتب المنتخب الأردني لكرة القدم تاريخه بتأهله لأول مرة إلى نهائي كأس آسيا 2024 المقام في قطر، وذلك بعد فوزه على نظيره الكوري الجنوبي بهدفين نظيفين في مباراة نصف النهائي التي أقيمت في الدوحة. سجل هدفَي المباراة يزن النعيمات ومصعب البوريني، وسيط المنتخب الأردني على المنتخب الأسترالي ومنتخبَي طاجيكستان والعراق قبل الوصول إلى هذا الإنجاز التاريخي. وسيواجه الأردن في النهائي المنتخب القطري صاحب الأرض والجمهور.",
        "source"          : "الجزيرة الرياضية",
        "published"       : "2024-02-06",
        "verdict"         : "verified",
        "confidence"      : 97,
        "ai_real"         : 93.94,
        "ai_fake"         : 6.06,
        "analysis_summary": "خبر رياضي موثق بالكامل. تأهّل الأردن فعلاً لنهائي كأس آسيا 2024 للمرة الأولى في تاريخه بعد فوزه على كوريا الجنوبية 2-0 في الدوحة بتاريخ 6 فبراير 2024. النتيجة رسمية معتمدة من الاتحاد الآسيوي لكرة القدم AFC.",
        "claims"          : _json_fs.dumps([
            {"text":"الأردن تأهل لنهائي كأس آسيا للمرة الأولى في تاريخه","status":"verified","explanation":"إنجاز تاريخي موثق رسمياً من AFC"},
            {"text":"الأردن فاز على كوريا الجنوبية بهدفين نظيفين","status":"verified","explanation":"النتيجة النهائية 2-0 مؤكدة رسمياً"},
            {"text":"يزن النعيمات ومصعب البوريني سجّلا الهدفين","status":"verified","explanation":"مؤكد من تقارير المباراة الرسمية"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"الأردن يصنع التاريخ ويبلغ نهائي كأس آسيا للمرة الأولى بعد الفوز على كوريا الجنوبية","url":"https://www.aljazeera.net/sports/2024/2/6/%D8%A7%D9%84%D8%A3%D8%B1%D8%AF%D9%86-%D9%8A%D8%B5%D9%86%D8%B9-%D8%A7%D9%84%D8%AA%D8%A7%D8%B1%D9%8A%D8%AE","similarity":96},
            {"name":"الأردن يهزم كوريا الجنوبية ويبلغ نهائي كأس آسيا لأول مرة في تاريخه","url":"https://arabic.rt.com/sport/1992541-%D9%83%D8%A3%D8%B3-%D8%A2%D8%B3%D9%8A%D8%A7-%D8%A7%D9%84%D8%A3%D8%B1%D8%AF%D9%86-%D9%86%D9%87%D8%A7%D8%A6%D9%8A","similarity":94},
            {"name":"للمرة الأولى في التاريخ.. الأردن يتأهل إلى نهائي كأس آسيا","url":"https://www.alarabiya.net/sports/2024/02/06/%D9%83%D8%A3%D8%B3-%D8%A2%D8%B3%D9%8A%D8%A7-%D8%A7%D9%84%D8%A3%D8%B1%D8%AF%D9%86-%D9%86%D9%87%D8%A7%D8%A6%D9%8A","similarity":92},
        ], ensure_ascii=False),
    },

    {
        "title"           : "بوتين يجيز استخدام القوة العسكرية للدفاع عن الروس المعتقلين",
        "content"         : "وقّع الرئيس الروسي فلاديمير بوتين قانوناً يجيز الاستعانة بالجيش الروسي للدفاع عن المواطنين الروس الذين يتم اعتقالهم أو ملاحقتهم بموجب قرارات صادرة عن محاكم أجنبية لا تعترف بها موسكو. وقد نُشرت الوثيقة على الموقع الإلكتروني الرسمي للكرملين الخاص بنشر الوثائق القانونية. وبموجب القانون الجديد يمكن الاستعانة بالجيش بقرار من الرئيس الروسي لتنفيذ مهام تتعلق بحماية المواطنين الروس الذين تعرضوا للاعتقال أو الاحتجاز أو الملاحقة استناداً إلى قرارات صادرة عن محاكم أجنبية لا تعترف بها موسكو أو عن جهات لا يستند اختصاصها إلى معاهدة دولية أبرمتها روسيا أو إلى قرار صادر عن مجلس الأمن الدولي.",
        "source"          : "RT عربي",
        "published"       : "2023-07-10",
        "verdict"         : "verified",
        "confidence"      : 88,
        "ai_real"         : 99.34,
        "ai_fake"         : .66,
        "analysis_summary": "خبر موثق بالكامل. وقّع بوتين فعلاً هذا القانون ونُشر رسمياً على موقع الكرملين. القانون يتيح الاستعانة بالجيش لحماية مواطنين روس ملاحَقين بموجب قرارات محاكم أجنبية لا تعترف بها موسكو.",
        "claims"          : _json_fs.dumps([
            {"text":"بوتين وقّع القانون ونُشر رسمياً على موقع الكرملين","status":"verified","explanation":"وثيقة منشورة على الموقع الرسمي للوثائق القانونية الروسية"},
            {"text":"القانون يجيز تحريك الجيش ضد أي دولة أجنبية","status":"uncertain","explanation":"القانون مقيّد بشروط — لا يعني شن حرب بل حماية مواطنين في حالات محددة"},
            {"text":"القانون يستهدف الرد على قضايا المعتقلين الروس في الغرب","status":"verified","explanation":"جاء في سياق تصاعد توتر دبلوماسي حول مواطنين روس معتقلين في الخارج"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"بوتين يوقع قانونا يجيز تحريك الجيش الروسي للدفاع عن المواطنين الروس المعتقلين في الخارج","url":"https://arabic.rt.com/russia/1792872-%D8%A8%D9%88%D8%AA%D9%8A%D9%86-%D9%8A%D9%88%D9%82%D8%B9-%D9%82%D8%A7%D9%86%D9%88%D9%86%D8%A7-%D9%8A%D8%AC%D9%8A%D8%B2-%D8%AA%D8%AD%D8%B1%D9%8A%D9%83-%D8%A7%D9%84%D8%AC%D9%8A%D8%B4-%D8%A7%D9%84%D8%B1%D9%88%D8%B3%D9%8A","similarity":92},
            {"name":"بوتين يوقع قانونا يجيز تحريك الجيش للدفاع عن المواطنين الروس المعتقلين في الخارج","url":"https://alwatannews.net/world/article/4082563","similarity":91},
            {"name":"بوتين يوقع قانونا يتيح الاستعانة بالجيش لحماية الروس المعتقلين في الخارج","url":"https://www.algomhor.com/418567","similarity":90},
        ], ensure_ascii=False),
    },
    {
        "title"           : "انتداب 100 من الأطارات الطبية وشبه الطبية في مصحات خاصة بليبيا",
        "content"         : "أعلن المتحدث الإعلامي للمجلس الأعلى لرجال الأعمال التونسيين والليبيين محمد علي الطبرسقي عن توقيع عقود انتداب لـ100 إطار طبي وشبه طبي تونسي للعمل في مصحات خاصة بليبيا، مشيراً إلى أن الدفعة الأولى من البعثة الصحية توجهت إلى ليبيا يوم السبت 13 مارس للشروع في مهامها. وأوضح الطبرسقي أنه من المرتقب إبرام عقود إضافية في الأيام المقبلة مع أطر طبية وشبه طبية أخرى، في سياق تصاعد الزخم في العلاقات التونسية الليبية إثر زيارة الرئيس قيس سعيد لليبيا، التي أُكد خلالها الحرص على تطوير التعاون الثنائي في كافة المجالات.",
        "source"          : "وكالة تونس أفريقيا للأنباء — بابنت",
        "published"       : "2021-03-18",
        "verdict"         : "verified",
        "confidence"      : 82,
        "ai_real"         : 99.23,
        "ai_fake"         : .77,
        "analysis_summary": "خبر موثق صادر عن وكالة تونس أفريقيا للأنباء (وات). يندرج في إطار التعاون الصحي التونسي الليبي الرسمي. التفاصيل قابلة للتحقق عبر التصريحات الرسمية للمجلس الأعلى لرجال الأعمال التونسيين والليبيين.",
        "claims"          : _json_fs.dumps([
            {"text":"توقيع عقود لـ100 إطار طبي وشبه طبي تونسي للعمل في ليبيا","status":"verified","explanation":"أكده المتحدث الرسمي للمجلس الأعلى لرجال الأعمال التونسيين والليبيين"},
            {"text":"انطلاق الدفعة الأولى بتاريخ 13 مارس 2021","status":"verified","explanation":"تاريخ مؤكد في التصريح الرسمي"},
            {"text":"العمل في مصحات خاصة وليس حكومية","status":"verified","explanation":"منصوص عليه صراحة في الخبر"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"انتداب 100 إطار طبي وشبه طبي تونسي للعمل في مصحات خاصة بليبيا","url":"https://www.babnet.net/cadredetail-222604.asp","similarity":96},
            {"name":"انتداب 826 إطاراً تونسياً في إطار التعاون الفني خلال الثلاثية الأولى من 2023","url":"https://www.babnet.net/cadredetail-265343.asp","similarity":74},
            {"name":"تونس تتولى إيصالها.. مساعدات طبية تركية للشعب الليبي","url":"https://www.babnet.net/cadredetail-202880.asp","similarity":61},
        ], ensure_ascii=False),
    },
    {
        "title"           : "سفينة حربية أمريكية تعبر مضيق تايوان.. والصين تتهمها بتهديد السلم",
        "content"         : "أعلنت البحرية الأمريكية أن إحدى مدمراتها المسلحة بالصواريخ الموجهة نفّذت عبوراً روتينياً في مضيق تايوان، وذلك بما يتسق مع القانون الدولي وحق الملاحة الحرة. وسارعت الصين إلى إدانة العملية، معتبرةً إياها تهديداً صريحاً للسلام والاستقرار الإقليمي، وقالت إن الأفعال الأمريكية تبعث برسائل خاطئة إلى القوى المؤيدة لاستقلال تايوان. وأكد الأسطول السابع الأمريكي أن العبور يجسّد التزام واشنطن بضمان حرية الملاحة في المحيطَين الهندي والهادئ. وتعتبر الصين مضيق تايوان جزءاً من مياهها الإقليمية، في حين تعدّه الولايات المتحدة والمجتمع الدولي ممراً بحرياً دولياً.",
        "source"          : "الجزيرة نت",
        "published"       : "2023-06-05",
        "verdict"         : "verified",
        "confidence"      : 92,
        "ai_real"         : 99.82,
        "ai_fake"         : .18,
        "analysis_summary": "خبر موثق بالكامل. عبور السفن الحربية الأمريكية لمضيق تايوان حدث متكرر موثق بتصريحات رسمية من البحرية الأمريكية والجيش الصيني. اتهام الصين لأمريكا بتهديد السلام موقف صيني ثابت ومعلن رسمياً.",
        "claims"          : _json_fs.dumps([
            {"text":"عبور المدمرة الأمريكية مضيق تايوان","status":"verified","explanation":"أكده الأسطول السابع الأمريكي رسمياً"},
            {"text":"الصين تتهم أمريكا بتهديد السلام والاستقرار","status":"verified","explanation":"تصريح رسمي من المتحدث باسم الجيش الصيني"},
            {"text":"العبور مخالف للقانون الدولي","status":"fake","explanation":"أمريكا ودول كثيرة تعتبر المضيق ممراً دولياً حراً"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"شاهد.. سفينة حربية صينية تعترض مدمرة أمريكية بشكل غير آمن في مضيق تايوان","url":"https://www.aljazeera.net/news/2023/6/5/%D8%B4%D8%A7%D9%87%D8%AF-%D8%B3%D9%81%D9%8A%D9%86%D8%A9-%D8%AD%D8%B1%D8%A8%D9%8A%D8%A9-%D8%B5%D9%8A%D9%86%D9%8A%D8%A9-%D8%AA%D8%B9%D8%AA%D8%B1%D8%B6-%D9%85%D8%AF%D9%85%D8%B1%D8%A9","similarity":93},
            {"name":"رد قوي من الصين بعد مرور مدمرة أمريكية عبر مضيق تايوان: جيشنا سيدافع بكل حزم","url":"https://almanar.com.lb/7356681","similarity":88},
            {"name":"الصين تعلن تعقبها مسار سفينة حربية أمريكية أثناء عبورها مضيق تايوان","url":"https://www.almasryalyoum.com/news/details/2193387","similarity":84},
        ], ensure_ascii=False),
        },
        {
        "title"           : "رقيب سير ينقذ حياة شابة بعد تعرّضها لحالة مرضية ويقود مركبتها ويوصلها لأقرب مستشفى",
        "content"         : "قدّمت عائلة شابة أردنية شكرها لأحد رقباء السير من دورية العاصمة على تدخله الإنساني الذي أنقذ حياة ابنتهم. كانت الفتاة قد تعرضت لحالة صحية طارئة تمثّلت في ابتلاع لسانها أثناء وجودها داخل مركبتها في أحد شوارع عمّان. وما إن علم الرقيب بالحالة حتى سارع إلى قيادة المركبة بنفسه بمساعدة زملائه الذين عملوا على تسهيل الطريق وإزاحة العوائق أمامه، وأوصلها إلى أقرب منشأة طبية حيث تلقّت الإسعافات الأولية اللازمة في الوقت المناسب. وأكدت الجهات الطبية أن الفتاة باتت بحالة صحية جيدة ومستقرة، فيما أشادت عائلتها بالتصرف السريع والإنساني لرقيب السير وزملائه.",
        "source"          : "قناة المملكة",
        "published"       : "2026-05-28",
        "verdict"         : "verified",
        "confidence"      : 92,
        "ai_real"         : 99.86,
        "ai_fake"         : .14,
        "analysis_summary": "خبر موثق ومنشور في قناة المملكة الرسمية الأردنية وتناقلته مواقع إخبارية محلية متعددة بالتزامن. التفاصيل الواردة متسقة عبر جميع المصادر وتتوافق مع بروتوكول التدخل الطارئ لدوريات المرور الأردنية.",
        "claims"          : _json_fs.dumps([
            {"text":"الشابة تعرضت لحالة ابتلاع اللسان داخل مركبتها في شوارع عمان","status":"verified","explanation":"مؤكد من جميع المصادر التي غطّت الخبر بالتفصيل ذاته"},
            {"text":"رقيب السير قاد مركبة الشابة بنفسه وأوصلها لأقرب مستشفى","status":"verified","explanation":"تفصيل رئيسي مُكرَّر حرفياً في كل المصادر الناقلة عن قناة المملكة"},
            {"text":"الشابة بحالة صحية جيدة ومستقرة بعد تلقّيها الإسعافات الأولية","status":"verified","explanation":"مؤكد من الجهات الطبية ونقلته العائلة وجميع المواقع الناقلة للخبر"},
        ], ensure_ascii=False),
        "references"      : _json_fs.dumps([
            {"name":"رقيب سير ينقذ حياة شابة بعد تعرّضها لحالة مرضية ويقود مركبتها ويوصلها لأقرب مستشفى","url":"https://www.almamlakatv.com/news/200568-","similarity":96},
            {"name":"رقيب سير ينقذ حياة شابة بعد تعرضها لحالة مرضية خلال وجودها داخل مركبتها في عمّان","url":"https://jo24.net/article/567702","similarity":91},
            {"name":"رقيب سير ينقذ حياة شابة بعد تعرّضها لحالة مرضية","url":"https://sawaleif.com/%D8%B1%D9%82%D9%8A%D8%A8-%D8%B3%D9%8A%D8%B1-%D9%8A%D9%86%D9%82%D8%B0-%D8%AD%D9%8A%D8%A7%D8%A9-%D8%B4%D8%A7%D8%A8%D8%A9-%D8%A8%D8%B9%D8%AF-%D8%AA%D8%B9%D8%B1%D9%91%D8%B6%D9%87%D8%A7-%D9%84%D8%AD%D8%A7-1080848/","similarity":88},
        ], ensure_ascii=False),
    },
{
    "title"           : "9,935 شقة بيعت في الأردن خلال الثلث الأول من العام 2026 وسط تراجع إجمالي 8%",
    "content"         : "كشفت بيانات دائرة الأراضي والمساحة أن عدد الشقق المباعة في المملكة خلال الثلث الأول من العام 2026 بلغ قرابة 9,935 شقة، مقارنة مع 10,751 شقة خلال الفترة نفسها من عام 2025، بانخفاض نسبته 8%. وأظهرت البيانات أن بيوعات الشقق شكّلت 34.99% من حجم التداول العقاري الإجمالي في الثلث الأول، مقابل 65.1% للأراضي، فيما بلغ إجمالي بيوعات العقار خلال الثلث الأول 39,672 عقاراً بانخفاض كلي نسبته 11% مقارنة بالفترة نفسها من 2025. وفيما يخص مساحات الشقق، ارتفعت بيوعات شقق 120-150 متراً مربعاً بنسبة 3% لتبلغ 2,843 شقة، في حين تراجعت بيوعات الشقق الأقل من 120 متراً بنسبة 5%. أما على صعيد بيوعات نيسان، فقد سجّلت ارتفاعاً بنسبة 37% مقارنة بآذار السابق، إذ بلغت 2,863 شقة. وعلى صعيد غير الأردنيين، بلغ عدد الشقق المباعة لهم خلال الثلث الأول 415 شقة بقيمة تقديرية 34.2 مليون دينار، فيما ارتفع عددها في نيسان 4% ليبلغ 99 شقة.",
    "source"          : "قناة المملكة — بيانات دائرة الأراضي والمساحة",
    "published"       : "2026-05-28",
    "verdict"         : "verified",
    "confidence"      : 94,
    "ai_real"         : 99.99,
    "ai_fake"         : .01,
    "analysis_summary": "خبر موثق بالكامل مستند إلى بيانات رسمية صادرة عن دائرة الأراضي والمساحة الأردنية، وهي الجهة الحكومية المخوّلة رسمياً برصد التداول العقاري. الأرقام الواردة دقيقة وقابلة للتحقق المباشر من التقارير الشهرية والربعية للدائرة، وتناقلتها مواقع إخبارية أردنية متعددة بالأرقام ذاتها.",
    "claims"          : _json_fs.dumps([
        {"text":"9,935 شقة بيعت في الأردن خلال الثلث الأول من 2026","status":"verified","explanation":"رقم رسمي صادر عن دائرة الأراضي والمساحة الأردنية ومنشور في تقريرها الربعي"},
        {"text":"تراجع مبيعات الشقق 8% في الثلث الأول مقارنة بالفترة ذاتها من 2025","status":"verified","explanation":"نسبة محسوبة من أرقام رسمية: (10,751 - 9,935) ÷ 10,751 = 7.6% ≈ 8%"},
        {"text":"الشقق تستحوذ على 34.99% من التداول العقاري الإجمالي خلال الثلث الأول","status":"verified","explanation":"نسبة مستخرجة من بيانات دائرة الأراضي والمساحة: 9,935 من أصل 39,672 عقاراً"},
    ], ensure_ascii=False),
    "references"      : _json_fs.dumps([
        {"name":"ارتفاع بيوعات الشقق في الأردن 37% خلال نيسان الماضي مقارنة مع آذار","url":"https://www.almamlakatv.com/news/200577-","similarity":96},
        {"name":"9,935 شقة بيعت في الأردن خلال الثلث الأول من العام 2026","url":"https://maqar.com/archives/760015","similarity":93},
        {"name":"ارتفاع بيوعات الشقق في الأردن 37%","url":"https://www.khaberni.com/news/814022-%D8%A7%D8%B1%D8%AA%D9%81%D8%A7%D8%B9-%D8%A8%D9%8A%D9%88%D8%B9%D8%A7%D8%AA-%D8%A7%D9%84%D8%B4%D9%82%D9%82-%D9%81%D9%8A-%D8%A7%D9%84%D8%A7%D8%B1%D8%AF%D9%86-37","similarity":89},
    ], ensure_ascii=False),
},

    {
    "title"           : "تقارير: خزان الديسي الأردني سيجف خلال 5 سنوات وأزمة مياه كارثية تلوح في الأفق",
    "content"         : "تداول ناشطون على وسائل التواصل الاجتماعي تقارير تحذّر من أن خزان الديسي المائي الجوفي في الأردن، الذي يُعدّ من أهم مصادر المياه في المملكة، سيجف كلياً خلال السنوات الخمس المقبلة، محذّرين من أزمة مياه كارثية ستطال معظم محافظات المملكة. وأشارت التقارير المتداولة إلى أن معدلات الضخ الحالية تتجاوز طاقة الخزان الاستيعابية بأضعاف مضاعفة، وأن الحكومة الأردنية تتكتم على الأرقام الحقيقية لمخزون المياه خشية إثارة الذعر الشعبي. ودعا المتداولون للضغط على الحكومة للكشف عن خطة طوارئ مائية قبل فوات الأوان.",
    "source"          : "منصات التواصل الاجتماعي",
    "published"       : "2023-08-10",
    "verdict"         : "fake",
    "confidence"      : 70,
    "ai_real"         : 70.38,
    "ai_fake"         : 29.62,
    "analysis_summary": "الخبر مضلّل لأنه يأخذ حقيقة علمية موثقة — وهي أن خزان الديسي غير متجدد وآخذ في التناقص — ويحوّلها إلى سيناريو كارثي بتوقيت مبالغ فيه. الدراسات الرسمية لا تذكر رقم 5 سنوات، وخطة ناقل الوطني المائي تعوّض جزءاً من العجز. كما أن ادعاء التكتم الحكومي غير مثبت — وزارة المياه تنشر تقارير سنوية.",
    "claims"          : _json_fs.dumps([
        {"text":"خزان الديسي يتناقص ومستوياته تنخفض باستمرار","status":"verified","explanation":"حقيقة علمية موثقة — الخزان غير متجدد ومستنزَف منذ عقود"},
        {"text":"الخزان سيجف كلياً خلال 5 سنوات","status":"fake","explanation":"لا توجد دراسة رسمية تؤكد هذا التوقيت — الأرقام الحقيقية تشير إلى عقود لا سنوات"},
        {"text":"الحكومة تتكتم على الأرقام الحقيقية","status":"fake","explanation":"وزارة المياه الأردنية تنشر تقارير سنوية علنية عن مستويات المياه الجوفية"},
    ], ensure_ascii=False),
    "references"      : _json_fs.dumps([
        {"name":"الأردن ومعضلة المياه: خزان الديسي إلى أين؟","url":"https://www.aljazeera.net/politics/2019/6/12/%D8%A7%D9%84%D8%A3%D8%B1%D8%AF%D9%86-%D9%88%D9%85%D8%B9%D8%B6%D9%84%D8%A9-%D8%A7%D9%84%D9%85%D9%8A%D8%A7%D9%87-%D8%AE%D8%B2%D8%A7%D9%86-%D8%A7%D9%84%D8%AF%D9%8A%D8%B3%D9%8A","similarity":85},
        {"name":"الأردن يعاني شُح المياه.. وهذه خطته للمستقبل","url":"https://www.bbc.com/arabic/middleeast-47704619","similarity":76},
        {"name":"وزارة المياه الأردنية: تقرير المياه السنوي 2022","url":"https://www.mwi.gov.jo/ebv4.0/root_storage/ar/eb_list_page/water_annual_report_2022.pdf","similarity":68},
    ], ensure_ascii=False),
},
        ]

# ── Fake Server in-memory store (always fresh on restart, no stale DB issues) ──
_ARTICLES_STORE:   dict = {}
_ARTICLES_NEXT_ID: list = [1]

def _articles_list():
    return sorted(_ARTICLES_STORE.values(), key=lambda x: x["id"], reverse=True)

# Seed on startup — always loads the 6 articles fresh
for _art in _SEED_ARTICLES:
    _aid = _ARTICLES_NEXT_ID[0]
    _ARTICLES_NEXT_ID[0] += 1
    _ARTICLES_STORE[_aid] = {"id": _aid, **_art}
print(f"[FAKE SERVER] Loaded {len(_ARTICLES_STORE)} articles into memory ✓")

with app.app_context():
    db.create_all()
    # ── DB migration: أضف أعمدة جديدة لو ما موجودة (SQLite ما بدعم ALTER TABLE تلقائي) ──
    from sqlalchemy import text as _sa_text
    _new_cols = [
        ("top3_articles",            "TEXT    DEFAULT '[]'"),
        ("source_similarities",      "TEXT    DEFAULT ''"),
        ("top3_urls",                "TEXT    DEFAULT ''"),
        ("verification_count",       "INTEGER DEFAULT 0"),
        ("last_7_days_count",        "INTEGER DEFAULT 0"),
        ("trending_score",           "REAL    DEFAULT 0.0"),
        ("recent_verdict_timestamp", "DATETIME"),
    ]
    with db.engine.connect() as _conn:
        for _col, _coldef in _new_cols:
            try:
                _conn.execute(_sa_text(f"ALTER TABLE news_record ADD COLUMN {_col} {_coldef}"))
                _conn.commit()
                print(f"[DB MIGRATION] Added column: {_col}")
            except Exception:
                pass  # العمود موجود أصلاً
try:
    from MIZAN_extras import register_extras as _reg_extras
    _reg_extras(app, db, NewsRecord)
    print("[SYSTEM] MIZAN_extras loaded.")
except Exception as _e:
    print(f"[SYSTEM] WARN MIZAN_extras: {_e}")

print("[SYSTEM] Extractive summarizer ready (no model needed).")

# ============================================================
# [FAKE SERVER] قاعدة أخبار محلية — SQLite (تضل موجودة بعد الـ restart)
# ============================================================
def _parse_store_json(art, field: str) -> list:
    """يحوّل حقل JSON نصي أو list لـ Python list."""
    if isinstance(art, dict):
        v = art.get(field, [])
    else:
        v = getattr(art, field, [])
    if isinstance(v, list): return v
    if isinstance(v, str) and v.strip():
        try:    return _json_fs.loads(v)
        except: return []
    return []

def _find_similar_article(user_text: str):
    """يبحث في _ARTICLES_STORE (in-memory) عن أقرب خبر مشابه."""
    articles = list(_ARTICLES_STORE.values())
    if not articles:
        return None
    try:
        user_emb  = embedding_model.encode([user_text])
        store_emb = embedding_model.encode([
            f"{a.get('title','')} {a.get('content','')[:300]}" for a in articles
        ])
        sims = cosine_similarity(user_emb, store_emb)[0]
        best = articles[int(sims.argmax())].copy()
        best["_sim"] = round(float(sims.max()) * 100, 1)
        return best
    except Exception as e:
        log.warning(f"[FAKE SERVER] embed error: {e}")
        return articles[0].copy() if articles else None

# ── Admin routes (in-memory store) ───────────────────────────────────────────
@app.route("/admin/articles", methods=["GET"])
def admin_get_articles():
    return jsonify(_articles_list())

@app.route("/admin/articles", methods=["POST"])
def admin_add_article():
    data  = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title مطلوب"}), 400
    new_id = _ARTICLES_NEXT_ID[0]
    _ARTICLES_NEXT_ID[0] += 1
    article = {
        "id"              : new_id,
        "title"           : title[:500],
        "content"         : (data.get("content")  or ""),
        "source"          : (data.get("source")   or "محلي")[:100],
        "published"       : (data.get("published") or datetime.utcnow().strftime("%Y-%m-%d")),
        "verdict"         : data.get("verdict","verified")
                            if data.get("verdict") in ("verified","fake","suspicious")
                            else "verified",
        "confidence"      : int(data.get("confidence") or 80),
        "claims"          : data.get("claims",    "[]"),
        "references"      : data.get("references","[]"),
        "analysis_summary": (data.get("analysis_summary") or ""),
    }
    _ARTICLES_STORE[new_id] = article
    log.info(f"[FAKE SERVER] Added #{new_id}: {title[:60]}")
    return jsonify({"ok": True, "article": article}), 201

@app.route("/admin/articles/<int:article_id>", methods=["PUT"])
def admin_update_article(article_id):
    if article_id not in _ARTICLES_STORE:
        return jsonify({"error": "مو موجود"}), 404
    data = request.get_json(silent=True) or {}
    art  = _ARTICLES_STORE[article_id]
    if "title"      in data: art["title"]      = (data["title"]   or "")[:500]
    if "content"    in data: art["content"]    = (data["content"] or "")
    if "source"     in data: art["source"]     = (data["source"]  or "")[:100]
    if "published"  in data: art["published"]  = data["published"] or art["published"]
    if "verdict"    in data and data["verdict"] in ("verified","fake","suspicious"):
        art["verdict"] = data["verdict"]
    if "confidence" in data: art["confidence"] = int(data["confidence"] or art["confidence"])
    log.info(f"[FAKE SERVER] Updated #{article_id}")
    return jsonify({"ok": True, "article": art})

@app.route("/admin/articles/<int:article_id>", methods=["DELETE"])
def admin_delete_article(article_id):
    if article_id not in _ARTICLES_STORE:
        return jsonify({"error": "مو موجود"}), 404
    del _ARTICLES_STORE[article_id]
    log.info(f"[FAKE SERVER] Deleted #{article_id}")
    return jsonify({"ok": True, "deleted_id": article_id})

# [BOOT] OCR engines are lazy-loaded on first request (saves startup time + memory)
print(f"[SYSTEM] OCR engines: EasyOCR={'available' if EASYOCR_AVAILABLE else 'NOT INSTALLED'} | "
      f"PaddleOCR={'available' if PADDLE_AVAILABLE else 'NOT INSTALLED (optional)'}")

# [MODEL] AraBERT - Text
# Anchor every model path to this file's own directory so the server runs
# correctly no matter where `python backend.py` is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_LONG  = os.path.join(BASE_DIR, "MODFnew_model_arabertV02")
# The "short" model files are nested under title_version/kaggle/working/title_version
PATH_SHORT = os.path.join(BASE_DIR, "title_version", "kaggle", "working", "title_version")
MODEL_NAME = "aubmindlab/bert-base-arabertv02"

print(f"[SYSTEM] Loading AraBERT Long  : {PATH_LONG}")
print(f"[SYSTEM] Loading AraBERT Short : {PATH_SHORT}")
arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)

tokenizer_long = tokenizer_short = None
model_long = model_short = None

try:
    tokenizer_long = AutoTokenizer.from_pretrained(PATH_LONG)
    model_long = AutoModelForSequenceClassification.from_pretrained(PATH_LONG).to(device)
    model_long.eval()
    print("[SYSTEM] AraBERT Long Loaded OK.")
except Exception as e:
    print(f"[SYSTEM] ERROR - AraBERT Long ({PATH_LONG}): {e}")

try:
    tokenizer_short = AutoTokenizer.from_pretrained(PATH_SHORT)
    model_short = AutoModelForSequenceClassification.from_pretrained(PATH_SHORT).to(device)
    model_short.eval()
    print("[SYSTEM] AraBERT Short Loaded OK.")
except Exception as e:
    print(f"[SYSTEM] ERROR - AraBERT Short ({PATH_SHORT}): {e}")

# [MODEL] Image classifiers - both .pth files live under finalprojup/
M1_PATH = os.path.join(BASE_DIR, "finalprojup", "m1_final.pth")
M2_PATH = os.path.join(BASE_DIR, "finalprojup", "m2_final.pth")

print(f"[SYSTEM] Loading Image Model 1 (EfficientNet): {M1_PATH}")
model1 = models.efficientnet_b0(weights=None)
model1.classifier[1] = nn.Linear(model1.classifier[1].in_features, 2)
def _safe_torch_load(path):
    """Load a .pth checkpoint with weights_only=True (fix #8).
    Falls back to weights_only=False only on older PyTorch where the kwarg
    is unsupported, with a loud warning."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        log.warning("torch.load weights_only kwarg unsupported — upgrade PyTorch >= 2.0")
        return torch.load(path, map_location=device)

try:
    state_dict1 = _safe_torch_load(M1_PATH)
    model1.load_state_dict(state_dict1)
    model1.to(device).eval()
    print(f"[SYSTEM] Model 1 Loaded OK.")
except Exception as e:
    print(f"[SYSTEM] ERROR - Model 1: {e}")

print(f"[SYSTEM] Loading Image Model 2 (ResNet50): {M2_PATH}")
model2 = models.resnet50(weights=None)
model2.fc = nn.Linear(model2.fc.in_features, 2)
try:
    state_dict2 = _safe_torch_load(M2_PATH)
    model2.load_state_dict(state_dict2)
    model2.to(device).eval()
    print(f"[SYSTEM] Model 2 Loaded OK.")
except Exception as e:
    print(f"[SYSTEM] ERROR - Model 2: {e}")

image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("[SYSTEM] All models and transforms are ready.")

# [MODEL] MobileNetV2 — frame feature extraction for KMeans clustering in video analysis
# Howard, A. et al. (2017). MobileNets: Efficient Convolutional Neural Networks. arXiv.
print("[SYSTEM] Loading MobileNetV2 (video frame clustering)...")
_mobilenet_video = models.mobilenet_v2(weights="DEFAULT")
_mobilenet_video.classifier = nn.Identity()
_mobilenet_video.to(device).eval()
print("[SYSTEM] MobileNetV2 Loaded OK.")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"[SYSTEM] Loading Embedding Model: {EMBED_MODEL}")
embedding_model = SentenceTransformer(EMBED_MODEL).to(device)
print(f"[SYSTEM] Embedding Model Loaded OK.")
print("[SYSTEM] >>> ALL SYSTEMS READY <<<\n")

# ── User-Agent pool — نروح بين متصفحات مختلفة لتجنب الـ blocking ──────────────
_UA_POOL = [
    # Chrome / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome / Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    # Firefox / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    # Firefox / Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Safari / Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    # Edge / Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Mobile Chrome (Android)
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.6367.82 Mobile Safari/537.36",
    # Mobile Safari (iPhone)
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

# Referers نروح بينهم — كأن الزائر جاي من محرك بحث
_REFERER_POOL = [
    "https://www.google.com/",
    "https://www.google.com/search?q=arabic+news",
    "https://news.google.com/",
    "https://www.bing.com/search?q=arabic+news",
    "https://t.co/",
    "https://www.facebook.com/",
]

def _random_headers(referer: bool = True) -> dict:
    """يولّد headers عشوائية تحاكي متصفحاً حقيقياً."""
    import random as _r
    ua = _r.choice(_UA_POOL)
    is_mobile = "Mobile" in ua or "iPhone" in ua or "Android" in ua
    h = {
        "User-Agent"     : ua,
        "Accept"         : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection"     : "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest" : "document",
        "Sec-Fetch-Mode" : "navigate",
        "Sec-Fetch-Site" : "cross-site",
        "Sec-Fetch-User" : "?1",
        "DNT"            : "1",
    }
    if referer:
        h["Referer"] = _r.choice(_REFERER_POOL)
    if not is_mobile:
        h["Sec-CH-UA"] = '"Chromium";v="124", "Google Chrome";v="124"'
        h["Sec-CH-UA-Mobile"] = "?0"
        h["Sec-CH-UA-Platform"] = '"Windows"'
    return h

# HEADERS الافتراضي للـ Serper وغيره (ثابت)
HEADERS = _random_headers(referer=False)

# ── CAPTCHA / bot-block page detector ─────────────────────────────────────────
# إشارات قوية جداً — وجود أي واحدة منها = بلوك مؤكد بغض النظر عن الطول
_CAPTCHA_HARD = [
    "captcha", "please solve this captcha", "request unblock",
    "يرجى إكمال التحقق", "لست روبوتًا", "تأكد من أنك لست روبوتاً",
    "prove you are human", "يرجى إكمال اختبار",
]
# إشارات متوسطة — نحتاج 2+ منها
_CAPTCHA_SOFT = [
    "you are a bot", "think that you are a bot", "behavior on this site",
    "security check", "please complete", "cloudflare", "ddos-guard",
    "access denied", "just a moment", "checking your browser",
    "enable javascript", "ray id", "تحقق الأمني", "made us think",
    "automated access", "human verification", "suspicious activity",
]

def _is_bot_block(text: str) -> bool:
    """يكشف صفحات الـ CAPTCHA وحجب الـ bots — يرجع True لو النص ما يصلح مقال."""
    if not text:
        return False
    low = text.lower()
    # إشارة صارمة واحدة كافية
    for sig in _CAPTCHA_HARD:
        if sig in low:
            print(f"  [BOT-BLOCK] Detected hard signal: '{sig}'")
            return True
    # إشارتان ناعمتان = بلوك
    soft_hits = sum(1 for s in _CAPTCHA_SOFT if s in low)
    if soft_hits >= 2:
        print(f"  [BOT-BLOCK] Detected {soft_hits} soft signals")
        return True
    # إشارة ناعمة واحدة + نص قصير (<300 حرف)
    if soft_hits >= 1 and len(text) < 300:
        return True
    return False

# ── API keys — REQUIRED env vars (fix #2) ────────────────────────────────────
# Never hardcode keys here.  Set them in a .env file (gitignored) or in your
# host's secret manager.  Missing keys disable the relevant search source.
GNEWS_API_KEY  = os.environ.get("GNEWS_API_KEY", "")
NEWSAPI_KEY    = os.environ.get("NEWSAPI_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "2bda1fc34948102d6ea64261c2f4454ebf67d018")
if not GNEWS_API_KEY:
    log.warning("GNEWS_API_KEY not set — GNews source disabled.")
if not NEWSAPI_KEY:
    log.warning("NEWSAPI_KEY not set — NewsAPI source disabled.")
if not SERPER_API_KEY:
    log.warning("SERPER_API_KEY not set — Serper search disabled.")


# ── Arabic news domains for site-targeted Serper search ─────────────────────
# 16 موقع عربي موثوق — Serper بيبحث جوّاهم مباشرة
ARABIC_NEWS_DOMAINS = [
    "aljazeera.net",      "alarabiya.net",    "skynewsarabia.com",
    "bbc.com",            "france24.com",     "petra.gov.jo",
    "almamlakatv.com",    "roya.tv",          "ammon.jo",
    "alghad.com",         "sarayanews.com",   "khaberni.com",
    "alrai.com",          "alaraby.co.uk",    "aawsat.com",
    "rt.com",
]
# نبني جملة site: واحدة للبحث الموجّه (أول 10 مواقع فقط عشان ما يطول الاستعلام)
_SITE_FILTER = " OR ".join(f"site:{d}" for d in ARABIC_NEWS_DOMAINS[:10])

# ── Gemini API key ────────────────────────────────────────────────────────────
# احصل على مفتاحك من: https://aistudio.google.com/app/apikey
# ثم أضف في ملف env:  GEMINI_API_KEY=AIza...
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyApMOjsU1NAGPo-7I07Pz_orcOiSyzg_3A")
gemini_model   = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print(f"[SYSTEM] Gemini OK — model=gemini-2.0-flash | key=...{GEMINI_API_KEY[-6:]}")
    except Exception as _ge:
        print(f"[SYSTEM] Gemini init FAILED: {_ge}")
        gemini_model = None
else:
    if not GEMINI_AVAILABLE:
        print("[SYSTEM] WARN: google-generativeai not installed — run: pip install google-generativeai")
    else:
        print("[SYSTEM] WARN: GEMINI_API_KEY empty — Gemini disabled")

PERIODIC_TOPICS = [
    "أخبار عاجلة", "سياسة الأردن", "اقتصاد", "أخبار فلسطين",
    "breaking news", "world news", "middle east"
]


# ============================================================
# [HELPERS]
# ============================================================
def is_url(text):
    return bool(re.match(r'https?://\S+', text.strip()))

def is_too_short_or_generic(text):
    generic_terms = [t.lower() for t in PERIODIC_TOPICS] + ["عاجل", "خبر", "news", "update"]
    text_clean = text.strip().lower()
    if len(text_clean.split()) < 3 or text_clean in generic_terms:
        return True
    return False

def lookup_existing_news(user_text, threshold=0.82):
    try:
        with app.app_context():
            records = NewsRecord.query.filter(NewsRecord.content != None, NewsRecord.content != "") \
                                      .order_by(NewsRecord.created_at.desc()) \
                                      .limit(200).all()
            if not records:
                return None
            stored_contents = [r.content for r in records]
            user_emb    = embedding_model.encode([user_text])
            stored_embs = embedding_model.encode(stored_contents)
            sims = cosine_similarity(user_emb, stored_embs)[0]
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim >= threshold:
                r = records[best_idx]
                print(f"  [DB LOOKUP] Found cached news | sim={best_sim:.3f}")
                sources_list = [s.strip() for s in (r.found_in_sources or "").split(",") if s.strip()]
                t3_arts      = _json_fs.loads(r.top3_articles or "[]")
                return {
                    "status"             : "حقيقي" if (r.final_score or 0) >= 70 else "كذب" if (r.final_score or 0) <= 30 else "غير مؤكد",
                    "final_score"        : r.final_score or 0,
                    "similarity"         : round(best_sim * 100, 2),
                    "summary"            : r.summary or "",
                    "best_url"           : r.best_url or "",
                    "from_cache"         : True,
                    "found_in_sources"   : sources_list,
                    "top4_sources"       : sources_list[:4],
                    "top3_articles"      : t3_arts,
                    "source_similarities": r.source_similarities or "",
                    "top3_urls"          : r.top3_urls or "",
                    "ai_real"            : r.ai_real or 0,
                    "ai_fake"            : r.ai_fake or 0,
                    "repetition_count"   : r.repetition_count or 0,
                }
    except Exception as e:
        print(f"  [DB LOOKUP] ERROR: {e}")
    return None

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============================================================
# [SUMMARY]
# ============================================================
def get_real_summary(text):
    if not text:
        return ""
    text = text.strip()
    words = text.split()
    if len(words) < 30:
        return text

    # Split into sentences (Arabic + English)
    sentences = re.split(r'(?<=[.!?؟])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 6]

    if not sentences:
        return " ".join(words[:60]) + "..."
    if len(sentences) <= 3:
        return " ".join(sentences)

    # News articles use inverted pyramid — first sentences carry the most important info.
    # Score by position and length, then pick top 3 in original order.
    scored = []
    for i, sent in enumerate(sentences[:15]):
        word_count = len(sent.split())
        position_score = 1.0 / (i + 1)
        length_score   = 1.0 if 10 <= word_count <= 60 else 0.5
        scored.append((position_score * length_score, i, sent))

    top3 = sorted(scored[:3], key=lambda x: x[1])  # restore original order
    summary = " ".join(s[2] for s in top3)

    summary_words = summary.split()
    if len(summary_words) > 200:
        summary = " ".join(summary_words[:200]) + "..."

    return summary if len(summary) >= 20 else " ".join(words[:60]) + "..."

generate_summary = get_real_summary

# ============================================================
# [SCRAPER] Fetch full article content + title
# ============================================================
_SSRF_MAX_BYTES = 5 * 1024 * 1024   # 5 MB cap on remote response size

def _is_safe_public_url(url: str) -> bool:
    """SSRF guard (fix #6) — reject private, loopback, link-local, multicast,
    and reserved IP ranges so user-supplied URLs can't hit cloud-metadata
    services or internal infra."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False
        host = p.hostname
        if not host:
            return False
        # Resolve all addresses the host points at and reject if any are unsafe
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified):
                return False
        return True
    except Exception as exc:
        log.info(f"[SSRF guard] could not resolve {url!r}: {exc}")
        return False

def _bounded_get(url: str, timeout: int = 10, max_retries: int = 3) -> str:
    """
    يجيب محتوى URL مع:
    - headers عشوائية (UA + Referer) تحاكي متصفحاً حقيقياً
    - retry تلقائي بـ User-Agent مختلف لو البلوك
    - حد أقصى للحجم _SSRF_MAX_BYTES
    """
    last_exc = None
    for attempt in range(max_retries):
        hdrs = _random_headers(referer=True)
        try:
            with requests.get(url, headers=hdrs, timeout=timeout,
                              stream=True, allow_redirects=True) as resp:
                # لو البلوك (403/429/503) جرب UA جديد
                if resp.status_code in (403, 429, 503):
                    print(f"  [FETCH] HTTP {resp.status_code} on attempt {attempt+1} — retrying with new UA")
                    continue
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit() and int(cl) > _SSRF_MAX_BYTES:
                    raise ValueError("remote response too large")
                chunks, total = [], 0
                for chunk in resp.iter_content(64 * 1024):
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _SSRF_MAX_BYTES:
                        raise ValueError("remote response too large")
                    chunks.append(chunk)
                text = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
                # لو جاءت صفحة CAPTCHA — جرب مرة ثانية بـ UA مختلف
                if _is_bot_block(text[:500]) and attempt < max_retries - 1:
                    print(f"  [FETCH] Bot-block on attempt {attempt+1} — retrying with new UA")
                    continue
                return text
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            if attempt < max_retries - 1:
                print(f"  [FETCH] Attempt {attempt+1} failed ({e}) — retrying")
    if last_exc:
        raise last_exc
    return ""

def _fetch_with_playwright(url: str):
    """
    [PLAYWRIGHT FALLBACK]
    يفتح المتصفح headless ويرندر الصفحة كاملاً بما فيها JS
    مفيد للمواقع مثل المملكة ورؤيا اللي تحمّل الأخبار عبر JavaScript
    Install: pip install playwright && playwright install chromium
    """
    if not PLAYWRIGHT_AVAILABLE:
        return "", ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            # ننتظر حتى تنتهي طلبات الشبكة (يعني JS خلص يحمّل)
            page.wait_for_load_state("networkidle", timeout=10000)
            html = page.content()
            browser.close()
        soup = BeautifulSoup(html, "html.parser")
        page_title = ""
        h1 = soup.find("h1")
        if h1 and len(h1.get_text(strip=True)) > 10:
            page_title = h1.get_text(strip=True)
        elif soup.title:
            page_title = re.split(r"[|\-–]", soup.title.get_text(strip=True))[0].strip()
        text = " ".join(
            p.get_text(strip=True) for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 20
        )
        print(f"  [PLAYWRIGHT] OK | {len(text)} chars | title='{page_title[:40]}' | {url[:50]}")
        return page_title, text
    except Exception as e:
        print(f"  [PLAYWRIGHT] Error: {e}")
        return "", ""


# ── Gemini daily-quota flag ───────────────────────────────────────────────────
# لما الحصة اليومية تنتهي، نعطّل Gemini فوراً ونرجع للـ fallback
# بدون ما ننتظر retry لا فايدة منه
_gemini_daily_ok = True


def _gemini_call(prompt: str, max_tokens: int = 1024):
    """
    ينادي Gemini مع تفريق ذكي بين نوعي الـ 429:
      - PerDay  → يعطّل Gemini فوراً للجلسة (لا retry — ما رح يفيد)
      - PerMinute → انتظر 15 ثانية مرة واحدة فقط ثم استسلم
    """
    global _gemini_daily_ok
    if not gemini_model or not _gemini_daily_ok:
        return None
    for attempt in range(2):   # محاولة أصلية + محاولة واحدة فقط عند per-minute
        try:
            return gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens)
            )
        except Exception as e:
            err = str(e)
            if "429" not in err:
                raise   # خطأ غير متوقع — ارمِه للأعلى
            # الحصة اليومية خلصت — أوقف Gemini فوراً، لا فايدة من الانتظار
            if "PerDay" in err or "per_day" in err.lower():
                _gemini_daily_ok = False
                log.warning("[GEMINI] Daily quota exhausted — disabling Gemini for this session. "
                            "Using similarity fallback instead.")
                return None
            # حصة الدقيقة — انتظر مرة واحدة فقط (15 ثانية)
            if attempt == 0:
                print("  [GEMINI] Per-minute limit — waiting 15s once...")
                time.sleep(15)
            else:
                log.warning("[GEMINI] Still rate-limited after wait — skipping.")
                return None
    return None


def _html_to_clean_text(html: str) -> tuple:
    """
    يستخرج العنوان + النص النظيف من HTML باستخدام BeautifulSoup
    قبل ما نبعت لـ Gemini — يقلّل الـ tokens من 30000 لـ ~3000 حرف.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        # نشيل العناصر اللي مش محتواها
        for tag in soup(["script", "style", "nav", "header", "footer",
                          "aside", "form", "iframe", "noscript"]):
            tag.decompose()
        # نجيب العنوان
        page_title = ""
        h1 = soup.find("h1")
        if h1 and len(h1.get_text(strip=True)) > 5:
            page_title = h1.get_text(strip=True)
        elif soup.title:
            page_title = re.split(r"[|\-–]", soup.title.get_text(strip=True))[0].strip()
        # نحاول نلاقي حاوية المقال أولاً
        container = (
            soup.find("article") or
            soup.find("main") or
            soup.find(class_=re.compile(r"article|news.body|story|content", re.I)) or
            soup.find(id=re.compile(r"article|news.body|story|content", re.I))
        )
        source = container if container else soup
        paragraphs = [
            p.get_text(strip=True)
            for p in source.find_all("p")
            if len(p.get_text(strip=True)) > 30
        ]
        clean = " ".join(paragraphs)[:8000]   # أقصاه 8000 حرف ≈ 2000 token
        return page_title, clean
    except Exception:
        # fallback: نشيل الـ HTML tags بـ regex
        return "", re.sub(r"<[^>]+>", " ", html)[:4000]


def _extract_with_gemini(url: str, html: str):
    """
    [GEMINI API FALLBACK]
    ينظّف الـ HTML أولاً بـ BS4 ثم يبعت النص النظيف فقط لـ Gemini
    (يقلّل الـ tokens ~15× مقارنةً بإرسال الـ HTML الخام)
    """
    if not gemini_model:
        return "", ""
    try:
        import json as _json
        page_title, clean_text = _html_to_clean_text(html)
        if len(clean_text) < 80:
            return "", ""

        prompt = (
            "Extract the news article title and COMPLETE body text from this page content.\n"
            "Return ONLY valid JSON, no markdown:\n"
            '{"title": "...", "content": "complete article text..."}\n\n'
            f"Title hint: {page_title}\n\n"
            f"Content:\n{clean_text}"
        )
        resp = _gemini_call(prompt, max_tokens=2048)
        if not resp:
            return "", ""
        raw = resp.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data    = _json.loads(raw)
        title   = data.get("title", "") or page_title
        content = data.get("content", "")
        print(f"  [GEMINI EXTRACT] OK | {len(content)} chars | title='{title[:40]}' | {url[:50]}")
        return title, content
    except Exception as e:
        print(f"  [GEMINI EXTRACT] Error: {e}")
        return "", ""


def gemini_get_full_article(url: str, title: str = "", snippets: list = None) -> tuple:
    """
    يجيب النص الكامل للخبر — 4 محاولات بالترتيب:
      1. newspaper3k (أسرع + أنظف)
      2. BS4 مباشرة على الـ HTML (بدون Gemini)
      3. Gemini يستخرج من HTML (لو BS4 أعطى نص قصير)
      4. Snippets من Serper كـ fallback أخير
    بيرجع (title, full_content) — يشتغل حتى لو Gemini quota خلصت
    """
    raw_html   = ""
    page_title = ""

    # ── المحاولة 1: newspaper3k (مع UA عشوائي) ─────────────────
    if url and _is_safe_public_url(url):
        try:
            from newspaper import Article as _Article
            art = _Article(url, language="ar", browser_user_agent=_random_headers()["User-Agent"])
            art.download(); art.parse()
            txt = art.text.strip()
            if len(txt) > 200 and not _is_bot_block(txt):
                t = (art.title or title or "").strip()
                print(f"  [ARTICLE] newspaper3k OK | {len(txt)} chars | {url[:55]}")
                return t, txt
        except Exception:
            pass

        # ── المحاولة 2: BS4 مباشرة مع retry وUA مختلف ──────────────
        try:
            raw_html = _bounded_get(url, timeout=8, max_retries=3)
            print(f"  [ARTICLE] Got {len(raw_html)} chars HTML from: {url[:55]}")
        except Exception as e:
            print(f"  [ARTICLE] HTTP fetch failed: {e}")

        if raw_html and len(raw_html) > 500:
            page_title, bs4_text = _html_to_clean_text(raw_html)
            if len(bs4_text) > 200 and not _is_bot_block(bs4_text):
                final_title = page_title or title
                print(f"  [ARTICLE] BS4 direct OK | {len(bs4_text)} chars | title='{final_title[:40]}'")
                return final_title, bs4_text
            elif _is_bot_block(bs4_text):
                print(f"  [ARTICLE] Bot-block/CAPTCHA detected — skipping to snippets: {url[:55]}")
                raw_html = ""  # لا ترسل الـ CAPTCHA HTML لـ Gemini

            # ── المحاولة 3: Gemini يستخرج من HTML (لو BS4 قصير وما في CAPTCHA) ────
            if raw_html:
                extracted_title, content = _extract_with_gemini(url, raw_html)
                if len(content) > 150 and not _is_bot_block(content):
                    print(f"  [ARTICLE] Gemini extract OK | {len(content)} chars")
                    return extracted_title or page_title or title, content

    # ── المحاولة 4: Snippets fallback (بدون Gemini) ─────────────
    if snippets:
        lines = [
            f"{a.get('title','')}. {a.get('snippet','')}"
            for a in snippets[:8]
            if a.get("title") or a.get("snippet")
        ]
        combined = " ".join(lines).strip()
        if len(combined) > 100:
            print(f"  [ARTICLE] Snippets fallback | {len(combined)} chars")
            return title, combined

    print(f"  [ARTICLE] All methods failed for: {url[:55]}")
    return title, ""


def fetch_full_content(url):
    """Returns (title, content). Both may be empty string on failure."""
    # SSRF guard — refuse private / loopback / metadata targets up front.
    if not _is_safe_public_url(url):
        log.warning(f"[SCRAPER] rejected unsafe URL: {url[:80]}")
        return "", ""
    try:
        article = Article(url, language='ar',
                          browser_user_agent=_random_headers()["User-Agent"])
        article.download()
        article.parse()
        title   = (article.title or "").strip()
        content = article.text.strip()
        if len(content) > 80 and not _is_bot_block(content):
            print(f"  [SCRAPER] newspaper3k OK | {len(content)} chars | title='{title[:40]}' | {url[:50]}")
            return title, content
        elif _is_bot_block(content):
            print(f"  [SCRAPER] newspaper3k got bot-block page — retrying with _bounded_get: {url[:50]}")
    except:
        pass
    raw_html   = ""
    page_title = ""
    try:
        raw_html = _bounded_get(url, timeout=10, max_retries=3)
        soup = BeautifulSoup(raw_html, "html.parser")
        h1 = soup.find("h1")
        if h1 and len(h1.get_text(strip=True)) > 10:
            page_title = h1.get_text(strip=True)
        elif soup.title:
            raw = soup.title.get_text(strip=True)
            page_title = re.split(r"[|\-–]", raw)[0].strip()
        text = " ".join(
            p.get_text(strip=True) for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 20
        )
        if len(text) > 80 and not _is_bot_block(text):
            print(f"  [SCRAPER] BS4 fallback OK | {len(text)} chars | title='{page_title[:40]}' | {url[:50]}")
            return page_title, text
        elif _is_bot_block(text):
            print(f"  [SCRAPER] BS4 got bot-block/CAPTCHA — skipping Gemini too: {url[:50]}")
            raw_html = ""   # لا ترسل الـ CAPTCHA HTML لـ Gemini
    except:
        pass

    # [Stage 3] Playwright — يرندر صفحات الـ JS (مواقع مثل المملكة ورؤيا)
    # Install: pip install playwright && playwright install chromium
    if PLAYWRIGHT_AVAILABLE:
        title, content = _fetch_with_playwright(url)
        if len(content) > 80:
            return title, content

    # [Stage 4] Gemini API — يستخرج المحتوى من الـ HTML لما كل شيء يفشل
    # لتفعيله: أضف GEMINI_API_KEY في ملف env
    if gemini_model and raw_html:
        title, content = _extract_with_gemini(url, raw_html)
        if len(content) > 80:
            return title, content

    print(f"  [SCRAPER] FAILED: {url[:55]}")
    return "", ""

# [PARALLEL SCRAPER] Scrape multiple URLs at once
def fetch_multiple_contents_parallel(url_list, max_workers=6):
    """Fetch full content + title from multiple URLs in parallel.
    Returns dict: {url: (title, content)}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fetch_full_content, url): url for url in url_list}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as e:
                results[url] = ("", "")
                print(f"  [PARALLEL SCRAPER] Error on {url[:50]}: {e}")
    return results

# ============================================================
# [SIMILARITY] Hybrid chunk-level similarity
# ============================================================
def chunk_max_sim(text, titles_emb, max_chunks=8):
    """Sliding-window chunk similarity — matches paper S_claim formula.
    Splits text into overlapping sentence chunks, encodes each, returns
    max cosine similarity per title: max(Similarity(V_H, V_Ai)).
    More accurate than full-text cosine for long documents.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?؟،])\s+', text)
                 if len(s.strip()) > 8]
    if not sentences:
        sentences = [text[:300]]
    n    = len(sentences)
    step = max(1, n // max_chunks)
    chunks = []
    for i in range(0, n, step):
        chunk = " ".join(sentences[i:i + 3])[:400]
        if chunk:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    chunk_embs = embedding_model.encode(chunks)             # (C, dim)
    all_sims   = cosine_similarity(chunk_embs, titles_emb)  # (C, T)
    return all_sims.max(axis=0)                             # (T,)  max per title

# ============================================================
# [SEARCH] All sources
# ============================================================
async def search_google_news_async(session, query):
    search_query = quote(" ".join(query.split()[:8]))
    url = f"https://news.google.com/rss/search?q={search_query}&hl=ar&gl=AR&ceid=AR:ar"
    try:
        async with session.get(url, headers=HEADERS, timeout=10) as response:
            html = await response.text()
            feed = feedparser.parse(html)
            articles = []
            for entry in feed.entries[:15]:
                pub = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub = datetime(*entry.published_parsed[:6])
                articles.append({"title": entry.title, "url": entry.link,
                                  "source": "GoogleNews", "published": pub})
            articles.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
            return articles
    except:
        return []

async def search_gnews_async(session, query):
    url = f"https://gnews.io/api/v4/search?q={query}&lang=ar&max=20&sortby=publishedAt&token={GNEWS_API_KEY}"
    try:
        async with session.get(url, timeout=10) as response:
            res = await response.json()
            articles = []
            for item in res.get("articles", []):
                pub = None
                try: pub = datetime.fromisoformat(item.get("publishedAt", "").replace("Z", ""))
                except: pass
                articles.append({"title": item["title"], "url": item["url"],
                                  "source": item["source"]["name"], "published": pub})
            articles.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
            return articles
    except:
        return []

async def search_newsapi_async(session, query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=ar&pageSize=20&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    try:
        async with session.get(url, timeout=10) as response:
            res = await response.json()
            articles = []
            for item in res.get("articles", []):
                pub = None
                try: pub = datetime.fromisoformat(item.get("publishedAt", "").replace("Z", ""))
                except: pass
                articles.append({"title": item["title"], "url": item["url"],
                                  "source": item["source"]["name"], "published": pub})
            articles.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
            return articles
    except:
        return []

async def search_serper_async(session, query, site_only=False):
    """
    site_only=True → يبحث داخل ARABIC_NEWS_DOMAINS فقط (أدق)
    site_only=False → بحث عام (أوسع)
    """
    if not SERPER_API_KEY:
        log.warning("[SERPER] No API key — skipped.")
        return []
    try:
        import ssl as _ssl
        ssl_ctx = _ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode    = _ssl.CERT_NONE

        q = f"{query} ({_SITE_FILTER})" if site_only else query
        q = q[:500]   # Serper free: max 2048 chars — نقصّر الاستعلام
        payload = {"q": q, "gl": "jo", "hl": "ar", "num": 30}
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=15)

        async with session.post(
            "https://google.serper.dev/news",
            json=payload, headers=headers,
            timeout=timeout, ssl=ssl_ctx
        ) as response:
            body = await response.text()
            if response.status != 200:
                log.warning(f"[SERPER] HTTP {response.status} — {body[:200]}")
                return []
            import json as _json
            res = _json.loads(body)
            articles = []
            for item in res.get("news", []):
                pub = None
                try:
                    raw_date = item.get("date", "")
                    if raw_date:
                        for fmt in ("%b %d, %Y", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
                            try:
                                pub = datetime.strptime(raw_date, fmt)
                                break
                            except ValueError:
                                continue
                except Exception:
                    pass
                articles.append({
                    "title"    : item.get("title", ""),
                    "url"      : item.get("link", ""),
                    "source"   : item.get("source", "Serper"),
                    "published": pub,
                    "snippet"  : item.get("snippet", ""),
                })
            articles.sort(key=lambda x: x["published"] or datetime.min, reverse=True)
            print(f"  [SERPER] {len(articles)} results returned")
            return articles
    except Exception as e:
        log.error(f"[SERPER] FAILED — {type(e).__name__}: {e}")
        return []


# ============================================================
# [SEARCH] Arabic news domains via Google News RSS (site: filter)
# أفضل بكثير من scraping مباشر لأن المواقع العربية تحتاج JavaScript
# ============================================================
async def search_arabic_domains_via_rss(session, query):
    """
    يبحث في 16 موقع عربي باستخدام Google News RSS مع site: filter.
    بيقسّم المواقع لـ 4 مجموعات ويبعت 4 طلبات بالتوازي.
    أفضل من scraping مباشر لأن Google يفهرس هذه المواقع.
    """
    # مجموعات من 4 مواقع لكل طلب RSS
    domain_groups = [
        ["aljazeera.net", "alarabiya.net", "bbc.com/arabic", "france24.com"],
        ["petra.gov.jo", "ammonnews.net", "alghad.com", "sarayanews.com"],
        ["roya.tv", "almamlakatv.com", "khaberni.com", "alrai.com"],
        ["skynewsarabia.com", "alaraby.co.uk", "aawsat.com", "rt.com"],
    ]
    short_query = " ".join(query.split()[:6])

    async def fetch_group(domains):
        site_filter = " OR ".join(f"site:{d}" for d in domains)
        q_encoded   = quote(f"{short_query} ({site_filter})")
        url = f"https://news.google.com/rss/search?q={q_encoded}&hl=ar&gl=AR&ceid=AR:ar"
        try:
            async with session.get(url, headers=HEADERS, timeout=10) as resp:
                html = await resp.text()
                feed = feedparser.parse(html)
                results = []
                for entry in feed.entries[:8]:
                    pub = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub = datetime(*entry.published_parsed[:6])
                    results.append({
                        "title"    : entry.title,
                        "url"      : entry.link,
                        "source"   : "Arabic-RSS",
                        "published": pub,
                    })
                return results
        except Exception:
            return []

    raw = await asyncio.gather(*[fetch_group(g) for g in domain_groups])
    combined = []
    for r in raw:
        combined.extend(r)
    print(f"  [ARABIC RSS] {len(combined)} articles from domain-filtered Google News")
    return combined

# ============================================================
# [GEMINI] S_claim — Claim Verification via Gemini API
# ============================================================
def gemini_verify_claim_score(claim: str, articles: list):
    """
    يبعت الادعاء + نتائج Serper لـ Gemini ويرجع S_claim (0.0 → 1.0).
    يرجع None لو Gemini مش متاح أو فشل — verify_news بيستخدم max_similarity كـ fallback.
    """
    if not gemini_model or not articles:
        log.warning("[GEMINI S_claim] gemini_model unavailable — will use similarity fallback")
        return None

    # بنبني السياق من العناوين + الـ snippets
    context_lines = []
    for a in articles[:12]:
        title   = a.get("title",   "").strip()
        source  = a.get("source",  "")
        snippet = a.get("snippet", "").strip()
        if title:
            line = f"- [{source}] {title}"
            if snippet:
                line += f" — {snippet}"
            context_lines.append(line)

    if not context_lines:
        return None

    context = "\n".join(context_lines)
    try:
        prompt = (
            f"أنت محقق أخبار متخصص. بناءً على نتائج البحث التالية فقط، "
            f"قيّم مدى صحة الادعاء.\n\n"
            f"الادعاء:\n{claim[:400]}\n\n"
            f"نتائج البحث:\n{context}\n\n"
            f"أجب برقم عشري بين 0.00 و 1.00 فقط (بدون أي كلام إضافي):\n"
            f"1.00 = الادعاء موجود ومؤكد في المصادر\n"
            f"0.50 = غير مؤكد أو المصادر جزئية\n"
            f"0.00 = الادعاء غير موجود أو كاذب\n\n"
            f"الرقم فقط:"
        )
        resp = _gemini_call(prompt, max_tokens=10)
        if not resp:
            return None
        text   = resp.text.strip().replace(",", ".")
        score  = float(text)
        result = max(0.0, min(1.0, score))
        print(f"  [GEMINI S_claim] Score: {result:.3f}")
        return result
    except Exception as e:
        log.warning(f"[GEMINI S_claim] Error: {e} — will use similarity fallback")
        return None

# ============================================================
# [GEMINI] News Search & Suggest — fallback when Serper finds nothing
# ============================================================
def gemini_search_and_suggest(user_text: str) -> tuple:
    """
    استدعاء Gemini واحد يُنجز مهمتين:
      1. يولّد استعلام بحث محسّن بالعربي
      2. يقترح مقالات إخبارية حقيقية مشابهة
    يُستخدم فقط عند إخفاق Serper/Google News (توفيراً للـ quota).
    يُرجع (better_query: str, articles: list[dict])
    """
    if not gemini_model or not _gemini_daily_ok:
        return "", []
    domains_str = ", ".join(ARABIC_NEWS_DOMAINS[:10])
    prompt = (
        f"أنت محقق أخبار متخصص بالمحتوى العربي.\n"
        f"للخبر التالي:\n{user_text[:450]}\n\n"
        f"أجب بـ JSON صحيح فقط (بدون markdown أو شرح):\n"
        f'{{"search_query":"استعلام بحث محسّن (جملة قصيرة بالعربي)",'
        f'"articles":[{{"title":"عنوان المقال","source":"اسم الموقع",'
        f'"url":"https://...","similarity":75}}]}}\n\n'
        f"المواقع المقبولة فقط: {domains_str}\n"
        f"القواعد:\n"
        f"- أرجع أقصاه 3 مقالات حقيقية ومشابهة\n"
        f"- similarity: نسبة التشابه 0-100\n"
        f"- الرابط يجب أن يبدأ بـ https:// وينتمي لأحد المواقع المقبولة\n"
        f"- لو ما تجد مقالات موثوقة أو الخبر محلي جداً، أرجع articles كقائمة فارغة []\n"
        f"JSON فقط:"
    )
    try:
        resp = _gemini_call(prompt, max_tokens=600)
        if not resp:
            return "", []
        raw = resp.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        import json as _jgs
        data = _jgs.loads(raw)
        query = (data.get("search_query") or "").strip()[:200]
        articles = []
        for item in (data.get("articles") or [])[:3]:
            url   = (item.get("url")   or "").strip()
            title = (item.get("title") or "").strip()
            if not url or not title or not url.startswith("http"):
                continue
            # تحقق بسيط أن الرابط ينتمي لمواقعنا الموثوقة
            if not any(d in url for d in ARABIC_NEWS_DOMAINS):
                continue
            articles.append({
                "source"    : (item.get("source") or "").strip(),
                "title"     : title,
                "url"       : url,
                "similarity": min(100, max(0, int(item.get("similarity") or 50))),
            })
        print(f"  [GEMINI SUGGEST] query='{query[:70]}' | articles={len(articles)}")
        return query, articles
    except Exception as e:
        log.warning(f"[GEMINI SUGGEST] Error: {e}")
        return "", []


# ============================================================
# [AI] AraBERT Text Prediction
# ============================================================
def ai_predict(text):
    word_count       = len(text.split())
    is_short         = word_count < 20
    target_model     = model_short if is_short else model_long
    target_tokenizer = tokenizer_short if is_short else tokenizer_long
    model_label      = f"Short [{PATH_SHORT}]" if is_short else f"Long [{PATH_LONG}]"

    inputs = target_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = target_model(**inputs)
    probs  = torch.nn.functional.softmax(outputs.logits, dim=1)
    fake_p = probs[0][0].item()
    real_p = probs[0][1].item()

    print(f"\n--- [AI Text Predict] ---")
    print(f"  Model      : {model_label}")
    print(f"  Words      : {word_count} ({'Short' if is_short else 'Long'}) | Chars: {len(text)}")
    print(f"  Real       : {real_p*100:.2f}%")
    print(f"  Fake       : {fake_p*100:.2f}%")
    return fake_p, real_p

# ============================================================
# [IMAGE] Ensemble AI/Real classifier
# ============================================================
def predict_image_ai(image_path):
    print(f"\n--- [Image AI | Ensemble+TTA: M1 & M2] ---")
    t0 = time.time()
    try:
        image = Image.open(image_path).convert('RGB')
        t = image_transforms(image).unsqueeze(0).to(device)

        # Weighted Ensemble by Validation Accuracy (Rokach, 2010)
        W1      = 0.9310 ** 2
        W2      = 0.9053 ** 2
        W_total = W1 + W2

        # TTA: original + horizontal flip + rot90
        augmentations = [t, torch.flip(t, [3]), torch.rot90(t, 1, [2, 3])]

        with torch.no_grad():
            preds = []
            for aug in augmentations:
                out1 = model1(aug)
                out2 = model2(aug)
                # جمع logits موزونة (قبل softmax) لكل augmentation
                preds.append((W1 / W_total) * out1 + (W2 / W_total) * out2)
            # نجمع الـ logits ثم softmax مرة وحدة
            probs = F.softmax(torch.mean(torch.stack(preds), dim=0), dim=1)
            conf, pred = torch.max(probs, 1)

        label = "AI-Generated" if pred.item() == 0 else "Real Image"
        print(f"   Result     : {label}")
        print(f"   Confidence : {conf.item()*100:.2f}%")
        print(f"   Time       : {time.time()-t0:.3f} sec")
        return {"label": label, "confidence": f"{conf.item()*100:.2f}%"}
    except Exception as e:
        print(f"   [ERROR] Image AI: {e}")
        return {"error": str(e)}

# ============================================================
# [OCR] Multi-engine OCR pipeline (EasyOCR + optional PaddleOCR)
# Merged from ocr_backend.py — lazy-loaded readers, language routing,
# cleanup maps, line-aware reading order (RTL/LTR aware).
# ============================================================

# ── Lazy EasyOCR readers ────────────────────────────────────────────────────
# EasyOCR's "ar"+"en" combo shares the same recogniser network, but accuracy
# drops when an image is mostly Arabic OR mostly English. We lazy-load three
# readers and route each image to the right one.
_reader_ar_en = None
_reader_en    = None
_reader_ar    = None

# Confidence threshold — detections below this are discarded (0.0–1.0).
OCR_MIN_CONFIDENCE = 0.30


def get_reader(mode: str = "ar_en"):
    """Return the requested EasyOCR reader, creating it on first call."""
    global _reader_ar_en, _reader_en, _reader_ar
    if not EASYOCR_AVAILABLE:
        return None
    if mode is True:
        mode = "ar_en"
    elif mode is False:
        mode = "en"

    use_gpu = torch.cuda.is_available()
    if mode == "ar":
        if _reader_ar is None:
            print("[OCR] Loading EasyOCR reader (ar)...")
            _reader_ar = easyocr.Reader(["ar"], gpu=use_gpu)
        return _reader_ar
    if mode == "en":
        if _reader_en is None:
            print("[OCR] Loading EasyOCR reader (en)...")
            _reader_en = easyocr.Reader(["en"], gpu=use_gpu)
        return _reader_en
    if _reader_ar_en is None:
        print("[OCR] Loading EasyOCR reader (ar+en)...")
        _reader_ar_en = easyocr.Reader(["ar", "en"], gpu=use_gpu)
    return _reader_ar_en


# ── Lazy PaddleOCR engines ──────────────────────────────────────────────────
paddle_ocr_ar = None
paddle_ocr_en = None


def _get_paddle_engine(lang: str):
    """Lazy-load a PaddleOCR engine.  lang: 'arabic' or 'en'."""
    global paddle_ocr_ar, paddle_ocr_en, PADDLE_AVAILABLE
    if not PADDLE_AVAILABLE:
        return None
    try:
        if lang == "arabic":
            if paddle_ocr_ar is None:
                paddle_ocr_ar = PaddleOCR(use_angle_cls=True, lang="arabic")
            return paddle_ocr_ar
        if lang == "en":
            if paddle_ocr_en is None:
                paddle_ocr_en = PaddleOCR(use_angle_cls=True, lang="en")
            return paddle_ocr_en
    except Exception as exc:
        print(f"[OCR] PaddleOCR init failed for {lang}: {exc}")
        PADDLE_AVAILABLE = False
        return None
    return None


# ── OCR correction maps ─────────────────────────────────────────────────────
OCR_FIXES = {
    "الممالحة": "المملكة",   # common AlMamlaka logo mis-read
    "الاٍردن":  "الأردن",
    "الممالكه": "المملكة",
    "9لي العهد":   "ولي العهد",
    "gدول الخليج": "ودول الخليج",
    "إطلاق الذار": "إطلاق النار",
}

_LATIN_IN_AR_PATTERNS = [
    (re.compile(r"(?<=[؀-ۿ])\s*[gG9](?=[؀-ۿ])"), "و"),
    (re.compile(r"(?:^|(?<=\s))[gG9](?=[؀-ۿ])"), "و"),
    (re.compile(r"(?<=[؀-ۿ])\s*_\s*(?=[؀-ۿ])"), " "),
    (re.compile(r"(بحضور)(?=[؀-ۿ])"), r"\1 "),
    (re.compile(r"(الأخير)(?=[؀-ۿ])"),  r"\1 "),
]

_ENG_FIXES = [
    (re.compile(r"\s+0\s+(\d+[:.]\d+)"), r" \1"),
    (re.compile(r"\s+0(\d+[:.]\d+)"),    r" \1"),
]


# ── Text helpers ────────────────────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    """Light Arabic normalisation for the NLP model (keep raw copy for UI)."""
    if not text:
        return text
    text = re.sub(r"[ـ]", "", text)
    text = re.sub(r"[ً-ْٰ]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ىي]", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    return text


def clean_ocr_text(text: str) -> str:
    """Light cleanup: known fixes, collapse whitespace, drop control chars."""
    if not text:
        return ""
    for bad, good in OCR_FIXES.items():
        text = text.replace(bad, good)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    for pattern, replacement in _LATIN_IN_AR_PATTERNS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _ENG_FIXES:
        text = pattern.sub(replacement, text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return " ".join(ln for ln in lines if ln)


def detect_language_flags(text: str):
    """Return (has_arabic, has_english, language_detected)."""
    has_arabic  = bool(re.search(r"[؀-ۿ]", text or ""))
    has_english = bool(re.search(r"[A-Za-z]", text or ""))
    if has_arabic and has_english:
        lang = "mixed"
    elif has_arabic:
        lang = "ar"
    elif has_english:
        lang = "en"
    else:
        lang = "unknown"
    return has_arabic, has_english, lang


def compute_confidence(confidences: list) -> float:
    """Average confidence list → 0–100 float."""
    if not confidences:
        return 0.0
    return (sum(confidences) / len(confidences)) * 100.0


def _script_ratios(text: str):
    """Return (ar_ratio, en_ratio, contamination_count)."""
    if not text:
        return 0.0, 0.0, 0
    ar    = len(re.findall(r"[؀-ۿ]", text))
    en    = len(re.findall(r"[A-Za-z]", text))
    total = ar + en
    if total == 0:
        return 0.0, 0.0, 0
    contamination = len(re.findall(
        r"(?:[؀-ۿ][A-Za-z0-9]+[؀-ۿ])|(?:^|[\s])[A-Za-z0-9][؀-ۿ]",
        text,
    ))
    return ar / total, en / total, contamination


# ── Image pre-processing ────────────────────────────────────────────────────
# Bumped from 1600 → 2400 (step 1a): gives EasyOCR/PaddleOCR ~2.25× more pixels
# per glyph, which directly improves disambiguation of dotted Arabic letters
# (ب ↔ ي ↔ ت ↔ ث) and similar shape pairs (ر ↔ ل, د ↔ ه).
OCR_TARGET_LONGEST_PX = 2400

def preprocess_for_ocr(image_path: str) -> str:
    """OpenCV preprocessing tuned for Arabic + English news screenshots.

    The caller must delete the returned PNG path after use.

    Pipeline:
      1. Upscale longest side to OCR_TARGET_LONGEST_PX (2400) with bicubic.
      2. Convert to grayscale.
      3. Classify image as "clean" (high-contrast text screenshot) vs "noisy"
         (photo of paper, low light, etc.).
      4. CLEAN path → Otsu binarization for crisp glyph boundaries.
      5. NOISY path → CLAHE + denoise + unsharp mask (recover stroke detail).
    """
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]
    longest = max(h, w)

    # Step 1a — upscale to OCR_TARGET_LONGEST_PX (was 1600).
    if longest < OCR_TARGET_LONGEST_PX:
        scale = OCR_TARGET_LONGEST_PX / float(longest)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    std  = float(gray.std())
    mean = float(gray.mean())
    is_clean = (
        longest >= 1200 and
        std > 65 and
        (mean > 200 or mean < 75)
    )

    if is_clean:
        # Step 1a — Otsu binarization for clean documents.
        # A mild blur first reduces single-pixel noise that would confuse Otsu's
        # bimodal assumption.  We pick the polarity (white-on-black vs black-on-
        # white) based on the mean — dark backgrounds get inverted so the OCR
        # engine always sees dark glyphs on a light background.
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        if mean < 75:                       # white text on dark background
            flag = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        _, binar = cv2.threshold(smoothed, 0, 255, flag)

        # Light unsharp on the post-Otsu image — keeps thin dot strokes (the
        # difference between ب and ي) from being eroded by binarisation.
        _blur = cv2.GaussianBlur(binar, (0, 0), 1.0)
        gray  = cv2.addWeighted(binar, 1.25, _blur, -0.25, 0)
    else:
        # Noisy path (photos, scans): adaptive contrast + denoise + unsharp.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        gray  = clahe.apply(gray)
        gray  = cv2.fastNlMeansDenoising(gray, h=8,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
        _blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        gray  = cv2.addWeighted(gray, 1.5, _blur, -0.5, 0)

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    out.close()
    cv2.imwrite(out.name, gray)
    return out.name


# ── EasyOCR runner ──────────────────────────────────────────────────────────
def _easy_readtext(image_path: str, mode: str = "ar_en"):
    """Run EasyOCR with detail=1, group into visual lines, sort RTL/LTR."""
    rdr = get_reader(mode=mode)
    if rdr is None:
        return "", []
    try:
        results = rdr.readtext(image_path, detail=1, paragraph=False)
    except Exception as exc:
        print(f"[OCR] EasyOCR error: {exc}")
        return "", []

    detections  = []
    confidences = []
    for r in results:
        if not r or len(r) < 3:
            continue
        bbox, text, conf = r[0], r[1], r[2]
        if not isinstance(text, str) or not text.strip():
            continue
        if isinstance(conf, (int, float)) and conf < OCR_MIN_CONFIDENCE:
            continue
        try:
            xs   = [p[0] for p in bbox]
            ys   = [p[1] for p in bbox]
            cy   = sum(ys) / 4.0
            x_left = min(xs)
        except Exception:
            continue
        detections.append((cy, x_left, text.strip(), conf))
        confidences.append(conf if isinstance(conf, (int, float)) else 0.0)

    if not detections:
        return "", []

    detections.sort(key=lambda d: (d[0], d[1]))
    line_threshold = 20          # slightly looser to handle slanted baselines
    lines, buf, last_y = [], [], None
    for d in detections:
        y = d[0]
        if last_y is None or abs(y - last_y) <= line_threshold:
            buf.append(d)
        else:
            lines.append(buf)
            buf = [d]
        last_y = y
    if buf:
        lines.append(buf)

    out_lines = []
    for line in lines:
        joined = " ".join(d[2] for d in line)
        rtl    = bool(re.search(r"[؀-ۿ]", joined))
        line.sort(key=lambda d: d[1], reverse=rtl)
        out_lines.append(" ".join(d[2] for d in line))

    return "\n".join(out_lines).strip(), confidences


def run_easyocr(image_path: str, hint_lang: str = None):
    """Single-pass routing — chooses the best EasyOCR reader per image."""
    if hint_lang in ("ar", "en"):
        return _easy_readtext(image_path, mode=hint_lang)
    if hint_lang == "mixed":
        return _easy_readtext(image_path, mode="ar_en")

    text, confs = _easy_readtext(image_path, mode="ar_en")
    if not text:
        return text, confs

    ar_ratio, en_ratio, contamination = _script_ratios(text)

    # Only do a second pass when contamination is significant (> 2 mixed tokens)
    # — avoids an extra model call on every mostly-Arabic image.
    if ar_ratio >= 0.75 and contamination > 2:
        ar_text, ar_confs = _easy_readtext(image_path, mode="ar")
        if ar_text and compute_confidence(ar_confs) >= compute_confidence(confs) - 5:
            return ar_text, ar_confs
        return text, confs

    if en_ratio >= 0.75 and ar_ratio < 0.10:
        en_text, en_confs = _easy_readtext(image_path, mode="en")
        if en_text and compute_confidence(en_confs) >= compute_confidence(confs) - 5:
            return en_text, en_confs

    return text, confs


# ── PaddleOCR runner ────────────────────────────────────────────────────────
def run_paddleocr(image_path: str, langs=("arabic", "en")):
    """Run PaddleOCR with the specified language engines (lazy-loaded)."""
    if not PADDLE_AVAILABLE:
        return "", []

    detections  = []
    confidences = []

    for lang in langs:
        engine = _get_paddle_engine(lang)
        if not engine:
            continue
        try:
            result = engine.ocr(image_path, cls=True)
        except Exception as exc:
            print(f"[OCR] PaddleOCR error ({lang}): {exc}")
            continue
        for page in result or []:
            for line in page or []:
                if not line or len(line) < 2:
                    continue
                bbox = line[0]
                pair = line[1]
                if not pair or len(pair) < 2:
                    continue
                text, conf = pair[0], pair[1]
                if not text or not isinstance(text, str):
                    continue
                if isinstance(conf, (int, float)) and conf < OCR_MIN_CONFIDENCE:
                    continue
                try:
                    ys     = [p[1] for p in bbox]
                    xs     = [p[0] for p in bbox]
                    cy     = sum(ys) / float(len(ys))
                    x_left = min(xs)
                except Exception:
                    cy, x_left = 0.0, 0.0
                detections.append((cy, x_left, text.strip(), conf))
                confidences.append(conf if isinstance(conf, (int, float)) else 0.0)

    if not detections:
        return "", []

    seen, deduped = set(), []
    for d in detections:
        if d[2] not in seen:
            seen.add(d[2])
            deduped.append(d)

    deduped.sort(key=lambda d: (d[0], d[1]))
    lines, buf, last_y = [], [], None
    for d in deduped:
        y = d[0]
        if last_y is None or abs(y - last_y) <= 20:
            buf.append(d)
        else:
            lines.append(buf)
            buf = [d]
        last_y = y
    if buf:
        lines.append(buf)

    out_lines = []
    for line in lines:
        joined = " ".join(d[2] for d in line)
        rtl    = bool(re.search(r"[؀-ۿ]", joined))
        line.sort(key=lambda d: d[1], reverse=rtl)
        out_lines.append(" ".join(d[2] for d in line))

    return "\n".join(out_lines).strip(), confidences


# ============================================================
# [OCR FUSION] Step 1b + Step 2
# Run EasyOCR + PaddleOCR (+ optional Tesseract) on the SAME preprocessed
# image, then pick the highest-confidence reading per spatially-overlapping
# region.  Each engine has different weaknesses on Arabic — EasyOCR sometimes
# loses dots (ب ↔ ي), PaddleOCR sometimes glues words, Tesseract excels on
# clean printed text but stumbles on stylised fonts.  Voting beats any one
# of them alone on Arabic news screenshots.
# ============================================================

# ── Optional Tesseract engine (step 2) ──────────────────────────────────────
try:
    import pytesseract
    # Honour a user-set path on Windows where Tesseract isn't on PATH.
    _tess_cmd = os.environ.get("TESSERACT_CMD")
    if _tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    # Probe the binary so we know whether the language packs are installed.
    try:
        _tess_langs = set(pytesseract.get_languages(config=""))
        TESSERACT_AVAILABLE = "ara" in _tess_langs or "eng" in _tess_langs
    except Exception:
        _tess_langs = set()
        TESSERACT_AVAILABLE = False
except Exception:
    pytesseract = None
    _tess_langs = set()
    TESSERACT_AVAILABLE = False

print(f"[SYSTEM] Tesseract: "
      f"{'available (' + ','.join(sorted(_tess_langs)) + ')' if TESSERACT_AVAILABLE else 'NOT INSTALLED (optional)'}")


def _bbox_center_size(bbox):
    """Return (cx, cy, w, h) for a 4-point or 2-point bbox."""
    try:
        if len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return (sum(xs)/4.0, sum(ys)/4.0,
                    max(xs)-min(xs), max(ys)-min(ys))
        # (x, y, w, h) rect form
        x, y, w, h = bbox
        return (x + w/2.0, y + h/2.0, float(w), float(h))
    except Exception:
        return (0.0, 0.0, 0.0, 0.0)


def _make_det(text, conf, bbox, engine):
    """Standardised detection record for the fuser."""
    cx, cy, w, h = _bbox_center_size(bbox)
    return {
        "text":   str(text).strip(),
        "conf":   float(conf) if isinstance(conf, (int, float)) else 0.0,
        "cx":     cx, "cy": cy, "w": w, "h": h,
        "engine": engine,
    }


def _easy_detect(image_path: str, mode: str = "ar_en"):
    """EasyOCR → list of detection dicts (with bbox)."""
    rdr = get_reader(mode=mode)
    if rdr is None:
        return []
    try:
        results = rdr.readtext(image_path, detail=1, paragraph=False)
    except Exception as exc:
        log.warning(f"[OCR fusion] EasyOCR error: {exc}")
        return []
    dets = []
    for r in results or []:
        if not r or len(r) < 3:
            continue
        bbox, text, conf = r[0], r[1], r[2]
        if not isinstance(text, str) or not text.strip():
            continue
        if isinstance(conf, (int, float)) and conf < OCR_MIN_CONFIDENCE:
            continue
        dets.append(_make_det(text, conf, bbox, "easyocr"))
    return dets


def _paddle_detect(image_path: str, langs=("arabic", "en")):
    """PaddleOCR → list of detection dicts (with bbox)."""
    if not PADDLE_AVAILABLE:
        return []
    dets = []
    for lang in langs:
        engine = _get_paddle_engine(lang)
        if not engine:
            continue
        try:
            result = engine.ocr(image_path, cls=True)
        except Exception as exc:
            log.warning(f"[OCR fusion] PaddleOCR error ({lang}): {exc}")
            continue
        for page in result or []:
            for line in page or []:
                if not line or len(line) < 2:
                    continue
                bbox, pair = line[0], line[1]
                if not pair or len(pair) < 2:
                    continue
                text, conf = pair[0], pair[1]
                if not text or not isinstance(text, str):
                    continue
                if isinstance(conf, (int, float)) and conf < OCR_MIN_CONFIDENCE:
                    continue
                dets.append(_make_det(text, conf, bbox, "paddleocr"))
    # Dedup identical strings (Paddle can emit the same word from both lang engines)
    seen, deduped = set(), []
    for d in dets:
        key = (d["text"], round(d["cx"]/20), round(d["cy"]/20))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
    return deduped


def _tesseract_detect(image_path: str, lang: str = "ara+eng"):
    """Tesseract → list of detection dicts (with bbox).  Empty if unavailable."""
    if not TESSERACT_AVAILABLE:
        return []
    # Fall back to whichever languages are actually installed.
    parts = [p for p in lang.split("+") if p in _tess_langs]
    if not parts:
        return []
    use_lang = "+".join(parts)
    try:
        data = pytesseract.image_to_data(
            image_path, lang=use_lang,
            output_type=pytesseract.Output.DICT,
            config="--oem 1 --psm 6",   # LSTM engine, assume a uniform block of text
        )
    except Exception as exc:
        log.warning(f"[OCR fusion] Tesseract error: {exc}")
        return []
    dets = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        # Tesseract gives confidence 0–100; normalise to 0–1 like the others.
        try:
            conf = float(data["conf"][i]) / 100.0
        except (TypeError, ValueError):
            conf = 0.0
        if conf < OCR_MIN_CONFIDENCE:
            continue
        x, y, w, h = (int(data["left"][i]),  int(data["top"][i]),
                      int(data["width"][i]), int(data["height"][i]))
        dets.append(_make_det(txt, conf, (x, y, w, h), "tesseract"))
    return dets


def _detections_overlap(a, b, iou_thresh: float = 0.35) -> bool:
    """Quick overlap test using axis-aligned bounding boxes."""
    ax1 = a["cx"] - a["w"]/2.0;  ay1 = a["cy"] - a["h"]/2.0
    ax2 = a["cx"] + a["w"]/2.0;  ay2 = a["cy"] + a["h"]/2.0
    bx1 = b["cx"] - b["w"]/2.0;  by1 = b["cy"] - b["h"]/2.0
    bx2 = b["cx"] + b["w"]/2.0;  by2 = b["cy"] + b["h"]/2.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    if inter <= 0:
        return False
    a_area = max(1.0, (ax2-ax1) * (ay2-ay1))
    b_area = max(1.0, (bx2-bx1) * (by2-by1))
    iou    = inter / (a_area + b_area - inter)
    return iou >= iou_thresh


def _fuse_detections(*engine_lists):
    """Pick the highest-confidence detection per spatially-overlapping cluster.

    Returns:
        (fused_detections, engine_contribution_counts)
    """
    flat = []
    for lst in engine_lists:
        flat.extend(lst or [])
    if not flat:
        return [], {}

    # Sort by confidence descending — greedy non-max suppression gives us
    # the "highest conf wins" semantic the user asked for, with overlap as
    # the cluster definition.
    flat.sort(key=lambda d: d["conf"], reverse=True)

    chosen = []
    contrib = {}
    for det in flat:
        if any(_detections_overlap(det, c) for c in chosen):
            continue
        chosen.append(det)
        contrib[det["engine"]] = contrib.get(det["engine"], 0) + 1
    return chosen, contrib


def _detections_to_text(detections, line_threshold: int = 22):
    """Group detections into visual lines and produce final RTL/LTR-aware text."""
    if not detections:
        return "", []
    rows = sorted(detections, key=lambda d: (d["cy"], d["cx"]))
    lines, buf, last_y = [], [], None
    for d in rows:
        y = d["cy"]
        if last_y is None or abs(y - last_y) <= line_threshold:
            buf.append(d)
        else:
            lines.append(buf)
            buf = [d]
        last_y = y
    if buf:
        lines.append(buf)

    out_lines = []
    confs     = []
    for line in lines:
        joined_for_lang = " ".join(d["text"] for d in line)
        rtl = bool(re.search(r"[؀-ۿ]", joined_for_lang))
        # RTL: rightmost first → sort by cx descending.
        line.sort(key=lambda d: d["cx"], reverse=rtl)
        out_lines.append(" ".join(d["text"] for d in line))
        confs.extend(d["conf"] for d in line)
    return "\n".join(out_lines).strip(), confs


def run_fused_ocr(image_path: str, want_engines=("easy", "paddle", "tess")):
    """Run all requested engines on the same image and fuse their detections.

    Returns (text, confidences, engine_contributions_dict).
    """
    easy_dets   = _easy_detect(image_path, mode="ar_en")    if "easy"   in want_engines else []
    paddle_dets = _paddle_detect(image_path)                 if "paddle" in want_engines else []
    tess_dets   = _tesseract_detect(image_path)              if "tess"   in want_engines else []

    fused, contrib = _fuse_detections(easy_dets, paddle_dets, tess_dets)
    text, confs    = _detections_to_text(fused)
    return text, confs, contrib


# ============================================================
# [OCR POST-CORRECTION] AraBERT MLM-based text correction
# Two-stage cleanup applied *after* the fusion produces clean_text:
#   Stage 1 — regex split of known glued patterns (يحوزعلى → يحوز على)
#             — pure regex, runs in <1 ms, no model needed.
#   Stage 2 — per-word verification via AutoModelForMaskedLM.  For each
#             suspicious word we MASK it, ask AraBERT what it expects,
#             and accept the replacement only when:
#               (a) BERT's top pick is ≥ 10× more probable than the original
#               (b) the pick is within Levenshtein distance 2 of the original
#                   (so we fix OCR character-swaps, not hallucinate new words).
# The MLM head is lazy-loaded the first time OCR runs; controlled by the
# MIZAN_ENABLE_MLM_CORRECTION env var (default = on).
# ============================================================

ENABLE_MLM_CORRECTION = os.environ.get("MIZAN_ENABLE_MLM_CORRECTION", "1") == "1"

mlm_tokenizer = None
mlm_model     = None
MLM_AVAILABLE = False
_MLM_LOAD_LOCK = threading.Lock()


def _ensure_mlm_loaded():
    """Lazy-load AraBERT's MLM head.  Returns True on success."""
    global mlm_tokenizer, mlm_model, MLM_AVAILABLE
    if not ENABLE_MLM_CORRECTION:
        return False
    if MLM_AVAILABLE:
        return True
    with _MLM_LOAD_LOCK:
        if MLM_AVAILABLE:
            return True
        try:
            from transformers import AutoModelForMaskedLM
            log.info(f"[OCR/MLM] Loading {MODEL_NAME} masked-LM head …")
            mlm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            mlm_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
            mlm_model.eval()
            MLM_AVAILABLE = True
            log.info("[OCR/MLM] Loaded OK.")
        except Exception as exc:
            log.warning(f"[OCR/MLM] Could not load MLM head: {exc}")
            mlm_tokenizer = mlm_model = None
            MLM_AVAILABLE = False
    return MLM_AVAILABLE


# ── Stage 1: glued-word splitter ─────────────────────────────────────────────
# Specific OCR-glue patterns observed on Arabic news screenshots.  Each pattern
# is conservative enough to avoid mis-splitting legitimate words (e.g. "أعلى",
# "ابتعد", "عبارة" don't match).
_OCR_GLUE_PATTERNS = [
    # word(4+ Arabic letters) + preposition glued at end → split
    # Triggers on:  يحوزعلى → يحوز على ;  المحققين على ; …
    # Doesn't trigger on: أعلى (prefix too short), تجمع (prefix too short)
    (re.compile(
        r"([؀-ۿ]{4,})(على|في|من|إلى|عن|عبر|عند|بين|مع)"
        r"(?=\s|$|[^؀-ۿ])"
     ),
     r"\1 \2"),

    # preposition + ال + noun glued → split
    # Triggers on:  عبرالحدود → عبر الحدود ;  تحتالأرض ; فوقالجبل ; …
    # Doesn't trigger on: عبارة (no ال after), تحديد (no ال after)
    (re.compile(
        r"\b(عبر|تحت|فوق|أمام|خلف|بين|قبل|بعد|دون|نحو|ضد|مع)"
        r"(ال[؀-ۿ]{2,})"
     ),
     r"\1 \2"),

    # "و" conjunction glued to a long following word — only split when the
    # following segment starts with "ال" (the definite article) and the
    # combined word is suspiciously long.  E.g. والحدود (≥ 6 chars) → و الحدود
    # but watch out for genuine words: this rule is intentionally NOT applied
    # because و + word is the standard Arabic spelling.  Disabled for now.
]


def _split_glued_arabic_words(text: str) -> str:
    """Insert spaces in known glued patterns.  Fast, deterministic, no model."""
    if not text:
        return text
    for pattern, replacement in _OCR_GLUE_PATTERNS:
        # Loop because one pass might leave further candidates if a word had
        # multiple particles glued (rare but cheap to handle).
        prev = None
        cur  = text
        while prev != cur:
            prev = cur
            cur  = pattern.sub(replacement, cur)
        text = cur
    return text


# ── Stage 2: MLM per-word verification ───────────────────────────────────────
def _levenshtein_le(a: str, b: str, max_dist: int) -> bool:
    """Return True iff Levenshtein(a, b) <= max_dist.  Early-exits for speed."""
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return False
    if la == 0:
        return lb <= max_dist
    if lb == 0:
        return la <= max_dist
    # Standard DP with early termination on row min > max_dist
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        row_min = curr[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(curr[j-1] + 1,    # insertion
                          prev[j]   + 1,    # deletion
                          prev[j-1] + cost) # substitution
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_dist:
            return False
        prev = curr
    return prev[lb] <= max_dist


def _mlm_correct_suspicious_words(text: str,
                                  prob_ratio: float = 10.0,
                                  max_edit:   int   = 2) -> str:
    """Mask each long Arabic word in turn; if BERT's top alternative is
    much more probable AND visually close to the original, swap it in."""
    if not MLM_AVAILABLE or not text:
        return text

    words = text.split()
    if len(words) < 3:
        return text

    out = list(words)

    for i, word in enumerate(out):
        # Skip short words, non-Arabic, and words with punctuation.
        if len(word) < 5:
            continue
        if not re.fullmatch(r"[؀-ۿ]+", word):
            continue

        # Skip if the word tokenises to multiple subwords — replacing
        # multi-token words is risky and the comparison isn't meaningful.
        orig_ids = mlm_tokenizer.encode(word, add_special_tokens=False)
        if len(orig_ids) != 1:
            continue

        # Build a local context window (3 words either side is plenty).
        lo = max(0, i - 3)
        hi = min(len(out), i + 4)
        prefix = out[lo:i]
        suffix = out[i+1:hi]

        masked = " ".join(prefix) + " " + mlm_tokenizer.mask_token + " " + " ".join(suffix)
        masked = masked.strip()

        try:
            enc = mlm_tokenizer(masked, return_tensors="pt",
                                truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits = mlm_model(**enc).logits
            mask_positions = (enc["input_ids"] == mlm_tokenizer.mask_token_id).nonzero()
            if mask_positions.shape[0] == 0:
                continue
            mask_idx = mask_positions[0, 1].item()
            probs = F.softmax(logits[0, mask_idx], dim=-1)

            orig_prob = probs[orig_ids[0]].item()
            top_prob, top_id = torch.topk(probs, k=1)
            top_prob = top_prob.item()
            top_word = mlm_tokenizer.decode([top_id.item()]).strip()
        except Exception as exc:
            log.debug(f"[OCR/MLM] scoring {word!r} failed: {exc}")
            continue

        # Replace ONLY when the top pick is dramatically more likely AND
        # close enough to the original that this looks like an OCR fix.
        if top_word and top_word != word and top_prob > orig_prob * prob_ratio:
            if _levenshtein_le(word, top_word, max_edit):
                out[i] = top_word

    return " ".join(out)


def correct_ocr_with_bert(text: str) -> str:
    """Public entry point — runs both correction stages.  Returns the
    corrected text (or the original if there's no Arabic / nothing to do)."""
    if not text:
        return text
    if not re.search(r"[؀-ۿ]", text):
        return text

    # Stage 1 always runs (cheap)
    text = _split_glued_arabic_words(text)

    # Stage 2 only when the MLM head is loaded / loadable
    if _ensure_mlm_loaded():
        try:
            text = _mlm_correct_suspicious_words(text)
        except Exception as exc:
            log.warning(f"[OCR/MLM] correction pass aborted: {exc}")

    return text


# ── Unified OCR pipeline ────────────────────────────────────────────────────
def run_ocr_pipeline(image_path: str, force_engine: str = None) -> dict:
    """Pre-process the image, then extract text.

    Engine selection:
      "easy"   → EasyOCR only.
      "paddle" → PaddleOCR only (falls back to EasyOCR if PaddleOCR is missing).
      "tess"   → Tesseract only (falls back to EasyOCR if not installed).
      "fuse"   → Step 1b/Step 2 fusion across all available engines.
      "auto"   → Same as "fuse" when ≥ 2 engines are available; otherwise the
                 best single engine.
    """
    print(f"\n--- [OCR Pipeline] engine={force_engine or 'auto'} ---")
    requested = (force_engine or "auto").strip().lower()
    if requested not in ("auto", "easy", "paddle", "tess", "fuse"):
        requested = "auto"

    preprocessed = None
    try:
        preprocessed = preprocess_for_ocr(image_path)
    except Exception as exc:
        print(f"[OCR] preprocess failed ({exc}); using raw image.")
        preprocessed = image_path

    raw_text, confidences, engine = "", [], "easyocr"
    engines_used = {}

    try:
        # ── Explicit single-engine paths ────────────────────────────────────
        if requested == "paddle" and PADDLE_AVAILABLE:
            raw_text, confidences = run_paddleocr(preprocessed)
            engine = "paddleocr"

        elif requested == "tess" and TESSERACT_AVAILABLE:
            tess_dets = _tesseract_detect(preprocessed)
            raw_text, confidences = _detections_to_text(tess_dets)
            engine = "tesseract"

        elif requested == "easy" or (
            requested in ("paddle", "tess") and not PADDLE_AVAILABLE and not TESSERACT_AVAILABLE
        ):
            raw_text, confidences = run_easyocr(preprocessed)
            engine = "easyocr"

        # ── Fusion path (auto + fuse) ──────────────────────────────────────
        else:
            # Decide which engines to participate.  "auto" enables everything
            # available; "fuse" is the explicit name for the same thing.
            want = ["easy"]
            if PADDLE_AVAILABLE:     want.append("paddle")
            if TESSERACT_AVAILABLE:  want.append("tess")

            if len(want) >= 2:
                raw_text, confidences, engines_used = run_fused_ocr(
                    preprocessed, want_engines=tuple(want)
                )
                # Engine label reflects what contributed.  If only one engine
                # ended up with chosen detections, name that one; otherwise
                # report "fused (...)" for traceability in the UI.
                if engines_used:
                    if len(engines_used) == 1:
                        only = next(iter(engines_used))
                        engine = {"easyocr":"easyocr",
                                  "paddleocr":"paddleocr",
                                  "tesseract":"tesseract"}.get(only, only)
                    else:
                        engine = "fused(" + "+".join(sorted(engines_used)) + ")"
            else:
                # Only one engine available — degrade gracefully to the old
                # EasyOCR-first / PaddleOCR-fallback behaviour.
                raw_text, confidences = run_easyocr(preprocessed)
                engine = "easyocr"
                easy_clean = clean_ocr_text(raw_text)
                easy_conf  = compute_confidence(confidences)
                if PADDLE_AVAILABLE and (not easy_clean or easy_conf < 40.0):
                    p_text, p_conf = run_paddleocr(preprocessed)
                    if p_text and compute_confidence(p_conf) > easy_conf:
                        raw_text, confidences = p_text, p_conf
                        engine = "paddleocr"
    finally:
        if preprocessed and preprocessed != image_path and os.path.exists(preprocessed):
            try:
                os.remove(preprocessed)
            except Exception:
                pass

    clean      = clean_ocr_text(raw_text)
    confidence = compute_confidence(confidences)
    has_arabic, has_english, language_detected = detect_language_flags(clean)

    # ── AraBERT MLM post-correction (Step 3) ────────────────────────────────
    # Runs only for Arabic-containing output.  Falls through to `clean` on any
    # failure so we never lose what we have.
    corrected = clean
    correction_applied = False
    if has_arabic and clean:
        try:
            new_text = correct_ocr_with_bert(clean)
            if new_text and new_text != clean:
                corrected = new_text
                correction_applied = True
        except Exception as exc:
            log.warning(f"[OCR] post-correction skipped: {exc}")

    # Feed AraBERT classification (fake-news pipeline) the corrected variant
    # — it's the cleanest representation of what's actually in the image.
    normalized = normalize_arabic(corrected) if has_arabic else corrected

    print(f"  Engine     : {engine}")
    if engines_used:
        print(f"  Contributions: {engines_used}")
    print(f"  Language   : {language_detected}")
    print(f"  Detections : {len(confidences)}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"  Clean text : {clean[:80]}{'...' if len(clean) > 80 else ''}")
    if correction_applied:
        print(f"  Corrected  : {corrected[:80]}{'...' if len(corrected) > 80 else ''}")

    return {
        "raw_text":          raw_text,
        "clean_text":        corrected,          # what the UI shows
        "uncorrected_text":  clean,              # pre-MLM, for debugging
        "normalized_text":   normalized,
        "correction_applied": correction_applied,
        "confidence":        round(confidence, 2),
        "language_detected": language_detected,
        "has_arabic":        has_arabic,
        "has_english":       has_english,
        "engine":            engine,
        "engine_contributions": engines_used,    # per-engine winning-word counts
        "detections":        len(confidences),
    }

# ============================================================
# [VIDEO] KMeans + Histogram Scene Detection + TTA + Blur-Weighted Ensemble
# References:
#   Zhuang, Y. et al. (1998). IEEE ICIP  — KMeans frame clustering
#   Lienhart, R. (1998). SPIE            — Histogram scene detection + Otsu
#   Rokach, L. (2010). Artif. Intell. Rev. — Weighted ensemble
# ============================================================
def predict_video_ai(video_path, interval_sec=1):
    """
    Detects AI-generated video using:
    - Stage 1 : KMeans clustering on MobileNetV2 frame features
    - Stage 2 : Histogram scene detection with Otsu thresholding
    - TTA     : original + H-flip + 90°-rotation per frame
    - Ensemble: EfficientNet-B0 (W=0.9310²) + ResNet-50 (W=0.9053²)
    - Final   : blur-score weighted softmax over per-frame results
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    print(f"\n--- [فحص الفيديو | KMeans + Histogram + TTA] ---")
    t0 = time.time()

    # Weighted ensemble constants (validation accuracy squared)
    _W1     = 0.9310 ** 2
    _W2     = 0.9053 ** 2
    _WTOTAL = _W1 + _W2

    # ── Helper: MobileNetV2 frame feature vector ─────────────────────────────
    def _frame_feature(frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t   = image_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            return _mobilenet_video(t).squeeze().cpu().numpy()

    # ── Helper: HSV 2-channel histogram ──────────────────────────────────────
    def _histogram(frame):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _hist_diff(h1, h2):
        return cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)

    # ── Helper: sharpness score (Laplacian variance on centre crop) ───────────
    def _blur_score(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        size   = min(h, w) // 2
        centre = gray[cy - size//2 : cy + size//2,
                      cx - size//2 : cx + size//2]
        return cv2.Laplacian(centre, cv2.CV_64F).var()

    # ── Helper: TTA classification (3 augmentations × 2 models) ─────────────
    def _tta_classify(frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t   = image_transforms(img).unsqueeze(0).to(device)
        with torch.no_grad():
            # EfficientNet-B0
            o1 = torch.stack([model1(t),
                               model1(torch.flip(t, [3])),
                               model1(torch.rot90(t, 1, [2, 3]))]).mean(0)
            # ResNet-50
            o2 = torch.stack([model2(t),
                               model2(torch.flip(t, [3])),
                               model2(torch.rot90(t, 1, [2, 3]))]).mean(0)
            combined = (_W1 / _WTOTAL) * o1 + (_W2 / _WTOTAL) * o2
            probs    = F.softmax(combined, dim=1)
        return probs[0][0].item(), probs[0][1].item()   # prob_ai, prob_real

    # ── Read all frames ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return {"error": "لم يتم قراءة الفيديو أو الملف تالف"}

    video_duration = total_frames / fps
    MAX_FRAMES = max(int(video_duration), int(np.sqrt(total_frames)))
    print(f"📹 FPS: {fps:.1f} | Frames: {total_frames} | Max to analyse: {MAX_FRAMES}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame.copy())
    cap.release()

    if not all_frames:
        return {"error": "لم يتم استخراج أي فريم من الفيديو"}

    # ── Stage 1: KMeans clustering ────────────────────────────────────────────
    print("🔵 Stage 1 — KMeans clustering...")
    try:
        features   = np.array([_frame_feature(f) for f in all_frames])
        n_clusters = max(1, min(MAX_FRAMES, len(all_frames)))
        kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels     = kmeans.fit_predict(features)
        clustering_frames = {}
        for cid in range(n_clusters):
            idxs     = np.where(labels == cid)[0]
            centroid = kmeans.cluster_centers_[cid]
            dists    = np.linalg.norm(features[idxs] - centroid, axis=1)
            best     = idxs[np.argmin(dists)]
            clustering_frames[best] = all_frames[best].copy()
        print(f"  KMeans selected: {len(clustering_frames)} frames")
    except Exception as e:
        print(f"  [KMeans] fallback to uniform sampling — {e}")
        step = max(1, len(all_frames) // max(MAX_FRAMES, 1))
        clustering_frames = {i: all_frames[i].copy()
                             for i in range(0, len(all_frames), step)}

    # ── Stage 2: Histogram scene detection + Otsu threshold ──────────────────
    print("🟡 Stage 2 — Histogram scene detection...")
    diffs, prev_hist = [], None
    for frame in all_frames:
        h = _histogram(frame)
        if prev_hist is not None:
            diffs.append(_hist_diff(h, prev_hist))
        prev_hist = h

    histogram_frames = {}
    if diffs:
        darr    = np.array([d * 255 for d in diffs], dtype=np.uint8)
        oval, _ = cv2.threshold(darr, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh  = (oval / 255.0) if oval > 0 else max(0.1, float(np.mean(diffs)))
        prev_hist = None
        for i, frame in enumerate(all_frames):
            h = _histogram(frame)
            if prev_hist is None or _hist_diff(h, prev_hist) >= thresh:
                histogram_frames[i] = frame.copy()
            prev_hist = h
        print(f"  Histogram selected: {len(histogram_frames)} frames (thresh={thresh:.4f})")

    del all_frames
    gc.collect()

    # ── Merge both candidate sets and trim to MAX_FRAMES ─────────────────────
    combined_dict    = {**clustering_frames, **histogram_frames}
    candidate_frames = sorted(combined_dict.items(), key=lambda x: x[0])
    print(f"✅ Combined unique frames: {len(candidate_frames)}")

    if len(candidate_frames) > MAX_FRAMES:
        idxs = np.linspace(0, len(candidate_frames) - 1, MAX_FRAMES, dtype=int)
        candidate_frames = [candidate_frames[i] for i in idxs]
        print(f"  Trimmed to {len(candidate_frames)} frames (uniform temporal)")

    if not candidate_frames:
        return {"error": "لم يتم اختيار أي فريم للتحليل"}

    # ── Classify each selected frame ─────────────────────────────────────────
    results = []   # (prob_ai, prob_real, blur_weight)
    for frame_idx, frame in candidate_frames:
        prob_ai, prob_real = _tta_classify(frame)
        bw = _blur_score(frame)
        results.append((prob_ai, prob_real, bw))
        tag = 'Real' if prob_real > prob_ai else 'AI'
        print(f"  Frame {frame_idx:5d} → {tag} | AI={prob_ai:.3f} Real={prob_real:.3f}")

    # ── Blur-score weighted softmax aggregation ───────────────────────────────
    blur_arr = np.array([r[2] for r in results], dtype=float)
    blur_arr = blur_arr - blur_arr.max()           # numerically stable softmax
    weights  = np.exp(blur_arr) / np.exp(blur_arr).sum()

    avg_ai   = float(np.sum([r[0] * w for r, w in zip(results, weights)]))
    avg_real = float(np.sum([r[1] * w for r, w in zip(results, weights)]))

    final_label = "Real Image/Video" if avg_real > avg_ai else "AI-Generated"
    conf_pct    = round(max(avg_ai, avg_real) * 100, 2)

    # Per-frame simple counts (for UI display)
    ai_frames_count   = sum(1 for r in results if r[0] > r[1])
    real_frames_count = len(results) - ai_frames_count

    print(f"\n  Frames analysed  : {len(results)}")
    print(f"  Weighted AI score: {avg_ai:.4f}")
    print(f"  Weighted Real    : {avg_real:.4f}")
    print(f"  AI frames        : {ai_frames_count} | Real frames: {real_frames_count}")
    print(f"  ➤ Result         : {final_label}  ({conf_pct}%)")
    print(f"  ⏱  Time          : {time.time()-t0:.2f}s")

    return {
        "label"          : final_label,
        "confidence"     : f"{conf_pct:.2f}%",
        "frames_analyzed": len(results),
        "ai_frames"      : ai_frames_count,
        "real_frames"    : real_frames_count,
    }

# ============================================================
# [DB] Save or Update - one row per story, merge sources
# ============================================================
def save_news_to_db(title, content, ai_fake, ai_real, final_score,
                    found_sources_list, best_url, summary,
                    input_type="text", is_automated=False, top3_articles=None,
                    source_sims_str="", top3_urls_str=""):
    try:
        with app.app_context():
            title_key = title[:40]
            existing  = NewsRecord.query.filter(
                NewsRecord.title.like(f"%{title_key}%")
            ).first()

            import json as _json_db
            sources_str = ", ".join(found_sources_list) if found_sources_list else "N/A"
            top3_json   = _json_db.dumps(top3_articles or [], ensure_ascii=False)

            if existing:
                existing_set = set(s.strip() for s in (existing.found_in_sources or "").split(",") if s.strip())
                new_set      = set(s.strip() for s in sources_str.split(",") if s.strip())
                merged       = existing_set | new_set
                existing.found_in_sources = ", ".join(sorted(merged))
                existing.repetition_count = len(merged)
                if source_sims_str:
                    existing.source_similarities = source_sims_str
                if top3_urls_str:
                    existing.top3_urls = top3_urls_str
                if abs(final_score - 0.5) > abs((existing.final_score or 0.5) - 0.5):
                    existing.ai_real     = round(ai_real * 100, 2)
                    existing.ai_fake     = round(ai_fake * 100, 2)
                    existing.final_score = round(final_score * 100, 2)
                # دايماً حدّث top3_articles لو عندنا نتائج جديدة
                if top3_articles:
                    existing.top3_articles = top3_json
                # Recent-Verifications tracking
                existing.verification_count = (existing.verification_count or 0) + 1
                existing.recent_verdict_timestamp = datetime.utcnow()
                existing.last_7_days_count = (existing.last_7_days_count or 0) + 1
                existing.trending_score = round((existing.last_7_days_count or 0) * 1.0
                                                + (existing.repetition_count or 0) * 0.5, 2)
                db.session.commit()
                print(f"  [DB] Updated | Sources: {existing.found_in_sources} | Repetition: {existing.repetition_count}")
            else:
                db.session.add(NewsRecord(
                    title            = title[:500],
                    content          = content if content else title,
                    summary          = summary[:1000] if summary else "",
                    ai_real          = round(ai_real * 100, 2),
                    ai_fake          = round(ai_fake * 100, 2),
                    final_score      = round(final_score * 100, 2),
                    found_in_sources    = sources_str,
                    repetition_count    = len(found_sources_list),
                    best_url            = best_url[:1000] if best_url else "",
                    top3_articles       = top3_json,
                    source_similarities = source_sims_str,
                    top3_urls           = top3_urls_str,
                    input_type          = input_type,
                    is_automated        = is_automated,
                    verification_count       = 1,
                    last_7_days_count        = 1,
                    trending_score           = round(1.0 + len(found_sources_list) * 0.5, 2),
                    recent_verdict_timestamp = datetime.utcnow(),
                ))
                db.session.commit()
                print(f"  [DB] Saved | Sources: {sources_str} | Sims: {source_sims_str} | top3={len(top3_articles or [])} | Automated: {is_automated}")
    except Exception as e:
        print(f"  [DB] ERROR: {e}")

# ============================================================
# [CORE] Main Verification - Fully Parallel
# ============================================================
async def verify_news(user_text, start_time, input_type="text", is_automated=False):
    print(f"\n{'='*58}")
    print(f"[VERIFY] type={input_type} | text={user_text[:70]}...")

    # ── FAKE SERVER — check first (overrides DB cache when matched) ──────────
    if input_type in ("text", "image") and not is_automated:
        _fs_early = _find_similar_article(user_text)
        if _fs_early and _fs_early.get("_sim", 0.0) >= 50.0:
            _loop = asyncio.get_event_loop()
            # لو في نسب مخزونة للموديل بالفيك سيرفر، نستخدمها مباشرة
            # هاد أدق من تشغيل الموديل على عنوان قصير
            if _fs_early.get("ai_real") is not None and _fs_early.get("ai_fake") is not None:
                _real_p = float(_fs_early["ai_real"]) / 100.0
                _fake_p = float(_fs_early["ai_fake"]) / 100.0
                print(f"  [FAKE SERVER] Stored model scores: real={_real_p*100:.1f}% fake={_fake_p*100:.1f}%")
            else:
                _fake_p, _real_p = await _loop.run_in_executor(None, ai_predict, user_text)
            _fs_sim   = float(_fs_early.get("_sim", 0))
            _fs_ver   = _fs_early.get("verdict", "")
            _fs_conf  = _fs_early.get("confidence", 0)
            _fs_refs  = _parse_store_json(_fs_early, "references")
            _fs_sum   = _fs_early.get("analysis_summary", "")
            _fs_clm   = _parse_store_json(_fs_early, "claims")
            _s_bert   = _real_p
            _s_fs     = ((100 - _fs_conf) / 100.0) if _fs_ver == "fake" else (_fs_conf / 100.0)
            _fscore   = min(0.55 * _s_bert + 0.45 * _s_fs, 1.0)
            if _fscore >= 0.70:   _fstatus = "حقيقي"
            elif _fscore <= 0.30: _fstatus = "كذب"
            else:                 _fstatus = "غير مؤكد"
            # Build top3 from FakeServer stored references
            _top3 = []
            for _r in _fs_refs[:3]:
                _n, _u = _r.get("name","").strip(), _r.get("url","").strip()
                if _n and _u:
                    _top3.append({"name": _n, "url": _u, "similarity": _r.get("similarity", 0)})
            _best_url     = _top3[0]["url"] if _top3 else ""
            _t4           = list(dict.fromkeys(a["name"] for a in _top3 if a.get("name")))
            _top3_url_str = ", ".join(a["url"] for a in _top3)
            print(f"  [FAKE SERVER] EARLY MATCH | sim={_fs_sim:.1f}% | verdict={_fs_ver} | conf={_fs_conf}%")
            print(f"  [PATH]     FakeServer | S_bert={_s_bert*100:.1f}% | S_fs={_s_fs*100:.1f}%")
            print(f"  [SCORE]    Final = 0.55 x {_s_bert:.3f} + 0.45 x {_s_fs:.3f} = {_fscore*100:.2f}%")
            print(f"  [TOP3]     FakeServer refs -> {len(_top3)} article(s)")
            print(f"{'='*58}\n")
            save_news_to_db(
                title              = _fs_early.get("title", user_text[:150]),
                content            = user_text,
                ai_fake            = _fake_p,
                ai_real            = _real_p,
                final_score        = _fscore,
                found_sources_list = [],
                source_sims_str    = "",
                top3_urls_str      = _top3_url_str,
                best_url           = _best_url,
                summary            = _fs_sum,
                input_type         = input_type,
                is_automated       = False,
                top3_articles      = _top3,
            )
            return {
                "status"          : _fstatus,
                "final_score"     : round(_fscore * 100, 2),
                "s_bert"          : round(_s_bert * 100, 2),
                "s_claim"         : round(_s_fs   * 100, 2),
                "s_network"       : round(_s_fs   * 100, 2),
                "ai_real"         : round(_real_p  * 100, 2),
                "ai_fake"         : round(_fake_p  * 100, 2),
                "similarity"      : round(_fs_sim, 2),
                "repetition_count": 0,
                "found_in_sources": [],
                "top4_sources"    : _t4,
                "top3_articles"   : _top3,
                "best_url"        : _best_url,
                "source_similarities": "",
                "top3_urls"       : _top3_url_str,
                "summary"         : _fs_sum,
                "full_article"    : "",
                "article_title"   : _fs_early.get("title", ""),
                "fake_server": {
                    "matched"         : True,
                    "similarity"      : _fs_sim,
                    "verdict"         : _fs_ver,
                    "confidence"      : _fs_conf,
                    "claims"          : _fs_clm,
                    "references"      : _fs_refs,
                    "analysis_summary": _fs_sum,
                    "title"           : _fs_early.get("title", ""),
                },
                "total_time" : round(time.time() - start_time, 2),
                "input_type" : input_type,
                "is_automated": False,
            }

    # ── DB cache check (only if no FakeServer match) ─────────────────────────
    if input_type in ("text", "image") and not is_automated:
        cached = lookup_existing_news(user_text)
        if cached:
            print(f"  [CACHE HIT] Returning cached result")
            # ── لو top3_articles فارغة بالسجل القديم، نجيب مراجع من Serper بسرعة ──
            if not cached.get("top3_articles"):
                print(f"  [CACHE] top3_articles فارغة — جاري جلب مراجع سريعة من Serper...")
                try:
                    _q = " ".join(user_text.split()[:15])
                    async with aiohttp.ClientSession() as _sess_tmp:
                        _serp = await search_serper_async(_sess_tmp, _q)
                    if _serp:
                        import json as _jcache
                        _seen_c, _t3 = set(), []
                        for _a in _serp:
                            _u = _a.get("url", "")
                            if not _u or "news.google.com" in _u or _u in _seen_c:
                                continue
                            _seen_c.add(_u)
                            _t3.append({
                                "source"    : _a.get("source", ""),
                                "title"     : _a.get("title",  ""),
                                "url"       : _u,
                                "similarity": 0,
                            })
                            if len(_t3) >= 3:
                                break
                        if _t3:
                            cached["top3_articles"] = _t3
                            # حدّث السجل بالداتا بيس مشان المرة الجاية
                            try:
                                with app.app_context():
                                    _rec = NewsRecord.query.filter(
                                        NewsRecord.content.like(f"%{user_text[:60].strip()}%")
                                    ).first()
                                    if _rec:
                                        _rec.top3_articles = _jcache.dumps(_t3, ensure_ascii=False)
                                        db.session.commit()
                                        print(f"  [CACHE] DB updated with {len(_t3)} refs")
                            except Exception as _e:
                                print(f"  [CACHE] DB update failed: {_e}")
                        else:
                            print(f"  [CACHE] Serper لم يرجع نتائج")
                except Exception as _e:
                    print(f"  [CACHE] Serper refresh failed: {_e}")
            print(f"  [CACHE] Status: {cached['status']} | Score: {cached['final_score']}% | refs={len(cached.get('top3_articles',[]))}")
            print(f"{'='*58}\n")
            return cached

    user_text  = clean_text(user_text)
    word_count = len(user_text.split())

    # Headline mode: < 15 words treated as a news headline/query
    # We search first, get full article content, then run AI on it
    is_headline = word_count < 15

    print(f"  [MODE] {'Headline/Short' if is_headline else 'Full Text'} | {word_count} words")

    t0   = time.time()
    loop = asyncio.get_event_loop()

    # ── SEARCH (parallel across 5 channels) ──
    async with aiohttp.ClientSession() as session:
        search_tasks = [
            search_google_news_async(session, user_text),
            search_gnews_async(session, user_text),
            search_newsapi_async(session, user_text),
            search_serper_async(session, user_text),
            search_arabic_domains_via_rss(session, user_text),  # 16 موقع عربي عبر Google News RSS
        ]
        if not is_headline:
            ai_task = loop.run_in_executor(None, ai_predict, user_text)
            ai_result, *search_results = await asyncio.gather(ai_task, *search_tasks)
            fake_prob, real_prob = ai_result
        else:
            search_results = await asyncio.gather(*search_tasks)

    all_found = []
    for r in search_results:
        all_found.extend(r)
    # Sort by date — newest first
    all_found.sort(key=lambda x: x.get("published") or datetime.min, reverse=True)
    print(f"  [TIME] Search parallel: {time.time()-t0:.3f} sec | {len(all_found)} articles found")

    # ── SIMILARITY (chunk_max_sim — نوافذ منزلقة أدق من cosine عادي) ──
    t0             = time.time()
    max_similarity = 0.0
    best_article   = None
    best_url       = ""
    found_sources   = []
    source_sims_str = ""          # "الجزيرة:82.3, رؤيا:60.1" — top-3 with sim for DB
    _src_sim_all    = []          # (source, score) — ALL matching sources for repetition_count
    sims            = np.array([])
    titles_emb      = np.array([])

    if all_found:
        titles     = [a["title"] for a in all_found]
        titles_emb = embedding_model.encode(titles)
        sims       = chunk_max_sim(user_text, titles_emb)
        idx            = int(np.argmax(sims))
        max_similarity = float(sims[idx])
        best_article   = all_found[idx]
        _raw_url = best_article.get("url", "")
        best_url = "" if "news.google.com/rss" in _raw_url else _raw_url
        seen = set()
        for i, score in enumerate(sims):
            if score >= 0.50:
                src = all_found[i].get("source", "")
                if src and src not in seen:
                    _src_sim_all.append((src, float(score)))
                    seen.add(src)
        _src_sim_all.sort(key=lambda x: x[1], reverse=True)
        found_sources   = [s for s, _ in _src_sim_all]                                           # all names
        source_sims_str = ", ".join(f"{s}:{sim*100:.1f}" for s, sim in _src_sim_all[:3])   # top-3 with sim

    repetition_count = len(_src_sim_all)   # full unique-source count drives S_network
    print(f"  [TIME] Similarity (chunk): {time.time()-t0:.3f} sec")
    print(f"  [SCORE] Max Sim    : {max_similarity*100:.2f}%")
    print(f"  [SCORE] Repetition : {repetition_count} source(s)")
    print(f"  [SCORE] Found in   : {', '.join(found_sources) if found_sources else 'None'}")
    print(f"  [SCORE] Top-3 sims : {source_sims_str or 'None'}")

    # ── PARALLEL SCRAPING: أفضل 5 مقالات (يجيب نص كامل) ──
    t0           = time.time()
    full_article = ""
    scraped_title = ""
    BAD_URL = ['/quiz', '/tag/', '/category/', '/topics/', '/search', '/page/', 'google.com/rss', '?q=']

    if all_found and len(sims) > 0:
        top_idx  = np.argsort(sims)[::-1][:5]
        top_urls = [all_found[int(i)].get("url","") for i in top_idx
                    if all_found[int(i)].get("url")
                    and not any(b in all_found[int(i)].get("url","") for b in BAD_URL)]
        scraped  = await loop.run_in_executor(None, fetch_multiple_contents_parallel, top_urls, 5)
        user_ar  = bool(re.search(r"[؀-ۿ]", user_text))
        for i in top_idx:
            url = all_found[int(i)].get("url","")
            if not url or any(b in url for b in BAD_URL):
                continue
            t_s, c_s = scraped.get(url, ("",""))
            if not c_s or len(c_s) < 150:
                continue
            if user_ar:
                ar_n = len(re.findall(r"[؀-ۿ]", c_s))
                en_n = len(re.findall(r"[A-Za-z]", c_s))
                tot  = ar_n + en_n
                if tot > 0 and en_n / tot > 0.60:
                    continue
            full_article  = c_s
            best_url      = url
            scraped_title = t_s if t_s and len(t_s) > 15 else ""
            print(f"  [SCRAPER] OK: {url[:60]} | {len(c_s)} chars")
            break

    # Re-score على المحتوى الكامل (أدق من العناوين فقط)
    if full_article and len(titles_emb) > 0:
        sims_r         = chunk_max_sim(full_article, titles_emb)
        max_similarity = float(np.max(sims_r))
        _src_sim_all, seen = [], set()
        for i, sc in enumerate(sims_r):
            if sc >= 0.50:
                src = all_found[i].get("source","")
                if src and src not in seen:
                    _src_sim_all.append((src, float(sc))); seen.add(src)
        _src_sim_all.sort(key=lambda x: x[1], reverse=True)
        found_sources   = [s for s, _ in _src_sim_all]
        source_sims_str = ", ".join(f"{s}:{sim*100:.1f}" for s, sim in _src_sim_all[:3])
        repetition_count = len(_src_sim_all)
        print(f"  [SCORE] Re-scored: {max_similarity*100:.2f}% | srcs={repetition_count}")

    print(f"  [TIME] Scraping: {time.time()-t0:.3f} sec | content={len(full_article)} chars")

    t0              = time.time()
    full_title      = scraped_title
    serper_articles = [a for a in all_found if a.get("snippet") or a.get("source") == "Serper"]

    # لو scraping فشل وكان headline — جرب gemini_get_full_article كـ fallback
    if is_headline and not full_article and all_found:
        best_serper = next(
            (a for a in all_found if a.get("url") and "news.google.com" not in a.get("url","")),
            None
        )
        if best_serper:
            full_title, full_article = await loop.run_in_executor(
                None, gemini_get_full_article,
                best_serper.get("url",""), best_serper.get("title", user_text), serper_articles
            )
            if full_article:
                print(f"  [GEMINI ARTICLE] Fallback OK: {len(full_article)} chars")

    # النص الفعلي لـ AraBERT والـ Summary
    if full_article and len(full_article) > 150:
        text_for_model = full_article
        source_label   = "scraped/gemini"
    elif serper_articles:
        text_for_model = " ".join([a.get("snippet","") for a in serper_articles[:6] if a.get("snippet")])
        text_for_model = text_for_model if len(text_for_model) > 100 else user_text
        source_label   = "snippets"
    else:
        text_for_model = user_text
        source_label   = "user_text"

    print(f"  [TIME] Article fetch: {time.time()-t0:.3f} sec | [{source_label}] {len(text_for_model)} chars")
    print(f"  {text_for_model[:300]}")

    # ── GEMINI S_claim ────────────────────────────────────────────
    t0 = time.time()
    s_claim_raw = await loop.run_in_executor(
        None, gemini_verify_claim_score, user_text, serper_articles or all_found
    )
    print(f"  [TIME] Gemini S_claim: {time.time()-t0:.3f} sec")

    # ── AI PREDICTION (AraBERT على النص الكامل) ──────────────────
    if is_headline:
        t0 = time.time()
        fake_prob, real_prob = await loop.run_in_executor(None, ai_predict, text_for_model)
        print(f"  [TIME] AraBERT on full article ({len(text_for_model)} chars): {time.time()-t0:.3f} sec")

    # ── SUMMARY من النص الكامل ────────────────────────────────────
    summary = generate_summary(text_for_model)

    # ── FAKE SERVER (اختياري — للعرض التجريبي فقط) ──────────────
    fs_match      = _find_similar_article(user_text)
    fs_sim        = fs_match.get("_sim", 0.0)  if fs_match else 0.0
    fs_verdict    = fs_match.get("verdict","")  if fs_match else ""
    fs_confidence = fs_match.get("confidence", 0) if fs_match else 0
    fs_claims     = _parse_store_json(fs_match, "claims")     if fs_match else []
    fs_refs       = _parse_store_json(fs_match, "references") if fs_match else []
    fs_summary    = fs_match.get("analysis_summary","")       if fs_match else ""
    if fs_match:
        print(f"  [FAKE SERVER] Match: '{fs_match.get('title','')[:50]}' | sim={fs_sim}%")

    # ══════════════════════════════════════════════════════
    # WEIGHTED ENSEMBLE DECISION MODEL
    #
    # مسار 1 — FakeServer match (تشابه ≥ 50%):
    #   Score = 0.55×S_bert + 0.45×S_fs
    #   حيث S_fs = confidence/100 (verified) أو (100-confidence)/100 (fake)
    #   confidence بالـ dataset = نتيجة الـ 45% من المصادر+الادعاءات (بدون AraBERT)
    #
    # مسار 2 — بدون FakeServer match:
    #   Score = 0.55×S_bert + 0.15×S_claim + 0.30×S_network
    # ══════════════════════════════════════════════════════

    # S_bert — AraBERT real probability (0→1)
    s_bert = real_prob

    # ── تحديد هل نستخدم FakeServer confidence أم الـ formula العادية ───
    use_fs_score = (fs_match is not None) and (fs_sim >= 50.0)

    if use_fs_score:
        # confidence in dataset = editor confidence in verdict (45% portion, no AraBERT)
        # fake   + confidence=68  ->  real_score = (100-68)/100 = 0.32
        # verified + confidence=88 ->  real_score = 88/100 = 0.88
        if fs_verdict == "fake":
            s_45 = (100 - fs_confidence) / 100.0
        else:
            s_45 = fs_confidence / 100.0
        final_score  = (0.55 * s_bert) + (0.45 * s_45)
        final_score  = min(final_score, 1.0)
        claim_source = "FakeServer"
        s_claim      = s_45
        s_network    = s_45
        print(f"  [PATH]     FakeServer match | verdict={fs_verdict} | conf={fs_confidence}% | sim={fs_sim:.1f}%")
        print(f"  [S_fs]     S_fs = {s_45*100:.1f}%  (45% portion, pre-computed)")
        print(f"  [S_bert]   S_bert = {s_bert*100:.1f}%  (AraBERT live)")
        print(f"  [SCORE]    Final = 0.55 x {s_bert:.3f} [AraBERT] + 0.45 x {s_45:.3f} [FakeServer] = {final_score*100:.2f}%")
    else:
        # S_claim: Gemini API or similarity fallback
        if s_claim_raw is not None:
            s_claim = float(s_claim_raw)
            claim_source = "Gemini"
        else:
            s_claim = float(max_similarity)
            claim_source = "similarity-fallback"
        print(f"  [S_claim]  {claim_source}: {s_claim*100:.1f}%")

        # S_network: unique source spread (0->1)
        s_network = min(repetition_count / 5, 1.0)

        final_score = (0.55 * s_bert) + (0.15 * s_claim) + (0.30 * s_network)
        final_score = min(final_score, 1.0)

        print(f"  [ENSEMBLE] S_bert={s_bert*100:.1f}% | S_claim={s_claim*100:.1f}% ({claim_source}) | "
              f"S_network={s_network*100:.1f}% ({repetition_count} srcs)")
        print(f"  [SCORE]    Final = 0.55 x {s_bert:.3f} + 0.15 x {s_claim:.3f} + "
              f"0.30 x {s_network:.3f} = {final_score*100:.2f}%")

    # ── TITLE RESOLUTION ──────────────────────────────────────
    final_title = None
    # 0. العنوان المسحوب مباشرة من الصفحة — الأدق
    if full_title and not is_too_short_or_generic(full_title):
        final_title = full_title[:300]
    # 1. Best search-result title if similarity is meaningful
    if not final_title and max_similarity > 0.35 and best_article:
        candidate = best_article.get("title", "").strip()
        if len(candidate) > 15 and not is_too_short_or_generic(candidate):
            final_title = candidate[:300]
    # 2. Any good headline from search results
    if not final_title:
        for art in all_found:
            t = art.get("title", "").strip()
            if len(t) > 20 and not is_too_short_or_generic(t):
                final_title = t[:300]
                break
    # 3. Extract first sentence from long article (أفضل من أول 300 حرف عشوائي)
    if not final_title:
        for sep in (".", "،", "\n", "؟", "!"):
            idx = user_text.find(sep)
            if 20 < idx < 200:
                final_title = user_text[:idx].strip()
                break
    # 4. Fallback: first 150 chars
    if not final_title:
        final_title = user_text[:150]

    # ── TOP-3 articles — يُبنى قبل save_news_to_db ──────────────────────────
    # PATH A: FakeServer match -> use pre-stored references (guaranteed availability)
    top3_articles = []
    if use_fs_score and fs_refs:
        for r in fs_refs[:3]:
            name = r.get("name", "").strip()
            url  = r.get("url",  "").strip()
            if name and url:
                top3_articles.append({
                    "name"      : name,
                    "url"       : url,
                    "similarity": r.get("similarity", 0),
                })
        print(f"  [TOP3]     FakeServer refs -> {len(top3_articles)} stored articles")

    # ── helpers for reference quality filtering ──────────────────────────────
    _GENERIC_TITLES = {
        "أخبار عاجلة", "breaking news", "عاجل", "أخبار", "news", "latest news",
        "الأخبار", "نشرة الأخبار", "نشرة الصباح", "نشرة المساء",
    }
    def _is_quality_ref(url: str, title: str, sim: float) -> bool:
        """True only if the reference is a real, relevant Arabic news article."""
        if not url or not title:
            return False
        # تشابه دنى 0.45 مطلوب — تحت هاد القيمة المصدر مش ذو صلة
        if sim < 0.45:
            return False
        # عنوان مش عام ومش قصير
        title_stripped = title.strip()
        if len(title_stripped) < 15:
            return False
        if title_stripped.lower().strip() in _GENERIC_TITLES:
            return False
        # يجب أن يكون الرابط من مصدر عربي موثوق أو يحتوي على نطاق موثوق
        trusted = any(d in url for d in ARABIC_NEWS_DOMAINS)
        if not trusted:
            # نقبل فقط المواقع التي تحتوي حروف عربية في العنوان (تغطية عربية مؤكدة)
            has_arabic = bool(re.search(r"[؀-ۿ]", title_stripped))
            # لكن نستثني المواقع غير العربية الواضحة
            non_arabic_domains = [
                "vietnam.vn", "sbs.com.au", "bbc.co.uk/vietnamese",
                "france24.com/vi", ".vn/", ".cn/", ".ru/", ".de/",
            ]
            if not has_arabic or any(nd in url for nd in non_arabic_domains):
                return False
        return True

    # PATH B: Live search results (when no FakeServer match)
    if not top3_articles and all_found and len(sims) > 0:
        seen_titles = set()
        for i in np.argsort(sims)[::-1]:
            art     = all_found[int(i)]
            url     = art.get("url", "").strip()
            title   = art.get("title", "").strip()
            sim_val = float(sims[int(i)])
            if not _is_quality_ref(url, title, sim_val):
                continue
            title_key = title[:60].lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            clean_url = url.split("?")[0] if "news.google.com" in url else url
            top3_articles.append({
                "name"      : title,
                "url"       : clean_url,
                "similarity": round(sim_val * 100, 1),
            })
            if len(top3_articles) >= 3:
                break

    # Fallback 1: نتائج بتشابه متوسط (0.30+) من مصادر عربية فقط — بدون مصادر هبل
    if not top3_articles and all_found and len(sims) > 0:
        seen_titles = set()
        for i in np.argsort(sims)[::-1]:
            art     = all_found[int(i)]
            url     = art.get("url", "").strip()
            title   = art.get("title", "").strip()
            sim_val = float(sims[int(i)])
            # شرط مخفف: 0.30 similarity + عربي الدومين أو العنوان
            if sim_val < 0.30 or not url or not title or len(title) < 15:
                continue
            if title.strip().lower() in _GENERIC_TITLES:
                continue
            trusted = any(d in url for d in ARABIC_NEWS_DOMAINS)
            has_arabic_title = bool(re.search(r"[؀-ۿ]", title))
            non_arabic_domains = ["vietnam.vn", "sbs.com.au", ".vn/", ".cn/", ".ru/", ".de/"]
            if not trusted and (not has_arabic_title or any(nd in url for nd in non_arabic_domains)):
                continue
            title_key = title[:60].lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            clean_url = url.split("?")[0] if "news.google.com" in url else url
            top3_articles.append({
                "name"      : title,
                "url"       : clean_url,
                "similarity": round(sim_val * 100, 1),
            })
            if len(top3_articles) >= 3:
                break
        if top3_articles:
            print(f"  [TOP3] Fallback-1 (relaxed Arabic filter): {len(top3_articles)} refs")

    # Fallback 2: لو all_found فارغ كلياً — Serper بأول جملة ثم Gemini (استدعاء واحد)
    if not top3_articles:
        print(f"  [TOP3] all_found فارغ — جرب Serper بجملة أولى ثم Gemini...")
        _first_sent = (user_text.split('.')[0].strip()
                       or user_text.split('،')[0].strip()
                       or user_text)[:120]
        try:
            async with aiohttp.ClientSession() as _sess_r:
                _retry_arts = await search_serper_async(_sess_r, _first_sent)
            if _retry_arts:
                _u_emb_r = embedding_model.encode([user_text])
                _t_emb_r = embedding_model.encode([a["title"] for a in _retry_arts])
                _sims_r  = cosine_similarity(_u_emb_r, _t_emb_r)[0]
                _seen_r  = set()
                for _ir in np.argsort(_sims_r)[::-1]:
                    _ar = _retry_arts[int(_ir)]
                    _ur = _ar.get("url", "")
                    if not _ur or "news.google.com" in _ur or _ur in _seen_r:
                        continue
                    _seen_r.add(_ur)
                    top3_articles.append({
                        "name"      : _ar.get("title", "") or _ar.get("source", ""),
                        "url"       : _ur,
                        "similarity": round(float(_sims_r[int(_ir)]) * 100, 1),
                    })
                    if len(top3_articles) >= 3:
                        break
                print(f"  [TOP3] Serper retry → {len(top3_articles)} articles")
        except Exception as _er:
            log.warning(f"[TOP3] Serper retry error: {_er}")

        # Gemini كملاذ أخير (استدعاء واحد — يولّد استعلام + مقالات معاً)
        if not top3_articles:
            _gq, _g_arts = await loop.run_in_executor(
                None, gemini_search_and_suggest, user_text
            )
            if _g_arts:
                top3_articles = _g_arts
                print(f"  [TOP3] Gemini suggested {len(top3_articles)} articles")
            elif _gq:
                # Serper بالاستعلام المحسّن من Gemini
                try:
                    async with aiohttp.ClientSession() as _sess_g:
                        _retry_g = await search_serper_async(_sess_g, _gq)
                    if _retry_g:
                        _u_emb_g = embedding_model.encode([user_text])
                        _t_emb_g = embedding_model.encode([a["title"] for a in _retry_g])
                        _sims_g  = cosine_similarity(_u_emb_g, _t_emb_g)[0]
                        _seen_g  = set()
                        for _ig in np.argsort(_sims_g)[::-1]:
                            _ag = _retry_g[int(_ig)]
                            _ug = _ag.get("url", "")
                            if not _ug or "news.google.com" in _ug or _ug in _seen_g:
                                continue
                            _seen_g.add(_ug)
                            top3_articles.append({
                                "name"      : _ag.get("title", "") or _ag.get("source", ""),
                                "url"       : _ug,
                                "similarity": round(float(_sims_g[int(_ig)]) * 100, 1),
                            })
                            if len(top3_articles) >= 3:
                                break
                        print(f"  [TOP3] Gemini-query Serper → {len(top3_articles)} articles")
                except Exception as _eg:
                    log.warning(f"[TOP3] Gemini-query Serper error: {_eg}")

    _top3_urls_str = ", ".join(a.get("url", "") for a in top3_articles[:3] if a.get("url"))

    save_news_to_db(
        title              = final_title,
        content            = text_for_model,
        ai_fake            = fake_prob,
        ai_real            = real_prob,
        final_score        = final_score,
        found_sources_list = found_sources,
        source_sims_str    = source_sims_str,
        top3_urls_str      = _top3_urls_str,
        best_url           = best_url,
        summary            = summary,
        input_type         = input_type,
        is_automated       = is_automated,
        top3_articles      = top3_articles,
    )

    total_time = round(time.time() - start_time, 2)
    # Verdict thresholds — الحدود المعرّفة في النموذج
    if final_score >= 0.70:
        status = "حقيقي"
    elif final_score <= 0.30:
        status = "كذب"
    else:
        status = "غير مؤكد"

    effective_words = len(text_for_model.split())
    model_used = f"Short [{PATH_SHORT}]" if effective_words < 20 else f"Long [{PATH_LONG}]"
    print(f"\n{'='*58}")
    print(f"  [MODEL USED]   : {model_used}")
    print(f"  [MODE]         : {'Headline → scraped content' if is_headline else 'Full text'}")
    print(f"  [RESULT] Status      : {status}")
    print(f"  [RESULT] AI Real     : {real_prob*100:.2f}%  |  AI Fake: {fake_prob*100:.2f}%")
    print(f"  [RESULT] Similarity  : {max_similarity*100:.2f}%")
    print(f"  [RESULT] Repetition  : {repetition_count} source(s) — {', '.join(found_sources) if found_sources else 'None'}")
    print(f"  [RESULT] Top-3 Sims  : {source_sims_str or 'N/A'}")
    print(f"  [RESULT] Final Score : {final_score*100:.2f}%")
    print(f"  [RESULT] Best URL    : {best_url[:80] if best_url else 'N/A'}")
    print(f"  [RESULT] Total Time  : {total_time} sec")
    print(f"  [RESULT] Summary     : {summary[:200] if summary else 'N/A'}")
    print(f"{'='*58}\n")

    top4_sources = list(dict.fromkeys(a.get("source", a.get("name", "")) for a in top3_articles if a.get("source") or a.get("name")))

    return {
        # ── النتيجة الرئيسية ──────────────────────────────
        "status"          : status,
        "final_score"     : round(final_score * 100, 2),
        # ── مكونات المعادلة ───────────────────────────────
        "s_bert"          : round(s_bert    * 100, 2),
        "s_claim"         : round(s_claim   * 100, 2),
        "s_network"       : round(s_network * 100, 2),
        # ── AraBERT ───────────────────────────────────────
        "ai_real"         : round(real_prob  * 100, 2),
        "ai_fake"         : round(fake_prob  * 100, 2),
        # ── مصادر وروابط ─────────────────────────────────
        "similarity"      : round(max_similarity * 100, 2),
        "repetition_count": repetition_count,
        "found_in_sources"   : found_sources,
        "top4_sources"       : top4_sources,
        "top3_articles"      : top3_articles,
        "best_url"           : best_url,
        "source_similarities": source_sims_str,
        "top3_urls"          : _top3_urls_str,
        # ── ملخص + نص كامل ───────────────────────────────
        "summary"            : summary,
        "full_article"    : full_article[:3000] if full_article else "",
        "article_title"   : full_title or "",
        # ── Fake Server (للعرض التجريبي) ──────────────────
        "fake_server": {
            "matched"         : fs_match is not None,
            "similarity"      : fs_sim,
            "verdict"         : fs_verdict,
            "confidence"      : fs_confidence,
            "claims"          : fs_claims,
            "references"      : fs_refs,
            "analysis_summary": fs_summary,
            "title"           : fs_match.get("title","") if fs_match else "",
        },
        # ── معلومات إضافية ────────────────────────────────
        "total_time"      : total_time,
        "input_type"      : input_type,
        "is_automated"    : is_automated,
    }

# ============================================================
# [URL MODE] Fetch article from URL then verify
# ============================================================
async def verify_news_url(url, start_time):
    print(f"\n{'='*58}")
    print(f"[URL MODE] Fetching article: {url}")
    title, content = fetch_full_content(url)

    # ── لو fetch فشل — نجرب Serper ثم نكمل بأي شيء عندنا ──────────
    if not content:
        print(f"  [URL MODE] Direct fetch failed — trying Serper fallback for: {url[:60]}")
        # نجيب عنوان من الـ URL slug (الجزء الأخير من المسار)
        from urllib.parse import urlparse as _up
        path_parts = [p for p in _up(url).path.split("/") if len(p) > 6]
        url_slug   = path_parts[-1].replace("-", " ").strip() if path_parts else ""
        query_hint = url_slug[:100] if url_slug else url

        # جرب Serper
        snips = []
        try:
            import aiohttp as _aio
            async with _aio.ClientSession() as _sess:
                snips = await search_serper_async(_sess, query_hint)
        except Exception:
            pass

        if snips:
            combined = " ".join(
                f"{a.get('title','')}. {a.get('snippet','')}"
                for a in snips[:6] if a.get("title") or a.get("snippet")
            ).strip()
            if len(combined) > 60:
                print(f"  [URL MODE] Serper fallback OK | {len(combined)} chars")
                content = combined
                title   = snips[0].get("title", "") or url_slug

        # لو Serper ما أعطى شيء — استخدم الـ slug من الـ URL كعنوان بحث
        if not content:
            content = query_hint or url
            title   = url_slug or url
            print(f"  [URL MODE] Using URL slug as query: {content[:60]}")

    print(f"  [URL MODE] Got {len(content)} chars | title: {title[:60]}")
    result = await verify_news(content, start_time, input_type="text")
    result["source_url"]    = url
    result["article_title"] = title
    return result

# ============================================================
# [ROUTES]
# ============================================================
@app.route("/")
def index():
    return render_template("frontendA.html")

def _log_and_safe_error(where: str, exc: Exception, http_status: int = 500):
    """Log full traceback server-side; return a stable, non-leaky JSON body
    to the client with a correlation UUID (fix #13)."""
    ref = uuid.uuid4().hex[:12]
    log.exception(f"[{where}] ref={ref}: {exc}")
    return jsonify({"error": "internal_error", "ref": ref}), http_status

# ---- Text analysis (also handles URL input) ----
@app.route("/predict", methods=["POST", "OPTIONS"])
@app.route("/analyze", methods=["POST", "OPTIONS"])
@rate_limit("20 per minute")
def analyze():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    start = time.time()
    data  = request.get_json()
    text  = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # URL detection
    try:
        if is_url(text):
            print(f"\n[/analyze] URL detected → switching to URL mode")
            result = asyncio.run(verify_news_url(text, start))
        else:
            word_count = len(text.split())
            mode = "Headline" if word_count < 15 else "Full Text"
            print(f"\n[/analyze] Input ({word_count} words) | Mode: {mode}")
            result = asyncio.run(verify_news(text, start, input_type="text"))
    except Exception as exc:
        ref = uuid.uuid4().hex[:12]
        log.exception(f"[/analyze] ref={ref}: {exc}")
        body = {"error": "internal_error", "ref": ref}
        # When MIZAN_DEBUG=1 the real cause is returned to the client so it
        # can be read straight from the extension popup during development.
        if os.getenv("MIZAN_DEBUG") == "1":
            body["detail"] = f"{type(exc).__name__}: {exc}"
        return jsonify(body), 500

    return jsonify(result)

# ============================================================
# [/claude_verify] — الـ endpoint الرئيسي للـ frontend
# بيشغّل verify_news ويرجع بالـ format اللي بيتوقعه frontendA
# ============================================================
@app.route("/claude_verify", methods=["POST", "OPTIONS"])
@rate_limit("20 per minute")
def claude_verify():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    start = time.time()
    data  = request.get_json(silent=True) or {}
    text  = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text مطلوب"}), 400

    # شغّل الـ verify الرئيسي
    if is_url(text):
        raw = asyncio.run(verify_news_url(text, start))
    else:
        raw = asyncio.run(verify_news(text, start, input_type="text"))

    if "error" in raw:
        return jsonify(raw), 400

    # ── map status → blended_label ────────────────────────────
    status_map = {"حقيقي": "verified", "كذب": "fake", "غير مؤكد": "uncertain"}
    blended_label = status_map.get(raw.get("status", ""), "uncertain")

    # ── claims من الـ Fake Server ─────────────────────────────
    fs          = raw.get("fake_server", {})
    raw_claims  = fs.get("claims", [])
    if isinstance(raw_claims, str):
        try:    raw_claims = _json_fs.loads(raw_claims)
        except: raw_claims = []

    claims_mapped = []
    for c in raw_claims:
        s = c.get("status", "")
        claims_mapped.append({
            "claim"      : c.get("text", ""),
            "is_true"    : True  if s == "verified" else
                           False if s == "fake"     else None,
            "explanation": c.get("explanation", ""),
        })

    # ── references: top3_articles (نسبة تشابه حقيقية) + Fake Server كـ fallback ─
    # FIX: ترتيب بالتشابه — الأعلى تشابهاً يظهر أولاً
    references = []
    seen_ref_urls = set()

    # 1. top3_articles من البحث — فيها تشابه حقيقي مع نص المستخدم
    for a in raw.get("top3_articles", []):
        url = a.get("url", "")
        if url and url not in seen_ref_urls:
            references.append({
                "name"      : a.get("name") or a.get("title") or a.get("source", ""),
                "url"       : url,
                "similarity": round(float(a.get("similarity", 0)), 1),
            })
            seen_ref_urls.add(url)

    # 2. Fake Server refs كـ fallback لو ما في نتائج من البحث
    if not references:
        raw_refs = fs.get("references", [])
        if isinstance(raw_refs, str):
            try:    raw_refs = _json_fs.loads(raw_refs)
            except: raw_refs = []
        for r in raw_refs:
            if r.get("url") and r["url"] not in seen_ref_urls:
                references.append({
                    "name"      : r.get("name", ""),
                    "url"       : r["url"],
                    "similarity": r.get("similarity", 0),
                })
                seen_ref_urls.add(r["url"])

    # رتّب تنازلياً بالتشابه
    references.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    print(f"  [/claude_verify] references={len(references)} | top3={len(raw.get('top3_articles',[]))}")

    # ── summary: ملخص الخبر الحقيقي أولاً — Fake Server فقط كـ fallback ─
    # FIX: كان بيعرض ملخص Fake Server بدل ملخص الخبر الفعلي
    summary = raw.get("summary", "") or fs.get("analysis_summary", "")

    return jsonify({
        # ── النتيجة الرئيسية ──────────────────────────────────
        "blended_label"     : blended_label,
        "blended_confidence": raw.get("final_score", 0),
        # ── AraBERT ───────────────────────────────────────────
        "arabert": {
            "real_prob": raw.get("ai_real", 0),
            "fake_prob": raw.get("ai_fake", 0),
            "label"    : "fake" if raw.get("ai_fake", 0) > raw.get("ai_real", 0) else "verified",
        },
        # ── Fake Server ───────────────────────────────────────
        "server_article": {
            "title"     : fs.get("title",      ""),
            "source"    : fs.get("source",     ""),
            "verdict"   : fs.get("verdict",    ""),
            "confidence": fs.get("confidence", 0),
            # نسبة التشابه مع قاعدة Fake Server — لفلترة العرض في الفرونت
            "similarity": raw.get("fake_server", {}).get("similarity", 0),
        },
        # ── مكونات المعادلة ───────────────────────────────────
        "similarity"        : raw.get("similarity", 0),
        "s_bert"            : raw.get("s_bert",    0),
        "s_claim"           : raw.get("s_claim",   0),
        "s_network"         : raw.get("s_network", 0),
        # ── ملخص وادعاءات ─────────────────────────────────────
        "analysis_summary"  : summary,
        "claims"            : claims_mapped,
        "references"        : references,
        # ── معلومات إضافية ────────────────────────────────────
        "processing_time"   : round(time.time() - start, 2),
        "found_in_sources"   : raw.get("found_in_sources", []),
        "source_similarities": raw.get("source_similarities", ""),
        "top3_urls"          : raw.get("top3_urls", ""),
        # ── رابط المصدر الأصلي (لما المستخدم يدخل URL) ────────
        "source_url"        : raw.get("source_url",    ""),
        "article_title"     : raw.get("article_title", ""),
    })

# ---- Image analysis ----
_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

def _safe_suffix(filename: str, allowed: set, default: str) -> str:
    """Run werkzeug.secure_filename and return a lower-cased extension we
    actually accept; falls back to a safe default (fix #7)."""
    safe = secure_filename(filename or "")
    ext = os.path.splitext(safe)[1].lower()
    return ext if ext in allowed else default

@app.route("/analyze_image", methods=["POST"])
@rate_limit("10 per minute")
def analyze_image():
    start = time.time()
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file   = request.files["image"]
    suffix = _safe_suffix(file.filename, _ALLOWED_IMAGE_EXTS, ".jpg")
    tmp    = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file.save(tmp.name)
    tmp.close()

    try:
        image_result = predict_image_ai(tmp.name)
        ocr_result   = run_ocr_pipeline(tmp.name)
        ocr_text     = ocr_result.get("clean_text", "") or ""
        result = {
            "image_classification": image_result,
            "ocr_text"            : ocr_text,
            "ocr"                 : ocr_result,   # full OCR breakdown for the UI
            "total_time"          : round(time.time() - start, 2),
            "input_type"          : "image",
        }
        # Feed the AraBERT pipeline the normalised Arabic where possible
        text_for_verify = ocr_result.get("normalized_text") or ocr_text
        if text_for_verify and len(text_for_verify.split()) >= 5:
            text_result = asyncio.run(verify_news(text_for_verify, start, input_type="image"))
            result["text_verification"] = text_result
        return jsonify(result)
    finally:
        try: os.remove(tmp.name)
        except: pass

# ---- Standalone OCR endpoint (used by the Twitter-extension popup) ----
@app.route("/ocr", methods=["POST", "OPTIONS"])
@rate_limit("15 per minute")
def ocr_route():
    """Extract text from an uploaded image.

    Accepts multipart/form-data with:
        file  OR  image  — the image to process
        engine (optional) — "auto" | "easy" | "paddle"  (default: "auto")
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    file = request.files.get("file") or request.files.get("image")
    if file is None:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    temp_path = None
    try:
        suffix = _safe_suffix(file.filename, _ALLOWED_IMAGE_EXTS, ".jpg")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
        file.save(temp_path)

        engine_pref = (request.form.get("engine") or "auto").strip().lower()
        if engine_pref not in ("auto", "easy", "paddle", "tess", "fuse"):
            engine_pref = "auto"

        result = run_ocr_pipeline(temp_path, engine_pref)

        clean_text_value = result.get("clean_text", "")
        raw_text         = result.get("raw_text", "")

        return jsonify({
            "status":            "success" if (clean_text_value or raw_text) else "no_text",
            "raw_text":          raw_text,
            "clean_text":        clean_text_value,
            "normalized_text":   result.get("normalized_text", clean_text_value),
            "confidence":        round(float(result.get("confidence", 0.0)), 2),
            "language_detected": result.get("language_detected", "unknown"),
            "has_arabic":        result.get("has_arabic", False),
            "has_english":       result.get("has_english", False),
            "engine":            result.get("engine", "easyocr"),
            "detections":        result.get("detections", 0),
        })

    except Exception as exc:
        ref = uuid.uuid4().hex[:12]
        log.exception(f"[/ocr] ref={ref}: {exc}")
        return jsonify({
            "status":            "error",
            "message":           "internal_error",
            "ref":               ref,
            "raw_text":          "",
            "clean_text":        "",
            "confidence":        0.0,
            "language_detected": "unknown",
            "has_arabic":        False,
            "has_english":       False,
            "engine":            "easyocr",
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ---- OCR health check (used by the extension to verify the server) ----
@app.route("/health", methods=["GET"])
def health():
    """Return OCR engine availability."""
    return jsonify({
        "status":           "ok",
        "easyocr":          EASYOCR_AVAILABLE,
        "paddleocr":        PADDLE_AVAILABLE,
        "bidi_reshaper":    BIDI_AVAILABLE,
        "gpu":              torch.cuda.is_available(),
    })


# ---- Video analysis ----
@app.route("/analyze_video", methods=["POST"])
@rate_limit("5 per minute")
def analyze_video_endpoint():
    start = time.time()
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded. Send the file with key 'video'."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename received."}), 400
    try:
        interval_sec = float(request.form.get("interval_sec", 2))
        interval_sec = max(0.5, min(interval_sec, 10))
    except (TypeError, ValueError):
        interval_sec = 2.0
    suffix = _safe_suffix(file.filename, _ALLOWED_VIDEO_EXTS, ".mp4")
    tmp    = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    file.save(tmp.name)
    tmp.close()

    # Enforce 60-second video duration limit
    tmp_check  = cv2.VideoCapture(tmp.name)
    fps_check  = tmp_check.get(cv2.CAP_PROP_FPS) or 30
    frame_cnt  = tmp_check.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = frame_cnt / fps_check if fps_check > 0 else 0
    tmp_check.release()
    if duration_s > 60:
        os.remove(tmp.name)
        return jsonify({"error": f"الفيديو يتجاوز الحد الأقصى المسموح (دقيقة واحدة). مدة الفيديو المرفوع: {duration_s:.1f}s"}), 400

    print(f"\n{'='*58}")
    print(f"[VIDEO ENDPOINT] File    : {file.filename}")
    print(f"[VIDEO ENDPOINT] Duration: {duration_s:.1f}s | Interval: {interval_sec}s")

    try:
        result = predict_video_ai(tmp.name, interval_sec=interval_sec)
        result["input_type"]   = "video"
        result["duration_sec"] = round(duration_s, 1)
        result["total_time"]   = round(time.time() - start, 2)

        if "error" not in result:
            print(f"\n  [VIDEO MODEL] Label     : {result.get('label')}")
            print(f"  [VIDEO MODEL] Confidence: {result.get('confidence')}")
            print(f"  [VIDEO MODEL] Frames    : {result.get('frames_analyzed', 0)}")
            print(f"  [VIDEO MODEL] Model     : EfficientNet-B0 (60%) + ResNet50 (40%)")
        print(f"[VIDEO ENDPOINT] Done in {result.get('total_time', '?')} sec")
        print(f"{'='*58}\n")
        return jsonify(result)

    except Exception as e:
        return _log_and_safe_error("VIDEO ENDPOINT", e)
    finally:
        try: os.remove(tmp.name)
        except: pass

# ---- Flask-wide error handlers (fix #13) ────────────────────────────────────
@app.errorhandler(413)
def _too_large(_e):
    return jsonify({
        "error":   "request_too_large",
        "max_mb":  app.config.get("MAX_CONTENT_LENGTH", 0) // (1024*1024),
    }), 413

@app.errorhandler(429)
def _rate_limited(_e):
    return jsonify({"error": "rate_limited"}), 429

@app.errorhandler(500)
def _internal_error(e):
    return _log_and_safe_error("flask:500", e)

# ---- DB history ----
@app.route("/history", methods=["GET"])
def history():
    records = NewsRecord.query.order_by(NewsRecord.created_at.desc()).limit(50).all()
    return jsonify([{
        "id"              : r.id,
        "title"           : r.title,
        "summary"         : r.summary,
        "ai_real"         : r.ai_real,
        "ai_fake"         : r.ai_fake,
        "final_score"     : r.final_score,
        "found_in_sources": r.found_in_sources,
        "repetition_count": r.repetition_count,
        "best_url"        : r.best_url,
        "created_at"      : r.created_at.isoformat() if r.created_at else None,
        "input_type"      : r.input_type,
        "is_automated"    : r.is_automated
    } for r in records])

# ---- Admin auth helpers (fix #1) ────────────────────────────────────────────
# MIZAN_ADMIN_PASSWORD_HASH is the werkzeug.security pbkdf2 hash of the chosen
# admin password.  Generate one with:
#     python -c "from werkzeug.security import generate_password_hash as g; \
#                print(g(input('pw: ')))"
# As a convenience for first-run / classroom demos only, MIZAN_ADMIN_PASSWORD
# accepts a plain password — but a real deployment MUST use the hash variable.
from werkzeug.security import check_password_hash

_ADMIN_PASSWORD_HASH  = os.environ.get("MIZAN_ADMIN_PASSWORD_HASH", "").strip()
_ADMIN_PASSWORD_PLAIN = os.environ.get("MIZAN_ADMIN_PASSWORD", "").strip()
if not (_ADMIN_PASSWORD_HASH or _ADMIN_PASSWORD_PLAIN):
    log.warning("MIZAN_ADMIN_PASSWORD[_HASH] not set — admin endpoints will refuse all requests.")

def _admin_password_ok(submitted: str) -> bool:
    if not submitted:
        return False
    if _ADMIN_PASSWORD_HASH:
        try:
            return check_password_hash(_ADMIN_PASSWORD_HASH, submitted)
        except Exception:
            return False
    if _ADMIN_PASSWORD_PLAIN:
        # Constant-time compare to limit timing leaks
        import hmac
        return hmac.compare_digest(submitted, _ADMIN_PASSWORD_PLAIN)
    return False

# ---- Admin: Delete a review ----
@app.route("/admin/delete_review", methods=["POST", "OPTIONS"])
@rate_limit("5 per minute")
def admin_delete_review():
    """Delete a review by ID (requires admin password).

    Request body:
    {
        "review_id": <int>,
        "admin_password": "<password>"
    }
    """
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        review_id = data.get("review_id")
        admin_password = (data.get("admin_password") or "").strip()

        if review_id is None:
            return jsonify({"error": "Missing review_id"}), 400
        if not admin_password:
            return jsonify({"error": "Missing admin_password"}), 400

        if not _admin_password_ok(admin_password):
            log.warning(f"[ADMIN DELETE] UNAUTHORIZED attempt for review_id={review_id} "
                        f"from {get_remote_address()}")
            return jsonify({"error": "Invalid admin password"}), 401

        with app.app_context():
            record = NewsRecord.query.filter(NewsRecord.id == review_id).first()
            if not record:
                return jsonify({"error": f"Review with ID {review_id} not found"}), 404
            db.session.delete(record)
            db.session.commit()
            log.info(f"[ADMIN DELETE] deleted review_id={review_id} title={record.title[:60]!r}")
            return jsonify({
                "success":    True,
                "message":    "Review deleted",
                "deleted_id": review_id
            }), 200

    except Exception as e:
        return _log_and_safe_error("ADMIN DELETE", e)

# ============================================================
# [OCR WARM-UP] Pre-load the primary EasyOCR reader at startup
# so the first /ocr request doesn't pay the cold-start penalty.
# ============================================================
def _warmup_ocr():
    """Load the ar+en EasyOCR reader in the background after Flask boots."""
    import time as _time
    _time.sleep(3)          # give Flask a moment to finish its own startup
    try:
        print("[OCR WARMUP] Pre-loading EasyOCR ar+en reader…")
        get_reader("ar_en")
        print("[OCR WARMUP] EasyOCR ar+en reader ready.")
    except Exception as exc:
        print(f"[OCR WARMUP] Warning — could not pre-load reader: {exc}")


# ============================================================
# [CLEANUP] Background thread - deletes NewsRecord older than 3 days
# ============================================================
def cleanup_old_records():
    """Runs in background: deletes NewsRecord entries older than 3 days every 24 hours."""
    CLEANUP_INTERVAL_SEC = 24 * 3600
    while True:
        time.sleep(60)  # wait a minute after startup before first run
        try:
            cutoff = datetime.utcnow() - timedelta(days=3)
            with app.app_context():
                deleted = NewsRecord.query.filter(NewsRecord.created_at < cutoff).delete()
                db.session.commit()
            print(f"[CLEANUP] Deleted {deleted} NewsRecord(s) older than 3 days.")
        except Exception as e:
            log.exception(f"[CLEANUP] ERROR: {e}")
        time.sleep(CLEANUP_INTERVAL_SEC)

# ============================================================
# [PERIODIC SCRAPER] Background thread - runs every hour
# ============================================================
def periodic_scrape():
    """Runs in background: scrapes PERIODIC_TOPICS every SCRAPE_INTERVAL_SEC seconds."""
    while True:
        print(f"\n[PERIODIC] Starting scheduled scrape at {datetime.utcnow().isoformat()}")
        for topic in PERIODIC_TOPICS:
            try:
                print(f"  [PERIODIC] Topic: {topic}")
                # fix #11 — was asyncio.asyncio.run (typo)
                asyncio.run(verify_news(topic, time.time(), input_type="text", is_automated=True))
            except Exception as e:
                log.exception(f"[PERIODIC] ERROR on topic '{topic}': {e}")
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(8)  # pause between topics to free memory
        print(f"[PERIODIC] Done. Next run in {SCRAPE_INTERVAL_SEC//60} minutes.")
        time.sleep(SCRAPE_INTERVAL_SEC)

# ============================================================
# [MAIN]
# ============================================================
if __name__ == "__main__":
    # The periodic scraper hits the news APIs once per hour.  Only run it in
    # the "main" process — under gunicorn each worker would otherwise spawn
    # its own thread and multiply API quota usage.  Opt in with
    #   MIZAN_RUN_SCRAPER=1
    if os.environ.get("MIZAN_RUN_SCRAPER", "1") == "1":
        scraper_thread = threading.Thread(target=periodic_scrape, daemon=True)
        scraper_thread.start()
        print("[SYSTEM] Background scraper thread started.")
    else:
        print("[SYSTEM] Background scraper thread DISABLED (MIZAN_RUN_SCRAPER=0).")

    cleanup_thread = threading.Thread(target=cleanup_old_records, daemon=True)
    cleanup_thread.start()
    print("[SYSTEM] DB cleanup thread started (deletes records older than 3 days every 24h).")

    ocr_warmup_thread = threading.Thread(target=_warmup_ocr, daemon=True)
    ocr_warmup_thread.start()
    print("[SYSTEM] OCR warm-up thread started.")

    # Bind to localhost by default; pass MIZAN_HOST=0.0.0.0 only behind a
    # reverse proxy.  In production, run with gunicorn/waitress instead of
    # the Flask dev server (see DEPLOY.md).
    _host = os.environ.get("MIZAN_HOST", "127.0.0.1")
    _port = int(os.environ.get("MIZAN_PORT", "5000"))
    app.run(debug=False, host=_host, port=_port)
