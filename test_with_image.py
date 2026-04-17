import time, asyncio, aiohttp, re, torch, numpy as np, requests, feedparser
from flask import Flask, render_template, request
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from urllib.parse import quote
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from newspaper import Article
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt

import easyocr
reader = easyocr.Reader(['ar']) 
# --------------------
app = Flask(__name__)
CORS(app)
##this is to make data set
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
#this to how to save the record in db
class NewsRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500))
    content = db.Column(db.Text)
    real_percentage = db.Column(db.Float) # نسبة الحقيقة
    fake_percentage = db.Column(db.Float) # نسبة الزيف
    source = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_automated = db.Column(db.Boolean, default=False) # هل هو سحب تلقائي؟

with app.app_context():
    db.create_all()

device = "cuda" if torch.cuda.is_available() else "cpu"

PATH_LONG = r"MODFnew_model_arabertV02"
PATH_SHORT = r"title_version" 
model_name = "aubmindlab/bert-base-arabertv02"
arabert_prep = ArabertPreprocessor(model_name=model_name)
tokenizer_long = AutoTokenizer.from_pretrained(PATH_LONG)
tokenizer_short = AutoTokenizer.from_pretrained(PATH_SHORT)

model_long = AutoModelForSequenceClassification.from_pretrained(PATH_LONG).to(device)
model_short = AutoModelForSequenceClassification.from_pretrained(PATH_SHORT).to(device)
model_long.eval()
model_short.eval()
image_model = models.resnet18()
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, 2) 
import zipfile
import io
image_model = models.resnet18()
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, 2)
target_file = r"final_model.pth" 

import torch
import torch.nn as nn
from torchvision import models

image_model = models.resnet18()
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, 2)

image_model = models.resnet18()
num_ftrs = image_model.fc.in_features
image_model.fc = nn.Linear(num_ftrs, 2)

index_file =  r"final_model.pth" 

try:
    print(f"🔄 Trying to bypass folder lock via index file...")
    
    with open(index_file, 'rb') as f:
        state_dict = torch.load(f, map_location=device, weights_only=False)
    
    if isinstance(state_dict, dict):
        image_model.load_state_dict(state_dict)
    else:
        image_model = state_dict
        
    image_model.to(device)
    image_model.eval()
    print(" FINALLY ")

except Exception as e:
    print(f" FIELD {e}")
    
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# ----------------------------------------

embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").to(device)
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
GNEWS_API_KEY = "32058a59bd84119c83f852932fcd0aff"  
NEWSAPI_KEY = "a2ecf7a09bc741a0afbc8243fb3b4fa3"

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

async def fetch_html(session, url):
    try:
        async with session.get(url, headers=HEADERS, timeout=10) as r:
            return await r.text()
    except:
        return ""

async def extract_article_text(url):
    try:
        async with aiohttp.ClientSession() as session:
            html = await fetch_html(session, url)
            if not html: return ""
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]): script.decompose()
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text() for p in paragraphs)
            return clean_text(text)
    except:
        return ""
def search_google_news(query):
    articles = []
    search_query = quote(" ".join(query.split()[:8]))
    url = f"https://news.google.com/rss/search?q={search_query}&hl=ar&gl=AR&ceid=AR:ar"
    feed = feedparser.parse(url)
    for entry in feed.entries[:10]: 
        articles.append({"title": entry.title, "url": entry.link, "source": getattr(entry, "source", {}).get("title", "GoogleNews")})
    return articles


def search_gnews(query):
    articles = []
    url = f"https://gnews.io/api/v4/search?q={query}&lang=ar&max=20&token={GNEWS_API_KEY}"
    try:
        res = requests.get(url).json()
        for item in res.get("articles", []):
            articles.append({"title": item["title"], "url": item["url"], "source": item["source"]["name"]})
    except: pass
    return articles

def search_newsapi(query):
    articles = []
    url = f"https://newsapi.org/v2/everything?q={query}&language=ar&pageSize=20&apiKey={NEWSAPI_KEY}"
    try:
        res = requests.get(url).json()
        for item in res.get("articles", []):
            articles.append({"title": item["title"], "url": item["url"], "source": item["source"]["name"]})
    except: pass
    return articles

SEARCH_SOURCES = [
    {"name": "بترا", "url": "https://petra.gov.jo/Include/InnerPage.jsp?ID=14&lang=ar&name=news&search={q}", "base": "https://petra.gov.jo"},
    {"name": "عمون", "url": "https://www.ammonnews.net/search?q={q}", "base": "https://www.ammonnews.net"},
    {"name": "سرايا", "url": "https://www.sarayanews.com/search.php?query={q}", "base": "https://www.sarayanews.com"},
    {"name": "رؤيا", "url": "https://roya.tv/search?q={q}", "base": "https://roya.tv"},
    {"name": "خبرني", "url": "https://www.khaberni.com/search?q={q}", "base": "https://www.khaberni.com"},
    {"name": "الغد", "url": "https://alghad.com/?s={q}", "base": "https://alghad.com"},
    {"name": "المملكة", "url": "https://www.almamlakatv.com/search?q={q}", "base": "https://www.almamlakatv.com"},
    {"name": "الرأي", "url": "http://alrai.com/search?q={q}", "base": "http://alrai.com"},
    {"name": "الجزيرة", "url": "https://www.aljazeera.net/search/{q}", "base": "https://www.aljazeera.net"},
    {"name": "بي بي سي", "url": "https://www.bbc.com/arabic/search?q={q}", "base": "https://www.bbc.com"}
]

async def search_local_sources(session, query):
    local_articles = []
    q_encoded = quote(query)
    for source in SEARCH_SOURCES:
        try:
            target_url = source["url"].format(q=q_encoded)
            async with session.get(target_url, headers=HEADERS, timeout=10) as response:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                for a in soup.find_all("a", href=True):
                    title = a.get_text(strip=True)
                    if len(title) > 25:
                        link = a['href']
                        full_url = link if link.startswith("http") else source["base"] + link
                        local_articles.append({"title": title, "url": full_url, "source": source["name"]})
        except: continue
    return local_articles

def fetch_full_content(url):
    try:
        article = Article(url, language='ar')
        article.download()
        article.parse()
        # تعديل: نرجع النص كاملاً للفحص بناءً على الطول
        return article.text if len(article.text) > 30 else None
    except:
        return None
    
async def collect_articles_async(user_text):
    articles = []
    articles.extend(search_google_news(user_text))
    articles.extend(search_gnews(user_text))
    articles.extend(search_newsapi(user_text))
   #  articles.extend(search_newsapi(user_text))
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_html(session, a["url"]) for a in articles]
        htmls = await asyncio.gather(*tasks)
        for i, html in enumerate(htmls):
            if not html: continue
            soup = BeautifulSoup(html, "html.parser")
            # تعديل: التأكد من سحب الخبر كاملاً
            text = clean_text(" ".join(p.get_text() for p in soup.find_all("p")))
            if len(text) > 100:
                articles[i]["text"] = text
                results.append(articles[i])
    return results

def ai_predict(text):
    word_count = len(text.split())
    
    if word_count < 20:
        print(f"\n[DEBUG] --> Using Headline Model (Short Text: {word_count} words)")
        target_model = model_short
        target_tokenizer = tokenizer_short
    else:
        print(f"\n[DEBUG] --> Using Long Text Model (Long Text: {word_count} words)")
        target_model = model_long
        target_tokenizer = tokenizer_long
 
    inputs = target_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = target_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    fake_score = probs[0][0].item()
    real_score = probs[0][1].item()

    print(f" Results from Model:")
    print(f" Fake Probability: {fake_score*100:.2f}%")
    print(f" Real Probability: {real_score*100:.2f}%")
    print("-" * 50)

    return fake_score, real_score


def predict_image_ai(image_path):
    start_time = time.time()
    try:
        image = Image.open(image_path).convert('RGB')
        image = image_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = image_model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            
        execution_time = round(time.time() - start_time, 2)
        
        label = "AI-Generated" if pred.item() == 0 else "Real Image"
        confidence = conf.item() * 100
        print(f" Image Analysis Result: {label} ({confidence:.2f}%)")
        print(f"⏱ Image Analysis Time: {execution_time} seconds")
        
        return {
            "label": label,
            "confidence": f"{confidence:.2f}%",
            "execution_time": execution_time
        }
    except Exception as e:
        print(f"Error in image prediction: {e}")
        return {"error": str(e)}
# ----------------------------------------

def save_news_to_db(title, content, fake_p, real_p, source, automated=False):
    try:
        print(">>> TRYING TO SAVE:", title)

        with app.app_context():
            exists = NewsRecord.query.filter_by(title=title).first()
            print("EXISTS:", exists)

            if not exists:
                new_entry = NewsRecord(
                    title=title[:500], 
                    content=content if content else title,
                    fake_percentage=round(float(fake_p) * 100, 2),
                    real_percentage=round(float(real_p) * 100, 2),
                    source=source,
                    is_automated=automated
                )
                db.session.add(new_entry)
                db.session.commit()
                print(">>> SAVED SUCCESSFULLY")
            else:
                print(">>> NOT SAVED: DUPLICATE TITLE")

    except Exception as e:
        print(f"Database Error: {e}")

async def collect_and_store_scraped_news(user_query):
    scraped_articles = search_google_news(user_query)
    for art in scraped_articles:
        full_content = fetch_full_content(art['url']) 
        content_to_use = full_content if full_content else art['title']
        # التقييم بناءً على المحتوى (سواء كان مقال طويل أو عنوان)
        f_p, r_p = ai_predict(content_to_use) 
        

        save_news_to_db(art['title'], content_to_use, f_p, r_p, art['source'], automated=True) 
    return scraped_articles

def find_similarity(user_text, articles):
    texts = [user_text] + [a["text"] for a in articles]
    embeddings = embedding_model.encode(texts)
    sims = cosine_similarity([embeddings[0]], embeddings[1:])[0]
    idx = np.argmax(sims)
    return articles[idx], sims[idx]

def verify_news(user_text, start_time, source_url=None):
    user_text = clean_text(user_text)

    # --- ميزة البحث في الداتا سيت أولاً (Cache) ---
    print(f"🔍 Searching in Database for: {user_text[:50]}...")
    cached_news = NewsRecord.query.filter(NewsRecord.title.like(f"%{user_text[:20]}%")).first()
    if cached_news:
        print("✅ Found in Dataset! Returning cached result.")
        ai_label = "Real" if cached_news.real_percentage > cached_news.fake_percentage else "Fake"
        status = " خبر حقيقي وموثق" if cached_news.real_percentage >= 70 else " خبر مشكوك فيه" if cached_news.real_percentage >= 40 else " خبر غالباً كاذب"
        return {
            "status": status + " (من قاعدة البيانات)",
            "ai_label": ai_label,
            "ai_probs": {"fake": f"{cached_news.fake_percentage}%", "real": f"{cached_news.real_percentage}%"},
            "final_weighted_score": f"{cached_news.real_percentage}%",
            "execution_time": round(time.time() - start_time, 2),
            "title": cached_news.title,
            "source": cached_news.source,
            "url": "#"
        }
    # -------------------------------------------
    
    fake_prob, real_prob = ai_predict(user_text)
    ai_label = "Real" if real_prob > fake_prob else "Fake"
    user_content_to_save = user_text
    if source_url:
        scraped_full = fetch_full_content(source_url)
        if scraped_full: user_content_to_save = scraped_full
    save_news_to_db(user_text[:100], user_content_to_save, fake_prob, real_prob, "User Input", automated=False)
    api_articles = search_google_news(user_text) 
    api_score, best_api_article = 0, None
    
    if api_articles:
        titles = [a["title"] for a in api_articles]
        user_emb = embedding_model.encode([user_text])
        titles_emb = embedding_model.encode(titles)
        sims = cosine_similarity(user_emb, titles_emb)[0]
        idx = np.argmax(sims)
        api_score, best_api_article = sims[idx], api_articles[idx]
        for art in api_articles[:3]:
            full_txt = fetch_full_content(art['url'])

            final_content = full_txt if full_txt else art['title']
            
            # التقييم هنا صار يعتمد على المحتوى المسحوب كاملاً (طويل/قصير)
            f_p, r_p = ai_predict(final_content)
            save_news_to_db(art['title'], final_content, f_p, r_p, art['source'], automated=True)

    final_score = (0.80 * real_prob) + (0.20 * float(api_score))
    
    if final_score >= 0.70: status = " خبر حقيقي وموثق"
    elif final_score >= 0.40: status = " خبر مشكوك فيه"
    else: status = " خبر غالباً كاذب"

    return {
        "status": status,
        "ai_label": ai_label, # الحين المتغير معرف 100%
        "ai_probs": {"fake": f"{fake_prob*100:.1f}%", "real": f"{real_prob*100:.1f}%"},
        "final_weighted_score": f"{final_score*100:.1f}%",
        "execution_time": round(time.time() - start_time, 2),
        "title": best_api_article["title"] if best_api_article else "لا يوجد خبر مشابه",
        "source": best_api_article["source"] if best_api_article else "N/A",
        "url": best_api_article["url"] if best_api_article else ""
    }

# --- كود مضاف: الـ Endpoint الخاص بفحص الصور مع OCR ---
@app.route("/predict_image", methods=["POST"])
def predict_image_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    temp_path = "temp_image.jpg"
    file.save(temp_path)
    
    # --- تحسين الصورة قبل الـ OCR ---
    from PIL import ImageOps
    img = Image.open(temp_path).convert('L') # تحويل لرمادي
    img = ImageOps.autocontrast(img) # زيادة التباين لبروز الحروف
    processed_path = "processed_temp.jpg"
    img.save(processed_path)
    # -------------------------------

    print("\n" + "="*50)
    print(f" 📥 استلام صورة جديدة للتحقق: {file.filename}")
    
    # 1. فحص الصورة بموديل ResNet18 (نستخدم الصورة الأصلية للموديل)
    img_result = predict_image_ai(temp_path)
    
    # 2. استخراج النص باستخدام OCR (نستخدم الصورة المعالجة processed_path)
    print(" 🔍 جاري استخراج النص المحسن من الصورة...")
    ocr_results = reader.readtext(processed_path, detail=0, paragraph=True) # detail=0 بجيب النص الصافي
    extracted_text = " ".join(ocr_results)
    
    # تنظيف النص المستخرج من الكلمات الزائدة مثل "مراسلة المملكة"
    clean_extracted = extracted_text.replace("المملحة", "المملكة").replace("مراسلة", "").strip()
    
    news_result = None
    if clean_extracted:
        print(f" 📝 النص المستخرج بعد التحسين: {clean_extracted}")
        start_time = time.time()
        news_result = verify_news(clean_extracted, start_time)
    
    # مسح الملفات المؤقتة
    for p in [temp_path, processed_path]:
        if os.path.exists(p): os.remove(p)
        
    return jsonify({
        "image_analysis": img_result,
        "extracted_text": clean_extracted,
        "text_analysis": news_result
    })

@app.route("/", methods=["GET", "POST"])
@app.route("/predict", methods=["POST", "OPTIONS"])
def index():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
    print("\n" + "="*50)
    print(" استلام طلب جديد للتحقق...")
    
    if request.is_json:
        data = request.get_json()
        user_text = data.get("news_text", "").strip()
        print(f" مصدر الطلب: إضافة المتصفح (JSON)")
    else:
        user_text = request.form.get("news_text", "").strip()
        print(f" مصدر الطلب: واجهة الموقع (Form)")

    if user_text:
        print(f"النص المستلم: {user_text[:100]}...") # عرض أول 100 حرف فقط
    print(" جاري استدعاء نموذج AraBERT للتحليل...")
    start_time = time.time()
    
    # استدعاء دالة التحقق (التي تبحث في الداتا سيت أولاً)
    result = verify_news(user_text, start_time)
    
    print(f" نتيجة التحليل: {result.get('status')}")
    print(f"نسبة الثقة (Confidence): {result.get('final_weighted_score')}")
    print(f"الوقت المستغرق: {round(time.time() - start_time, 2)} ثانية")
    print("="*50 + "\n")

    if request.is_json or request.path == "/predict":
        return jsonify(result)
        
    return render_template("digita pro.html", result=result, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
