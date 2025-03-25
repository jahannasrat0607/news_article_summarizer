from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import requests
from bs4 import BeautifulSoup
import uvicorn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
from pathlib import Path
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for audio files
audio_dir = Path("static/audio")
audio_dir.mkdir(parents=True, exist_ok=True)

# Mount static files to serve audio files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def fetch_news(company_name, max_results=10):
    """Fetch news articles using NewsAPI"""
    try:
        api_key = "1949b3d75fa1494791dc4a3e9db37b07"  # Get from newsapi.org
        url = f"https://newsapi.org/v2/everything?q={company_name}&pageSize={max_results}&apiKey={api_key}"

        headers = {
            "User-Agent": "NewsSummarizer/1.0",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data['status'] != 'ok' or not data.get('articles'):
            return {"error": "No articles found from NewsAPI"}

        articles = []
        for article in data['articles']:
            articles.append({
                "title": article.get('title', 'No title available'),
                "link": article.get('url', ''),
                "content": article.get('content', '')[:2000]  # Truncate very long content
            })

        return articles[:max_results]  # Ensure we don't exceed max_results

    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI request failed: {str(e)}")
        return {"error": f"Failed to fetch news: {str(e)}"}
    except Exception as e:
        logger.error(f"Error processing NewsAPI response: {str(e)}")
        return {"error": f"Error processing news: {str(e)}"}


def fetch_full_article(url):
    """Fallback function to scrape full article if NewsAPI content is insufficient"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text().split()) > 15]
        article_text = "\n".join(paragraphs)

        return article_text if len(article_text) > 150 else None
    except Exception as e:
        logger.warning(f"Failed to fetch article {url}: {e}")
        return None


def summarize_text(text):
    """Summarizes the article content dynamically based on input length."""
    if not text or len(text) < 100:
        return "⚠️ Summary not available."

    max_len = min(len(text) // 3, 200)  # Dynamic max length
    try:
        summary = summarizer(text[:1024], max_length=max_len, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return "⚠️ Summary not available."


def translate_to_hindi(text):
    """Translates text from English to Hindi."""
    try:
        if not text or text.startswith("⚠️"):
            return text

        # Split long text into chunks (Google Translator has limits)
        chunks = [text[i:i + 5000] for i in range(0, len(text), 5000)]
        translated_chunks = []

        for chunk in chunks:
            translated = GoogleTranslator(source='en', target='hi').translate(chunk)
            translated_chunks.append(translated)

        return " ".join(translated_chunks)
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text


def initialize_sentiment_analyzer():
    """Initialize sentiment analyzer with proper configuration"""
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


def analyze_sentiment(text, model, tokenizer):
    """Strict three-category sentiment analysis"""
    if not text or len(text.strip()) < 20:
        return "Neutral", 0.0

    try:
        inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
        sentiments = ["Negative", "Neutral", "Positive"]
        return sentiments[probs.argmax()], float(probs.max())
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return "Neutral", 0.0


def extract_topics(text):
    """Simple topic extraction from text"""
    topics = []
    keywords = {
        "Electric Vehicles": ["ev", "electric vehicle", "electric car", "tesla"],
        "Stock Market": ["stock", "share", "market cap", "invest"],
        "Innovation": ["innovate", "breakthrough", "new tech", "technology"],
        "Regulations": ["regulat", "law", "policy", "government"],
        "Autonomous Vehicles": ["self-driving", "autonomous", "fsd", "autopilot"],
        "Battery": ["battery", "range", "charging", "power"],
        "CEO": ["musk", "elon", "ceo", "executive"],
        "Competition": ["compet", "rival", "ford", "gm", "volkswagen"]
    }

    text_lower = text.lower()
    for topic, terms in keywords.items():
        if any(term in text_lower for term in terms):
            topics.append(topic)

    return topics if topics else ["General"]


def text_to_speech(text, filename):
    """Converts text to Hindi speech and saves as MP3"""
    if not text or text.startswith("⚠️"):
        return None

    try:
        # Clean filename to remove invalid characters
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
        output_path = audio_dir / safe_filename

        # Skip if file already exists
        if output_path.exists():
            return f"/static/audio/{safe_filename}"

        tts = gTTS(text=text[:5000], lang='hi', slow=False)  # Limit text length
        tts.save(output_path)
        return f"/static/audio/{safe_filename}"
    except Exception as e:
        logger.error(f"Text-to-speech failed: {e}")
        return None


def generate_final_output(news_articles, company_name):
    """Generates the final output in the specified JSON format"""
    if isinstance(news_articles, dict) and "error" in news_articles:
        return news_articles  # Return API errors directly

    articles_output = []
    model, tokenizer = initialize_sentiment_analyzer()

    for idx, article in enumerate(news_articles[:10]):  # Process max 10 articles
        try:
            # Use NewsAPI content if available, otherwise fallback to scraping
            text = article.get("content") or fetch_full_article(article.get("link", "")) or article.get("title", "")

            sentiment, confidence = analyze_sentiment(text, model, tokenizer)
            topics = extract_topics(text)
            summary = summarize_text(text)
            summary_hindi = translate_to_hindi(summary)

            # Generate audio file with unique name
            tts_filename = f"summary_{company_name}_{idx}.mp3"
            audio_path = text_to_speech(summary_hindi, tts_filename)

            articles_output.append({
                "Title": article.get("title", "No title available"),
                "Summary": summary,
                "Summary_Hindi": summary_hindi,
                "Sentiment": sentiment,
                "Topics": topics,
                "Audio": f"http://127.0.0.1:8000{audio_path}" if audio_path else None,
                "Link": article.get("link", "")
            })

        except Exception as e:
            logger.error(f"Error processing article {idx}: {e}")
            continue

    return {
        "Company": company_name,
        "Articles": articles_output,
        "Count": len(articles_output)
    }


@app.get("/")
def read_root():
    return {
        "message": "News Summarizer API",
        "endpoints": {
            "/fetch_news?company=NAME": "Get summarized news about a company"
        }
    }


@app.get("/fetch_news")
def get_news(company: str):
    """Main endpoint to fetch and process news"""
    logger.info(f"Fetching news for: {company}")
    news_articles = fetch_news(company)
    return generate_final_output(news_articles, company)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)