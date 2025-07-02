import os
import pandas as pd
from fpdf import FPDF
import json
from huggingface_hub import InferenceClient

SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
KEYWORD_MODEL = "ml6team/keyphrase-extraction-distilbert-inspec"

def get_hf_client(api_key=None):
    api_key = api_key or os.getenv("HF_API_KEY")
    return InferenceClient(token=api_key)

def analyze_sentiment(texts, api_key=None):
    """Batch sentiment analysis using Hugging Face InferenceClient. Includes error messages and debug prints."""
    results = []
    client = get_hf_client(api_key)
    # Debug: print masked API key
    masked_key = (api_key[:4] + "..." + api_key[-4:]) if api_key else "None"
    print(f"[DEBUG] Using API key: {masked_key}")
    for text in texts:
        error_message = None
        try:
            data = client.text_classification(text, model=SENTIMENT_MODEL)
            print(f"[DEBUG] Raw result for text '{text}': {data}")
            if isinstance(data, list) and len(data) > 0:
                label = data[0]["label"]
                score = data[0]["score"]
            else:
                label = "unknown"
                score = 0.0
                error_message = f"Unexpected response: {data}"
        except Exception as e:
            label = "error"
            score = 0.0
            error_message = str(e)
        if not error_message:
            error_message = ""
        results.append({
            "text": text,
            "sentiment": label,
            "confidence": score,
            "error_message": error_message
        })
    return results

def extract_keywords(text, api_key=None):
    client = get_hf_client(api_key)
    try:
        data = client.text_generation(text, model=KEYWORD_MODEL, max_new_tokens=32)
        # The output may need parsing depending on the model's output format
        # For keyphrase-extraction-distilbert-inspec, the output is a string of keywords
        if isinstance(data, str):
            # Split by comma or newlines
            keywords = [kw.strip() for kw in data.replace("\n", ",").split(",") if kw.strip()]
        else:
            keywords = []
    except Exception:
        keywords = []
    return keywords

def export_to_csv(results, filename="results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    return filename

def export_to_json(results, filename="results.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return filename

def export_to_pdf(results, filename="results.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sentiment Analysis Results", ln=True, align="C")
    for item in results:
        pdf.multi_cell(0, 10, txt=f"Text: {item['text']}\nSentiment: {item['sentiment']} (Confidence: {item['confidence']:.2f})\nKeywords: {', '.join(item.get('keywords', []))}\nError: {item.get('error_message', '')}")
        pdf.ln(2)
    pdf.output(filename)
    return filename