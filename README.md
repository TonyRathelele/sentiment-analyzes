An interactive web dashboard for analyzing sentiment in text data, built with Streamlit and Hugging Face Transformers.

## Features
- **Text Input**: Enter text directly or upload CSV/TXT files for batch analysis.
- **Multi-class Sentiment**: Classifies texts as Positive, Negative, or Neutral with confidence scores.
- **Keyword Extraction**: Highlights key phrases influencing sentiment.
- **Batch Processing**: Analyze multiple texts at once.
- **Visualizations**: 
  - Sentiment distribution (bar and pie charts)
  - Sentiment confidence line graph
  - Comparative analysis placeholder
- **Explanations**: Shows why each text received its sentiment score.
- **Export**: Download results as CSV, JSON, or PDF.
- **History**: Sidebar stores previous analyses for easy review.
- **New Analysis**: Start fresh with a single click.
- **Error Handling**: Handles invalid input and API/model errors gracefully.

## Installation
1. Clone this repository or copy the files to your project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided local URL in your browser.
3. Choose input mode (Direct Text Entry or File Upload).
4. Enter or upload your texts, then click **Send** (for direct entry) or upload a file (auto-analyzes).
5. View results, visualizations, and explanations.
6. Export results or review previous analyses from the sidebar.
7. Click **New Analysis** to start over.

## Model Info
- Uses [CardiffNLP Twitter-RoBERTa-base Sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) for sentiment analysis.
- Keyword extraction via [YAKE](https://github.com/LIAAD/yake).
- Confidence scores indicate model certainty (closer to 1.0 = more confident).

## Limitations
- The sentiment model is trained on social media data and may not generalize to all domains.
- Confidence scores below 0.7 should be interpreted with caution.
- Keyword extraction is heuristic and may not match model reasoning.
- For large files, processing is batched for efficiency.
- Always review results, especially for critical applications.

## License
This project is for educational and demonstration purposes.
