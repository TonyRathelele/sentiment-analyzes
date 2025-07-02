import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import yake
from fpdf import FPDF
import json
import datetime

# --- Custom CSS for Professional Gradient Background ---
st.markdown(
    """
    <style>
    /* Gradient background for main app */
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%) !important;
        min-height: 100vh;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }
    /* Card-like look for main blocks */
    .stMarkdown, .stDataFrame, .stButton, .stTextArea, .stFileUploader, .stRadio, .stDownloadButton, .stSubheader, .stInfo {
        background: rgba(255,255,255,0.85) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 24px rgba(44, 62, 80, 0.08) !important;
        padding: 1.2em !important;
        margin-bottom: 1.2em !important;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e0eafc 100%) !important;
        border-top-right-radius: 18px;
        border-bottom-right-radius: 18px;
        box-shadow: 2px 0 16px rgba(44, 62, 80, 0.06);
    }
    /* Title and headers */
    .stTitle, .stSubheader, .stSidebar .stTitle {
        color: #2d3a4a !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #4f8cff 0%, #3358e0 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(44, 62, 80, 0.10);
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #3358e0 0%, #4f8cff 100%) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

df_results = None  # Always use this for display/export

# --- Sidebar ---
st.sidebar.title('Sentiment Analysis Dashboard')

# --- New Analysis Button ---
if st.sidebar.button('New Analysis'):
    # Clear all relevant session state for a blank start, but do not rerun or stop
    for key in list(st.session_state.keys()):
        if key not in ['history']:
            del st.session_state[key]
    st.session_state['new_analysis'] = True

# --- History in Sidebar ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.sidebar.subheader('Analysis History')
history = st.session_state['history']
if history:
    for i, entry in enumerate(reversed(history)):
        label = f"{entry['timestamp']} | {entry['n_texts']} texts"
        if st.sidebar.button(label, key=f'hist_{i}'):
            st.session_state['view_history'] = len(history) - 1 - i
else:
    st.sidebar.write('No history yet.')

# --- Main Title ---
st.title('Interactive Sentiment Analysis Dashboard')

# --- Input Section ---
if 'new_analysis' in st.session_state and st.session_state['new_analysis']:
    input_mode = st.radio('Choose input mode:', ['Direct Text Entry', 'File Upload'], key='new_input_mode')
    user_texts = ''
    texts = []
    uploaded_file = None
    send_clicked = False
    st.session_state['new_analysis'] = False
else:
    input_mode = st.radio('Choose input mode:', ['Direct Text Entry', 'File Upload'])

# --- Sentiment Analysis and Keyword Extraction ---
def analyze_texts(texts):
    try:
        sentiment_pipe = pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-roberta-base-sentiment-latest',
            tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest'
        )
    except Exception as e:
        st.error(f"Failed to load sentiment model: {e}")
        return []
    kw_extractor = yake.KeywordExtractor(top=5, stopwords=None)
    results = []
    try:
        sentiment_results = sentiment_pipe(texts, batch_size=8)
    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
        return []
    for text, sent in zip(texts, sentiment_results):
        keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
        # Simple explanation: highlight keywords and sentiment
        explanation = (
            f"The text was classified as {sent['label']} with confidence {sent['score']:.2f}. "
            f"Key phrases influencing this: {', '.join(keywords) if keywords else 'N/A'}."
        )
        results.append({
            'text': text,
            'sentiment': sent['label'],
            'confidence': sent['score'],
            'keywords': keywords,
            'explanation': explanation
        })
    return results

# --- Run Analysis and Set df_results ---
if input_mode == 'Direct Text Entry':
    user_texts = st.text_area('Enter text(s) (one per line):')
    texts = [t.strip() for t in user_texts.split('\n') if t.strip()]
    send_clicked = st.button('Send')
else:
    uploaded_file = st.file_uploader('Upload a CSV or TXT file', type=['csv', 'txt'])
    texts = []
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            texts = df.iloc[:,0].astype(str).tolist()
        else:
            texts = [line.decode('utf-8').strip() for line in uploaded_file.readlines() if line.strip()]
    send_clicked = True  # Always auto-analyze for file upload

if texts and send_clicked:
    st.write('Ready to analyze', len(texts), 'texts.')
    with st.spinner('Analyzing...'):
        results = analyze_texts(texts)
    if results:
        df_results = pd.DataFrame(results)
        # Save to history
        st.session_state['history'].append({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_texts': len(texts),
            'df': df_results.copy()
        })
        st.session_state['view_history'] = len(st.session_state['history']) - 1

# --- Show current or selected history ---
if 'view_history' in st.session_state and st.session_state['history']:
    idx = st.session_state['view_history']
    df_results = st.session_state['history'][idx]['df']

# --- Display, Visualize, and Export if df_results exists ---
if df_results is not None and not df_results.empty:
    st.dataframe(df_results)
    # --- Visualization: Sentiment Distribution ---
    st.subheader('Sentiment Distribution')
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x='sentiment', data=df_results, ax=ax[0])
    ax[0].set_title('Sentiment Count')
    sentiment_counts = df_results['sentiment'].value_counts()
    ax[1].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    ax[1].set_title('Sentiment Proportion')
    st.pyplot(fig)

    # --- Line Graph: Sentiment on X, Text Index on Y ---
    st.subheader('Sentiment by Text Index')
    # Map sentiment to custom numeric values for plotting
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 5}
    df_results['sentiment_num'] = df_results['sentiment'].map(sentiment_map)
    fig_line, ax_line = plt.subplots(figsize=(8, 4))
    # Plot a single, thicker, more prominent line connecting all points (text index on x, sentiment on y)
    ax_line.plot(
        df_results.index,
        df_results['sentiment_num'],
        marker='o',
        linestyle='-',
        color='#1a237e',  # deep blue
        linewidth=3,
        label='Sentiment Progression'
    )
    # Color markers by sentiment
    colors = df_results['sentiment'].map({'negative': '#e74c3c', 'neutral': '#f1c40f', 'positive': '#2ecc71'})
    ax_line.scatter(df_results.index, df_results['sentiment_num'], c=colors, s=100, zorder=3, edgecolor='#222', linewidth=1.5, label='Sentiment Dots')
    # Set y-ticks to new sentiment values and labels
    ax_line.set_yticks([0, 1, 5])
    ax_line.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax_line.set_xlabel('Text Index')
    ax_line.set_ylabel('Sentiment')
    ax_line.set_title('Sentiment by Text Index')
    ax_line.legend()
    st.pyplot(fig_line)

    # --- Comparative Analysis (if multiple sources) ---
    st.subheader('Comparative Analysis')
    st.info('Upload or enter texts from different sources to compare sentiment distributions.')

    # --- Export Options ---
    st.subheader('Export Results')
    # CSV Export
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', csv_data, file_name='sentiment_results.csv', mime='text/csv')
    # JSON Export
    json_data = df_results.to_json(orient='records', force_ascii=False, indent=2)
    st.download_button('Download JSON', json_data, file_name='sentiment_results.json', mime='application/json')
    # PDF Export
    def create_pdf(df):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Sentiment Analysis Results', ln=True, align='C')
        pdf.set_font('Arial', '', 10)
        col_width = pdf.w / (len(df.columns) + 1)
        row_height = pdf.font_size * 1.5
        # Header
        for col in df.columns:
            pdf.cell(col_width, row_height, str(col), border=1)
        pdf.ln(row_height)
        # Rows
        for _, row in df.iterrows():
            for item in row:
                cell = str(item)
                if len(cell) > 30:
                    cell = cell[:27] + '...'
                # Remove non-latin1 characters for FPDF compatibility
                cell = cell.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(col_width, row_height, cell, border=1)
            pdf.ln(row_height)
        return pdf.output(dest='S').encode('latin1')
    pdf_data = create_pdf(df_results)
    st.download_button('Download PDF', pdf_data, file_name='sentiment_results.pdf', mime='application/pdf')
else:
    st.info('Please enter or upload text data to begin.')

# --- Model Limitations and Confidence Threshold Info ---
st.markdown("""
---
### Model Limitations & Confidence Thresholds
- **Model**: The sentiment analysis uses a transformer model trained on social media data (Twitter). It may not generalize perfectly to all domains (e.g., formal reviews, technical documents).
- **Confidence Scores**: Scores close to 1.0 indicate high model certainty. Scores below 0.7 should be interpreted with caution.
- **Neutral Class**: The model may sometimes misclassify subtle or mixed sentiments as neutral.
- **Keyword Extraction**: Keywords are extracted using YAKE, which may not always align with the model's internal reasoning.
- **Batch Size**: For large datasets, processing is done in batches for efficiency, but API/model rate limits may apply.
- **Responsible Use**: Always review results, especially for critical applications. The model may reflect biases present in its training data.

For more information, see the [CardiffNLP Twitter-RoBERTa documentation](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).
""")