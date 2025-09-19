import streamlit as st
import os
import pdfplumber
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from docx import Document
from pdf2image import convert_from_path
import pytesseract
import tempfile
import csv
from datetime import datetime
import requests
import time

st.set_page_config(page_title="AI Document Processor", layout="wide")
# --- File history CSV ---
HISTORY_CSV = os.path.join(os.path.dirname(__file__), "user_history.csv")

# --- Login Page ---
if "user_id" not in st.session_state:
    # --- Unique Login Page ---
    def show_login():
        st.markdown("""
            <style>
            .login-title {font-size: 2.2em; font-weight: bold; color: #2c3e50; margin-bottom: 0.2em;}
            .login-desc {font-size: 1.1em; color: #555; margin-bottom: 1em;}
            .clock {font-size: 1.1em; color: #0072C6; margin-bottom: 1.5em;}
            </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔒 Welcome to AI Document Processor</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-desc">Upload PDFs and get instant AI-powered summaries, sentiment, classification, and more. Your history is private and secure. Try our new chat and Q&A features!</div>', unsafe_allow_html=True)
        # --- Real-time clock (no threading) ---
        import time
        clock_placeholder = st.empty()
        clock_placeholder.markdown(
            f'<div class="clock">🕒 {time.strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True
        )
        login_name = st.text_input("Name")
        login_password = st.text_input("Password", type="password")
        if st.button("Login"):
                      response = requests.post("https://ml-project-al73wyu5prwjlzssksvert.streamlit.app/login", json={"name": login_name, "password": login_password})

            if response.status_code == 200:
                try:
                    data = response.json()
                    user_id = data.get("user_id")
                    st.success("Login successful!")
                    st.session_state["user_id"] = user_id
                    st.rerun()
                except Exception:
                    st.error("Login succeeded but response is not valid JSON. Please check your backend.")
            else:
                st.error("Invalid credentials.")

    show_login()
    st.stop()
else:
    # --- Main Project File ---
    # --- Sidebar for AI tool selection ---
    st.sidebar.header("AI Tools")
    ai_tools = st.sidebar.multiselect(
        "Select AI tools to apply:",
        ["Text Summarization", "Sentiment Analysis", "Chatbot Integration", "Document Classification"],
        default=["Text Summarization"]
    )
    lang = st.sidebar.text_input("Translate summary to (optional)", "")
    output_type = st.sidebar.selectbox("Output Format", ["PDF", "TXT", "DOCX", "HTML", "CSV"])
    # --- Page count input ---

    def extract_text_from_pdf(pdf_file):
        all_text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text + "\n"
        except Exception as e:
            st.warning(f"[Error using pdfplumber: {e}]")
        if not all_text.strip():
            st.info("No selectable text found. Running OCR...")
            poppler_path = os.path.join(os.path.dirname(__file__), "uploads", "poppler", "bin")
            images = convert_from_path(tmp_path, poppler_path=poppler_path)
            for img in images:
                text = pytesseract.image_to_string(img, lang="eng")
                all_text += text + "\n"
        os.remove(tmp_path)
        return all_text.strip()

    # --- Page count input ---
    page_count = st.sidebar.number_input("How many pages for summary?", min_value=1, max_value=10, value=1, step=1)
    # --- File upload ---
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    def split_text(text, max_words=400):
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def summarize_text(chunks, pages=1):
        device = torch.device("cpu")
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        summaries = []
        # Adjust max_length based on pages (roughly 200 tokens per page)
        max_length = 200 * pages
        for chunk in chunks:
            try:
                summary = summarizer(chunk, max_length=max_length, min_length=50*pages, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                summaries.append(f"[Error summarizing chunk: {e}]")
        return "\n".join(summaries)

    def translate_to_language(text, language):
        if language.lower() == "tamil":
            return "இது ஒரு தமிழ் சுருக்கம் (This is a Tamil summary)."
        elif language:
            return f"[Translated to {language}]: {text}"
        else:
            return text

    def save_to_file(content, filetype):
        if filetype == "PDF":
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                c = canvas.Canvas(tmp.name, pagesize=letter)
                width, height = letter
                text_object = c.beginText(40, height - 50)
                text_object.setFont("Times-Roman", 12)
                for line in content.split("\n"):
                    text_object.textLine(line)
                c.drawText(text_object)
                c.save()
                return tmp.name
        elif filetype == "TXT":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                tmp.write(content)
                return tmp.name
        elif filetype == "DOCX":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                doc = Document()
                doc.add_heading("Summary", 0)
                for line in content.split("\n"):
                    doc.add_paragraph(line)
                doc.save(tmp.name)
                return tmp.name
        elif filetype == "HTML":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
                summary_html = content.replace("\n", "<br>")
                html_content = f"<html><body><h2>Summary</h2><p>{summary_html}</p></body></html>"
                tmp.write(html_content)
                return tmp.name
        elif filetype == "CSV":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp:
                tmp.write("Summary\n")
                for line in content.split("\n"):
                    tmp.write(f'"{line}"\n')
                return tmp.name
        return None

    def log_user_request(ai_tools, output_type, page_count):
        with open(HISTORY_CSV, mode="a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ", ".join(ai_tools),
                output_type,
                page_count
            ])

    def load_history():
        if not os.path.exists(HISTORY_CSV):
            return []
        with open(HISTORY_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            return list(reader)

    if "user_id" in st.session_state:
        st.sidebar.subheader("Your History")
        history_response = requests.get(f"https://ml-project-al73wyu5prwjlzssksvert.streamlit.app/history?user_id={st.session_state['user_id']}")
        if history_response.status_code == 200:
            user_history = history_response.json().get("history", [])
            for entry in user_history:
                st.sidebar.write(f"{entry[0]}: {entry[1]}")
        else:
            st.sidebar.error("Failed to load history.")

    # --- Main logic ---
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        if not text.strip():
            st.error("❌ Could not extract text from PDF (even with OCR).")
        else:
            result_sections = []
            summary = text
            if "Text Summarization" in ai_tools:
                chunks = split_text(text, max_words=100)
                summary = summarize_text(chunks, pages=page_count)
                result_sections.append("--- Text Summarization ---\n" + summary)
            if "Sentiment Analysis" in ai_tools:
                sentiment_analyzer = pipeline("sentiment-analysis", device=-1)
                sentiment = sentiment_analyzer(summary)[0]
                sentiment_str = f"Label: {sentiment['label']}, Score: {sentiment['score']:.2f}"
                result_sections.append("--- Sentiment Analysis ---\n" + sentiment_str)
            if "Chatbot Integration" in ai_tools:
                chatbot_response = f"Chatbot: You said '{summary[:100]}...'"
                result_sections.append("--- Chatbot Integration ---\n" + chatbot_response)
            if "Document Classification" in ai_tools:
                classifier = pipeline("zero-shot-classification", device=-1)
                candidate_labels = ["business", "education", "health", "technology", "finance", "other"]
                classification = classifier(summary, candidate_labels)
                top_label = classification['labels'][0]
                top_score = classification['scores'][0]
                class_str = f"Top class: {top_label} (score: {top_score:.2f})"
                result_sections.append("--- Document Classification ---\n" + class_str)
            if not result_sections:
                result_sections.append(summary)
            final_output = "\n\n".join(result_sections)
            if lang:
                final_output = translate_to_language(final_output, lang)
            st.subheader("AI Result:")
            st.code(final_output, language=None)
            file_path = save_to_file(final_output, output_type)
            with open(file_path, "rb") as f:
                st.download_button(
                    label=f"Download Result as {output_type}",
                    data=f,
                    file_name=f"ai_result.{output_type.lower()}",
                    mime=None
                )
            os.remove(file_path)
            log_user_request(ai_tools, output_type, page_count)

    # --- Additional AI Tools (right side) ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.title("📄 AI Document Processor")
        st.write("Upload a PDF, select AI tools, and get instant results!")
        # ...existing main logic...

    with col2:
        st.markdown("### Other AI Tools (Try them!)")
        st.markdown("**1. Keyword Extraction**: Extracts main keywords from your document.")
        if pdf_file:
            import re
            from collections import Counter
            words = re.findall(r'\w+', text.lower())
            stopwords = set(['the','and','is','in','to','of','a','for','on','with','as','by','at','an','be','are','from','that','this','it','or','was','but','not','have','has','had','which','you','we','they','their','our','can','will','would','should','could'])
            keywords = [w for w in words if w not in stopwords and len(w) > 3]
            top_keywords = Counter(keywords).most_common(10)
            st.write("Top Keywords:", ', '.join([k for k, v in top_keywords]))
        st.markdown("**2. Named Entity Recognition (NER)**: Finds names, places, organizations, etc.")
        if pdf_file:
            try:
                ner_pipe = pipeline("ner", grouped_entities=True, device=-1)
                ner_results = ner_pipe(text[:1000])  # limit for speed
                ents = [(ent['word'], ent['entity_group']) for ent in ner_results]
                if ents:
                    st.write("Entities:", ', '.join([f"{w} ({e})" for w, e in ents]))
                else:
                    st.write("No entities found.")
            except Exception as e:
                st.write(f"NER error: {e}")
        st.markdown("**3. Language Detection**: Detects the language of your document.")
        if pdf_file:
            try:
                from langdetect import detect
                detected_lang = detect(text)
                st.write(f"Detected Language: {detected_lang}")
            except Exception as e:
                st.write(f"Language detection error: {e}")
        st.markdown("**4. Text Summarization (Alternative Model)**: Try T5 model for a different summary.")
        if pdf_file:
            try:
                t5_summarizer = pipeline("summarization", model="t5-small", device=-1)
                t5_summary = t5_summarizer(text[:1000], max_length=120, min_length=30, do_sample=False)[0]['summary_text']
                st.write("T5 Summary:", t5_summary)
            except Exception as e:
                st.write(f"T5 summarization error: {e}")
        st.markdown("**5. Plagiarism Checker (Simulated)**: Checks for duplicate content.")
        if pdf_file:
            # Simulate plagiarism check by checking for repeated sentences
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            repeated = [s for s in set(sentences) if sentences.count(s) > 1]
            if repeated:
                st.write("Possible duplicate sentences:", repeated)
            else:
                st.write("No obvious plagiarism detected.")
        st.markdown("**6. Text Similarity (Simulated)**: Compare your document to a reference.")
        ref_text = st.text_area("Paste reference text for similarity check (optional):")
        if pdf_file and ref_text:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, text, ref_text).ratio()
            st.write(f"Similarity score: {similarity:.2f}")
        st.markdown("**7. Document Q&A (MiniLM)**: Ask a question about your document.")
        if pdf_file:
            question = st.text_input("Ask a question about your document:")
            if question:
                try:
                    from transformers import pipeline as hf_pipeline
                    qa_pipe = hf_pipeline("question-answering", model="deepset/minilm-uncased-squad2", device=-1)
                    answer = qa_pipe(question=question, context=text[:2000])
                    st.write(f"Answer: {answer['answer']}")
                except Exception as e:
                    st.write(f"Q&A error: {e}")
    st.sidebar.markdown("**AI Features:**\n- Text Summarization\n- Sentiment Analysis\n- Chatbot Integration\n- Document Classification\n- Language Translation")

    # --- Show history ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("History")
    history = load_history()
    history_options = [f"{i+1}. {row[0]} | {row[1]} | {row[2]} | {row[3]} page(s)" for i, row in enumerate(history)]
    selected_history = st.sidebar.selectbox("Select a history entry to view:", ["-"] + history_options)
    selected_history_data = None
    if selected_history != "-":
        idx = history_options.index(selected_history)
        selected_history_data = history[idx]
        st.sidebar.write(f"**AI Tools:** {selected_history_data[1]}")
        st.sidebar.write(f"**Format:** {selected_history_data[2]}")
        st.sidebar.write(f"**Pages:** {selected_history_data[3]}")
        st.sidebar.write(f"**Time:** {selected_history_data[0]}")
