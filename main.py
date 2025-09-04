import streamlit as st
from sentence_transformers import SentenceTransformer, util
import re
import fitz  # PyMuPDF for PDFs
import docx
import plotly.graph_objects as go
def extract_text_from_pdf(file) -> str:
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()

def score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "lightgreen"}
            ]},
        title={'text': "Fit Score (%)"}
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_skills(skills, title, color):
    st.markdown(f"### {title}")
    if not skills:
        st.write("None")
        return
    for skill in skills:
        st.markdown(
            f"<span style='background-color:{color}; color:white; "
            f"padding:5px 12px; border-radius:15px; margin:4px; "
            f"display:inline-block;'>{skill}</span>",
            unsafe_allow_html=True
        )
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume and compare it with a job description to see skill matches and overall fit score.")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This AI Resume Analyzer uses NLP embeddings to measure similarity "
             "between your resume and a job description. It also highlights matched and missing skills.")
    st.markdown("**Tech stack:** Streamlit, Sentence Transformers, PyMuPDF, Docx, Plotly")

uploaded_resume = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
jd_text = st.text_area("Paste Job Description:")

if st.button("🔍 Analyze Resume"):
    if uploaded_resume is None or not jd_text.strip():
        st.error("Please upload a resume and paste a job description.")
    else:
        # Extract text
        if uploaded_resume.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_resume)
        elif uploaded_resume.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(uploaded_resume)
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Normalize
        resume_text_norm = normalize(resume_text)
        jd_text_norm = normalize(jd_text)

        # Embedding similarity
        emb = model.encode([resume_text_norm, jd_text_norm], convert_to_tensor=True)
        score = util.cos_sim(emb[0], emb[1]).item()
        sim_score = round((score + 1) / 2 * 100, 2)  # 0–100 scale

        # Skill match (simple demo)
        SKILLS = {"python", "sql", "java", "aws", "tensorflow", "pytorch", "docker"}
        resume_tokens = set(re.findall(r"[a-zA-Z+#.]+", resume_text_norm))
        hits = SKILLS.intersection(resume_tokens)
        missing = SKILLS - hits

        # Tabs for Results vs Resume Preview
        tab1, tab2 = st.tabs(["📊 Results", "📜 Resume Preview"])

        with tab1:
            st.subheader("✅ Fit Score")
            score_gauge(sim_score)

            display_skills(hits, "Matched Skills", "green")
            display_skills(missing, "Missing Skills", "crimson")

        with tab2:
            st.subheader("Extracted Resume Text")
            st.text_area("Resume Text",
                         resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""),
                         height=400)

