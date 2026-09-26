"""Streamlit research dashboard. Start with: python -m streamlit run app.py"""
import html
import json
from datetime import timedelta
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import sparse

from crime_nlp.config import ROOT, PROCESSED, ARTIFACTS, MODELS
from crime_nlp.text import evidence_answer

st.set_page_config(page_title="Narrative Intelligence | Crime NLP", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
PALETTE = ["#087F75", "#3E6DA8", "#DA9C42", "#9167AD", "#DC7B65", "#60AEB0", "#768BC1", "#9DAD71"]
px.defaults.color_discrete_sequence = PALETTE
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1540px}
[data-testid="stSidebar"] {background:#10283b}
[data-testid="stSidebar"] * {color:#e5edf4}
[data-testid="stSidebar"] hr {border-color:#355062}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#182c3e}
[data-testid="stMetric"] {background:white;border:1px solid #e2e8ef;border-radius:12px;padding:18px 22px}
[data-testid="stMetricLabel"] {color:#6a7888;font-size:.8rem}
[data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-weight:800;font-size:1.85rem}
div.stButton>button[kind="primary"] {border-radius:8px;font-weight:600}
.eyebrow {font-size:11px;letter-spacing:.18em;font-weight:700;color:#078578;text-transform:uppercase;margin-bottom:8px}
.subtitle {color:#6d7b8b;font-size:15px;margin-top:-10px;margin-bottom:24px}
.brand {font-family:'Manrope',sans-serif;font-weight:800;font-size:23px;line-height:1.25;margin:8px 0 6px}
.brand-sub {font-size:10px;letter-spacing:.18em;color:#8fb1c2!important;margin-bottom:30px}
.hero {background:linear-gradient(115deg,#132f42,#174e52);border-radius:16px;padding:28px 34px;color:#fff;margin:12px 0 26px}
.hero h2 {color:white;font-size:26px;margin:0 0 8px;font-weight:700}
.hero p {color:#bfd5de;margin:0;max-width:760px;font-size:14px;line-height:1.7}
.badge {display:inline-block;font-size:10px;letter-spacing:.09em;padding:5px 9px;border-radius:5px;background:#e5f3ee;color:#087565;font-weight:700}
.note {padding:14px 18px;background:#edf3f7;border-left:3px solid #6b93a8;border-radius:0 8px 8px 0;font-size:13px;line-height:1.6;color:#4e6375}
.topic-card {background:white;border:1px solid #e1e8ef;border-radius:12px;padding:20px;min-height:160px;margin-bottom:15px}
.topic-card h3 {font-size:18px;margin:0 0 12px}.topic-card p {color:#627486;font-size:14px;line-height:1.8}
.section-label {font-size:11px;letter-spacing:.12em;font-weight:700;color:#6b7a89;text-transform:uppercase}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load_data(stamp):
    df = pd.read_parquet(PROCESSED / "dashboard.parquet")
    df["event_time"] = pd.to_datetime(df.event_time)
    return df


@st.cache_resource
def search_resources(stamp):
    return joblib.load(MODELS / "tfidf.joblib").named_steps["vectorizer"], sparse.load_npz(ARTIFACTS / "search_matrix.npz")


@st.cache_data(max_entries=64, show_spinner=False)
def cached_analysis(text, use_bert=False):
    from crime_nlp.inference import analyze
    return analyze(text, use_bert=use_bert)


def read_artifact(name):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def heading(eyebrow, title, subtitle):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)


def chart(fig, height=350):
    fig.update_layout(height=height, margin=dict(l=5, r=15, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans, sans-serif", color="#526679", size=12), legend_title_text="", hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e5ebf0", zeroline=False)
    st.plotly_chart(fig, width="stretch")


st.sidebar.markdown('<div class="brand">◈ Narrative<br>Intelligence</div><div class="brand-sub">CRIME ANALYTICS LAB</div>', unsafe_allow_html=True)
st.sidebar.markdown("University research project")
page = st.sidebar.radio("Workspace", ["Overview", "Case explorer", "Narrative lab", "Model evaluation", "Topics", "Pipeline & syllabus"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("DATA PROVENANCE")
st.sidebar.markdown("**Synthetic reports**\n\nFictional people, places and events. For NLP research and demonstration.")
st.sidebar.caption("Spark · BERT · CRF · spaCy")

dashboard_path = PROCESSED / "dashboard.parquet"
if not dashboard_path.exists():
    heading("Project setup", "Your research workspace", "Decoding Crime Narratives using NLP and Big Data Analytics")
    st.info("Generate the dataset and run the models to populate this dashboard.")
    st.code("python scripts/download_resources.py\npython run_pipeline.py\npython -m streamlit run app.py", language="bash")
    st.stop()

stamp = dashboard_path.stat().st_mtime_ns
data = load_data(stamp)
evaluation = read_artifact("evaluation.json")
ner_evaluation = read_artifact("ner_evaluation.json")
topics = read_artifact("topics.json")
spark_metrics = read_artifact("spark_metrics.json")


def filters(frame, key):
    a, b, c = st.columns([1.25, 1.25, 1.6])
    cats = a.multiselect("Crime categories", sorted(frame.crime_type.unique()), key=key + "_cats", placeholder="All categories")
    districts = b.multiselect("Districts", sorted(frame.district.unique()), key=key + "_districts", placeholder="All districts")
    date_range = c.date_input("Reporting period", value=(frame.event_time.min().date(), frame.event_time.max().date()), key=key + "_dates")
    mask = pd.Series(True, index=frame.index)
    if cats:
        mask &= frame.crime_type.isin(cats)
    if districts:
        mask &= frame.district.isin(districts)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        mask &= frame.event_time.between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1] + timedelta(days=1)), inclusive="left")
    return frame.loc[mask]


def show_analysis(result, key):
    from crime_nlp.inference import highlight_entities, language_model
    from spacy import displacy
    top = st.columns(4)
    top[0].metric("TF-IDF prediction", result["crime_type"])
    top[1].metric("VADER polarity", f"{result['sentiment']['compound']:+.2f}")
    top[2].metric("Threat-language cues", result["threat"]["level"])
    top[3].metric("Extracted entities", len(result["entities"]))
    if "bert_prediction" in result:
        st.info(f"BERT sequence classification: {result['bert_prediction']}")
    tabs = st.tabs(["Entities & summary", "Syntax & morphology", "Evidence search", "Analysis details"])
    with tabs[0]:
        st.markdown("#### Annotated narrative")
        st.caption("Predicted by the trained CRF: SUSPECT · VICTIM · LOCATION · WEAPON. Allegations in text are not verified facts.")
        st.markdown(highlight_entities(result["text"], result["entities"]), unsafe_allow_html=True)
        st.markdown("#### Case summary")
        st.write(result["summary"])
        st.caption("Extractive TextRank: source sentences ranked by similarity, with a small lead-sentence prior.")
        if result["entities"]:
            st.dataframe(pd.DataFrame(result["entities"])[["text", "label", "start", "end"]], hide_index=True, width="stretch")
    with tabs[1]:
        st.markdown("#### How the narrative is structured")
        doc = language_model()(result["text"])
        sents = list(doc.sents)
        selected = st.selectbox("Sentence to inspect", range(len(sents)), format_func=lambda i: sents[i].text[:130], key=key + "_sentence")
        markup = displacy.render(sents[selected].as_doc(), style="dep", options={"compact": True, "distance": 115, "color": "#203b4d", "bg": "#ffffff", "font": "Arial"}, page=False)
        components.html('<div style="overflow-x:auto;background:#fff;border-radius:10px;padding:14px">' + markup + '</div>', height=350, scrolling=True)
        st.dataframe(pd.DataFrame(result["tokens"]), hide_index=True, width="stretch")
        st.caption("spaCy predicts POS tags, lemmas, morphology and dependency heads; these annotations are not evaluated against a gold syntax corpus here.")
    with tabs[2]:
        question = st.text_input("Find supporting evidence in this case", placeholder="e.g. weapon, witness statement, camera footage", key=key + "_question")
        if question:
            answer = evidence_answer(question, result["text"])
            st.write(answer["answer"])
            st.caption(f"Lexical similarity: {answer['score']:.3f}. Retrieves source text; it does not generate a factual answer.")
    with tabs[3]:
        st.write("**Threat-language heuristic:**", ", ".join(result["threat"]["cues"]) or "No unnegated cues found")
        st.caption("This cue score is a demonstration heuristic. VADER sentiment measures polarity separately; neither score estimates real-world danger.")
        st.write("**Generic spaCy entities:**")
        st.dataframe(pd.DataFrame(result["generic_entities"], columns=["text", "label"]), hide_index=True, width="stretch")
        st.download_button("Download analysis JSON", json.dumps(result, indent=2, ensure_ascii=False), file_name="narrative_analysis.json", mime="application/json", key=key + "_download")


if page == "Overview":
    heading("Research workspace / overview", "From narratives to insight", "Decoding Crime Narratives using NLP and Big Data Analytics")
    st.markdown('<div class="hero"><h2>Every report has a story. Find the patterns.</h2><p>Explore a corpus of fictional case narratives, follow recurring themes, and inspect the language behind each report. A complete journey from distributed processing to explainable text analysis.</p></div>', unsafe_allow_html=True)
    selected = filters(data, "overview")
    a, b, c, d = st.columns(4)
    a.metric("REPORTS IN VIEW", f"{len(selected):,}")
    b.metric("CRIME CATEGORIES", selected.crime_type.nunique())
    c.metric("DISTRICTS", selected.district.nunique())
    d.metric("BERT TEST MACRO-F1", f"{evaluation['models']['BERT']['macro_f1']:.1%}")
    st.caption("Counts use the current filters and synthetic source labels. Model score uses the fixed held-out test set.")
    if selected.empty:
        st.info("No reports match these filters. Expand the reporting period or categories.")
        st.stop()
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("#### Reporting patterns over time")
        monthly = selected.groupby(["month", "crime_type"], observed=True).size().reset_index(name="Reports")
        chart(px.area(monthly, x="month", y="Reports", color="crime_type", labels={"month": "", "crime_type": "Crime category"}), 360)
    with right:
        st.markdown("#### Composition of the corpus")
        counts = selected.crime_type.value_counts().rename_axis("Category").reset_index(name="Reports")
        chart(px.bar(counts.sort_values("Reports"), x="Reports", y="Category", orientation="h", color="Reports", color_continuous_scale=["#b3dcd3", "#087f75"]).update_layout(coloraxis_showscale=False), 360)
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("#### Categories across fictional districts")
        heat = pd.crosstab(selected.district, selected.crime_type)
        chart(px.imshow(heat, color_continuous_scale="Teal", aspect="auto", labels=dict(x="", y="", color="Reports")), 320)
    with right:
        st.markdown("#### Language polarity")
        chart(px.histogram(selected, x="sentiment", nbins=25, labels={"sentiment": "VADER compound score"}, color_discrete_sequence=["#3e6da8"]), 320)
    st.markdown('<div class="note"><b>Reading these charts:</b> the corpus is generated, so the trends reflect the generator rather than real crime rates. All people, places and events are fictional.</div>', unsafe_allow_html=True)

elif page == "Case explorer":
    heading("Research workspace / case explorer", "Explore the case library", "Search, filter and inspect the evidence in individual narratives.")
    query = st.text_input("Search narratives", placeholder="Try: unauthorized account access, stolen phone, fire, or a report ID")
    selected = filters(data, "explorer")
    limit = st.slider("Results to display", 5, 100, 20, step=5)
    if query.strip():
        vectorizer, matrix = search_resources(stamp)
        scores = np.asarray((matrix @ vectorizer.transform([query]).T).toarray()).ravel()
        subset = selected.copy()
        subset["relevance"] = scores[subset.index]
        id_matches = subset.report_id.str.contains(query.strip(), case=False, regex=False)
        subset.loc[id_matches, "relevance"] = 1.
        selected = subset[subset.relevance > .02].sort_values(["relevance", "report_id"], ascending=[False, True])
    else:
        selected = selected.sort_values("event_time", ascending=False)
    st.caption(f"{len(selected):,} matching reports · TF-IDF ranked search · all records are synthetic")
    if selected.empty:
        st.info("No matching reports. Try different words or broaden the filters.")
        st.stop()
    display = selected.head(limit)
    st.dataframe(display[["report_id", "crime_type", "district", "reported_at", "predicted_crime", "threat_level"]].rename(columns={"crime_type": "Source label", "predicted_crime": "TF-IDF prediction", "threat_level": "Language cues"}), hide_index=True, width="stretch")
    choice = st.selectbox("Open a report", display.report_id.tolist(), format_func=lambda report_id: f"{report_id} · {display.loc[display.report_id == report_id, 'crime_type'].iloc[0]}")
    row = display.loc[display.report_id == choice].iloc[0]
    st.divider()
    st.markdown(f"#### {choice}")
    st.caption(f"{row.district} · {row.reported_at} · {row.split} split")
    with st.spinner("Analyzing the narrative..."):
        result = cached_analysis(row.narrative)
    show_analysis(result, "case")
    st.download_button("Export search results CSV", display[["report_id", "narrative", "crime_type", "district", "predicted_crime"]].to_csv(index=False), "search_results.csv", "text/csv")

elif page == "Narrative lab":
    heading("Research workspace / narrative lab", "Read between the lines", "Analyze your own text with the trained NLP pipeline.")
    st.caption("Models are trained on fictional English narratives. Results on other writing styles may be unreliable. Uploaded text is analyzed in memory and is not saved by the app.")
    upload = st.file_uploader("Optional: open a text case file", type=["txt"])
    default = data.iloc[0].narrative
    if upload:
        if upload.size > 120000:
            st.error("Use a UTF-8 text file smaller than 120 KB.")
            st.stop()
        try:
            default = upload.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError:
            st.error("The uploaded text must use UTF-8 encoding.")
            st.stop()
    text = st.text_area("Case narrative", value=default, height=220, max_chars=30000, key="lab_text_" + (upload.name if upload else "sample"))
    use_bert = st.checkbox("Include BERT sequence classification", value=True)
    if st.button("Analyze narrative", type="primary"):
        if not text.strip():
            st.warning("Enter a narrative first.")
        else:
            with st.spinner("Extracting entities, syntax, topics and context..."):
                # Avoid caching user submissions across browser sessions.
                from crime_nlp.inference import analyze
                st.session_state["lab_result"] = analyze(text, use_bert)
    if "lab_result" in st.session_state:
        show_analysis(st.session_state["lab_result"], "lab")

elif page == "Model evaluation":
    heading("Research workspace / model evaluation", "Measure what the models learn", "Real metrics from a fixed test set of previously unseen scenario templates.")
    protocol = evaluation["protocol"]
    a, b, c, d = st.columns(4)
    a.metric("TRAIN NARRATIVES", f"{protocol['train_rows']:,}")
    b.metric("VALIDATION NARRATIVES", f"{protocol['validation_rows']:,}")
    c.metric("TEST NARRATIVES", f"{protocol['test_rows']:,}")
    d.metric("CRF ENTITY F1", f"{ner_evaluation['strict_entity_f1']:.1%}")
    st.info("Templates are disjoint across train, validation and test. Vectorizers and Word2Vec are fit only on training text. BERT checkpoints are selected by validation macro-F1. These are synthetic benchmark scores.")
    scores = pd.DataFrame([{"Model": name, "Accuracy": item["accuracy"], "Macro F1": item["macro_f1"], "Weighted F1": item["weighted_f1"]} for name, item in evaluation["models"].items()])
    chart(px.bar(scores.melt(id_vars="Model", var_name="Metric", value_name="Score"), x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1.05]), 340)
    st.dataframe(scores.style.format({"Accuracy": "{:.3f}", "Macro F1": "{:.3f}", "Weighted F1": "{:.3f}"}), hide_index=True, width="stretch")
    chosen = st.selectbox("Inspect model", list(evaluation["models"]))
    m = evaluation["models"][chosen]
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Where predictions agree and differ")
        chart(px.imshow(np.array(m["confusion_matrix"]), x=m["labels"], y=m["labels"], text_auto=True, color_continuous_scale="Teal", labels={"x": "Predicted class", "y": "True class", "color": "Reports"}), 460)
    with right:
        st.markdown("#### Per-class performance")
        per_class = pd.DataFrame(m["classification_report"]).T
        st.dataframe(per_class.loc[m["labels"], ["precision", "recall", "f1-score", "support"]].round(3), width="stretch")
    st.markdown("#### Entity extraction: strict span evaluation")
    entity_rows = {k: v for k, v in ner_evaluation["report"].items() if k in ner_evaluation["labels"]}
    st.dataframe(pd.DataFrame(entity_rows).T.round(3), width="stretch")
    st.caption(f"CRF evaluated on {ner_evaluation['test_rows']} held-out narratives. An entity must have both the correct span and label to count as correct.")
    st.caption("CRF sentence-context features were refined after initial error analysis. This is a development benchmark, not an untouched external test.")
    if "ablation" in ner_evaluation:
        with st.expander("CRF context ablation and development disclosure"):
            st.json(ner_evaluation["ablation"])
    st.markdown("#### BERT training history")
    chart(px.line(pd.DataFrame(evaluation["models"]["BERT"]["history"]), x="epoch", y=["loss", "validation_macro_f1"], markers=True), 250)
    st.markdown("#### Distributed MLlib baseline")
    st.json(spark_metrics["mllib"], expanded=True)
    st.caption("MLlib uses the full training/test partitions, so its sample sizes differ from the four-model comparison above.")
    st.download_button("Download measured metrics", json.dumps(evaluation, indent=2), "evaluation.json", "application/json")

elif page == "Topics":
    heading("Research workspace / recurring themes", "Discover recurring language", "Latent Dirichlet Allocation groups words that occur together in the narratives.")
    a, b, c = st.columns(3)
    a.metric("LATENT TOPICS", len(topics["topics"]))
    b.metric("HELD-OUT PERPLEXITY", f"{topics['held_out_perplexity']:.1f}")
    c.metric("TOP-TERM DIVERSITY", f"{topics['topic_diversity']:.1%}")
    st.caption("Topic numbers are arbitrary; themes suggest narrative patterns for review, not verified modus operandi. Perplexity is comparable only under the same vocabulary and corpus protocol.")
    columns = st.columns(2)
    for topic in topics["topics"]:
        with columns[topic["id"] % 2]:
            st.markdown(f'<div class="topic-card"><span class="badge">TOPIC {topic["id"] + 1:02d}</span><h3>{html.escape(" · ".join(topic["terms"][:3]))}</h3><p>{html.escape(" / ".join(topic["terms"][3:]))}</p></div>', unsafe_allow_html=True)
    selected_topic = st.selectbox("Explore a topic", range(len(topics["topics"])), format_func=lambda i: f"Topic {i + 1:02d}: {', '.join(topics['topics'][i]['terms'][:4])}")
    topic_data = data[data.topic_id == selected_topic]
    chart(px.line(topic_data.groupby("month", observed=True).size().reset_index(name="Reports"), x="month", y="Reports", markers=True), 280)
    st.dataframe(topic_data.nlargest(8, "topic_probability")[["report_id", "narrative", "topic_probability"]], hide_index=True, width="stretch")

else:
    heading("Research workspace / pipeline", "A traceable research pipeline", "From generated raw reports to distributed processing, NLP models and an interactive dashboard.")
    st.markdown("**Generate → Spark SQL cleaning → partitioned Parquet → model training → corpus enrichment → dashboard**")
    a, b, c, d = st.columns(4)
    a.metric("RAW ROWS", f"{spark_metrics['raw_rows']:,}")
    b.metric("INVALID ROWS REMOVED", f"{spark_metrics['invalid_rows_removed']:,}")
    c.metric("DUPLICATES REMOVED", f"{spark_metrics['duplicates_removed']:,}")
    d.metric("CLEAN REPORTS", f"{spark_metrics['clean_rows']:,}")
    st.caption(spark_metrics["deployment_note"])
    mapping = pd.DataFrame([
        ["Big Data 1", "Architecture, Hadoop/MapReduce context", "docs/REPORT.md; Spark execution and partitioned Parquet"],
        ["Big Data 2", "Scala, collections and functional programming", "scala/CrimeAnalytics.scala: case classes, map/filter, traits, matching, Spark SQL"],
        ["Big Data 3", "Spark, relational/distributed data, transformations/actions", "crime_nlp/spark_pipeline.py: repartition, filter, window, SQL groupBy, count, Parquet"],
        ["Big Data 4", "Spark MLlib pipelines, classification and evaluation", "RegexTokenizer → StopWords → HashingTF → IDF → LogisticRegression"],
        ["NLP 1", "Syntax, morphology and linguistic preprocessing", "spaCy tokens, lemmas, morphology, POS, dependencies"],
        ["NLP 2", "BoW, TF-IDF, neural word representations", "CountVectorizer, TfidfVectorizer, Gensim skip-gram Word2Vec"],
        ["NLP 3", "Transformers, sequence models, topic modeling", "Fine-tuned BERT + linear-chain CRF BIO tagging + LDA"],
        ["NLP 4", "NER, parsing, sentiment, summarization, evaluation", "CRF entities, spaCy, NLTK VADER, TextRank, macro-F1 and strict entity F1"],
    ], columns=["Syllabus", "Requirement", "Implementation"])
    st.dataframe(mapping, hide_index=True, width="stretch")
    with st.expander("Spark execution details", expanded=True):
        st.json(spark_metrics)
    with st.expander("Dataset provenance and evaluation limits"):
        st.json(read_artifact("dataset_manifest.json"))
        st.markdown("No real crime dataset or real-world crime-rate claim is included. Sentiment and threat heuristics have no labeled evaluation set. Summaries are extractive and require review; entity roles are predictions from fictional training examples.")
    st.markdown("**Reproduce the pipeline**")
    st.code("python scripts/download_resources.py\npython run_pipeline.py --records 60000 --master 'local[2]'\npython -m pytest -q\npython -m streamlit run app.py", language="bash")
    st.caption("Documentation: README.md · docs/REPORT.md · docs/SYLLABUS_MAPPING.md · docs/DEMO_GUIDE.md")

st.divider()
st.caption("DECODING CRIME NARRATIVES  /  NLP & BIG DATA ANALYTICS  /  SYNTHETIC RESEARCH CORPUS")
