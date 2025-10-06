import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import json
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# ============================================================================
# INITIAL CONFIGURATION
# ============================================================================

load_dotenv()

st.set_page_config(
    page_title="NASA Space Biology Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GROQ_MODEL = 'llama-3.3-70b-versatile'

# Configure Groq
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found in .env file")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedding_model():
    with st.spinner("🤖 Loading embedding model..."):
        return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/publicaciones.csv')
        embeddings = np.load('data/corpus_embeddings.npy')
        
        if len(df) != len(embeddings):
            st.error("❌ Error: Number of publications doesn't match embeddings")
            st.stop()
        
        return df, embeddings
    except FileNotFoundError as e:
        st.error(f"❌ Error: Data files not found. {str(e)}")
        st.info("""
        **Steps to resolve:**
        1. Make sure you have the file `data/publicaciones.csv`
        2. Run `python process_pdfs.py` or `python quick_download.py` first
        """)
        st.stop()

# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def semantic_search(query, top_k=5):
    model = load_embedding_model()
    df, corpus_embeddings = load_data()
    
    query_embedding = model.encode(query, convert_to_tensor=False)
    
    similarities = np.dot(corpus_embeddings, query_embedding) / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    
    results = []
    for idx, score in zip(top_indices, top_scores):
        result = df.iloc[idx].to_dict()
        result['similarity_score'] = float(score)
        results.append(result)
    
    return results

# ============================================================================
# AI FUNCTIONS WITH GROQ
# ============================================================================

def generate_summary(text, title, mode="academic"):
    """Generate summary using Groq/Llama"""
    
    if not text or len(text.strip()) < 50:
        return "⚠️ Abstract too short or unavailable to generate summary."
    
    # Truncate text
    max_chars = 2500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    if mode == "academic":
        prompt = f"""You are an expert in NASA space bioscience.

Title: {title}

Abstract: {text}

Summarize this scientific publication in 3 key points:
1. Methodology and experimental design
2. Main results with specific data
3. Implications for space exploration

Use precise scientific terminology. Each point: 2-3 sentences."""
    else:  # outreach
        prompt = f"""You are a science communicator specializing in space.

Title: {title}

Abstract: {text}

Explain this space research for high school students in 3 simple points:
1. What experiment was done? (as if explaining to a friend)
2. What did they discover? (with everyday examples)
3. Why is it important for space travel?

Use simple language, analogies, and avoid technical jargon."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating summary: {str(e)}"

def extract_entities(text, title):
    """Extract entities using Groq/Llama"""
    
    max_chars = 1500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""Analyze this scientific text about space biology.

Title: {title}

Text: {text}

Extract in JSON format:
- "organism": Organism studied
- "condition": Space condition (microgravity, radiation, etc.)
- "key_finding": Main finding (max 15 words)
- "methodology": Method used

If info is missing, use "Not specified".
Respond ONLY with JSON, no markdown."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        content = content.replace('```json', '').replace('```', '').strip()
        
        return json.loads(content)
    except:
        return {
            "organism": "N/A",
            "condition": "N/A",
            "key_finding": "Not available",
            "methodology": "N/A"
        }

def generate_chat_response(prompt, context, mode="academic"):
    """Generate chat response using Groq/Llama"""
    
    max_context = 4000
    if len(context) > max_context:
        context = context[:max_context] + "\n...(more papers available)"
    
    if mode == "academic":
        system_prompt = f"""You are an expert researcher in NASA space biology with access to 607 scientific papers.

Most relevant papers for this query:
{context}

INSTRUCTIONS:
- These are only 3 examples of the 607 available papers
- Cite specific papers: "According to Paper 1..."
- If papers are not relevant, suggest rephrasing the question
- Use precise scientific terminology
- Be conversational but accurate"""
    else:  # outreach
        system_prompt = f"""You are a science communicator specializing in space with access to 607 NASA papers.

Most relevant papers:
{context}

INSTRUCTIONS:
- These are 3 examples of the 607 available papers
- Respond in a friendly and clear manner
- Use analogies when possible
- Be enthusiastic and educational"""
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {str(e)[:150]}"

def generate_citation(result, format="apa7"):
    """Generate citation in different formats"""
    title = result.get('title', 'No title')
    authors = result.get('authors', 'N/A')
    year = result.get('year', 'N/A')
    url = result.get('source_url', '')
    
    if format == "apa7":
        if authors != 'N/A':
            citation = f"{authors} ({year}). {title}. NASA Space Biology Archive."
        else:
            citation = f"{title} ({year}). NASA Space Biology Archive."
        
        if url:
            citation += f" {url}"
        
        return citation
    
    elif format == "bibtex":
        safe_key = re.sub(r'[^a-zA-Z0-9]', '', title[:20])
        return f"""@article{{{safe_key}{year},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{{year}}},
  journal = {{NASA Space Biology Archive}},
  url = {{{url}}}
}}"""
    
    elif format == "plain":
        return f"{authors}. \"{title}\". {year}."

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_year_distribution(df):
    """Chart of distribution by years"""
    year_counts = df['year'].value_counts().sort_index()
    
    fig = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        title="📅 Distribution of Publications by Year",
        labels={'x': 'Year', 'y': 'Number of Publications'},
        color=year_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=350)
    return fig

# ============================================================================
# GLOSSARY
# ============================================================================

GLOSSARY = {
    "Microgravity": "Very low gravity condition, nearly zero, like astronauts experience in orbit.",
    "Transcriptomics": "Study of all genes that are activated in a cell at a given moment.",
    "ISS": "International Space Station.",
    "Arabidopsis": "Small plant commonly used in scientific research.",
    "C. elegans": "Microscopic worm used in biological studies.",
    "Gene expression": "Process by which genes produce proteins.",
    "Cosmic radiation": "High-energy particles from space that can damage cells.",
    "EMCS": "European Modular Cultivation System - cultivation system on the ISS."
}

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    st.title("🚀 NASA Space Biology Knowledge Engine")
    st.markdown("""
    Explore **NASA scientific publications** using advanced AI.  
    Semantic search · Automatic summaries · Entity extraction
    """)
    
    df, embeddings = load_data()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("👤 User Mode")
        user_mode = st.radio(
            "Select your profile:",
            ["👨‍🎓 Academic", "🎓 Outreach"],
            help="Academic: technical terminology / Outreach: simple language"
        )
        
        mode = "academic" if "Academic" in user_mode else "outreach"
        
        st.divider()
        
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        st.subheader("🤖 AI Options")
        show_summary = st.checkbox("Generate summaries", value=True)
        show_entities = st.checkbox("Extract entities", value=True)
        show_citation = st.checkbox("Show citations", value=True)
        
        st.divider()
        
        st.subheader("🔍 Filters")
        years = df['year'].dropna().unique()
        if len(years) > 0:
            year_filter = st.multiselect("Filter by year", options=sorted(years, reverse=True), default=[])
        else:
            year_filter = []
        
        st.divider()
        
        st.markdown(f"""
        ### 📊 Statistics
        - **Total publications**: {len(df):,}
        - **LLM Model**: Llama 3.3 70B
        - **Embeddings**: MiniLM-L6-v2
        - **Dimension**: {embeddings.shape[1]}D
        """)
        
        with st.expander("📖 Scientific Glossary"):
            for term, definition in GLOSSARY.items():
                st.markdown(f"**{term}**: {definition}")
        
        st.divider()
        st.markdown("Developed for the **NASA Space Biology Challenge** Guadalajara 2025 🇲🇽")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "💬 Chatbot", "📊 Visualizations", "📚 Explorer"])
    
    # ========================================================================
    # TAB 1: SEARCH
    # ========================================================================
    
    with tab1:
        st.header("🔍 Intelligent Search")
        
        query = st.text_input("What do you want to research?", placeholder="e.g., effects of microgravity on plants")
        
        st.markdown("**Quick examples:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💡 Radiation & DNA", use_container_width=True):
                query = "effects of space radiation on DNA"
        with col2:
            if st.button("🌱 Plants in space", use_container_width=True):
                query = "experiments with plants in microgravity"
        with col3:
            if st.button("🧬 Immune system", use_container_width=True):
                query = "changes in astronaut immune system"
        with col4:
            if st.button("🔬 C. elegans", use_container_width=True):
                query = "C elegans studies in space"
        
        if query and len(query.strip()) > 0:
            with st.spinner("🔎 Searching..."):
                results = semantic_search(query, top_k=top_k)
            
            if year_filter:
                results = [r for r in results if r.get('year') in year_filter]
            
            if not results:
                st.warning("⚠️ No results found")
                return
            
            st.success(f"✅ Found **{len(results)} relevant results**")
            
            avg_score = sum(r['similarity_score'] for r in results) / len(results)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Results", len(results))
            with col2:
                st.metric("🎯 Average Relevance", f"{avg_score:.1%}")
            with col3:
                st.metric("🥇 Best Match", f"{results[0]['similarity_score']:.1%}")
            
            st.divider()
            
            for i, result in enumerate(results, 1):
                with st.expander(f"**{i}. {result['title']}** · Similarity: {result['similarity_score']:.1%}", expanded=(i == 1)):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**✍️ Authors:** {result.get('authors', 'N/A')}")
                    with col2:
                        st.metric("📅 Year", result.get('year', 'N/A'))
                    with col3:
                        st.metric("🎯 Match", f"{result['similarity_score']:.1%}")
                    
                    if result.get('source_url'):
                        st.markdown(f"🔗 [View original publication]({result['source_url']})")
                    
                    st.divider()
                    
                    # Summary
                    if show_summary:
                        st.markdown("### 📝 AI-Generated Summary")
                        if mode == "outreach":
                            st.info("💡 **Outreach Mode**: Simplified explanation")
                        
                        summary_placeholder = st.empty()
                        summary_placeholder.info("⏳ Generating summary...")
                        
                        summary = generate_summary(result.get('abstract_text', ''), result['title'], mode)
                        
                        summary_placeholder.empty()
                        st.write(summary)
                        st.divider()
                    
                    # Entities
                    if show_entities:
                        st.markdown("### 🏷️ Extracted Entities")
                        
                        entities_placeholder = st.empty()
                        entities_placeholder.info("⏳ Extracting entities...")
                        
                        entities = extract_entities(result.get('abstract_text', ''), result['title'])
                        
                        entities_placeholder.empty()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🔬 Organism", entities.get('organism', 'N/A'))
                            st.metric("🌌 Condition", entities.get('condition', 'N/A'))
                        with col2:
                            st.metric("🔬 Methodology", entities.get('methodology', 'N/A'))
                            st.markdown(f"**💡 Finding:** {entities.get('key_finding', 'N/A')}")
                        
                        st.divider()
                    
                    # Citations
                    if show_citation:
                        st.markdown("### 📚 Citations")
                        citation_format = st.radio("Format:", ["APA 7", "BibTeX", "Plain text"], key=f"cite_{i}", horizontal=True)
                        fmt_map = {"APA 7": "apa7", "BibTeX": "bibtex", "Plain text": "plain"}
                        citation = generate_citation(result, fmt_map[citation_format])
                        st.code(citation, language="text")
                        if st.button("📋 Copy citation", key=f"copy_{i}"):
                            st.success("✅ Copied to clipboard (simulated)")
                        st.divider()
                    
                    with st.expander("📄 View full abstract"):
                        st.write(result.get('abstract_text', 'Not available'))
        else:
            st.info("""
            👆 **Type a query above** or use the example buttons.
            
            **Query examples:**
            - "How does microgravity affect plant growth?"
            - "Effects of cosmic radiation on human cells"
            - "Cardiovascular system adaptations in space"
            """)
    
    # ========================================================================
    # TAB 2: CHATBOT
    # ========================================================================
    
    with tab2:
        st.header("💬 Space Biology Conversational Assistant")
        st.markdown(f"Ask me anything about the {len(df)} NASA papers. **Current mode:** {user_mode}")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.messages = []
                st.rerun()
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Sources consulted"):
                        for source in message["sources"]:
                            st.markdown(f"**• {source['title']}** ({source['year']})")
                            st.caption(f"Authors: {source['authors']}")
        
        if prompt := st.chat_input("e.g., How does microgravity affect plants?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner(f"🔎 Searching {len(df)} papers..."):
                results = semantic_search(prompt, top_k=3)
            
            context = ""
            sources = []
            for i, r in enumerate(results, 1):
                context += f"\n[Paper {i}]\nTitle: {r['title']}\nAuthors: {r.get('authors', 'N/A')}\nYear: {r.get('year', 'N/A')}\n"
                abstract = r.get('abstract_text', '')
                if abstract and len(abstract) > 100:
                    context += f"Abstract: {abstract[:800]}...\n\n"
                sources.append({'title': r['title'], 'authors': r.get('authors', 'N/A'), 'year': r.get('year', 'N/A')})
            
            with st.spinner("💭 Generating response..."):
                assistant_response = generate_chat_response(prompt, context, mode)
            
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                if sources:
                    with st.expander("📚 Sources consulted"):
                        for source in sources:
                            st.markdown(f"**• {source['title']}** ({source['year']})")
                            st.caption(f"Authors: {source['authors']}")
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_response, "sources": sources})
    
    # ========================================================================
    # TAB 3: VISUALIZATIONS
    # ========================================================================
    
    with tab3:
        st.header("📊 Corpus Visual Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df.columns:
                fig_years = create_year_distribution(df)
                st.plotly_chart(fig_years, use_container_width=True)
        with col2:
            st.metric("📚 Total Publications", len(df))
            if 'year' in df.columns:
                years_range = df['year'].dropna()
                if len(years_range) > 0:
                    st.metric("📅 Year Range", f"{int(years_range.min())} - {int(years_range.max())}")
        st.divider()
        st.info("💡 **Note**: Organism and condition visualizations require processing all papers with AI.")
    
    # ========================================================================
    # TAB 4: EXPLORER
    # ========================================================================
    
    with tab4:
        st.header("📚 Publications Explorer")
        st.markdown("Browse all available publications:")
        display_cols = ['title', 'authors', 'year']
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols].head(50), use_container_width=True, height=400)
        st.info(f"Showing the first 50 of {len(df)} publications")

if __name__ == "__main__":
    main()