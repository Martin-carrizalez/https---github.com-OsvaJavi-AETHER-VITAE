import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import json
import plotly.express as px
import re

# ============================================================================
# 1. DICCIONARIO DE TRADUCCIONES (i18n)
# ============================================================================

TRANSLATIONS = {
    "page_title": {"es": "Motor de Conocimiento de Biología Espacial", "en": "Space Biology Knowledge Engine"},
    "page_subtitle": {"es": "Explora **publicaciones científicas** de la NASA usando IA avanzada.", "en": "Explore NASA's **scientific publications** using advanced AI."},
    "search_tab": {"es": "🔍 Búsqueda", "en": "🔍 Search"},
    "chatbot_tab": {"es": "💬 Chatbot", "en": "💬 Chatbot"},
    "viz_tab": {"es": "📊 Visualizaciones", "en": "📊 Visualizations"},
    "explorer_tab": {"es": "📚 Explorador", "en": "📚 Explorer"},
    "sidebar_header": {"es": "⚙️ Configuración", "en": "⚙️ Settings"},
    "language_selector": {"es": "Idioma / Language", "en": "Language / Idioma"},
    "user_mode_header": {"es": "👤 Modo de Usuario", "en": "👤 User Mode"},
    "user_mode_radio": {"es": "Selecciona tu perfil:", "en": "Select your profile:"},
    "academic_mode": {"es": "👨‍🎓 Académico", "en": "👨‍🎓 Academic"},
    "outreach_mode": {"es": "🎓 Divulgación", "en": "🎓 Outreach"},
    "academic_help": {"es": "Académico: terminología técnica / Divulgación: lenguaje simple", "en": "Academic: technical terminology / Outreach: simple language"},
    "results_slider": {"es": "Número de resultados", "en": "Number of results"},
    "ai_options_header": {"es": "🤖 Opciones de IA", "en": "🤖 AI Options"},
    "gen_summary_checkbox": {"es": "Generar resúmenes", "en": "Generate summaries"},
    "extract_entities_checkbox": {"es": "Extraer entidades", "en": "Extract entities"},
    "show_citations_checkbox": {"es": "Mostrar citaciones", "en": "Show citations"},
    "filters_header": {"es": "🔍 Filtros", "en": "🔍 Filters"},
    "year_filter": {"es": "Filtrar por año", "en": "Filter by year"},
    "stats_header": {"es": "📊 Estadísticas", "en": "📊 Statistics"},
    "total_pubs": {"es": "Total publicaciones", "en": "Total publications"},
    "search_header": {"es": "Búsqueda Inteligente", "en": "Intelligent Search"},
    "search_placeholder": {"es": "¿Qué quieres investigar?", "en": "What do you want to research?"},
    "quick_examples": {"es": "Ejemplos rápidos:", "en": "Quick examples:"},
    "found_results": {"es": "Encontrados **{count} resultados** relevantes", "en": "Found **{count} relevant** results"},
    "no_results": {"es": "⚠️ No se encontraron resultados", "en": "⚠️ No results found"},
    "avg_relevance": {"es": "Relevancia Promedio", "en": "Average Relevance"},
    "best_match": {"es": "Mejor Match", "en": "Best Match"},
    "authors": {"es": "Autores", "en": "Authors"},
    "year": {"es": "Año", "en": "Year"},
    "original_link": {"es": "Ver publicación original", "en": "View original publication"},
    "summary_header": {"es": "📝 Resumen Generado por IA", "en": "📝 AI-Generated Summary"},
    "outreach_mode_info": {"es": "💡 **Modo Divulgación**: Explicación simplificada", "en": "💡 **Outreach Mode**: Simplified explanation"},
    "generating_summary": {"es": "Generando resumen...", "en": "Generating summary..."},
    "entities_header": {"es": "🏷️ Entidades Extraídas", "en": "🏷️ Extracted Entities"},
    "extracting_entities": {"es": "Extrayendo entidades...", "en": "Extracting entities..."},
    "organism": {"es": "Organismo", "en": "Organism"},
    "condition": {"es": "Condición", "en": "Condition"},
    "methodology": {"es": "Metodología", "en": "Methodology"},
    "finding": {"es": "Hallazgo", "en": "Finding"},
    "citations_header": {"es": "📚 Citaciones", "en": "📚 Citations"},
    "copy_citation": {"es": "📋 Copiar citación", "en": "📋 Copy citation"},
    "copied_success": {"es": "✅ Copiado al portapapeles (simulado)", "en": "✅ Copied to clipboard (simulated)"},
    "abstract_expander": {"es": "📄 Ver abstract completo", "en": "📄 View full abstract"},
    "chatbot_header": {"es": "Asistente Conversacional de Biología Espacial", "en": "Conversational Space Biology Assistant"},
    "chatbot_subheader": {"es": "Pregúntame cualquier cosa sobre los {count} papers de la NASA.", "en": "Ask me anything about the {count} NASA papers."},
    "clear_chat_button": {"es": "🗑️ Limpiar", "en": "🗑️ Clear"},
    "chat_sources": {"es": "📚 Fuentes consultadas", "en": "📚 Sources consulted"},
    "chat_placeholder": {"es": "Ej: ¿Cómo afecta la microgravedad a las plantas?", "en": "Ex: How does microgravity affect plants?"},
    "searching_papers": {"es": "🔎 Buscando en {count} papers...", "en": "🔎 Searching in {count} papers..."},
    "generating_response": {"es": "💭 Generando respuesta...", "en": "💭 Generating response..."},
}

# ============================================================================
# 2. CONFIGURACIÓN Y CARGA DE DATOS
# ============================================================================

def t(key, **kwargs):
    """Función de traducción"""
    return TRANSLATIONS.get(key, {}).get(st.session_state.lang, key).format(**kwargs)

# --- Configuración de Página ---
st.set_page_config(
    page_title="NASA Space Biology Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estado de Sesión ---
if 'lang' not in st.session_state:
    st.session_state.lang = 'es'

# --- Carga de Claves y Modelos ---
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GROQ_MODEL = 'llama-3.1-70b-versatile'

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please add it to your .env file or Streamlit secrets.")
    st.stop()
groq_client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/publicaciones.csv')
        embeddings = np.load('data/corpus_embeddings.npy')
        if len(df) != len(embeddings):
            st.error("Data-embedding mismatch.")
            st.stop()
        return df, embeddings
    except FileNotFoundError:
        st.error("Data files not found. Please run the data processing scripts first.")
        st.stop()

# ============================================================================
# 3. LÓGICA DE BÚSQUEDA Y IA (MODIFICADA PARA IDIOMA)
# ============================================================================

def semantic_search(query, top_k=5):
    model = load_embedding_model()
    df, corpus_embeddings = load_data()
    query_embedding = model.encode(query, convert_to_tensor=False)
    similarities = np.dot(corpus_embeddings, query_embedding) / (np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding))
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx, score in zip(top_indices, similarities[top_indices]):
        result = df.iloc[idx].to_dict()
        result['similarity_score'] = float(score)
        results.append(result)
    return results

def generate_summary(text, title, mode, language):
    if not text or len(text.strip()) < 50:
        return "⚠️ Abstract too short or unavailable to generate summary."
    
    text = text[:2500]
    
    if language == 'es':
        if mode == "academico":
            prompt = f"Eres un experto en biociencia espacial de la NASA. Título: {title}. Abstract: {text}. Resume esta publicación en español en 3 puntos clave (Metodología, Resultados, Implicaciones). Usa terminología precisa."
        else:
            prompt = f"Eres un divulgador científico. Título: {title}. Abstract: {text}. Explica esta investigación en español para estudiantes en 3 puntos simples (¿Qué se hizo?, ¿Qué se descubrió?, ¿Por qué es importante?). Usa lenguaje fácil y analogías."
    else: # English
        if mode == "academico":
            prompt = f"You are a NASA space bioscience expert. Title: {title}. Abstract: {text}. Summarize this publication in English in 3 key points (Methodology, Main Results, Implications for Space Exploration). Use precise scientific terminology."
        else:
            prompt = f"You are a science communicator. Title: {title}. Abstract: {text}. Explain this research in English for high school students in 3 simple points (What was done?, What was discovered?, Why is it important?). Use simple language and analogies."

    try:
        response = groq_client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.4, max_tokens=600)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating summary: {e}"

def generate_chat_response(prompt, context, mode, language):
    context = context[:4000]
    lang_instruction = "Responde en español." if language == 'es' else "Respond in English."
    
    if mode == "academico":
        system_prompt = f"You are a NASA space bioscience research expert with access to 607 papers. Relevant papers for this query:\n{context}\nINSTRUCTIONS:\n- Base your answer ONLY on the provided papers.\n- Cite the paper you use: 'According to the paper...'.\n- Be conversational but scientifically accurate. {lang_instruction}"
    else:
        system_prompt = f"You are a friendly science communicator. Relevant papers:\n{context}\nINSTRUCTIONS:\n- Explain the topic in a simple and enthusiastic way.\n- Use analogies if possible.\n- Base your answer on the provided context. {lang_instruction}"
        
    try:
        response = groq_client.chat.completions.create(model=GROQ_MODEL, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], temperature=0.5, max_tokens=1000)
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error: {e}"

# (Otras funciones como extract_entities, generate_citation, etc. se mantienen igual pero se llamarán desde la UI traducida)
# ... [El resto de tus funciones auxiliares como extract_entities, generate_citation, create_year_distribution]

# ============================================================================
# 4. INTERFAZ DE USUARIO (TRADUCIDA)
# ============================================================================

def main():
    st.title(f"🚀 {t('page_title')}")
    st.markdown(t('page_subtitle'))
    
    df, embeddings = load_data()
    
    with st.sidebar:
        st.header("⚙️ Settings") # Header no se traduce para consistencia
        
        lang_choice = st.radio("Language / Idioma", ["Español 🇪🇸", "English 🇺🇸"])
        st.session_state.lang = 'es' if "Español" in lang_choice else 'en'
        
        st.subheader(t('user_mode_header'))
        user_mode_options = [t('academic_mode'), t('outreach_mode')]
        user_mode = st.radio(t('user_mode_radio'), user_mode_options, help=t('academic_help'))
        mode = "academico" if t('academic_mode') in user_mode else "divulgacion"
        
        st.divider()
        top_k = st.slider(t('results_slider'), 1, 20, 5)
        
        st.subheader(t('ai_options_header'))
        show_summary = st.checkbox(t('gen_summary_checkbox'), value=True)
        show_entities = st.checkbox(t('extract_entities_checkbox'), value=True)
        show_citation = st.checkbox(t('show_citations_checkbox'), value=True)
        
        # ... [El resto de tu sidebar, usando t() para cada texto]
        st.divider()
        st.markdown(f"""
        ### {t('stats_header')}
        - **{t('total_pubs')}**: {len(df):,}
        - **LLM Model**: Llama 3.1 70B
        - **Embeddings**: MiniLM-L6-v2
        """)

    tabs = st.tabs([t('search_tab'), t('chatbot_tab'), t('viz_tab'), t('explorer_tab')])
    
    with tabs[0]: # Pestaña de Búsqueda
        st.header(t('search_header'))
        query = st.text_input(t('search_placeholder'))

        if query:
            results = semantic_search(query, top_k=top_k)
            st.success(t('found_results', count=len(results)))
            # ... [El resto de tu lógica para mostrar resultados, usando t() para las etiquetas]
            for i, result in enumerate(results, 1):
                with st.expander(f"**{i}. {result['title']}** · Similarity: {result['similarity_score']:.1%}", expanded=(i==1)):
                    # ...
                    if show_summary:
                        st.markdown(f"### {t('summary_header')}")
                        if mode == "divulgacion":
                            st.info(t('outreach_mode_info'))
                        
                        # Se pasa el idioma a la función de la IA
                        summary = generate_summary(result.get('abstract_text', ''), result['title'], mode, st.session_state.lang)
                        st.write(summary)
                    # ...
    
    with tabs[1]: # Pestaña del Chatbot
        st.header(t('chatbot_header'))
        st.markdown(t('chatbot_subheader', count=len(df)))
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if st.button(t('clear_chat_button')):
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(t('chat_placeholder')):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(t('searching_papers', count=len(df))):
                    results = semantic_search(prompt, top_k=3)
                    context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract_text', '')[:500]}..." for r in results])
                
                with st.spinner(t('generating_response')):
                    # Se pasa el idioma a la función de la IA
                    response = generate_chat_response(prompt, context, mode, st.session_state.lang)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

    # ... [El resto de tus pestañas, usando t() para todos los textos]

if __name__ == "__main__":
    main()