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
# CONFIGURACIÓN INICIAL
# ============================================================================

load_dotenv()

st.set_page_config(
    page_title="NASA Space Biology Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
GROQ_MODEL = 'llama-3.3-70b-versatile'

# Configurar Groq
if not GROQ_API_KEY:
    st.error("⚠️ No se encontró GROQ_API_KEY en el archivo .env")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================================
# FUNCIONES DE CACHÉ
# ============================================================================

@st.cache_resource
def load_embedding_model():
    with st.spinner("🤖 Cargando modelo de embeddings..."):
        return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/publicaciones.csv')
        embeddings = np.load('data/corpus_embeddings.npy')
        
        if len(df) != len(embeddings):
            st.error("❌ Error: El número de publicaciones no coincide con los embeddings")
            st.stop()
        
        return df, embeddings
    except FileNotFoundError as e:
        st.error(f"❌ Error: No se encontraron los archivos de datos. {str(e)}")
        st.info("""
        **Pasos para resolver:**
        1. Asegúrate de tener el archivo `data/publicaciones.csv`
        2. Ejecuta primero `python process_pdfs.py` o `python quick_download.py`
        """)
        st.stop()

# ============================================================================
# FUNCIONES DE BÚSQUEDA
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
# FUNCIONES DE IA CON GROQ
# ============================================================================

def generate_summary(text, title, mode="academico"):
    """Generar resumen usando Groq/Llama"""
    
    if not text or len(text.strip()) < 50:
        return "⚠️ Abstract muy corto o no disponible para generar resumen."
    
    # Truncar texto
    max_chars = 2500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    if mode == "academico":
        prompt = f"""Eres un experto en biociencia espacial de la NASA.

Título: {title}

Abstract: {text}

Resume esta publicación científica en 3 puntos clave:
1. Metodología y diseño experimental
2. Resultados principales con datos específicos
3. Implicaciones para la exploración espacial

Usa terminología científica precisa. Cada punto: 2-3 oraciones."""
    else:
        prompt = f"""Eres un divulgador científico especializado en espacio.

Título: {title}

Abstract: {text}

Explica esta investigación espacial para estudiantes de secundaria en 3 puntos simples:
1. ¿Qué experimento se hizo? (como si se lo explicaras a un amigo)
2. ¿Qué descubrieron? (con ejemplos cotidianos)
3. ¿Por qué es importante para viajar al espacio?

Usa lenguaje simple, analogías y evita jerga técnica."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error al generar resumen: {str(e)}"

def extract_entities(text, title):
    """Extraer entidades usando Groq/Llama"""
    
    max_chars = 1500
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""Analiza este texto científico sobre biología espacial.

Título: {title}

Texto: {text}

Extrae en formato JSON:
- "organism": Organismo estudiado
- "condition": Condición espacial (microgravedad, radiación, etc.)
- "key_finding": Hallazgo principal (máximo 15 palabras)
- "methodology": Método usado

Si falta info, usa "No especificado".
Responde SOLO el JSON sin markdown."""

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
            "key_finding": "No disponible",
            "methodology": "N/A"
        }

def generate_chat_response(prompt, context, mode="academico"):
    """Generar respuesta de chat usando Groq/Llama"""
    
    max_context = 4000
    if len(context) > max_context:
        context = context[:max_context] + "\n...(más papers disponibles)"
    
    if mode == "academico":
        system_prompt = f"""Eres un experto investigador en biología espacial de la NASA con acceso a 607 papers científicos.

Papers más relevantes para esta consulta:
{context}

INSTRUCCIONES:
- Estos son solo 3 ejemplos de los 607 papers disponibles
- Cita específicamente qué paper usas: "Según el Paper 1..."
- Si los papers no son relevantes, sugiere reformular la pregunta
- Usa terminología científica precisa
- Sé conversacional pero exacto"""
    else:
        system_prompt = f"""Eres un divulgador científico especializado en espacio con acceso a 607 papers de la NASA.

Papers más relevantes:
{context}

INSTRUCCIONES:
- Estos son 3 ejemplos de los 607 papers disponibles
- Responde de forma amigable y clara
- Usa analogías cuando sea posible
- Sé entusiasta y educativo"""
    
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
    """Genera citación en diferentes formatos"""
    title = result.get('title', 'Sin título')
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
# VISUALIZACIONES
# ============================================================================

def create_year_distribution(df):
    """Gráfica de distribución por años"""
    year_counts = df['year'].value_counts().sort_index()
    
    fig = px.bar(
        x=year_counts.index,
        y=year_counts.values,
        title="📅 Distribución de Publicaciones por Año",
        labels={'x': 'Año', 'y': 'Número de Publicaciones'},
        color=year_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=350)
    return fig

# ============================================================================
# GLOSARIO
# ============================================================================

GLOSARIO = {
    "Microgravedad": "Condición de gravedad muy reducida, casi cero, como la que experimentan los astronautas en órbita.",
    "Transcriptómica": "Estudio de todos los genes que se activan en una célula en un momento dado.",
    "ISS": "International Space Station - Estación Espacial Internacional.",
    "Arabidopsis": "Planta pequeña usada comúnmente en investigación científica.",
    "C. elegans": "Gusano microscópico usado en estudios biológicos.",
    "Expresión génica": "Proceso por el cual los genes producen proteínas.",
    "Radiación cósmica": "Partículas de alta energía del espacio que pueden dañar células.",
    "EMCS": "European Modular Cultivation System - sistema de cultivo en la ISS."
}

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    st.title("🚀 Motor de Conocimiento de Biología Espacial")
    st.markdown("""
    Explora **publicaciones científicas** de la NASA usando IA avanzada.  
    Búsqueda semántica · Resúmenes automáticos · Extracción de entidades
    """)
    
    df, embeddings = load_data()
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.subheader("👤 Modo de Usuario")
        user_mode = st.radio(
            "Selecciona tu perfil:",
            ["👨‍🎓 Académico", "🎓 Divulgación"],
            help="Académico: terminología técnica / Divulgación: lenguaje simple"
        )
        
        mode = "academico" if "Académico" in user_mode else "divulgacion"
        
        st.divider()
        
        top_k = st.slider("Número de resultados", min_value=1, max_value=20, value=5)
        
        st.subheader("🤖 Opciones de IA")
        show_summary = st.checkbox("Generar resúmenes", value=True)
        show_entities = st.checkbox("Extraer entidades", value=True)
        show_citation = st.checkbox("Mostrar citaciones", value=True)
        
        st.divider()
        
        st.subheader("🔍 Filtros")
        years = df['year'].dropna().unique()
        if len(years) > 0:
            year_filter = st.multiselect("Filtrar por año", options=sorted(years, reverse=True), default=[])
        else:
            year_filter = []
        
        st.divider()
        
        st.markdown(f"""
        ### 📊 Estadísticas
        - **Total publicaciones**: {len(df):,}
        - **Modelo LLM**: Llama 3.3 70B
        - **Embeddings**: MiniLM-L6-v2
        - **Dimensión**: {embeddings.shape[1]}D
        """)
        
        with st.expander("📖 Glosario Científico"):
            for term, definition in GLOSARIO.items():
                st.markdown(f"**{term}**: {definition}")
        
        st.divider()
        st.markdown("Desarrollado para el **NASA Space Biology Challenge** Guadalajara 2025 🇲🇽")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Búsqueda", "💬 Chatbot", "📊 Visualizaciones", "📚 Explorador"])
    
    with tab1:
        st.header("🔍 Búsqueda Inteligente")
        
        query = st.text_input("¿Qué quieres investigar?", placeholder="Ej: efectos de la microgravedad en plantas")
        
        st.markdown("**Ejemplos rápidos:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💡 Radiación y ADN", use_container_width=True):
                query = "efectos de la radiación espacial en el ADN"
        with col2:
            if st.button("🌱 Plantas en espacio", use_container_width=True):
                query = "experimentos con plantas en microgravedad"
        with col3:
            if st.button("🧬 Sistema inmune", use_container_width=True):
                query = "cambios en el sistema inmune de astronautas"
        with col4:
            if st.button("🔬 C. elegans", use_container_width=True):
                query = "estudios con C elegans en el espacio"
        
        if query and len(query.strip()) > 0:
            with st.spinner("🔎 Buscando..."):
                results = semantic_search(query, top_k=top_k)
            
            if year_filter:
                results = [r for r in results if r.get('year') in year_filter]
            
            if not results:
                st.warning("⚠️ No se encontraron resultados")
                return
            
            st.success(f"✅ Encontrados **{len(results)} resultados** relevantes")
            
            avg_score = sum(r['similarity_score'] for r in results) / len(results)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Resultados", len(results))
            with col2:
                st.metric("🎯 Relevancia Promedio", f"{avg_score:.1%}")
            with col3:
                st.metric("🥇 Mejor Match", f"{results[0]['similarity_score']:.1%}")
            
            st.divider()
            
            for i, result in enumerate(results, 1):
                with st.expander(f"**{i}. {result['title']}** · Similitud: {result['similarity_score']:.1%}", expanded=(i == 1)):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**✍️ Autores:** {result.get('authors', 'N/A')}")
                    with col2:
                        st.metric("📅 Año", result.get('year', 'N/A'))
                    with col3:
                        st.metric("🎯 Match", f"{result['similarity_score']:.1%}")
                    
                    if result.get('source_url'):
                        st.markdown(f"🔗 [Ver publicación original]({result['source_url']})")
                    
                    st.divider()
                    
                    if show_summary:
                        st.markdown("### 📝 Resumen Generado por IA")
                        if mode == "divulgacion":
                            st.info("💡 **Modo Divulgación**: Explicación simplificada")
                        
                        with st.spinner("Generando resumen..."):
                            summary = generate_summary(result.get('abstract_text', ''), result['title'], mode)
                        st.write(summary)
                        st.divider()
                    
                    if show_entities:
                        st.markdown("### 🏷️ Entidades Extraídas")
                        with st.spinner("Extrayendo entidades..."):
                            entities = extract_entities(result.get('abstract_text', ''), result['title'])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🔬 Organismo", entities.get('organism', 'N/A'))
                            st.metric("🌌 Condición", entities.get('condition', 'N/A'))
                        with col2:
                            st.metric("🔬 Metodología", entities.get('methodology', 'N/A'))
                            st.markdown(f"**💡 Hallazgo:** {entities.get('key_finding', 'N/A')}")
                        
                        st.divider()
                    
                    if show_citation:
                        st.markdown("### 📚 Citaciones")
                        citation_format = st.radio("Formato:", ["APA 7", "BibTeX", "Texto plano"], key=f"cite_{i}", horizontal=True)
                        fmt_map = {"APA 7": "apa7", "BibTeX": "bibtex", "Texto plano": "plain"}
                        citation = generate_citation(result, fmt_map[citation_format])
                        st.code(citation, language="text")
                        if st.button("📋 Copiar citación", key=f"copy_{i}"):
                            st.success("✅ Copiado al portapapeles (simulado)")
                        st.divider()
                    
                    with st.expander("📄 Ver abstract completo"):
                        st.write(result.get('abstract_text', 'No disponible'))
        else:
            st.info("👆 **Escribe una consulta arriba** o usa los botones de ejemplo.")
    
    with tab2:
        st.header("💬 Asistente Conversacional de Biología Espacial")
        st.markdown(f"Pregúntame cualquier cosa sobre los {len(df)} papers de la NASA. **Modo actual:** {user_mode}")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        col1, col2 = st.columns([6, 1])
        with col2:
            if st.button("🗑️ Limpiar"):
                st.session_state.messages = []
                st.rerun()
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 Fuentes consultadas"):
                        for source in message["sources"]:
                            st.markdown(f"**• {source['title']}** ({source['year']})")
                            st.caption(f"Autores: {source['authors']}")
        
        if prompt := st.chat_input("Ej: ¿Cómo afecta la microgravedad a las plantas?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner(f"🔎 Buscando en {len(df)} papers..."):
                results = semantic_search(prompt, top_k=3)
            
            context = ""
            sources = []
            for i, r in enumerate(results, 1):
                context += f"\n[Paper {i}]\nTítulo: {r['title']}\nAutores: {r.get('authors', 'N/A')}\nAño: {r.get('year', 'N/A')}\n"
                abstract = r.get('abstract_text', '')
                if abstract and len(abstract) > 100:
                    context += f"Resumen: {abstract[:800]}...\n\n"
                sources.append({'title': r['title'], 'authors': r.get('authors', 'N/A'), 'year': r.get('year', 'N/A')})
            
            with st.spinner("💭 Generando respuesta..."):
                assistant_response = generate_chat_response(prompt, context, mode)
            
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                if sources:
                    with st.expander("📚 Fuentes consultadas"):
                        for source in sources:
                            st.markdown(f"**• {source['title']}** ({source['year']})")
                            st.caption(f"Autores: {source['authors']}")
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_response, "sources": sources})
    
    with tab3:
        st.header("📊 Análisis Visual del Corpus")
        col1, col2 = st.columns(2)
        with col1:
            if 'year' in df.columns:
                fig_years = create_year_distribution(df)
                st.plotly_chart(fig_years, use_container_width=True)
        with col2:
            st.metric("📚 Total de Publicaciones", len(df))
            if 'year' in df.columns:
                years_range = df['year'].dropna()
                if len(years_range) > 0:
                    st.metric("📅 Rango de Años", f"{int(years_range.min())} - {int(years_range.max())}")
        st.divider()
        st.info("💡 **Nota**: Las visualizaciones de organismos y condiciones requieren procesar todos los papers con IA.")
    
    with tab4:
        st.header("📚 Explorador de Publicaciones")
        st.markdown("Navega por todas las publicaciones disponibles:")
        display_cols = ['title', 'authors', 'year']
        available_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[available_cols].head(50), use_container_width=True, height=400)
        st.info(f"Mostrando las primeras 50 de {len(df)} publicaciones")

if __name__ == "__main__":
    main()