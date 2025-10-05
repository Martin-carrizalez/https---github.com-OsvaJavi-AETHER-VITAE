import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv
import json

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="NASA Space Biology Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MODEL_NAME = 'llama-3.3-70b-versatile'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# ============================================================================
# FUNCIONES DE CACHÉ - Se ejecutan una sola vez
# ============================================================================

@st.cache_resource
def get_groq_client():
    """Inicializa el cliente de Groq"""
    if not GROQ_API_KEY:
        st.error("⚠️ No se encontró GROQ_API_KEY en el archivo .env")
        st.stop()
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_embedding_model():
    """Carga el modelo de embeddings (se ejecuta una sola vez)"""
    with st.spinner("🤖 Cargando modelo de embeddings..."):
        return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def load_data():
    """Carga los datos y embeddings (se ejecuta una sola vez)"""
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
        2. Ejecuta primero `python create_embeddings.py` para generar los embeddings
        """)
        st.stop()

# ============================================================================
# FUNCIONES DE BÚSQUEDA
# ============================================================================

def semantic_search(query, top_k=5):
    """
    Realiza búsqueda semántica sobre el corpus de publicaciones
    
    Args:
        query: Consulta del usuario en lenguaje natural
        top_k: Número de resultados a retornar
        
    Returns:
        Lista de diccionarios con los resultados y sus scores
    """
    model = load_embedding_model()
    df, corpus_embeddings = load_data()
    
    # Convertir query a embedding
    query_embedding = model.encode(query, convert_to_tensor=False)
    
    # Calcular similitud coseno
    similarities = np.dot(corpus_embeddings, query_embedding) / (
        np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Obtener top_k resultados
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    
    # Preparar resultados
    results = []
    for idx, score in zip(top_indices, top_scores):
        result = df.iloc[idx].to_dict()
        result['similarity_score'] = float(score)
        results.append(result)
    
    return results

# ============================================================================
# FUNCIONES DE IA CON GROQ
# ============================================================================

def generate_summary(text, title):
    """
    Genera un resumen conciso de 3 puntos usando Groq/Llama
    
    Args:
        text: Texto del abstract
        title: Título de la publicación
        
    Returns:
        String con el resumen generado
    """
    client = get_groq_client()
    
    prompt = f"""Eres un experto en biociencia espacial de la NASA. 

Título: {title}

Texto: {text}

Resume esta publicación científica en 3 puntos clave concisos:
1. El experimento u observación realizada
2. Los resultados principales encontrados
3. La implicación para la exploración espacial humana

Usa lenguaje claro y accesible. Cada punto debe tener máximo 2 oraciones.
No uses encabezados como "1.", "2.", etc. Solo escribe los puntos separados por saltos de línea."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error al generar resumen: {str(e)}"

def extract_entities(text, title):
    """
    Extrae entidades clave del texto en formato JSON
    
    Args:
        text: Texto del abstract
        title: Título de la publicación
        
    Returns:
        Diccionario con las entidades extraídas
    """
    client = get_groq_client()
    
    prompt = f"""Analiza el siguiente texto científico sobre biología espacial.

Título: {title}

Texto: {text}

Extrae las siguientes entidades en formato JSON:
- "organism": El organismo principal estudiado (ej: plantas, ratones, C. elegans, humanos)
- "condition": La condición espacial estudiada (ej: microgravedad, radiación, vuelo espacial)
- "key_finding": El hallazgo más importante en una frase corta (máximo 15 palabras)

Si alguna entidad no está presente, usa "No especificado".

Responde ÚNICAMENTE con el JSON, sin markdown, sin código, solo el JSON puro.
Ejemplo: {{"organism": "plantas", "condition": "microgravedad", "key_finding": "..."}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        content = response.choices[0].message.content.strip()
        # Limpiar posibles markdown
        content = content.replace('```json', '').replace('```', '').strip()
        
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "organism": "Error en formato",
            "condition": "Error en formato",
            "key_finding": "No se pudo extraer información"
        }
    except Exception as e:
        return {
            "organism": "Error",
            "condition": "Error",
            "key_finding": str(e)
        }

# ============================================================================
# INTERFACE PRINCIPAL
# ============================================================================

def main():
    # Header
    st.title("🚀 Motor de Conocimiento de Biología Espacial")
    st.markdown("""
    Explora **608 publicaciones científicas** de la NASA usando IA avanzada.  
    Búsqueda semántica · Resúmenes automáticos · Extracción de entidades
    """)
    
    # Cargar datos una vez
    df, embeddings = load_data()
    
    # ========================================================================
    # SIDEBAR - Configuración
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        top_k = st.slider(
            "Número de resultados",
            min_value=1,
            max_value=20,
            value=5,
            help="Cuántas publicaciones mostrar"
        )
        
        st.subheader("🤖 Opciones de IA")
        show_summary = st.checkbox("Generar resúmenes", value=True)
        show_entities = st.checkbox("Extraer entidades", value=True)
        
        st.divider()
        
        st.markdown(f"""
        ### 📊 Estadísticas
        - **Total publicaciones**: {len(df):,}
        - **Modelo LLM**: llama-3.3-70b-versatile
        - **Embeddings**: MiniLM-L6-v2
        - **Dimensión**: {embeddings.shape[1]}D
        """)
        
        st.divider()
        
        st.markdown("""
        ### 💡 Consejos
        - Usa preguntas naturales
        - Sé específico en tu búsqueda
        - Prueba diferentes términos
        """)
        
        st.markdown("""
        ---
        Desarrollado para el **NASA Space Biology Challenge**  
        Guadalajara 2025 🇲🇽
        """)
    
    # ========================================================================
    # ÁREA PRINCIPAL - Búsqueda
    # ========================================================================
    
    st.header("🔍 Búsqueda Inteligente")
    
    # Input de búsqueda
    query = st.text_input(
        "¿Qué quieres investigar?",
        placeholder="Ej: efectos de la microgravedad en plantas",
        help="Escribe tu consulta en lenguaje natural",
        key="search_query"
    )
    
    # Botones de ejemplo
    st.markdown("**Ejemplos rápidos:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💡 Radiación y ADN", use_container_width=True):
            query = "efectos de la radiación espacial en el ADN"
            st.rerun()
    
    with col2:
        if st.button("🌱 Plantas en espacio", use_container_width=True):
            query = "experimentos con plantas en microgravedad"
            st.rerun()
    
    with col3:
        if st.button("🧬 Sistema inmune", use_container_width=True):
            query = "cambios en el sistema inmune de astronautas"
            st.rerun()
    
    with col4:
        if st.button("🔬 C. elegans", use_container_width=True):
            query = "estudios con C elegans en el espacio"
            st.rerun()
    
    # ========================================================================
    # PROCESAMIENTO Y RESULTADOS
    # ========================================================================
    
    if query and len(query.strip()) > 0:
        
        # Realizar búsqueda
        with st.spinner("🔎 Buscando en 608 publicaciones..."):
            results = semantic_search(query, top_k=top_k)
        
        if not results:
            st.warning("⚠️ No se encontraron resultados para tu búsqueda")
            return
        
        st.success(f"✅ Encontrados **{len(results)} resultados** relevantes")
        
        # Mostrar métricas generales
        avg_score = sum(r['similarity_score'] for r in results) / len(results)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Resultados", len(results))
        with col2:
            st.metric("🎯 Relevancia Promedio", f"{avg_score:.1%}")
        with col3:
            st.metric("🥇 Mejor Match", f"{results[0]['similarity_score']:.1%}")
        
        st.divider()
        
        # Mostrar cada resultado
        for i, result in enumerate(results, 1):
            
            with st.expander(
                f"**{i}. {result['title']}** · Similitud: {result['similarity_score']:.1%}",
                expanded=(i == 1)  # Primer resultado expandido
            ):
                
                # Metadata básica
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**✍️ Autores:** {result.get('authors', 'N/A')}")
                with col2:
                    st.metric("📅 Año", result.get('year', 'N/A'))
                with col3:
                    st.metric("🎯 Match", f"{result['similarity_score']:.1%}")
                
                # Link a fuente original (si existe)
                if result.get('source_url') and result['source_url'] != '':
                    st.markdown(f"🔗 [Ver publicación original]({result['source_url']})")
                
                st.divider()
                
                # Resumen con IA
                if show_summary:
                    st.markdown("### 📝 Resumen Generado por IA")
                    with st.spinner("Generando resumen..."):
                        summary = generate_summary(
                            result.get('abstract_text', ''),
                            result['title']
                        )
                    st.info(summary)
                    st.divider()
                
                # Extracción de entidades
                if show_entities:
                    st.markdown("### 🏷️ Entidades Extraídas")
                    with st.spinner("Extrayendo entidades clave..."):
                        entities = extract_entities(
                            result.get('abstract_text', ''),
                            result['title']
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🔬 Organismo", entities.get('organism', 'N/A'))
                        st.metric("🌌 Condición Espacial", entities.get('condition', 'N/A'))
                    with col2:
                        st.markdown("**💡 Hallazgo Principal:**")
                        st.markdown(f"> {entities.get('key_finding', 'N/A')}")
                    
                    st.divider()
                
                # Abstract completo (colapsado)
                with st.expander("📄 Ver abstract completo"):
                    st.write(result.get('abstract_text', 'No disponible'))
    
    else:
        # Estado inicial - mostrar instrucciones
        st.info("""
        👆 **Escribe una consulta arriba** o usa los botones de ejemplo para comenzar.
        
        **Ejemplos de consultas:**
        - "¿Cómo afecta la microgravedad al crecimiento de plantas?"
        - "Efectos de la radiación cósmica en células humanas"
        - "Adaptaciones del sistema cardiovascular en el espacio"
        - "Experimentos con bacterias en la ISS"
        """)

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()