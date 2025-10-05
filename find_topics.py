import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os

def find_top_words(csv_path, top_n=20):
    """
    Lee un CSV con publicaciones, limpia el texto y encuentra las
    palabras o temas más comunes.
    """
    print("="*60)
    print("🔎 ANALIZADOR DE TEMAS Y PALABRAS CLAVE")
    print("="*60 + "\n")

    # --- 1. Cargar los datos ---
    print(f"📂 Cargando publicaciones desde '{csv_path}'...")
    try:
        df = pd.read_csv(csv_path)
        # Manejar posibles valores nulos en el texto
        df['title'] = df['title'].fillna('')
        df['abstract_text'] = df['abstract_text'].fillna('')
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{csv_path}'.")
        return

    # --- 2. Preparar el corpus de texto ---
    # Unimos el título y el resumen para un análisis más completo
    corpus = df['title'] + ' ' + df['abstract_text']
    print(f"✅ {len(corpus)} documentos listos para analizar.\n")

    # --- 3. Limpiar el texto y contar palabras ---
    print("🧹 Limpiando texto y contando palabras clave...")

    # Lista de "stopwords": palabras comunes a ignorar.
    # Añadimos palabras comunes en ciencia que no aportan significado de "tema".
    stop_words_custom = list(CountVectorizer(stop_words='english').get_stop_words())
    stop_words_custom.extend([
        'study', 'results', 'methods', 'conclusions', 'introduction',
        'background', 'purpose', 'discussion', 'significance',
        'figure', 'table', 'data', 'analysis', 'group', 'groups', 'using',
        'shown', 'found', 'used', 'may', 'however', 'also', 'et', 'al'
    ])

    # Usamos CountVectorizer para tokenizar, limpiar y contar todo en un paso
    vectorizer = CountVectorizer(
        stop_words=stop_words_custom,
        max_df=0.85,  # Ignorar palabras que aparecen en más del 85% de los docs
        min_df=3,     # Ignorar palabras que aparecen en menos de 3 docs
        ngram_range=(1, 2) # Analizar palabras individuales y pares de palabras (bigramas)
    )

    word_count_matrix = vectorizer.fit_transform(corpus)
    
    # Sumar las ocurrencias de cada palabra
    word_counts = pd.DataFrame({
        'word': vectorizer.get_feature_names_out(),
        'count': word_count_matrix.sum(axis=0).getA1()
    }).sort_values('count', ascending=False)

    # --- 4. Mostrar los resultados ---
    print("\n" + "="*60)
    print(f"🏆 TOP {top_n} TEMAS Y PALABRAS MÁS REPETIDAS")
    print("="*60)
    print(word_counts.head(top_n).to_string(index=False))

if __name__ == "__main__":
    # Asegúrate de que este sea el nombre de tu archivo CSV final y completo
    final_csv_file = 'data/publicaciones.csv' 
    
    if os.path.exists(final_csv_file):
        find_top_words(final_csv_file, top_n=25)
    else:
        print(f"❌ Error: El archivo '{final_csv_file}' no existe.")
        print("   Por favor, actualiza el nombre del archivo en el script si es necesario.")