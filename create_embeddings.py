"""
Script para generar embeddings de las publicaciones de la NASA
EJECUTAR UNA SOLA VEZ antes de lanzar la aplicación

Uso: python create_embeddings.py
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def create_embeddings():
    print("\n" + "="*60)
    print("🚀 GENERADOR DE EMBEDDINGS - NASA SPACE BIOLOGY")
    print("="*60 + "\n")
    
    # 1. Verificar que existe la carpeta data
    if not os.path.exists('data'):
        print("❌ Error: No existe la carpeta 'data'")
        print("   Crea la carpeta: mkdir data")
        return
    
    # 2. Cargar datos
    print("📂 Paso 1/4: Cargando publicaciones...")
    try:
        df = pd.read_csv('data/publicaciones.csv')
        print(f"   ✅ {len(df)} publicaciones cargadas")
    except FileNotFoundError:
        print("   ❌ Error: No se encontró data/publicaciones.csv")
        print("   Asegúrate de tener el archivo con las columnas:")
        print("   - id, title, authors, year, abstract_text, source_url")
        return
    except Exception as e:
        print(f"   ❌ Error al cargar datos: {str(e)}")
        return
    
    # Verificar columnas requeridas
    required_cols = ['title', 'abstract_text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"   ❌ Faltan columnas requeridas: {missing_cols}")
        return
    
    # 3. Cargar modelo de embeddings
    print("\n🤖 Paso 2/4: Cargando modelo de embeddings...")
    print("   (Esto puede tardar en la primera ejecución)")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"   ❌ Error al cargar modelo: {str(e)}")
        return
    
    # 4. Preparar textos
    print("\n📝 Paso 3/4: Preparando textos...")
    texts = []
    for idx, row in df.iterrows():
        # Combinar título y abstract para mejor contexto
        title = str(row['title']) if pd.notna(row['title']) else ""
        abstract = str(row['abstract_text']) if pd.notna(row['abstract_text']) else ""
        
        # Si no hay abstract, usar solo título
        if abstract:
            text = f"{title}. {abstract}"
        else:
            text = title
        
        texts.append(text)
    
    print(f"   ✅ {len(texts)} textos preparados")
    
    # 5. Generar embeddings
    print("\n⚡ Paso 4/4: Generando embeddings...")
    print("   (Esto puede tardar varios minutos dependiendo del tamaño)")
    
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True  # Normalizar para similitud coseno
        )
        print(f"   ✅ Embeddings generados")
    except Exception as e:
        print(f"   ❌ Error al generar embeddings: {str(e)}")
        return
    
    # 6. Guardar embeddings
    print("\n💾 Guardando embeddings...")
    try:
        output_path = 'data/corpus_embeddings.npy'
        np.save(output_path, embeddings)
        
        # Verificar que se guardó correctamente
        test_load = np.load(output_path)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"   ✅ Embeddings guardados en: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Error al guardar: {str(e)}")
        return
    
    # 7. Resumen final
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*60)
    print(f"""
📊 ESTADÍSTICAS FINALES:
   
   • Total de documentos:        {len(embeddings):,}
   • Dimensión de embeddings:    {embeddings.shape[1]}D
   • Tamaño del archivo:         {file_size_mb:.2f} MB
   • Archivo generado:           {output_path}
   
🚀 SIGUIENTE PASO:
   
   Ejecuta la aplicación con:
   streamlit run app.py
   
💡 NOTA: No necesitas ejecutar este script de nuevo a menos que:
   - Agregues nuevas publicaciones
   - Cambies el modelo de embeddings
   - Los datos originales cambien
    """)

if __name__ == "__main__":
    create_embeddings()