"""
Script para procesar PDFs de la NASA desde Google Drive
Extrae texto, genera embeddings y los guarda para la app
"""

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import requests
from tqdm import tqdm
import time

# ============================================================================
# OPCIÓN 1: PDFs en Google Drive (carpeta pública)
# ============================================================================

def download_pdf_from_drive(file_id, output_path):
    """
    Descarga un PDF desde Google Drive usando su file_id
    
    Args:
        file_id: ID del archivo en Google Drive
        output_path: Ruta donde guardar el PDF
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Error descargando {file_id}: {str(e)}")
        return False

def extract_text_from_pdf(pdf_path):
    """
    Extrae texto de un PDF
    
    Args:
        pdf_path: Ruta al archivo PDF
        
    Returns:
        String con el texto extraído
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extraer texto de todas las páginas
            for page in reader.pages:
                text += page.extract_text() + " "
            
            # Limpieza básica
            text = text.replace('\n', ' ').replace('\r', ' ')
            text = ' '.join(text.split())  # Eliminar espacios múltiples
            
            return text
    except Exception as e:
        print(f"Error procesando {pdf_path}: {str(e)}")
        return ""

def process_pdfs_from_drive_list(drive_links_file):
    """
    Procesa una lista de links de Google Drive
    
    Args:
        drive_links_file: CSV con columnas: title, drive_link, year, authors
    """
    print("="*60)
    print("🚀 PROCESADOR DE PDFs - NASA SPACE BIOLOGY")
    print("="*60 + "\n")
    
    # Crear carpeta temporal para PDFs
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Leer lista de papers
    print("📂 Cargando lista de publicaciones...")
    df = pd.read_csv(drive_links_file, engine='python')
    print(f"✅ {len(df)} publicaciones encontradas\n")
    
    # Preparar lista de datos
    publications = []
    
    print("📥 Descargando y procesando PDFs...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        
        # Extraer file_id del link de Drive
        drive_link = row['drive_link']
        
        # Formatos de link de Drive:
        # https://drive.google.com/file/d/FILE_ID/view
        # https://drive.google.com/open?id=FILE_ID
        if '/d/' in drive_link:
            file_id = drive_link.split('/d/')[1].split('/')[0]
        elif 'id=' in drive_link:
            file_id = drive_link.split('id=')[1].split('&')[0]
        else:
            print(f"⚠️ Link inválido: {drive_link}")
            continue
        
        # Descargar PDF temporalmente
        pdf_path = os.path.join(temp_dir, f"paper_{idx}.pdf")
        
        if download_pdf_from_drive(file_id, pdf_path):
            # Extraer texto
            text = extract_text_from_pdf(pdf_path)
            
            if text:
                publications.append({
                    'id': idx,
                    'title': row.get('title', f"Paper {idx}"),
                    'authors': row.get('authors', 'N/A'),
                    'year': row.get('year', 'N/A'),
                    'abstract_text': text[:5000],  # Primeros 5000 caracteres
                    'full_text': text,
                    'source_url': drive_link
                })
            
            # Eliminar PDF temporal
            os.remove(pdf_path)
        
        # Pausa para no saturar Drive
        time.sleep(0.5)
    
    # Guardar datos procesados
    print("\n💾 Guardando datos procesados...")
    df_processed = pd.DataFrame(publications)
    df_processed.to_csv('data/publicaciones.csv', index=False)
    print(f"✅ Guardado: data/publicaciones.csv ({len(df_processed)} papers)")
    
    # Generar embeddings
    print("\n⚡ Generando embeddings...")
    generate_embeddings(df_processed)
    
    # Limpiar carpeta temporal
    os.rmdir(temp_dir)
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)

# ============================================================================
# OPCIÓN 2: PDFs en carpeta local
# ============================================================================

def process_pdfs_from_folder(folder_path):
    """
    Procesa PDFs desde una carpeta local
    
    Args:
        folder_path: Ruta a la carpeta con los PDFs
    """
    print("="*60)
    print("🚀 PROCESADOR DE PDFs - CARPETA LOCAL")
    print("="*60 + "\n")
    
    # Listar PDFs
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    print(f"📂 Encontrados {len(pdf_files)} PDFs\n")
    
    publications = []
    
    print("📖 Procesando PDFs...")
    for idx, pdf_file in enumerate(tqdm(pdf_files)):
        pdf_path = os.path.join(folder_path, pdf_file)
        
        # Extraer texto
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            # Extraer título del nombre del archivo
            title = pdf_file.replace('.pdf', '').replace('_', ' ')
            
            publications.append({
                'id': idx,
                'title': title,
                'authors': 'N/A',
                'year': 'N/A',
                'abstract_text': text[:5000],
                'full_text': text,
                'source_url': ''
            })
    
    # Guardar datos
    print("\n💾 Guardando datos procesados...")
    df = pd.DataFrame(publications)
    df.to_csv('data/publicaciones.csv', index=False)
    print(f"✅ Guardado: data/publicaciones.csv ({len(df)} papers)")
    
    # Generar embeddings
    print("\n⚡ Generando embeddings...")
    generate_embeddings(df)
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)

# ============================================================================
# OPCIÓN 3: API de NASA (si existe endpoint)
# ============================================================================

def fetch_from_nasa_api(api_endpoint):
    """
    Descarga papers desde API de NASA
    """
    print("🌐 Conectando con API de NASA...")
    
    try:
        response = requests.get(api_endpoint)
        response.raise_for_status()
        data = response.json()
        
        publications = []
        for item in data:
            publications.append({
                'id': item.get('id'),
                'title': item.get('title'),
                'authors': item.get('authors'),
                'year': item.get('year'),
                'abstract_text': item.get('abstract', ''),
                'full_text': item.get('full_text', ''),
                'source_url': item.get('url', '')
            })
        
        df = pd.DataFrame(publications)
        df.to_csv('data/publicaciones.csv', index=False)
        
        print(f"✅ {len(df)} publicaciones descargadas")
        generate_embeddings(df)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# ============================================================================
# GENERAR EMBEDDINGS
# ============================================================================

def generate_embeddings(df):
    """
    Genera embeddings del dataframe
    """
    print("🤖 Cargando modelo de embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Preparar textos
    texts = []
    for _, row in df.iterrows():
        text = f"{row['title']}. {row['abstract_text']}"
        texts.append(text)
    
    # Generar embeddings
    print("⚡ Generando embeddings (puede tardar varios minutos)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    
    # Guardar
    np.save('data/corpus_embeddings.npy', embeddings)
    print(f"✅ Embeddings guardados: {embeddings.shape}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 NASA SPACE BIOLOGY - PROCESADOR DE PAPERS\n")
    print("Selecciona una opción:")
    print("1. Procesar desde lista de Google Drive (CSV con links)")
    print("2. Procesar desde carpeta local con PDFs")
    print("3. Descargar desde API de NASA\n")
    
    opcion = input("Opción (1/2/3): ").strip()
    
    if opcion == "1":
        csv_file = input("Ruta al CSV con links de Drive: ").strip()
        process_pdfs_from_drive_list(csv_file)
        
    elif opcion == "2":
        folder = input("Ruta a la carpeta con PDFs: ").strip()
        process_pdfs_from_folder(folder)
        
    elif opcion == "3":
        api_url = input("URL de la API: ").strip()
        fetch_from_nasa_api(api_url)
        
    else:
        print("❌ Opción inválida")