"""
Script para enriquecer el CSV de la NASA con metadatos reales
Extrae autores, año y abstract de las páginas de NCBI
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import re
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_metadata_from_ncbi(url):
    """
    Extrae metadatos completos de una página de NCBI/PMC
    
    Returns:
        dict con title, authors, year, abstract_text
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # ===== EXTRAER AUTORES =====
        authors_list = []
        
        # Método 1: Buscar en div.authors-list
        authors_div = soup.find('div', class_='authors-list')
        if authors_div:
            for author in authors_div.find_all('a', class_='full-name'):
                authors_list.append(author.text.strip())
        
        # Método 2: Buscar en div.contrib-group (alternativo)
        if not authors_list:
            contrib_group = soup.find('div', class_='contrib-group')
            if contrib_group:
                for author in contrib_group.find_all('a'):
                    name = author.text.strip()
                    if name and len(name) > 2:
                        authors_list.append(name)
        
        authors = ", ".join(authors_list) if authors_list else "N/A"
        
        # ===== EXTRAER AÑO =====
        year = "N/A"
        
        # Método 1: Buscar en span.cit
        cit_span = soup.find('span', class_='cit')
        if cit_span:
            year_match = re.search(r'\b(19|20)\d{2}\b', cit_span.text)
            if year_match:
                year = year_match.group(0)
        
        # Método 2: Buscar en meta tags
        if year == "N/A":
            date_meta = soup.find('meta', {'name': 'citation_publication_date'})
            if date_meta and date_meta.get('content'):
                year_match = re.search(r'\b(19|20)\d{2}\b', date_meta['content'])
                if year_match:
                    year = year_match.group(0)
        
        # ===== EXTRAER TÍTULO =====
        title = "N/A"
        
        # Método 1: h1.content-title
        title_h1 = soup.find('h1', class_='content-title')
        if title_h1:
            title = title_h1.text.strip()
        
        # Método 2: meta tag
        if title == "N/A":
            title_meta = soup.find('meta', {'name': 'citation_title'})
            if title_meta and title_meta.get('content'):
                title = title_meta['content'].strip()
        
        # ===== EXTRAER ABSTRACT =====
        abstract_text = ""
        
        # Método 1: div.abstract
        abstract_div = soup.find('div', class_='abstract')
        if abstract_div:
            # Eliminar el título "Abstract" si existe
            abstract_title = abstract_div.find('h2')
            if abstract_title:
                abstract_title.decompose()
            
            abstract_text = abstract_div.get_text(separator=' ', strip=True)
        
        # Método 2: section#abstract
        if not abstract_text:
            abstract_section = soup.find('section', id='abstract')
            if abstract_section:
                abstract_text = abstract_section.get_text(separator=' ', strip=True)
        
        # Método 3: Buscar en meta tags
        if not abstract_text:
            abstract_meta = soup.find('meta', {'name': 'citation_abstract'})
            if abstract_meta and abstract_meta.get('content'):
                abstract_text = abstract_meta['content'].strip()
        
        # Limpieza del abstract
        abstract_text = re.sub(r'\s+', ' ', abstract_text).strip()
        
        return {
            'title': title,
            'authors': authors,
            'year': year,
            'abstract_text': abstract_text,
            'source_url': url
        }
        
    except requests.RequestException as e:
        print(f"\n  Error de conexión: {url[:50]}... - {e}")
        return None
    except Exception as e:
        print(f"\n  Error al parsear: {url[:50]}... - {e}")
        return None

def enrich_nasa_csv(input_file, output_csv='data/publicaciones.csv', 
                    output_embeddings='data/corpus_embeddings.npy'):
    """
    Proceso completo: enriquece CSV y genera embeddings
    """
    print("="*60)
    print("NASA PAPERS - EXTRACTOR DE METADATOS")
    print("="*60 + "\n")
    
    # Crear carpeta data si no existe
    import os
    os.makedirs('data', exist_ok=True)
    
    # Cargar CSV original
    print(f"Cargando {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Total de papers: {len(df)}\n")
    
    # Agregar columnas si no existen
    if 'authors' not in df.columns:
        df['authors'] = 'N/A'
    if 'year' not in df.columns:
        df['year'] = 'N/A'
    if 'abstract_text' not in df.columns:
        df['abstract_text'] = ''
    if 'source_url' not in df.columns:
        df['source_url'] = ''
    
    # Extraer metadatos
    print("Extrayendo metadatos de NCBI...\n")
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Procesando"):
        link = row.get('Link', row.get('link', ''))
        
        if not link or pd.isna(link):
            failed += 1
            continue
        
        metadata = extract_metadata_from_ncbi(link)
        
        if metadata:
            df.at[idx, 'title'] = metadata['title'] if metadata['title'] != 'N/A' else row.get('Title', 'N/A')
            df.at[idx, 'authors'] = metadata['authors']
            df.at[idx, 'year'] = metadata['year']
            df.at[idx, 'abstract_text'] = metadata['abstract_text']
            df.at[idx, 'source_url'] = metadata['source_url']
            successful += 1
        else:
            failed += 1
        
        # Pausa para no saturar el servidor
        time.sleep(0.5)
    
    print(f"\n\nResultados:")
    print(f"  Exitosos: {successful}")
    print(f"  Fallidos: {failed}")
    
    # Guardar CSV enriquecido
    print(f"\nGuardando {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    # Generar embeddings
    print("\nGenerando embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = []
    for _, row in df.iterrows():
        title = str(row['title']) if pd.notna(row['title']) else ""
        abstract = str(row['abstract_text']) if pd.notna(row['abstract_text']) else ""
        
        if abstract:
            text = f"{title}. {abstract}"
        else:
            text = title
        
        texts.append(text)
    
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    
    # Guardar embeddings
    print(f"\nGuardando {output_embeddings}...")
    np.save(output_embeddings, embeddings)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    print(f"\nArchivos generados:")
    print(f"  - {output_csv}")
    print(f"  - {output_embeddings}")
    print(f"\nEstadísticas finales:")
    print(f"  - Papers procesados: {len(df)}")
    print(f"  - Con autores: {df['authors'].notna().sum()}")
    print(f"  - Con año: {df['year'].notna().sum()}")
    print(f"  - Con abstract: {(df['abstract_text'].str.len() > 0).sum()}")
    
    print("\nPróximo paso:")
    print("  streamlit run app.py")

def validate_csv(csv_file):
    """
    Valida que el CSV tenga las columnas necesarias
    """
    df = pd.read_csv(csv_file)
    
    # Verificar columnas
    required = ['Title', 'Link']
    missing = [col for col in required if col not in df.columns and col.lower() not in df.columns]
    
    if missing:
        print(f"Error: Faltan columnas {missing}")
        return False
    
    print(f"CSV válido: {len(df)} papers encontrados")
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NASA SPACE BIOLOGY - PROCESADOR DE METADATOS")
    print("="*60 + "\n")
    
    print("Este script:")
    print("  1. Lee el CSV de la NASA con links a NCBI")
    print("  2. Extrae autores, año y abstracts")
    print("  3. Genera embeddings")
    print("  4. Crea archivos listos para la app\n")
    
    input_file = input("Ruta al CSV de la NASA (default: SB_publication_PMC.csv): ").strip()
    
    if not input_file:
        input_file = "SB_publication_PMC.csv"
    
    if validate_csv(input_file):
        print("\nIniciando proceso...")
        time.sleep(1)
        enrich_nasa_csv(input_file)
    else:
        print("\nFormato esperado del CSV:")
        print("Title,Link")
        print("Paper 1,https://www.ncbi.nlm.nih.gov/pmc/articles/...")
        print("Paper 2,https://www.ncbi.nlm.nih.gov/pmc/articles/...")