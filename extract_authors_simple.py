"""
Versión SIMPLIFICADA y GARANTIZADA para extraer autores
Usa SOLO los meta tags que confirmamos que existen
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import re

def extract_from_ncbi_simple(url):
    """
    Extrae autores, año y abstract usando SOLO meta tags
    (el método más confiable según el diagnóstico)
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # ===== AUTORES - Método de META TAGS (100% confiable) =====
        author_metas = soup.find_all('meta', {'name': 'citation_author'})
        authors_list = [meta['content'].strip() for meta in author_metas if meta.get('content')]
        authors = ", ".join(authors_list) if authors_list else "N/A"
        
        # ===== AÑO =====
        year = "N/A"
        
        # Buscar en meta tag de fecha
        date_meta = soup.find('meta', {'name': 'citation_publication_date'})
        if date_meta and date_meta.get('content'):
            year_match = re.search(r'\b(19|20)\d{2}\b', date_meta['content'])
            if year_match:
                year = year_match.group(0)
        
        # Si no encuentra, buscar en citation_date
        if year == "N/A":
            date_meta2 = soup.find('meta', {'name': 'citation_date'})
            if date_meta2 and date_meta2.get('content'):
                year_match = re.search(r'\b(19|20)\d{2}\b', date_meta2['content'])
                if year_match:
                    year = year_match.group(0)
        
        # ===== TÍTULO =====
        title = "N/A"
        title_meta = soup.find('meta', {'name': 'citation_title'})
        if title_meta and title_meta.get('content'):
            title = title_meta['content'].strip()
        
        # Reemplaza la sección de ABSTRACT con esto:

        # ===== ABSTRACT =====
        abstract = ""

        # Método 1: Meta tag
        abstract_meta = soup.find('meta', {'name': 'citation_abstract'})
        if abstract_meta and abstract_meta.get('content'):
            abstract = abstract_meta['content'].strip()

        # Método 2: Div con clase 'abstract'
        if not abstract:
            abstract_div = soup.find('div', class_='abstract')
            if abstract_div:
                # Remover título "Abstract" si existe
                for title_tag in abstract_div.find_all(['h2', 'h3', 'h4']):
                    title_tag.decompose()
                abstract = abstract_div.get_text(separator=' ', strip=True)

        # Método 3: Section con id 'abstract'
        if not abstract:
            abstract_section = soup.find('section', id='abstract')
            if abstract_section:
                abstract = abstract_section.get_text(separator=' ', strip=True)

        # Método 4: Cualquier div que contenga "abstract" en su clase
        if not abstract:
            for div in soup.find_all('div', class_=re.compile(r'abstract', re.I)):
                text = div.get_text(separator=' ', strip=True)
                if len(text) > 100:  # Asegurar que sea texto sustancial
                    abstract = text
                    break

        # Limpieza
        abstract = re.sub(r'\s+', ' ', abstract).strip()
        abstract = abstract.replace('Abstract', '').strip()
        
        return {
            'title': title,
            'authors': authors,
            'year': year,
            'abstract_text': abstract,
            'source_url': url
        }
        
    except Exception as e:
        print(f"\n  Error en {url[:50]}...: {e}")
        return None

def process_csv(input_file, output_file='data/publicaciones_fixed.csv'):
    """
    Procesa el CSV y genera uno nuevo con autores correctos
    """
    print("="*60)
    print("EXTRACTOR SIMPLE DE AUTORES (Solo Meta Tags)")
    print("="*60 + "\n")
    
    # Crear carpeta data
    import os
    os.makedirs('data', exist_ok=True)
    
    # Leer CSV
    print(f"Leyendo {input_file}...")
    df = pd.read_csv(input_file)
    
    # Si ya tiene una columna Link, usarla; si no, buscar otras variantes
    link_col = None
    for col in ['Link', 'link', 'source_url', 'url']:
        if col in df.columns:
            link_col = col
            break
    
    if not link_col:
        print("Error: No se encontró columna de links")
        return
    
    print(f"Total de papers: {len(df)}\n")
    print("Procesando...\n")
    
    # Procesar cada fila
    success = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row[link_col]
        
        if pd.isna(url) or not url:
            failed += 1
            continue
        
        metadata = extract_from_ncbi_simple(url)
        
        if metadata:
            # Actualizar datos
            if metadata['title'] != 'N/A':
                df.at[idx, 'title'] = metadata['title']
            
            df.at[idx, 'authors'] = metadata['authors']
            df.at[idx, 'year'] = metadata['year']
            df.at[idx, 'abstract_text'] = metadata['abstract_text']
            df.at[idx, 'source_url'] = metadata['source_url']
            
            success += 1
        else:
            failed += 1
        
        # Pausa para no saturar
        time.sleep(0.3)
    
    # Guardar
    print(f"\n\nGuardando {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    print(f"\nEstadísticas:")
    print(f"  Exitosos: {success}")
    print(f"  Fallidos: {failed}")
    print(f"  Total: {len(df)}")
    
    # Verificar autores
    with_authors = df[df['authors'] != 'N/A'].shape[0]
    print(f"\n  Papers con autores: {with_authors} ({with_authors/len(df)*100:.1f}%)")
    
    print(f"\n📄 Archivo generado: {output_file}")
    print("\nPrimeros 5 registros:")
    print(df[['title', 'authors', 'year']].head())
    
    return df

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("EXTRACTOR SIMPLE - 100% Meta Tags")
    print("="*60 + "\n")
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("Archivo CSV a procesar (default: data/publicaciones.csv): ").strip()
        if not input_file:
            input_file = "data/publicaciones.csv"
    
    print(f"\nProcesando: {input_file}\n")
    process_csv(input_file)