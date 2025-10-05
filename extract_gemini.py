"""
Versión DEFINITIVA del extractor de metadatos.
Usa Selenium para controlar un navegador real y renderizar JavaScript.
Es más lento pero mucho más confiable.
"""

import pandas as pd
from tqdm import tqdm
import time
import re
import os

# --- Importaciones de Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    """Configura el navegador Chrome invisible (headless)."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Para que no se abra una ventana visible
    chrome_options.add_argument("--log-level=3") # Para reducir los mensajes en la terminal
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # Instala o actualiza el driver de Chrome automáticamente
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_with_selenium(driver, url):
    """
    Extrae metadatos de una URL usando Selenium para cargar la página completa.
    """
    try:
        driver.get(url)
        # Espera un poco para que el JavaScript cargue todo el contenido
        time.sleep(2) 
        
        # Obtener el HTML renderizado
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # --- Extracción de metadatos (la misma lógica robusta de antes) ---
        
        # Título
        title_meta = soup.find('meta', {'name': 'citation_title'})
        title = title_meta['content'].strip() if title_meta and title_meta.get('content') else "N/A"

        # Autores
        author_metas = soup.find_all('meta', {'name': 'citation_author'})
        authors_list = [meta['content'].strip() for meta in author_metas if meta.get('content')]
        authors = ", ".join(authors_list) if authors_list else "N/A"
        
        # Año
        year = "N/A"
        date_meta = soup.find('meta', {'name': 'citation_publication_date'})
        if date_meta and date_meta.get('content'):
            year_match = re.search(r'\b(19|20)\d{2}\b', date_meta['content'])
            if year_match: year = year_match.group(0)

        # Abstract (con los métodos de respaldo)
        abstract = ""
        abstract_meta = soup.find('meta', {'name': 'citation_abstract'})
        if abstract_meta and abstract_meta.get('content'):
            abstract = abstract_meta['content'].strip()
        if not abstract:
            abstract_div = soup.find('div', class_=re.compile(r'abstract-content|abstract', re.I))
            if abstract_div:
                abstract = abstract_div.get_text(separator=' ', strip=True)

        return {
            'title': title, 'authors': authors, 'year': year,
            'abstract_text': abstract.replace('Abstract', '').strip(),
            'source_url': url
        }

    except Exception as e:
        print(f"\n   Error procesando {url[:60]}...: {e}")
        return None

def process_csv_final(input_file, output_file='data/publicaciones_final.csv'):
    print("="*60)
    print("EXTRACTOR DE METADATOS DEFINITIVO (con Selenium)")
    print("="*60 + "\n")

    print("Configurando navegador... (esto puede tardar la primera vez)")
    driver = setup_driver()
    print("Navegador listo.\n")

    df = pd.read_csv(input_file)
    if 'abstract_text' not in df.columns:
        df['abstract_text'] = ""

    link_col = next((col for col in ['Link', 'link'] if col in df.columns), None)
    if not link_col:
        print("Error: No se encontró columna de links.")
        driver.quit()
        return

    print(f"Procesando {len(df)} papers. Esto será lento pero seguro, por favor sé paciente.\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Para ahorrar tiempo, solo procesamos los que no tienen abstract
        if pd.notna(row.get('abstract_text')) and len(str(row.get('abstract_text'))) > 50:
            continue

        url = row[link_col]
        if pd.isna(url) or not url:
            continue

        metadata = extract_with_selenium(driver, url)
        
        if metadata:
            df.at[idx, 'title'] = metadata['title']
            df.at[idx, 'authors'] = metadata['authors']
            df.at[idx, 'year'] = metadata['year']
            df.at[idx, 'abstract_text'] = metadata['abstract_text']
            df.at[idx, 'source_url'] = metadata['source_url']
            
        # Guardar progreso cada 10 filas para no perder el trabajo
        if (idx + 1) % 10 == 0:
            df.to_csv(output_file, index=False)
            
    driver.quit() # Cierra el navegador al final
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("PROCESO FINALIZADO")
    print("="*60)
    with_abstracts = df[df['abstract_text'].str.strip().str.len() > 50].shape[0]
    print(f"\n   Papers con abstract: {with_abstracts} ({with_abstracts/len(df)*100:.1f}%)")
    print(f"\n📄 Archivo final generado: {output_file}")

if __name__ == "__main__":
    from bs4 import BeautifulSoup # Necesario añadir esta importación
    input_csv = "data/publicaciones_fixed.csv" # Cambia esto si tu archivo se llama diferente
    process_csv_final(input_csv)