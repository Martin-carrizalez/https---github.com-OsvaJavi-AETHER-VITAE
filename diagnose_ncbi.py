"""
Script de diagnóstico para ver la estructura HTML de NCBI
y encontrar dónde están realmente los autores
"""

import requests
from bs4 import BeautifulSoup

def diagnose_page(url):
    """
    Analiza una página de NCBI y muestra toda la información relevante
    """
    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO DE: {url}")
    print(f"{'='*60}\n")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # ===== META TAGS =====
    print("1. META TAGS DE AUTORES:")
    author_metas = soup.find_all('meta', {'name': 'citation_author'})
    if author_metas:
        print(f"   Encontrados {len(author_metas)} meta tags:")
        for i, meta in enumerate(author_metas[:5], 1):
            print(f"   {i}. {meta.get('content')}")
    else:
        print("   ❌ No se encontraron meta tags citation_author")
    
    # ===== BÚSQUEDA POR CLASES COMUNES =====
    print("\n2. DIVS CON CLASES RELACIONADAS A AUTORES:")
    for class_name in ['authors', 'authors-list', 'contrib-group', 'contributors', 'author-list']:
        divs = soup.find_all('div', class_=class_name)
        if divs:
            print(f"   ✅ Encontrado div.{class_name}:")
            for div in divs[:2]:
                text = div.get_text(strip=True)[:100]
                print(f"      Contenido: {text}...")
    
    # ===== BÚSQUEDA EN HEADER =====
    print("\n3. HEADER/ARTICLE-META:")
    header = soup.find('div', class_='article-meta')
    if not header:
        header = soup.find('header')
    
    if header:
        print("   ✅ Encontrado header/article-meta")
        # Buscar todos los links
        links = header.find_all('a')
        print(f"   Links encontrados: {len(links)}")
        for link in links[:5]:
            text = link.get_text(strip=True)
            if text and 5 < len(text) < 50:
                print(f"      - {text}")
    else:
        print("   ❌ No se encontró header")
    
    # ===== TODOS LOS SPANS CON NOMBRES =====
    print("\n4. SPANS QUE PODRÍAN SER AUTORES:")
    spans = soup.find_all('span', class_=lambda x: x and ('author' in x.lower() or 'contrib' in x.lower()))
    if spans:
        print(f"   Encontrados {len(spans)} spans:")
        for span in spans[:5]:
            print(f"      - Clase: {span.get('class')}")
            print(f"        Texto: {span.get_text(strip=True)[:50]}")
    
    # ===== ESTRUCTURA COMPLETA DEL HEADER =====
    print("\n5. HTML DEL ÁREA DE AUTORES (primeros 1000 caracteres):")
    article_header = soup.find('div', {'class': lambda x: x and 'article' in x.lower()})
    if article_header:
        html_preview = str(article_header)[:1000]
        print(f"   {html_preview}...")
    else:
        print("   ❌ No se encontró div con 'article'")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # URLs de ejemplo de tu CSV
    test_urls = [
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4136787/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3630201/",
        "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11988870/"
    ]
    
    print("\nEste script analizará 3 páginas de ejemplo para encontrar")
    print("dónde están exactamente los autores en el HTML\n")
    
    for url in test_urls:
        try:
            diagnose_page(url)
            print("\nPresiona Enter para continuar...")
            input()
        except Exception as e:
            print(f"❌ Error: {e}\n")