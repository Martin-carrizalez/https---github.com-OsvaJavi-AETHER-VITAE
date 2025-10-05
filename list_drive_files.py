"""
Script para listar todos los PDFs de una carpeta pública de Google Drive
y generar automáticamente el CSV para process_pdfs.py

IMPORTANTE: La carpeta de Drive debe ser PÚBLICA (cualquiera con el enlace)
"""

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pandas as pd
import os
import pickle

# Scopes necesarios (solo lectura)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def authenticate_drive():
    """
    Autentica con Google Drive API
    
    Requiere crear credenciales en Google Cloud Console:
    1. Ve a https://console.cloud.google.com
    2. Crea un proyecto
    3. Habilita Google Drive API
    4. Crea credenciales OAuth 2.0
    5. Descarga el JSON como 'credentials.json'
    """
    creds = None
    
    # Token guardado de sesiones previas
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # Si no hay credenciales válidas, hacer login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("❌ Error: No se encontró credentials.json")
                print("\n📋 PASOS PARA OBTENER CREDENCIALES:")
                print("1. Ve a https://console.cloud.google.com")
                print("2. Crea un proyecto nuevo")
                print("3. Habilita 'Google Drive API'")
                print("4. Ve a 'Credenciales' → 'Crear credenciales' → 'OAuth 2.0'")
                print("5. Descarga el JSON y guárdalo como 'credentials.json'")
                print("6. Vuelve a ejecutar este script")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Guardar credenciales para la próxima vez
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def list_files_in_folder(folder_id):
    """
    Lista todos los archivos PDF en una carpeta de Drive
    
    Args:
        folder_id: ID de la carpeta (lo sacas de la URL)
        
    Returns:
        Lista de diccionarios con info de cada archivo
    """
    print("🔐 Autenticando con Google Drive...")
    creds = authenticate_drive()
    
    if not creds:
        return []
    
    print("✅ Autenticación exitosa")
    print(f"🔍 Listando archivos en carpeta {folder_id}...\n")
    
    try:
        service = build('drive', 'v3', credentials=creds)
        
        # Query para buscar PDFs en la carpeta
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        
        files = []
        page_token = None
        
        while True:
            response = service.files().list(
                q=query,
                spaces='drive',
                fields='nextPageToken, files(id, name, createdTime, webViewLink)',
                pageToken=page_token,
                pageSize=1000
            ).execute()
            
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            
            if page_token is None:
                break
        
        print(f"✅ Encontrados {len(files)} archivos PDF\n")
        return files
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []

def extract_metadata_from_filename(filename):
    """
    Intenta extraer información del nombre del archivo
    Ajusta según el formato de tus archivos
    """
    # Ejemplo: "2023_Smith_Effect_of_Microgravity.pdf"
    # O: "Microgravity Study - Smith et al 2023.pdf"
    
    # Remover extensión
    name = filename.replace('.pdf', '')
    
    # Extraer año (buscar 4 dígitos)
    import re
    year_match = re.search(r'(19|20)\d{2}', name)
    year = year_match.group(0) if year_match else 'N/A'
    
    # Título (simplificado)
    title = name.replace('_', ' ').replace('-', ' ')
    title = ' '.join(title.split())  # Limpiar espacios múltiples
    
    return {
        'title': title,
        'year': year,
        'authors': 'N/A'  # Se puede extraer si el formato es consistente
    }

def create_csv_from_drive(folder_id, output_file='nasa_papers_links.csv'):
    """
    Crea el CSV con todos los PDFs de la carpeta
    """
    print("="*60)
    print("🚀 NASA PAPERS - LISTADOR DE GOOGLE DRIVE")
    print("="*60 + "\n")
    
    # Listar archivos
    files = list_files_in_folder(folder_id)
    
    if not files:
        print("⚠️ No se encontraron archivos")
        return
    
    # Crear DataFrame
    papers = []
    
    print("📝 Procesando metadata...\n")
    for file in files:
        metadata = extract_metadata_from_filename(file['name'])
        
        papers.append({
            'title': metadata['title'],
            'authors': metadata['authors'],
            'year': metadata['year'],
            'drive_link': file['webViewLink'],
            'file_id': file['id'],
            'original_filename': file['name']
        })
    
    df = pd.DataFrame(papers)
    
    # Guardar CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ CSV generado: {output_file}")
    print(f"📊 Total de papers: {len(df)}")
    print("\n📋 Primeros 5 registros:")
    print(df[['title', 'year', 'drive_link']].head())
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    print("\n🚀 SIGUIENTE PASO:")
    print("→ python process_pdfs.py")
    print("   Selecciona opción 1")
    print(f"   Indica: {output_file}")

def extract_folder_id_from_url(url):
    """
    Extrae el folder_id de la URL de Drive
    """
    # URL formato: https://drive.google.com/drive/folders/FOLDER_ID
    import re
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

# ============================================================================
# MÉTODO ALTERNATIVO SIN API (Para carpetas públicas)
# ============================================================================

def list_public_folder_simple(folder_url):
    """
    Método simplificado para carpetas públicas
    No requiere autenticación pero puede ser menos confiable
    """
    import requests
    from bs4 import BeautifulSoup
    
    print("🌐 Método simplificado (sin autenticación)...")
    print("⚠️ Solo funciona con carpetas PÚBLICAS\n")
    
    try:
        # Intentar acceder a la carpeta
        response = requests.get(folder_url)
        response.raise_for_status()
        
        # Este método es limitado porque Drive usa JavaScript
        print("⚠️ Limitación: Drive carga archivos con JavaScript")
        print("📋 RECOMENDACIÓN:")
        print("1. Usa el método con API (más confiable)")
        print("2. O crea el CSV manualmente")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 GOOGLE DRIVE FOLDER LISTER - NASA PAPERS")
    print("="*60 + "\n")
    
    print("Opciones:")
    print("1. Listar carpeta usando Google Drive API (RECOMENDADO)")
    print("2. Método simplificado para carpeta pública (experimental)")
    print("3. Crear plantilla vacía para llenar manualmente\n")
    
    opcion = input("Selecciona opción (1/2/3): ").strip()
    
    if opcion == "1":
        print("\n📋 Necesitas la URL de tu carpeta de Drive")
        print("Ejemplo: https://drive.google.com/drive/folders/1ABC123DEF456\n")
        
        folder_url = input("URL de la carpeta: ").strip()
        folder_id = extract_folder_id_from_url(folder_url)
        
        if folder_id:
            print(f"\n✅ Folder ID: {folder_id}\n")
            create_csv_from_drive(folder_id)
        else:
            print("❌ URL inválida. Debe ser: https://drive.google.com/drive/folders/FOLDER_ID")
    
    elif opcion == "2":
        folder_url = input("\nURL de la carpeta pública: ").strip()
        list_public_folder_simple(folder_url)
        
    elif opcion == "3":
        # Crear plantilla vacía
        template = pd.DataFrame({
            'title': ['Ejemplo: Microgravity Effects on Plants'],
            'authors': ['Smith J, Garcia M'],
            'year': [2023],
            'drive_link': ['https://drive.google.com/file/d/1ABC123/view']
        })
        template.to_csv('nasa_papers_template.csv', index=False)
        print("✅ Plantilla creada: nasa_papers_template.csv")
        print("📝 Llena el archivo con tus 600 papers")
    
    else:
        print("❌ Opción inválida")
    
    print("\n💡 Nota: Si tienes problemas con la API, usa la opción 3")
    print("   y llena el CSV manualmente con los links de Drive")