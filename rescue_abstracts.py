import pandas as pd
from groq import Groq
from tqdm import tqdm
import os
from dotenv import load_dotenv

def rescue_abstracts_with_ai(final_csv_path, old_csv_path, output_csv_path):
    """
    Usa el full_text de un CSV antiguo para rellenar los abstracts faltantes
    en un CSV final, usando una IA para limpiar y extraer el resumen.
    """
    print("="*60)
    print("🚀 SCRIPT DE RESCATE DE ABSTRACTS CON IA")
    print("="*60 + "\n")

    # --- 1. Cargar la API Key y el cliente de Groq ---
    load_dotenv()
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    except Exception as e:
        print("❌ Error: No se pudo configurar el cliente de Groq. Revisa tu API key en el archivo .env")
        return

    # --- 2. Cargar los dos archivos CSV ---
    print(f"📂 Cargando archivo final: {final_csv_path}")
    df_final = pd.read_csv(final_csv_path)
    
    print(f"📦 Cargando archivo antiguo con texto completo: {old_csv_path}")
    df_old = pd.read_csv(old_csv_path)

    # Asegurarse de que las columnas de texto no tengan valores nulos
    df_final['abstract_text'] = df_final['abstract_text'].fillna('')
    df_old['full_text'] = df_old['full_text'].fillna('')
    
    # --- 3. Identificar los abstracts que faltan ---
    missing_abstracts_df = df_final[df_final['abstract_text'].str.strip().str.len() < 50]
    if missing_abstracts_df.empty:
        print("\n✅ ¡Felicidades! No faltan abstracts. No hay nada que rescatar.")
        return

    print(f"\n🔍 Se encontraron {len(missing_abstracts_df)} abstracts para rescatar. ¡Manos a la obra!\n")

    # --- 4. Iterar y rescatar ---
    for idx, row in tqdm(missing_abstracts_df.iterrows(), total=len(missing_abstracts_df), desc="Rescatando abstracts"):
        title_to_find = row['title']
        
        # Buscar el paper correspondiente en el archivo antiguo por título
        old_row = df_old[df_old['title'] == title_to_find]
        
        if not old_row.empty and old_row.iloc[0]['full_text']:
            full_text = old_row.iloc[0]['full_text']
            
            try:
                # Pedirle a la IA que extraiga el abstract del texto en bruto
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "Eres un asistente experto en análisis de documentos científicos. Tu tarea es leer un texto completo y desordenado de un paper y extraer únicamente el texto del resumen (abstract), limpiándolo de cualquier texto sobrante o metadatos."
                        },
                        {
                            "role": "user",
                            "content": f"Extrae el resumen (abstract) del siguiente texto: '{full_text[:4000]}'" # Usamos los primeros 4000 caracteres para no exceder límites
                        }
                    ],
                    model="llama-3.1-8b-instant",
                )
                
                clean_abstract = chat_completion.choices[0].message.content
                
                # Actualizar el DataFrame final con el abstract rescatado
                df_final.at[idx, 'abstract_text'] = clean_abstract
                
            except Exception as e:
                # Si la IA falla, lo dejamos en blanco para no detener el proceso
                pass 
                
    # --- 5. Guardar el resultado final ---
    df_final.to_csv(output_csv_path, index=False)
    
    print("\n" + "="*60)
    print("✅ PROCESO DE RESCATE COMPLETADO")
    print("="*60)
    final_count = df_final[df_final['abstract_text'].str.strip().str.len() > 50].shape[0]
    print(f"\n   Total de papers con abstract ahora: {final_count} ({final_count/len(df_final)*100:.1f}%)")
    print(f"\n📄 Archivo final y completo guardado en: {output_csv_path}")

if __name__ == "__main__":
    # Asegúrate de que los nombres de los archivos sean correctos
    final_csv = 'data/publicaciones_final.csv'
    old_csv_with_full_text = 'data/publicacionescompl.csv'
    output_csv = 'data/publicaciones_COMPLETO.csv'
    
    rescue_abstracts_with_ai(final_csv, old_csv_with_full_text, output_csv)