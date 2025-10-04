🚀 Motor de Conocimiento de Biología Espacial - NASA Challenge
📋 Descripción del Proyecto
Motor de búsqueda inteligente basado en IA que permite explorar y analizar las 608 publicaciones científicas de la NASA sobre biología espacial. Utiliza búsqueda semántica, resúmenes automatizados y extracción de entidades clave para facilitar el acceso al conocimiento científico.
✨ Características Principales

🔍 Búsqueda Semántica: Encuentra publicaciones por significado, no solo por palabras clave
📝 Resúmenes Automáticos: Genera resúmenes concisos de artículos científicos usando Groq + Llama
🏷️ Extracción de Entidades: Identifica automáticamente organismos, condiciones espaciales y hallazgos clave
📊 Visualizaciones Interactivas: Dashboard intuitivo con estadísticas y gráficos
⚡ Respuestas en Tiempo Real: Procesamiento rápido gracias a embeddings pre-calculados

🛠️ Tecnologías Utilizadas

Frontend: Streamlit
IA/LLM: Groq API (Llama 3)
Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
Procesamiento: Pandas, NumPy
Lenguaje: Python 3.9+

📁 Estructura del Proyecto
/nasa-knowledge-engine
├── data/
│   ├── publicaciones.csv          # Datos de las 608 publicaciones
│   └── corpus_embeddings.npy      # Embeddings pre-calculados
├── notebooks/
│   └── exploracion.ipynb          # Análisis exploratorio (opcional)
├── app.py                         # Aplicación principal de Streamlit
├── create_embeddings.py           # Script para generar embeddings
├── requirements.txt               # Dependencias del proyecto
├── .env                          # API keys (no subir a Git)
├── .gitignore                    # Archivos a ignorar
└── README.md                     # Este archivo
🚀 Instalación y Configuración
1. Clonar el Repositorio
bashgit clone https://github.com/tu-usuario/nasa-knowledge-engine.git
cd nasa-knowledge-engine
2. Crear Entorno Virtual
bashpython -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
3. Instalar Dependencias
bashpip install -r requirements.txt
4. Configurar API Key de Groq
Crea un archivo .env en la raíz del proyecto:
envGROQ_API_KEY=tu-api-key-aqui

💡 Obtén tu API key gratis en console.groq.com

5. Preparar los Datos
Coloca el archivo publicaciones.csv en la carpeta /data con las siguientes columnas:

id: Identificador único
title: Título de la publicación
authors: Autores
year: Año de publicación
abstract_text: Resumen/abstract
source_url: URL de la fuente (opcional)

6. Generar Embeddings
bashpython create_embeddings.py
Este proceso puede tardar algunos minutos. Generará el archivo corpus_embeddings.npy.
7. Ejecutar la Aplicación
bashstreamlit run app.py
La aplicación se abrirá automáticamente en tu navegador en http://localhost:8501
💡 Cómo Usar la Aplicación

Buscar: Escribe tu consulta en lenguaje natural (ej. "efectos de la microgravedad en plantas")
Explorar Resultados: Revisa las publicaciones más relevantes ordenadas por similitud semántica
Leer Resúmenes: Obtén un resumen de 3 puntos clave generado automáticamente
Ver Entidades: Consulta organismos, condiciones y hallazgos extraídos automáticamente
Filtrar: Usa la barra lateral para refinar tu búsqueda por año, organismo, etc.

📊 Ejemplos de Consultas

"¿Cómo afecta la radiación espacial al ADN?"
"Experimentos con plantas en microgravedad"
"Cambios en el sistema inmune de astronautas"
"Estudios con C. elegans en el espacio"

🎯 Desafíos Técnicos Resueltos

Procesamiento Eficiente: Uso de embeddings pre-calculados y caché de Streamlit
Prompt Engineering: Diseño de prompts específicos para resúmenes y extracción de entidades
Búsqueda Semántica: Implementación de similitud coseno para resultados relevantes
Experiencia de Usuario: Interface intuitiva con carga asíncrona y feedback visual

🚧 Próximas Mejoras

 Filtros avanzados por tipo de organismo y condición espacial
 Visualización de red de relaciones entre publicaciones
 Exportación de resultados en PDF
 Chat interactivo con el corpus completo
 Soporte multiidioma

👥 Autor
Desarrollado para el NASA Space Biology Challenge - Guadalajara 2025
📄 Licencia
Este proyecto es de código abierto y está disponible bajo la licencia MIT.
🙏 Agradecimientos

NASA por proporcionar acceso a las publicaciones científicas
Groq por su API ultrarrápida de LLMs
Comunidad de Streamlit por sus excelentes herramientas

📧 Contacto
¿Preguntas o sugerencias? Abre un issue en GitHub o contáctame en [tu-email]

