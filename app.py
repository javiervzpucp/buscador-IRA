import os
import requests
import spacy
import unicodedata
import streamlit as st
from dotenv import load_dotenv  

# Cargar variables de entorno
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GRAPHDB_SERVER = "https://72c5a4103094.ngrok.app/"
REPO_NAME = "IRA"
SPARQL_ENDPOINT = f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Cargar modelo spaCy
modelo_es = "es_core_news_sm"
if not spacy.util.is_package(modelo_es):
    os.system(f"python -m spacy download {modelo_es}")
nlp = spacy.load(modelo_es)

# 📌 Función para consultar GraphDB
def query_graphdb(sparql_query):
    params = {"query": sparql_query}
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(SPARQL_ENDPOINT, params=params, headers=headers)
    return response.json() if response.status_code == 200 else None

# 📌 Función para limpiar caracteres UTF-8 mal codificados
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text).strip()
    
    # Corrección de caracteres mal codificados
    replacements = {
        "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
        "Ã±": "ñ", "Ã": "Á", "Ã‰": "É", "Ã": "Í", "Ã“": "Ó",
        "Ãš": "Ú", "Ã‘": "Ñ", "â€“": "–", "â€”": "—", "â€œ": "“",
        "â€�": "”", "â€¢": "•", "â€¦": "…", "Âº": "º", "Âª": "ª",
        "Ã¼": "ü", "Ãœ": "Ü", "Ã€": "À", "Ãˆ": "È", "ÃŒ": "Ì",
        "Ã’": "Ò", "Ã™": "Ù", "\x93": '"', "\x94": '"',
        "\x91": "'", "\x92": "'", "\xad": "-", "3n": "ón",
        "\xada": "á", "Ã\xada": "í", "Ã\xad": "í"
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text

# 📌 Extraer entidades clave de la pregunta
def extract_entities(question):
    doc = nlp(question)
    entities = {"date": None, "creator": None, "subject": None}

    for ent in doc.ents:
        if ent.label_ == "DATE":
            entities["date"] = ent.text
        elif ent.label_ == "PERSON":
            entities["creator"] = ent.text
        elif ent.label_ in ["LOC", "GPE", "ORG", "MISC"]:
            entities["subject"] = ent.text

    return entities

# 📌 Generar consulta SPARQL
def generate_sparql_query(question):
    entities = extract_entities(question)

    sparql_query = """
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT ?title ?date ?creator ?subject WHERE {
        ?doc dc:title ?title .
    """

    if entities["date"]:
        sparql_query += f'\n        ?doc dc:date ?date . FILTER(CONTAINS(LCASE(?date), "{entities["date"].lower()}")) .'
    if entities["creator"]:
        sparql_query += f'\n        ?doc dc:creator ?creator . FILTER(CONTAINS(LCASE(?creator), "{entities["creator"].lower()}")) .'
    if entities["subject"]:
        sparql_query += f'\n        ?doc dc:subject ?subject . FILTER(CONTAINS(LCASE(?subject), "{entities["subject"].lower()}")) .'

    sparql_query += "\n    } LIMIT 20"
    
    return sparql_query.strip()

# 📌 Resumir documentos obtenidos de GraphDB
def summarize_documents(documents):
    if not documents:
        return "No se encontraron documentos relevantes en la base de datos."

    grouped_by_subject = {}
    for doc in documents:
        title = clean_text(doc.get("title", {}).get("value", "Título desconocido"))
        subject = clean_text(doc.get("subject", {}).get("value", "Sin tema"))
        date = clean_text(doc.get("date", {}).get("value", "Fecha desconocida"))
        creator = clean_text(doc.get("creator", {}).get("value", "Autor desconocido"))

        if subject not in grouped_by_subject:
            grouped_by_subject[subject] = set()
        grouped_by_subject[subject].add(f"{title} ({date}) - Autor: {creator}")

    # 📌 Generar un resumen estructurado
    summary = "Aquí tienes un resumen de los documentos encontrados:\n\n"
    for subject, titles in grouped_by_subject.items():
        summary += f"📌 **{subject}**:\n"
        for title in sorted(titles)[:5]:  # Mostrar solo los primeros 5 de cada tema
            summary += f"- {title}\n"
        summary += "\n"

    return summary.strip()

# 📌 Consultar Hugging Face API con Mistral
def ask_mistral(question, context):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {
        "inputs": f"Pregunta: {question}\nInformación relevante:\n{context}",
        "parameters": {"max_new_tokens": 250, "temperature": 0.3}
    }

    response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]['generated_text']
        return "⚠️ Respuesta inesperada del modelo."
    
    return "⚠️ Error en Hugging Face API"

# 📌 Procesar la pregunta y generar la respuesta
def ask_question(question):
    sparql_query = generate_sparql_query(question)
    graphdb_results = query_graphdb(sparql_query)

    # Convertir resultados en un solo párrafo estructurado
    context = summarize_documents(graphdb_results["results"]["bindings"]) if graphdb_results else "No hay información disponible."

    return ask_mistral(question, context)

# 📌 Interfaz en Streamlit
st.title("RAG con GraphDB y Mixtral")
st.markdown("🔎 **Pregunta sobre los [documentos almacenados](https://www.ontotext.com/products/graphdb/) y obtén respuestas con [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).**")

pregunta = st.text_area("Pregunta", placeholder="Ejemplo: ¿Qué documentos son de Lima?")
if st.button("🔎 Consultar"):
    respuesta = ask_question(pregunta)
    st.text_area("Respuesta", respuesta, height=300)
