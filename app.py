import os
import requests
import spacy
import unicodedata
import streamlit as st
import rdflib
import re
from dotenv import load_dotenv  
from rdflib.plugins.sparql import prepareQuery

# ------------------ Configuración inicial ------------------
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
RDF_FILE = "dataset.ttl"

# Cargar grafo RDF desde archivo local
g = rdflib.Graph()
g.parse(RDF_FILE, format="turtle")

# Cargar modelo spaCy para análisis de texto
modelo_es = "es_core_news_sm"
if not spacy.util.is_package(modelo_es):
    os.system(f"python -m spacy download {modelo_es}")
nlp = spacy.load(modelo_es)

# 📌 Función para consultar el grafo RDF local con SPARQL
def query_rdf(sparql_query):
    try:
        q = prepareQuery(sparql_query)
        results = g.query(q)
        return [{
            "title": str(row.title),
            "date": str(row.date) if row.date else "Fecha desconocida",
            "creator": str(row.creator) if row.creator else "Autor desconocido",
            "subject": str(row.subject) if row.subject else "Sin tema"
        } for row in results]
    except Exception as e:
        print(f"Error en consulta SPARQL: {e}")
        return []

# 📌 Función para limpiar caracteres mal codificados en UTF-8
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).strip()
    replacements = {
        "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
        "Ã±": "ñ", "Ã": "Á", "Ã‰": "É", "Ã": "Í", "Ã“": "Ó",
        "Ãš": "Ú", "Ã‘": "Ñ", "â€“": "–", "â€”": "—", "â€œ": "“",
        "â€�": "”", "â€¢": "•", "â€¦": "…", "Âº": "º", "Âª": "ª",
        "Ã¼": "ü", "Ãœ": "Ü"
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

# 📌 Extraer entidades clave de la pregunta
def extract_year(question):
    match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", question)
    return match.group(0) if match else None

# 📌 Generar consulta SPARQL
def generate_sparql_query(question):
    year = extract_year(question)
    sparql_query = """
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT ?title ?date ?creator ?subject WHERE {
        ?doc dc:title ?title .
        OPTIONAL { ?doc dc:date ?date }
        OPTIONAL { ?doc dc:creator ?creator }
        OPTIONAL { ?doc dc:subject ?subject }
    """
    if year:
        sparql_query += f'\n    FILTER(?date = "{year}"^^<http://www.w3.org/2001/XMLSchema#gYear>)'
    
    sparql_query += "\n} LIMIT 20"
    return sparql_query.strip()

# 📌 Resumir documentos obtenidos del grafo RDF
def summarize_documents(documents, year_filter=None):
    if not documents:
        return f"**No se encontraron documentos para el año {year_filter}.**"

    summary = f"## Documentos del año {year_filter}\n\n"
    grouped_docs = {}

    for doc in documents:
        title = clean_text(doc["title"])
        subject = clean_text(doc["subject"])
        date = clean_text(doc["date"])
        creator = clean_text(doc["creator"])

        # Agrupar por tema
        if subject not in grouped_docs:
            grouped_docs[subject] = []
        grouped_docs[subject].append(f"- **{title}** ({date}) - Autor: {creator}")

    # Construir el resumen ordenado por tema
    for subject, docs in grouped_docs.items():
        summary += f"### {subject}\n" + "\n".join(docs) + "\n\n"

    return summary.strip()

# 📌 Consultar Hugging Face API con Mixtral
def ask_mistral(question, context):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {
        "inputs": f"Pregunta: {question}\nInformación relevante:\n{context}",
        "parameters": {"max_new_tokens": 250, "temperature": 0.3}
    }
    response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=data)
    return response.json()[0].get("generated_text", "Error en la generación de respuesta.") if response.status_code == 200 else "Error en Hugging Face API"

# 📌 Procesar la pregunta y generar la respuesta
def ask_question(question):
    year = extract_year(question)
    sparql_query = generate_sparql_query(question)
    rdf_results = query_rdf(sparql_query)
    context = summarize_documents(rdf_results, year) if rdf_results else "**No hay información disponible.**"
    return ask_mistral(question, context)

# ------------------ Interfaz Streamlit ------------------

# **Encabezado con descripción**
st.title("Explora documentos históricos utilizando RAG y [Mixtral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)")
st.markdown(
    """
    - **Fuente de datos:** [Lima y personajes peruanos, PUCP - Instituto Riva-Agüero](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/RFZZNY&version=1.0)  
    - **Realiza consultas en lenguaje natural sobre documentos históricos.**
    """
)

# **Campo de pregunta**
pregunta = st.text_area("Escribe tu pregunta:", placeholder="Ejemplo: ¿Qué documentos son de 1906?")

# **Botón para consultar**
if st.button("Buscar"):
    with st.spinner("Buscando información..."):
        respuesta = ask_question(pregunta)
        # **Mostrar respuesta en Markdown**
        st.markdown(f"## Respuesta Generada\n\n{respuesta}")
