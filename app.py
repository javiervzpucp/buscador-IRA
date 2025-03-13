import os
import json
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
METADATA_FILE = "metadata.json"

# Cargar grafo RDF desde archivo local
g = rdflib.Graph()
g.parse(RDF_FILE, format="turtle")

# Cargar metadatos del JSON
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# 📌 Extraer categorías clave del JSON
def get_relevant_categories():
    citation_fields = metadata["datasetVersion"]["metadataBlocks"]["citation"]["fields"]
    keywords = [
        kw["keywordValue"]["value"]
        for field in citation_fields if field["typeName"] == "keyword"
        for kw in field["value"]
    ]
    return keywords if keywords else ["Historia", "Fotografía", "Lima"]

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

# 📌 Generar un resumen en formato de texto
def generate_summary_text(documents, year_filter=None):
    if not documents:
        return f"No se encontraron documentos para el año {year_filter}."

    categories = get_relevant_categories()
    summary_text = f"Se han encontrado {len(documents)} documentos correspondientes al año {year_filter}. "

    categorized_docs = {category: 0 for category in categories}
    uncategorized_count = 0

    for doc in documents:
        subject = doc["subject"]
        assigned = False
        for category in categories:
            if category.lower() in subject.lower():
                categorized_docs[category] += 1
                assigned = True
                break
        if not assigned:
            uncategorized_count += 1

    for category, count in categorized_docs.items():
        if count > 0:
            summary_text += f"En la categoría de {category}, se encontraron {count} documentos. "

    if uncategorized_count > 0:
        summary_text += f"Además, hay {uncategorized_count} documentos sin una categoría específica."

    return summary_text.strip()

# 📌 Generar la lista detallada de documentos sin modificar el formato previo
def generate_detailed_list(documents, year_filter=None):
    if not documents:
        return f"**No se encontraron documentos para el año {year_filter}.**"

    summary = f"## Documentos detallados para el año {year_filter}\n\n"
    grouped_docs = {}

    for doc in documents:
        title = doc["title"]
        subject = doc["subject"]
        date = doc["date"]
        creator = doc["creator"]

        if subject not in grouped_docs:
            grouped_docs[subject] = []
        grouped_docs[subject].append(f"- **{title}** ({date}) - Autor: {creator}")

    for subject, docs in grouped_docs.items():
        summary += f"### {subject}\n" + "\n".join(docs) + "\n\n"

    return summary.strip()

# 📌 Generar respuesta con Mixtral
def ask_mistral(question, summary_text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    data = {
        "inputs": f"Pregunta: {question}\nInformación relevante:\n{summary_text}",
        "parameters": {"max_new_tokens": 250, "temperature": 0.3}
    }
    response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=data)
    return response.json()[0].get("generated_text", "Error en la generación de respuesta.") if response.status_code == 200 else "Error en Hugging Face API"

# 📌 Procesar la pregunta y generar la respuesta en tres partes
def ask_question(question):
    year = extract_year(question)
    sparql_query = generate_sparql_query(question)
    rdf_results = query_rdf(sparql_query)
    
    # Generar resumen en formato de texto
    summary_text = generate_summary_text(rdf_results, year)
    
    # Generar respuesta con Mixtral basada en el resumen
    mixtral_response = ask_mistral(question, summary_text)
    
    # Generar la lista detallada de documentos
    detailed_list = generate_detailed_list(rdf_results, year)
    
    return summary_text, mixtral_response, detailed_list

# ------------------ Interfaz Streamlit ------------------

# **Encabezado con descripción**
st.title("RAG con RDF y Mixtral")
st.markdown(
    """
    **Explora documentos históricos utilizando RDF y [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)**  
    - **Fuente de datos:** [PUCP - Instituto Riva-Agüero](https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/RFZZNY&version=1.0)  
    - **Realiza consultas en lenguaje natural sobre documentos históricos.**
    """
)

# **Campo de pregunta**
pregunta = st.text_area("Escribe tu pregunta:", placeholder="Ejemplo: ¿Qué documentos son de 1906?")

# **Botón para consultar**
if st.button("Buscar"):
    with st.spinner("Buscando información..."):
        summary_text, mixtral_response, detailed_list = ask_question(pregunta)
        st.markdown(f"### Resumen de los Documentos\n\n{summary_text}")
        st.markdown(f"### Respuesta Generada con Mixtral\n\n{mixtral_response}")
        st.markdown(f"## Respuesta Detallada\n\n{detailed_list}")
