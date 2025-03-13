import os
import requests
from dotenv import load_dotenv  

# Cargar variables de entorno
load_dotenv()
DATAVERSE_API_KEY = os.getenv("DATAVERSE_API_KEY")
DATASET_PERSISTENT_ID = "hdl:20.500.12534/RFZZNY" ## https://datos.pucp.edu.pe/dataset.xhtml?persistentId=hdl:20.500.12534/RFZZNY&version=1.0
DATAVERSE_URL = "https://datos.pucp.edu.pe"

# ruta
DATASET_PATH = "dataset"

def download_dataset():
    """Descarga el dataset desde Dataverse PUCP."""
    url = f"{DATAVERSE_URL}/api/access/dataset/:persistentId?persistentId={DATASET_PERSISTENT_ID}"
    headers = {"X-Dataverse-key": DATAVERSE_API_KEY}

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Lanza un error si la solicitud falla

        with open(DATASET_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Dataset descargado exitosamente en {DATASET_PATH}")

    except requests.exceptions.RequestException as e:
        print(f"Error al descargar dataset: {e}")

download_dataset()

import zipfile
import os

DATASET_PATH_ZIP = "dataset"  # Archivo ZIP
EXTRACTION_PATH = "dataset_extracted/"  # Carpeta de extracción

# Crear la carpeta de extracción si no existe
os.makedirs(EXTRACTION_PATH, exist_ok=True)

# Extraer el ZIP
with zipfile.ZipFile(DATASET_PATH_ZIP, "r") as zip_ref:
    zip_ref.extractall(EXTRACTION_PATH)

print(f"Archivo extraído en: {EXTRACTION_PATH}")
print("Contenido extraído:", os.listdir(EXTRACTION_PATH))

import rdflib
import pandas as pd
import urllib.parse
import chardet

# Detectar encoding del archivo antes de cargarlo
archivo_tab = "dataset_extracted/1. Lima y personajes peruanos - PUCP - IRA - Base de datos.tab"

with open(archivo_tab, "rb") as f:
    raw_data = f.read(100000)  # Leer una parte del archivo para detectar encoding
    detected_encoding = chardet.detect(raw_data)["encoding"]
    print(f"Encoding detectado: {detected_encoding}")

# Cargar el archivo asegurando UTF-8
df = pd.read_csv(archivo_tab, encoding=detected_encoding, sep="\t", on_bad_lines="skip", dtype=str)

# Crear un grafo RDF
g = rdflib.Graph()

# Definir prefijos RDF
g.bind("dc", rdflib.Namespace("http://purl.org/dc/elements/1.1/"))
g.bind("dcterms", rdflib.Namespace("http://purl.org/dc/terms/"))
g.bind("foaf", rdflib.Namespace("http://xmlns.com/foaf/0.1/"))
g.bind("ira", rdflib.Namespace("http://ira.pucp.edu.pe/ontology#"))

# Base URI
base_uri = "http://ira.pucp.edu.pe/resource/"

# Función para sanitizar URIs y evitar errores
def sanitize_uri(text):
    if pd.isna(text) or text.strip() == "":
        return None  # Ignorar valores vacíos
    text = text.strip()
    text = text.replace(" ", "_")  # Reemplazar espacios con "_"
    text = text.translate(str.maketrans({"\\": "", "\"": "", "[": "", "]": "", "/": "_"}))  # Limpiar caracteres no válidos
    return urllib.parse.quote(text)  # Codificar caracteres especiales en UTF-8

# Convertir cada fila en RDF
for _, row in df.iterrows():
    titulo_sanitizado = sanitize_uri(row.get("dc.title[es_ES]", ""))

    if titulo_sanitizado:
        doc_uri = rdflib.URIRef(base_uri + titulo_sanitizado)

        g.add((doc_uri, rdflib.RDF.type, rdflib.URIRef("http://purl.org/dc/terms/BibliographicResource")))

        # Agregar propiedades solo si no están vacías
        if row.get("dc.title[es_ES]"):
            g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/title"), rdflib.Literal(row["dc.title[es_ES]"], lang="es")))

        if row.get("dc.contributor.author"):
            g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/creator"), rdflib.Literal(row["dc.contributor.author"], lang="es")))

        # Modifica el bloque del campo 'dc.date.issued' así:
        if row.get("dc.date.issued"):
            # Convertir a string y sanitizar
            fecha_str = str(row["dc.date.issued"]).strip()  # <--- Conversión clave aquí
    
            if fecha_str and len(fecha_str) >= 4 and fecha_str[:4].isdigit():
                año = fecha_str[:4]
                g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/date"), 
                       rdflib.Literal(año, datatype=rdflib.XSD.gYear)))
            else:
                print(f"⚠️ Formato de fecha inválido: '{fecha_str}'")

        if row.get("dc.description[es_ES]"):
            g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/description"), rdflib.Literal(row["dc.description[es_ES]"], lang="es")))

        if row.get("dc.language.iso[es_ES]"):
            g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/language"), rdflib.Literal(row["dc.language.iso[es_ES]"])))

        if row.get("dc.publisher"):
            g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/publisher"), rdflib.Literal(row["dc.publisher"], lang="es")))

        if row.get("dc.subject[es_ES]"):
            g.add((doc_uri, rdflib.URIRef("http://purl.org/dc/elements/1.1/subject"), rdflib.Literal(row["dc.subject[es_ES]"], lang="es")))

# Guardar en archivo Turtle asegurando UTF-8
rdf_output = "dataset.ttl"
g.serialize(destination=rdf_output, format="turtle", encoding="utf-8")

print(f"✅ RDF guardado en UTF-8 en {rdf_output}")