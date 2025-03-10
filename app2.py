import os
import requests
import spacy
import unicodedata
import streamlit as st
from dotenv import load_dotenv  

st.set_page_config(page_title="Sistema Documental Histórico", layout="wide")

# Cargar variables de entorno
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
GRAPHDB_SERVER = "https://f469362ea5c7.ngrok.app/"
REPO_NAME = "IRA"
SPARQL_ENDPOINT = f"{GRAPHDB_SERVER}/repositories/{REPO_NAME}"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

@st.cache_resource
def load_spacy_model():
    modelo_es = "es_core_news_sm"
    if not spacy.util.is_package(modelo_es):
        os.system(f"python -m spacy download {modelo_es}")
    return spacy.load(modelo_es)

nlp = load_spacy_model()

@st.cache_data(ttl=3600)
def query_graphdb(sparql_query):
    params = {"query": sparql_query}
    headers = {"Accept": "application/sparql-results+json"}
    try:
        response = requests.get(SPARQL_ENDPOINT, params=params, headers=headers, timeout=10)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error conectando a GraphDB: {str(e)}")
        return None

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode('latin-1', 'ignore').decode('utf-8', 'ignore').strip()
    replacements = {"Ã¡": "á", "Ã©": "é", "Ã±": "ñ", "Âº": "º"}
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    return text

def generate_sparql_query(question):
    base_query = """
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT ?title ?date ?creator ?subject ?description WHERE {
        ?doc dc:title ?title .
        OPTIONAL { ?doc dc:date ?date . }
        OPTIONAL { ?doc dc:creator ?creator . }
        OPTIONAL { ?doc dc:subject ?subject . }
        OPTIONAL { ?doc dc:description ?description . }
    }
    ORDER BY DESC(?date) 
    LIMIT 30
    """
    
    if "más relevantes" in question.lower() or "comunes" in question.lower():
        return base_query.replace("LIMIT 30", "LIMIT 50")
    
    doc = nlp(question.lower())
    keywords = [ent.text for ent in doc.ents] + [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN"]]
    
    if keywords:
        filters = " || ".join([f'regex(str(?title), "{term}", "i")' for term in set(keywords)])
        return base_query.replace("WHERE {", f"WHERE {{ FILTER({filters}) ")
    
    return base_query

@st.cache_data(ttl=600)
def ask_mistral(context):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    prompt = f"""
    [INST] Como experto en historia peruana del siglo XIX, genera:
    1. Un párrafo resumen con los aspectos clave
    2. Lista de puntos importantes
    3. Contexto histórico relevante
    
    Información a analizar:
    {context}
    [/INST]
    """
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
        headers=headers,
        json={"inputs": prompt, "parameters": {"max_new_tokens": 600}}
    )
    if response.status_code == 200:
        return response.json()[0]['generated_text'].split("[/INST]")[-1].strip()
    return "Error generando el resumen"

def process_question(question):
    sparql_query = generate_sparql_query(question)
    graphdb_results = query_graphdb(sparql_query)
    
    if not graphdb_results:
        return "No se encontraron resultados. Intente con otros términos.", ""
    
    context = []
    references = []
    for result in graphdb_results['results']['bindings'][:15]:
        title = clean_text(result.get('title', {}).get('value', ''))
        date = clean_text(result.get('date', {}).get('value', ''))
        subject = clean_text(result.get('subject', {}).get('value', ''))
        description = clean_text(result.get('description', {}).get('value', ''))
        
        context.append(f"Título: {title}\nFecha: {date}\nTema: {subject}\nDescripción: {description[:200]}...")
        references.append(f"- {title} ({date}) - {subject}")
    
    return ask_mistral("\n\n".join(context)), "\n".join(references)

def generate_suggested_questions():
    sparql_query = """
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    SELECT DISTINCT ?subject (COUNT(?doc) as ?count) WHERE {
        ?doc dc:subject ?subject 
    } GROUP BY ?subject ORDER BY DESC(?count) LIMIT 8
    """
    results = query_graphdb(sparql_query)
    
    if not results or 'results' not in results:
        return [
            "¿Cuáles son los documentos más relevantes?",
            "¿Qué temas históricos son más comunes?"
        ]
    
    questions = []
    templates = [
        "¿Qué información existe sobre {subject}?",
        "¿Cómo se aborda el tema de {subject}?",
        "¿Qué documentos mencionan {subject}?",
        "¿Qué autores escribieron sobre {subject}?"
    ]
    
    for i, item in enumerate(results['results']['bindings']):
        subject = clean_text(item['subject']['value'])
        count = item['count']['value']
        questions.append({
            "text": templates[i % len(templates)].format(subject=subject),
            "key": f"q_{hash(subject)}_{i}"
        })
        if len(questions) >= 6:
            break
            
    return questions

# Interfaz mejorada
st.title("Análisis de Documentos Históricos")

st.markdown("### Preguntas sugeridas:")
suggested_questions = generate_suggested_questions()
cols = st.columns(3)
for i, q in enumerate(suggested_questions):
    with cols[i % 3]:
        st.button(
            q["text"],
            key=q["key"],
            on_click=lambda t=q["text"]: st.session_state.update({"pregunta": t})
        )

pregunta = st.text_input("Escribe tu pregunta:", value=getattr(st.session_state, 'pregunta', ''))
if st.button("Analizar documentos"):
    if pregunta:
        with st.spinner('Buscando y analizando...'):
            resumen, referencias = process_question(pregunta)
            
            st.markdown("## Resumen analítico")
            st.markdown(f"""
            <div style='
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                margin-bottom: 2rem;
            '>
            {resumen}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("## Documentos relacionados")
            st.markdown(referencias)
    else:
        st.warning("Por favor ingresa una pregunta")