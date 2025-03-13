import json
import faiss
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_API_TOKEN")

# Cargar el JSON-LD con los datos de las lenguas
json_file_path = "grambank_simple.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    json_ld_data = json.load(f)

# Filtrar solo las entradas que sean lenguas
languages = [entry for entry in json_ld_data if isinstance(entry, dict) and "http://purl.org/linguistics#Language" in entry.get("@type", [])]

# Extraer información textual de las lenguas para embeddings
documents = []
for entry in languages:
    label = entry.get("http://www.w3.org/2000/01/rdf-schema#label", [{}])[0].get("@value", "")
    glottocode = entry.get("http://purl.org/linguistics#glottocode", [{}])[0].get("@value", "")
    if label:  # Asegurar que el label no esté vacío
        description = f"Lengua: {label}, Glottocode: {glottocode}"
        documents.append(description)

# Inicializar el modelo de embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convertir los documentos en embeddings
vectorstore = FAISS.from_texts(documents, embedding_model)

# Guardar la base de datos FAISS
faiss_db_path = "quechua_rag.index"
vectorstore.save_local(faiss_db_path)

# Cargar modelo Mixtral-8x7B-Instruct de Hugging Face con optimización
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_API_KEY, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=HUGGINGFACE_API_KEY,
    trust_remote_code=True
)

# Crear pipeline de generación de texto con optimización
mixtral_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Integrar Mixtral con LangChain
llm = HuggingFacePipeline(pipeline=mixtral_pipeline)

# Configurar el sistema RAG
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# Ejemplo de pregunta
query = "Describe las lenguas Quechua"
response = qa_chain.run(query)
print(response)