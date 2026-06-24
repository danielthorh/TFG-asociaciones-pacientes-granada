"""
evaluar.py — Script de evaluación para el TFG.

Genera artículos en dos configuraciones para los mismos temas:
  - GENÉRICO: llama3.2 sin contexto recuperado (RAG desactivado).
  - RAG: llama3.2 + contexto recuperado de la asociación indicada.

No toca servidor.py, interfaz.html ni generar.py. Es una pieza aparte
pensada solo para producir los datos de tu comparativa, separada de la
herramienta que usará la asociación.

Uso:
    python evaluar.py
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from datetime import datetime
import csv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")



CASOS_EVALUACION = [
    {"asociacion": "SEQUIM", "tema": "qué es la sensibilidad química múltiple", "tipo": "condicion"},
    {"asociacion": "SEQUIM", "tema": "convivir con la sensibilidad química múltiple", "tipo": "convivencia"},

    {"asociacion": "ASPRODES", "tema": "qué es la sordera o discapacidad auditiva", "tipo": "condicion"},
    {"asociacion": "ASPRODES", "tema": "convivir con un hijo sordo", "tipo": "convivencia"},

    {"asociacion": "PARKINSON", "tema": "qué es el Parkinson", "tipo": "condicion"},
    {"asociacion": "PARKINSON", "tema": "convivir con una persona con Parkinson", "tipo": "convivencia"},

    {"asociacion": "DOWN", "tema": "qué es el síndrome de Down", "tipo": "condicion"},
    {"asociacion": "DOWN", "tema": "convivir con una persona con síndrome de Down", "tipo": "convivencia"},

    {"asociacion": "ACCU", "tema": "qué es la enfermedad de Crohn", "tipo": "condicion"},
    {"asociacion": "ACCU", "tema": "convivir con la colitis ulcerosa en el día a día", "tipo": "convivencia"},

    {"asociacion": "GAEN", "tema": "qué es el Alzheimer", "tipo": "condicion"},
    {"asociacion": "GAEN", "tema": "cuidar a una persona con demencia en casa", "tipo": "convivencia"},
]

ESTRUCTURAS = {
    "convivencia": """
TÍTULO:
INTRODUCCIÓN:
¿QUÉ NECESITO SABER?:
¿QUÉ PUEDO HACER EN EL DÍA A DÍA?:
ERRORES COMUNES A EVITAR:
CONCLUSIÓN:
PALABRAS CLAVE SEO:
""",
    "condicion": """
TÍTULO:
INTRODUCCIÓN:
¿QUÉ ES?:
SÍNTOMAS O SEÑALES DE ALERTA:
¿QUÉ PUEDO HACER?:
CONCLUSIÓN:
PALABRAS CLAVE SEO:
"""
}

# Prompt RAG: incluye el contexto recuperado de la asociación.
PROMPT_RAG = """
Eres un redactor especializado en salud para asociaciones de pacientes.
Tu escritura es clara, empática y accesible para personas sin conocimientos médicos.

REGLAS IMPORTANTES:
- Usa lenguaje inclusivo y no estigmatizante en todo momento
- Evita el modelo deficitario: menciona también fortalezas y capacidades de las personas
- Presta atención a la gramática: comprueba el género de cada sustantivo antes de escribir
- Da ejemplos concretos y situaciones cotidianas, no consejos abstractos
- No uses el término "riesgo" para describir características o rasgos de una condición
- Para las palabras clave SEO incluye términos long-tail específicos y útiles

Usa este contexto de la asociación:
{context}

Escribe un artículo de blog completo sobre: {question}

Sigue EXACTAMENTE esta estructura:
{estructura}

Tono: cercano y esperanzador. Entre 400 y 600 palabras.
"""

# Prompt GENÉRICO: mismas reglas e instrucciones, pero SIN sección de contexto.
PROMPT_GENERICO = """
Eres un redactor especializado en salud para asociaciones de pacientes.
Tu escritura es clara, empática y accesible para personas sin conocimientos médicos.

REGLAS IMPORTANTES:
- Usa lenguaje inclusivo y no estigmatizante en todo momento
- Evita el modelo deficitario: menciona también fortalezas y capacidades de las personas
- Presta atención a la gramática: comprueba el género de cada sustantivo antes de escribir
- Da ejemplos concretos y situaciones cotidianas, no consejos abstractos
- No uses el término "riesgo" para describir características o rasgos de una condición
- Para las palabras clave SEO incluye términos long-tail específicos y útiles

Escribe un artículo de blog completo sobre: {question}

Sigue EXACTAMENTE esta estructura:
{estructura}

Tono: cercano y esperanzador. Entre 400 y 600 palabras.
"""


def formatear_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generar_rag(tema: str, tipo: str, asociacion: str, llm) -> str:
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"asociacion": asociacion}}
    )

    estructura = ESTRUCTURAS[tipo]
    prompt = PromptTemplate(
        input_variables=["context", "question", "estructura"],
        template=PROMPT_RAG
    )

    cadena = (
        {
            "context": retriever | formatear_docs,
            "question": RunnablePassthrough(),
            "estructura": lambda _: estructura
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return cadena.invoke(tema)


def generar_generico(tema: str, tipo: str, llm) -> str:
    estructura = ESTRUCTURAS[tipo]
    prompt = PromptTemplate(
        input_variables=["question", "estructura"],
        template=PROMPT_GENERICO
    )

    cadena = (
        {
            "question": RunnablePassthrough(),
            "estructura": lambda _: estructura
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return cadena.invoke(tema)


def guardar(carpeta: str, nombre_base: str, contenido: str) -> str:
    os.makedirs(carpeta, exist_ok=True)
    fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = f"{carpeta}/{fecha}_{nombre_base[:40].replace(' ', '_')}.txt"
    with open(nombre, "w", encoding="utf-8") as f:
        f.write(contenido)
    return nombre


def registrar_manifiesto(modo: str, asociacion: str, tema: str, tipo: str, archivo: str):
    """Registra sin truncar qué se generó, para poder mapear cada archivo
    con su tema/asociación real más adelante (los nombres de archivo se
    truncan a 40 caracteres y no son fiables para eso)."""
    ruta_manifiesto = "./evaluacion/manifest.csv"
    os.makedirs("./evaluacion", exist_ok=True)
    existe = os.path.isfile(ruta_manifiesto)
    with open(ruta_manifiesto, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["modo", "asociacion", "tema", "tipo", "archivo"])
        writer.writerow([modo, asociacion, tema, tipo, archivo])


def ejecutar_evaluacion():
    llm = OllamaLLM(model="llama3.2", temperature=0.7, base_url=OLLAMA_URL)

    for i, caso in enumerate(CASOS_EVALUACION, start=1):
        tema = caso["tema"]
        tipo = caso["tipo"]
        asociacion = caso["asociacion"]

        print(f"\n[{i}/{len(CASOS_EVALUACION)}] {asociacion} — {tema}")

        print("  → Generando GENÉRICO...")
        articulo_generico = generar_generico(tema, tipo, llm)
        ruta_generico = guardar("./evaluacion/generico", tema, articulo_generico)
        registrar_manifiesto("generico", asociacion, tema, tipo, ruta_generico)
        print(f"    guardado en {ruta_generico}")

        print("  → Generando RAG...")
        articulo_rag = generar_rag(tema, tipo, asociacion, llm)
        ruta_rag = guardar("./evaluacion/rag", f"{asociacion}_{tema}", articulo_rag)
        registrar_manifiesto("rag", asociacion, tema, tipo, ruta_rag)
        print(f"    guardado en {ruta_rag}")

    print(f"\n✅ Evaluación completada: {len(CASOS_EVALUACION)} casos procesados.")
    print("   Resultados en ./evaluacion/generico/ y ./evaluacion/rag/")


if __name__ == "__main__":
    ejecutar_evaluacion()