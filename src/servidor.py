from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

app = Flask(__name__)
CORS(app)


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

PROMPT_ARTICULO = """
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


PROMPT_REVISION = """Eres un editor experto en español especializado en contenido de salud y discapacidad.
Revisa el siguiente artículo y corrige:

1. Errores gramaticales (concordancia de género, número, tiempos verbales)
2. Frases sin sentido o mal construidas
3. Lenguaje estigmatizante o medicalizado inadecuado
4. Consejos demasiado vagos: hazlos más concretos si puedes
5. Palabras clave SEO demasiado genéricas: mejóralas con términos long-tail

No cambies la estructura ni el tono general del artículo.
Devuelve SOLO el artículo corregido, sin explicaciones ni comentarios previos.

Artículo a revisar:
{articulo}"""


def formatear_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def seleccionar_estructura(tipo: str) -> str:
    if tipo == "convivencia":
        return ESTRUCTURAS["convivencia"]
    return ESTRUCTURAS["condicion"]


def revisar_con_gemini(articulo: str) -> str:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    respuesta = model.generate_content(
        PROMPT_REVISION.format(articulo=articulo)
    )
    return respuesta.text



@app.route("/")
def index():
    return send_from_directory(".", "interfaz.html")


@app.route("/generar", methods=["POST"])
def generar():
    datos = request.json
    tema = datos.get("tema", "").strip()
    tipo = datos.get("tipo", "condicion")
    asociacion = datos.get("asociacion", "").strip()
    revisar_gemini = bool(datos.get("revisar_gemini", False))

    if not tema:
        return jsonify({"error": "El tema no puede estar vacío"}), 400

    if not asociacion:
        return jsonify({"error": "Tienes que indicar la asociación"}), 400

    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"asociacion": asociacion}}
        )
        llm = OllamaLLM(model="llama3.2", temperature=0.7, base_url=OLLAMA_URL)
        estructura = seleccionar_estructura(tipo)

        prompt = PromptTemplate(
            input_variables=["context", "question", "estructura"],
            template=PROMPT_ARTICULO
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

        # Dato de evaluación: borrador puro de llama3.2 (genérico o RAG)
        articulo_borrador = cadena.invoke(tema)

        os.makedirs("./articulos", exist_ok=True)
        fecha = datetime.now().strftime("%Y%m%d_%H%M")
        base_nombre = f"{fecha}_{tema[:30].replace(' ', '_')}"

        nombre_borrador = f"./articulos/{base_nombre}_borrador.txt"
        with open(nombre_borrador, "w", encoding="utf-8") as f:
            f.write(articulo_borrador)

        respuesta = {
            "articulo_borrador": articulo_borrador,
            "archivo_borrador": nombre_borrador,
        }

        # Post-proceso opcional con Gemini, fuera del alcance evaluado
        if revisar_gemini:
            articulo_revisado = revisar_con_gemini(articulo_borrador)
            nombre_revisado = f"./articulos/{base_nombre}_revisado_gemini.txt"
            with open(nombre_revisado, "w", encoding="utf-8") as f:
                f.write(articulo_revisado)

            respuesta["articulo_revisado_gemini"] = articulo_revisado
            respuesta["archivo_revisado"] = nombre_revisado

        return jsonify(respuesta)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=False)