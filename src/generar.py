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


def seleccionar_estructura(tema: str) -> str:
    """Elige la estructura según el tipo de tema introducido."""
    tema_lower = tema.lower()
    palabras_convivencia = ["convivir", "convivencia", "relacionarse", "vivir con", "entender a", "apoyar a"]

    if any(p in tema_lower for p in palabras_convivencia):
        print("📋 Estructura seleccionada: convivencia")
        return ESTRUCTURAS["convivencia"]
    else:
        print("📋 Estructura seleccionada: condición médica")
        return ESTRUCTURAS["condicion"]


def revisar_con_gemini(articulo: str) -> str:
    """Post-proceso editorial fuera del alcance evaluado del TFG. No usar
    su salida como dato de la comparativa genérico vs. RAG."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    respuesta = model.generate_content(
        PROMPT_REVISION.format(articulo=articulo)
    )
    return respuesta.text


def generar_articulo(tema: str, asociacion: str, revisar_gemini: bool = False):
    """
    revisar_gemini=False (por defecto): el dato que se guarda y se evalúa
    es el borrador puro de llama3.2 (genérico o RAG, según uses retriever).
    revisar_gemini=True: además se genera una versión revisada con Gemini,
    guardada en un fichero aparte, claramente etiquetada como fuera de la
    comparativa.
    asociacion: nombre exacto de la subcarpeta en ./documentos/ (debe
    coincidir con metadata["asociacion"] generado por indexar.py).
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # Filtra el contexto recuperado para que solo venga de la asociación elegida
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"asociacion": asociacion}}
    )
    llm = OllamaLLM(model="llama3.2", temperature=0.7, base_url=OLLAMA_URL)

    estructura = seleccionar_estructura(tema)

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

    # 1. Generar con llama3.2 — este es el dato de la evaluación
    print(f"\n✍️  Generando artículo sobre: {tema}\n")
    articulo_borrador = cadena.invoke(tema)
    print("📄 Borrador generado (llama3.2).")

    os.makedirs("./articulos", exist_ok=True)
    fecha = datetime.now().strftime("%Y%m%d_%H%M")
    base_nombre = f"{fecha}_{tema[:30].replace(' ', '_')}"

    nombre_borrador = f"./articulos/{base_nombre}_borrador.txt"
    with open(nombre_borrador, "w", encoding="utf-8") as f:
        f.write(articulo_borrador)

    print(articulo_borrador)
    print(f"\n💾 Borrador (dato de evaluación) guardado en: {nombre_borrador}")

    # 2. Revisión opcional con Gemini — NUNCA usar como dato de evaluación
    if revisar_gemini:
        print("\n🔍 Revisando con Gemini (post-proceso fuera del alcance del TFG)...\n")
        articulo_revisado = revisar_con_gemini(articulo_borrador)

        nombre_revisado = f"./articulos/{base_nombre}_revisado_gemini.txt"
        with open(nombre_revisado, "w", encoding="utf-8") as f:
            f.write(articulo_revisado)

        print(articulo_revisado)
        print(f"\n💾 Versión revisada (NO es dato de evaluación) guardada en: {nombre_revisado}")


def elegir_asociacion() -> str:
    """Lista las subcarpetas de ./documentos/ y deja elegir una por número,
    para evitar errores de tecleo que no coincidan con metadata["asociacion"]."""
    base = "./documentos"
    carpetas = sorted(
        nombre for nombre in os.listdir(base)
        if os.path.isdir(os.path.join(base, nombre))
    )
    print("\nAsociaciones disponibles:")
    for i, nombre in enumerate(carpetas, start=1):
        print(f"  {i}. {nombre}")

    while True:
        eleccion = input("Elige el número de la asociación → ").strip()
        if eleccion.isdigit() and 1 <= int(eleccion) <= len(carpetas):
            return carpetas[int(eleccion) - 1]
        print("Número no válido, intenta de nuevo.")


if __name__ == "__main__":
    tema = input("¿Sobre qué tema quieres el artículo? → ")
    asociacion = elegir_asociacion()
    quiere_revision = input("¿Quieres también una versión revisada con Gemini? (s/n) → ").strip().lower() == "s"
    generar_articulo(tema, asociacion, revisar_gemini=quiere_revision)