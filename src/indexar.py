from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import sys
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DOCUMENTOS_DIR = "./documentos/"


def cargar_documentos_por_asociacion(solo_asociacion: str = None):
    """
    Carga los PDFs recorriendo cada subcarpeta de ./documentos/ por separado
    y etiqueta cada documento con metadata["asociacion"] = nombre de la
    subcarpeta. Si se pasa solo_asociacion, carga únicamente esa carpeta.
    """
    documentos = []

    if not os.path.isdir(DOCUMENTOS_DIR):
        raise FileNotFoundError(f"No existe la carpeta {DOCUMENTOS_DIR}")

    subcarpetas = sorted(
        nombre for nombre in os.listdir(DOCUMENTOS_DIR)
        if os.path.isdir(os.path.join(DOCUMENTOS_DIR, nombre))
    )

    if solo_asociacion:
        if solo_asociacion not in subcarpetas:
            raise ValueError(
                f"'{solo_asociacion}' no existe en {DOCUMENTOS_DIR}. "
                f"Carpetas disponibles: {', '.join(subcarpetas)}"
            )
        subcarpetas = [solo_asociacion]

    if not subcarpetas:
        print(f"⚠️  No se encontraron subcarpetas de asociación dentro de {DOCUMENTOS_DIR}")

    for nombre_asociacion in subcarpetas:
        ruta_asociacion = os.path.join(DOCUMENTOS_DIR, nombre_asociacion)
        loader = DirectoryLoader(
            ruta_asociacion,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        docs_asociacion = loader.load()

        for doc in docs_asociacion:
            doc.metadata["asociacion"] = nombre_asociacion

        print(f"   → {nombre_asociacion}: {len(docs_asociacion)} páginas")
        documentos.extend(docs_asociacion)

    return documentos


def indexar_documentos(solo_asociacion: str = None):
    print("📄 Cargando documentos por asociación...")
    documentos = cargar_documentos_por_asociacion(solo_asociacion)
    print(f"   → TOTAL: {len(documentos)} páginas cargadas")

    print("✂️  Dividiendo en chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100  
    )
    # split_documents conserva metadata (incluido "asociacion") en cada chunk
    chunks = splitter.split_documents(documentos)
    print(f"   → {len(chunks)} chunks creados")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)

    if solo_asociacion:

        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        print(f"🗑️  Borrando chunks anteriores de '{solo_asociacion}'...")
        vectorstore.delete(where={"asociacion": solo_asociacion})

        print("🔢 Creando embeddings y añadiendo a ChromaDB...")
        if chunks:
            vectorstore.add_documents(chunks)
    else:
        print("🔢 Creando embeddings y guardando en ChromaDB (todas las asociaciones)...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    print("✅ Indexación completada (cada chunk lleva metadata['asociacion'])")


if __name__ == "__main__":

    asociacion_arg = sys.argv[1] if len(sys.argv) > 1 else None
    indexar_documentos(asociacion_arg)