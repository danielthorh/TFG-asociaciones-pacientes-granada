"""
analizar.py — Calcula métricas para la comparativa genérico vs. RAG del TFG.

SE EJECUTA EN TU VENV LOCAL DE WINDOWS, NO DENTRO DE DOCKER.
No forma parte del pipeline de producción de la app — es solo para el
análisis de datos de tu TFG, y usa dependencias (rouge-score) que no
tiene sentido meter en la imagen que usará la asociación.

Instala antes, en tu venv ya activado:
    pip install rouge-score pypdf google-generativeai python-dotenv

Requiere que ./evaluacion/manifest.csv exista (lo genera evaluar.py).
Ejecuta esto DESDE LA CARPETA src/, junto a evaluar.py:
    cd src
    python analizar.py
"""

import os
import csv
import json
import time
from pathlib import Path
from rouge_score import rouge_scorer
from pypdf import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

DOCUMENTOS_DIR = Path("./documentos")
EVALUACION_DIR = Path("./evaluacion")
MANIFIESTO = EVALUACION_DIR / "manifest.csv"
SALIDA_METRICAS = EVALUACION_DIR / "metricas.csv"
SALIDA_RUBRICA = EVALUACION_DIR / "rubrica_humana_VACIA.csv"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# El juez es una herramienta de evaluación externa, no parte del pipeline

JUEZ = genai.GenerativeModel("gemini-2.5-flash")

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

PROMPT_JUEZ_RAG = """Eres un evaluador experto en contenido de salud para asociaciones de pacientes.
Te doy un CONTEXTO recuperado de documentos reales de una asociación, y un ARTÍCULO generado a partir de ese contexto.

Evalúa en una escala de 1 a 5 (5 = mejor):
- faithfulness: ¿el artículo se mantiene fiel al contexto, sin inventar datos que lo contradigan?
- answer_relevancy: ¿el artículo responde de forma directa y útil al tema pedido?

CONTEXTO:
{contexto}

ARTÍCULO:
{articulo}

TEMA PEDIDO: {tema}

Devuelve SOLO un JSON con EXACTAMENTE estas claves en inglés, sin traducirlas,
sin texto adicional ni comillas triples:
{{"faithfulness": <1-5>, "answer_relevancy": <1-5>, "justificacion": "<una frase breve>"}}
"""

PROMPT_JUEZ_GENERICO = """Eres un evaluador experto en contenido de salud para asociaciones de pacientes.
Te doy un ARTÍCULO generado por un LLM SIN ningún contexto de documentos reales, solo con su conocimiento general.

Evalúa en una escala de 1 a 5 (5 = mejor):
- answer_relevancy: ¿el artículo responde de forma directa y útil al tema pedido?
- general_accuracy_percibida: basándote en tu propio conocimiento, ¿el contenido parece médicamente razonable y sin errores evidentes?

ARTÍCULO:
{articulo}

TEMA PEDIDO: {tema}

Devuelve SOLO un JSON con este formato exacto, sin texto adicional ni comillas triples:
{{"answer_relevancy": <1-5>, "general_accuracy_percibida": <1-5>, "justificacion": "<una frase breve>"}}
"""


def extraer_texto_asociacion(nombre_asociacion: str, cache: dict) -> str:
    """Extrae y concatena el texto de los PDFs de una asociación. Se usa
    como referencia para ROUGE: mide si el artículo se ancla en los
    documentos reales (groundedness), no es el uso clásico de ROUGE
    contra un resumen de referencia humano — déjalo claro en la memoria."""
    if nombre_asociacion in cache:
        return cache[nombre_asociacion]

    carpeta = DOCUMENTOS_DIR / nombre_asociacion
    textos = []
    for pdf_path in carpeta.glob("**/*.pdf"):
        try:
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                textos.append(page.extract_text() or "")
        except Exception as e:
            print(f"   ⚠️  Error leyendo {pdf_path.name}: {e}")

    texto_completo = "\n".join(textos)
    cache[nombre_asociacion] = texto_completo
    return texto_completo


def calcular_rouge(articulo: str, referencia: str) -> dict:
    if not referencia.strip():
        return {"rouge1": None, "rougeL": None}
    scores = scorer.score(referencia, articulo)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def juzgar_con_llm(prompt: str, max_intentos: int = 3) -> dict:
    for intento in range(max_intentos):
        try:
            respuesta = JUEZ.generate_content(prompt)
            texto = respuesta.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(texto)
        except Exception as e:
            print(f"   ⚠️  Reintentando juez LLM ({intento + 1}/{max_intentos}): {e}")
            time.sleep(2)
    return {}


def main():
    if not MANIFIESTO.is_file():
        print(f"❌ No existe {MANIFIESTO}. Tienes que correr evaluar.py (versión con manifiesto) primero.")
        return

    with open(MANIFIESTO, encoding="utf-8") as f:
        casos = list(csv.DictReader(f))

    print(f"📊 Analizando {len(casos)} artículos según el manifiesto...\n")

    cache_referencias = {}
    filas = []

    for i, caso in enumerate(casos, start=1):
        modo = caso["modo"]
        asociacion = caso["asociacion"]
        tema = caso["tema"]
        archivo = Path(caso["archivo"])

        if not archivo.is_file():
            print(f"[{i}/{len(casos)}] ⚠️  No encuentro {archivo}, lo salto.")
            continue

        articulo = archivo.read_text(encoding="utf-8")
        referencia = extraer_texto_asociacion(asociacion, cache_referencias)
        rouge = calcular_rouge(articulo, referencia)

        print(f"[{i}/{len(casos)}] Juzgando ({modo}) {archivo.name}...")

        if modo == "rag":
            juicio = juzgar_con_llm(PROMPT_JUEZ_RAG.format(
                contexto=referencia[:6000],
                articulo=articulo,
                tema=tema
            ))
            faithfulness = (
                juicio.get("faithfulness")
                or juicio.get("fidelidad")
                or juicio.get("Faithfulness")
            )
            filas.append({
                "modo": "rag",
                "archivo": archivo.name,
                "asociacion": asociacion,
                "tema": tema,
                "palabras": len(articulo.split()),
                "rouge1_vs_fuente": rouge["rouge1"],
                "rougeL_vs_fuente": rouge["rougeL"],
                "faithfulness": faithfulness,
                "answer_relevancy": juicio.get("answer_relevancy"),
                "general_accuracy_percibida": None,
                "justificacion_juez": juicio.get("justificacion"),
            })
        else:
            juicio = juzgar_con_llm(PROMPT_JUEZ_GENERICO.format(
                articulo=articulo,
                tema=tema
            ))
            filas.append({
                "modo": "generico",
                "archivo": archivo.name,
                "asociacion": asociacion,
                "tema": tema,
                "palabras": len(articulo.split()),
                "rouge1_vs_fuente": rouge["rouge1"],
                "rougeL_vs_fuente": rouge["rougeL"],
                "faithfulness": None,
                "answer_relevancy": juicio.get("answer_relevancy"),
                "general_accuracy_percibida": juicio.get("general_accuracy_percibida"),
                "justificacion_juez": juicio.get("justificacion"),
            })

    # ── Guardar métricas automáticas ──
    EVALUACION_DIR.mkdir(parents=True, exist_ok=True)
    campos = ["modo", "archivo", "asociacion", "tema", "palabras",
              "rouge1_vs_fuente", "rougeL_vs_fuente", "faithfulness",
              "answer_relevancy", "general_accuracy_percibida", "justificacion_juez"]
    with open(SALIDA_METRICAS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(filas)
    print(f"\n✅ Métricas automáticas guardadas en {SALIDA_METRICAS}")

    # ── Plantilla vacía para rúbrica humana (SummEval-style) ──
    campos_rubrica = ["modo", "archivo", "asociacion", "tema",
                       "coherencia_1a5", "factualidad_1a5", "fluidez_1a5",
                       "relevancia_1a5", "lenguaje_no_estigmatizante_1a5", "comentarios"]
    with open(SALIDA_RUBRICA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos_rubrica)
        writer.writeheader()
        for caso in casos:
            writer.writerow({
                "modo": caso["modo"],
                "archivo": caso["archivo"],
                "asociacion": caso["asociacion"],
                "tema": caso["tema"],
                "coherencia_1a5": "",
                "factualidad_1a5": "",
                "fluidez_1a5": "",
                "relevancia_1a5": "",
                "lenguaje_no_estigmatizante_1a5": "",
                "comentarios": "",
            })
    print(f"✅ Plantilla de rúbrica humana (vacía, rellénala tú) en {SALIDA_RUBRICA}")


if __name__ == "__main__":
    main()
