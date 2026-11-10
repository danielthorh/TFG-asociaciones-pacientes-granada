# Generador de artículos con IA para PAPGRA

Sistema RAG (*Retrieval-Augmented Generation*) **100 % local** que genera artículos divulgativos para las asociaciones de pacientes de **PAPGRA** (Federación Granadina de Asociaciones de Pacientes), a partir de la documentación propia de cada asociación.

Proyecto desarrollado como Trabajo Fin de Grado del Grado en Información y Contenidos Digitales — <!-- TODO: universidad --> — bajo la tutela de Ginés <!-- TODO: apellidos -->.

> **Coste recurrente: 0 €.** Los modelos se ejecutan en local con Ollama y los documentos de las asociaciones nunca salen de la máquina (salvo que se active la revisión editorial opcional con Gemini).

---

## Índice

- [Motivación](#motivación)
- [Arquitectura](#arquitectura)
- [Stack](#stack)
- [Instalación](#instalación)
- [Uso](#uso)
- [Configuración](#configuración)
- [Evaluación y resultados](#evaluación-y-resultados)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Líneas de mejora](#líneas-de-mejora)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Licencia](#licencia)

---

## Motivación

PAPGRA agrupa **31 asociaciones de pacientes** con poco personal y sin presupuesto para herramientas de IA de pago. Necesitaban producir contenido divulgativo fiable sobre sus patologías sin enviar documentación interna a servicios externos. Este sistema:

- Genera artículos anclados en los documentos de cada asociación (no en el conocimiento genérico del modelo).
- Funciona sin conexión y sin suscripciones.
- Se distribuye como contenedores Docker que el personal puede levantar con Docker Desktop.

## Arquitectura

```
                ┌──────────────────────────┐
 Documentos ──► │  Indexado (LangChain)    │
 por asociación │  chunk 500 / overlap 100 │
                └────────────┬─────────────┘
                             │ nomic-embed-text
                             ▼
                ┌──────────────────────────┐
                │  ChromaDB (1 instancia,  │
                │  metadato = asociación)  │
                └────────────┬─────────────┘
                             │ top-k = 5, filtrado por asociación
 Petición  ──►  ┌────────────▼─────────────┐
 (tema +        │  Flask (servidor.py)     │
 asociación)    │  + Llama 3.2 3B (Ollama) │
                └────────────┬─────────────┘
                             │ (opcional)
                             ▼
                ┌──────────────────────────┐
                │ Revisión editorial       │
                │ Gemini 2.5 Flash         │
                └────────────┬─────────────┘
                             ▼
                         Artículo
```

Las 31 asociaciones conviven en **una sola colección de ChromaDB**; cada fragmento lleva un metadato con su asociación y la recuperación se filtra por él. Evita mantener 31 bases vectoriales.

## Stack

| Componente | Tecnología |
|---|---|
| LLM de generación | Llama 3.2 3B vía [Ollama](https://ollama.com) |
| Embeddings | `nomic-embed-text` (Ollama) |
| Base vectorial | ChromaDB |
| Orquestación RAG | LangChain |
| Backend | Flask |
| Revisión editorial (opcional) | Gemini 2.5 Flash |
| Despliegue | Docker Compose (servicios `ollama` + `app`) |

## Instalación

### Requisitos

- Docker y Docker Compose (o Docker Desktop en Windows/macOS)
- ~8 GB de RAM libres <!-- TODO: verificar -->
- GPU NVIDIA opcional (acelera la inferencia, no es imprescindible)

### Pasos

```bash
git clone https://github.com/<!-- TODO: usuario -->/<!-- TODO: repo -->.git
cd <!-- TODO: repo -->

# Copia y edita las variables de entorno
cp .env.example .env

# Levanta Ollama y la aplicación
docker compose up -d

# Descarga los modelos dentro del contenedor de Ollama
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
```

La interfaz queda disponible en `http://localhost:5000` <!-- TODO: confirmar puerto -->.

## Uso

1. Coloca los documentos de cada asociación en `<!-- TODO: carpeta -->`.
2. Indexa el corpus:
   ```bash
   <!-- TODO: comando de indexado, p. ej. docker compose exec app python indexar.py -->
   ```
3. Abre la interfaz web, elige asociación y tema, y genera el artículo.
4. (Opcional) Activa la revisión editorial con Gemini si has configurado `GEMINI_API_KEY`.

## Configuración

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `OLLAMA_HOST` | URL del servidor Ollama | `http://ollama:11434` |
| `GEMINI_API_KEY` | Clave para la revisión editorial (opcional) | — |

Parámetros RAG utilizados en el TFG:

| Parámetro | Valor |
|---|---|
| `chunk_size` | 500 |
| `chunk_overlap` | 100 |
| `k` (fragmentos recuperados) | 5 |

> **Nota sobre Docker:** `OllamaEmbeddings` y `OllamaLLM` deben recibir `base_url` explícito leído de `OLLAMA_HOST`. Sin él, LangChain apunta a `localhost:11434`, que dentro del contenedor de la app no es Ollama, y todo falla. Ya está corregido en `servidor.py`.

## Evaluación y resultados

Se seleccionaron **6 asociaciones × 12 temas × 2 configuraciones** (RAG vs. generación genérica sin contexto) = **24 artículos** evaluados.

| Métrica | Diferencia RAG − genérico |
|---|---|
| Factualidad | **+0,79** |
| Fluidez | **+0,93** |
| Relevancia | **−0,58** |
| ROUGE | Prácticamente idéntico |
| Fidelidad (*faithfulness*) RAG | 2,00 / 5 |

**Lectura honesta:** el RAG hace los artículos más correctos y mejor redactados, pero **menos relevantes** al tema pedido. La causa principal identificada es la calidad de la recuperación: el indexador incorporó índices y tablas de contenido de los documentos, que se recuperan como "contexto" sin aportar información. Eso también explica la fidelidad baja.

## Limitaciones conocidas

- Fidelidad baja (2/5) por fragmentos no sustantivos (índices, tablas de contenido) en el corpus indexado.
- Llama 3.2 3B es un modelo pequeño: elegido por poder ejecutarse en equipos modestos, a costa de calidad.
- Evaluación sobre 24 artículos: suficiente para detectar tendencias, no para conclusiones estadísticamente robustas.
- La revisión con Gemini rompe la premisa de "todo local"; por eso es opcional.
- El despliegue en la nube (Oracle Cloud) se descartó por falta de capacidad; la distribución actual es local vía Docker Desktop.

## Líneas de mejora

- Filtrar contenido no sustantivo (índices, portadas, tablas de contenido) antes de indexar.
- *Chunking* semántico en lugar de por número de caracteres.
- Enriquecer el corpus de cada asociación.
- Probar modelos de embeddings más potentes.
- Validar el contexto recuperado antes de generar (descartar recuperaciones irrelevantes).

## Estructura del repositorio

<!-- TODO: ajustar a la estructura real -->
```
.
├── servidor.py          # Backend Flask + pipeline RAG
├── docker-compose.yml   # Servicios ollama + app
├── Dockerfile
├── requirements.txt
├── .env.example
└── ...
```

## Licencia

<!-- TODO: elegir licencia (MIT, GPL-3.0…) y añadir el archivo LICENSE -->

---

**Autor:** <!-- TODO: nombre --> · Trabajo Fin de Grado, <!-- TODO: año y universidad -->
