# klaro

# **An E2E pipeline for generating educational videos using AI and automation**

This project generates educational videos from web sources using a **knowledge graph + RAG pipeline** with narration and subtitles.

---

## Project Structure

```
├── pipeline.py                 # Main pipeline
├── {query}/                    # Cached json files
├── requirements.txt            # Python dependencies
├── .env                        # API keys

```

---

## Features

- **Knowledge Graph Construction:** Builds hierarchical nodes from web sources based on a query.
- **RAG Retrieval:** Retrieves relevant content using embeddings from `SentenceTransformer`.
- **Narration Script Generation:** Uses OpenAI API to produce structured slide content.
- **Audio Generation:** Converts narration to speech with **ElevenLabs** with word-level timestamps.
- **Caching:** Stores knowledge graphs, scripts, and audio locally to reduce redundant API calls.

---

## Workflow

```mermaid
flowchart TD
    A[Start: User Query] --> B[Web Search via Tavily API]
    B --> C[Extract & Clean Content from URLs]
    C --> D[Summarization using BART]
    D --> E[Keyword Extraction via KeyBERT]
    E --> F[Build Knowledge Graph Node hierarchy]
    F --> G[Flatten Graph & Generate Embeddings]
    G --> H[Retrieve Context via RAG]
    H --> I[Generate Question Set and Narration Script using OpenAI API]
    I --> J[Text-to-Speech via ElevenLabs]
    J --> K[Save Audio Locally]
    K --> L[Generate Timestamped Slide Content using Normalized Alignment from the Audio]

```

> The diagram above shows the end-to-end pipeline from a query to the final output.
> 

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rachitbhandarii/klaro.git
cd klaro
```

2. Set up a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add environment variables in `.env`:

```
OPENAI_API_KEY=<your_openai_api_key>
ELEVENLABS_API_KEY=<your_elevenlabs_api_key>
TAVILY_API_KEY=<your_tavily_api_key>
```
---

## Usage

1. Set your topic in `pipeline.py`:

```python
query = "Russia leaves Nuclear Arms Treaty with US"
```

2. Run the main script:

```bash
python pipeline.py
```

---
