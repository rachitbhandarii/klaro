from tavily import TavilyClient
import os
import json
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Any
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from litellm import completion
from pydantic import BaseModel
from langchain.schema import Document
import numpy as np
from elevenlabs.client import ElevenLabs
import base64
from pydub import AudioSegment
import io

# Util methods

# Load environment variables
load_dotenv()

# ............................................................................

query = os.getenv("VIDEO_TOPIC").__str__()
if not query:
    raise ValueError("VIDEO_TOPIC environment variable not set!")

# ............................................................................

# Helper methods

def safe_filename(query: str) -> str:
    # Replace non-alphanumeric characters with dashes
    return re.sub(r'[^a-zA-Z0-9]+', '-', query.lower()).strip('-')

def get_filepath(name: str, results_dir: str, extension: str = "json") -> str:
    path_to_default_dir = os.path.join("public", safe_filename(query))
    results_dir = os.path.join(path_to_default_dir, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{safe_filename(name)}.{extension}"
    filepath = os.path.join(results_dir, filename)
    
    return filepath

def save_audio(base64_str: str, filepath: str, silence_duration: int = 1000):
    audio_bytes = base64.b64decode(base64_str)
    sound = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    silence = AudioSegment.silent(duration=silence_duration)
    modified = silence + sound
    modified.export(filepath, format="mp3")
    return modified.duration_seconds

# ............................................................................

# AI generation methods and classes for structured response

class Outline(BaseModel):
    topic: str
    subtopics: list[str]

class Outlines(BaseModel):
    outline: list[Outline]

class NarrationItem(BaseModel):
    questions: list[str]
    content: str

class Content(BaseModel):
    point: str
    start_time: float

class SlideContent(BaseModel):
    slide: list[Content]

def get_structured_response(sys_msg: str, prompt: str, format: BaseModel ,retry: int = 1):

    response = completion(
        model="azure/gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": format.__name__.lower(),
                "schema": format.model_json_schema()
            }
        }
    )

    if (response is None):
        if retry > 3:
            raise Exception("Failed to generate output after multiple retries")
        else:
            print(f"Retrying outline generation... Attempt {retry + 1}")
            return get_structured_response(sys_msg, prompt, format, retry + 1)
        
    return json.loads(response.choices[0].message.content)

# ............................................................................

# Function to create an outline for a video based on a query

def create_outline(query: str, graph_summary: str) -> List[Dict[str, Any]]:

    filepath = get_filepath(query, "narration-outline")
    
    narration_outline = []
    
    if os.path.exists(filepath):
        print("Outline already exists. Loading from file...")
        with open(filepath, "r", encoding="utf-8") as f:
                narration_outline = json.load(f)
        return narration_outline

    sys_msg = "You are a pedagogy specialist. We want to create an informative video on current affairs for UPSC aspirants. Create a structured outline for such a video by eliminiating redundancies (a single subtopic for a single keyword) and unrelated content (not related to current affairs and UPSC preparation) as well as restructuring the order of topics and subtopics (no topic/subtopic should be repeated or related to another topic/subtopic). Each Topic's Subtopics should be linked to the topic. Create an extensive outline (go into the depth)."
    narration_outline = get_structured_response(sys_msg, "\n".join(graph for graph in graph_summary), Outlines)["outline"]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(narration_outline, f, ensure_ascii=False, indent=2)

    return narration_outline

# ............................................................................

# Keyword extraction using KeyBERT and SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
kw_model = KeyBERT(model=embedding_model)

def extract_keywords(text: str) -> str:
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 4),
        stop_words="english",
        top_n=1  # extract more first
    )
    keywords = [kw for kw, score in keywords]
    return keywords[0]

# ............................................................................

# Summarization using Hugging Face Transformers

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def summarize_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    word_count = len(text.split())
    if word_count < 30:
        return text

    # Tokenizer-level max length for BART = 1024
    max_input_tokens = 900  

    # Dynamically set output lengths
    max_len = min(200, max(50, int(word_count * 0.5)))
    min_len = min(60, max(20, int(word_count * 0.25)))

    try:
        # Instead of passing raw text directly, tokenize first
        inputs = summarizer.tokenizer(
            text,
            max_length=max_input_tokens,
            truncation=True,
            return_tensors="pt"
        ).to(summarizer.model.device)

        summary_ids = summarizer.model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            do_sample=False
        )
        return summarizer.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print("Summarization failed:", e)
        return text


# ............................................................................

# Web scraping and content extraction

# Initialize a session with retry logic
# Use requests with retry logic to handle transient errors
# To avoid issues with rate limiting or temporary server errors.
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0 Safari/537.36"
    )
}

# We can scrape the content of these URLs and extract the relevant information

def extract_main_content(url: str, chunk_size: int = 800, overlap: int = 50) -> List[str]:

    try:
        html = session.get(url, headers=headers, timeout=10)
        html.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return []
    
    soup = BeautifulSoup(html.text, "html.parser")
    # Remove scripts and styles
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(paragraphs)

    if not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split into words
    words = text.split()

    # Chunk with overlap
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    if (len(chunks) > 20):
        return chunks[:20]

    return chunks

# ............................................................................

# Function to perform web search using Tavily API

def web_search(query: str) -> List[str]:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    filepath = get_filepath(query, "web-search-results")

    # Check cache first
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            response = json.load(f)
    else:
        response = tavily.search(query, num_results=3, language="en")
        with open(filepath, "w") as f:
            json.dump(response, f, indent=2)

    urls = [result["url"] for result in response["results"][:3]]
    return urls

# ............................................................................

# Classes for building a Knowledge Graph against a query

@dataclass
class Node:
    def __init__(self, content: str, summary: str = None, keywords: str = None, url: str = None, level: int = None):
        self.summary = summary
        self.keywords = keywords or ""
        self.level = level if level is not None else 1
        self.content = content
        self.url = url
        self.children = []
    
    def to_dict(self):
        return {
            "content": self.content,
            "summary": self.summary,
            "keywords": self.keywords,
            "url": self.url,
            "level": self.level,
            "children": [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        node = cls(
            content=data["content"],
            summary=data.get("summary"),
            keywords=data.get("keywords", ""),
            url=data.get("url"),
            level=data.get("level", 1)
            
        )
        for child_data in data.get("children", []):
            child_node = cls.from_dict(child_data)
            node.children.append(child_node)
        return node
    
class KnowledgeGraphBuilder:
    def __init__(self, query: str, maxLevel: int = 3):
        self.maxLevel = maxLevel
        self.src: Node = None

        if query:  # only build if query is given
            self.src = Node(content=query, keywords=query, summary=query)
            self.build_graph(self.src)
    
    def build_graph(self, parent : Node = None) -> None:
        if parent is None:
            parent = self.src
        
        urls = web_search(parent.keywords)

        for url in urls:

            contents = extract_main_content(url)
            print(len(contents), "contents found for", parent.keywords)

            for content in contents:
                child = Node(content=content, url=url, level=parent.level + 1)
                child.summary = summarize_text(content)
                print("Summary for child:", child.summary)
                child.keywords = extract_keywords(child.summary)
                print("Keywords for child:", child.keywords)
                parent.children.append(child)
                if child.level < self.maxLevel:
                    self.build_graph(child)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.src.to_dict() if self.src else {}

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraphBuilder":
        kg = cls(query=None)
        kg.src = Node.from_dict(data)
        return kg

    @classmethod
    def load_json(cls, filepath: str) -> "KnowledgeGraphBuilder":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

def get_knowledge_graph(query: str, results_dir: str = "knowledge-graph-dump") -> KnowledgeGraphBuilder:
    filepath = get_filepath(query, results_dir)

    if os.path.exists(filepath):
        print("Knowledge graph already exists. Loading from file...")
        return KnowledgeGraphBuilder.load_json(filepath)
    else:
        knowledge_graph = KnowledgeGraphBuilder(query=query)
        print("Knowledge Graph:", knowledge_graph.to_dict())
        knowledge_graph.save_json(filepath)
        return knowledge_graph

def traverse_graph(node: Node) -> str:
    graph = []
    for child in node.children:
        result = "  " * child.level + f"Node (Level {child.level}): {child.keywords}\n"
        for sub_child in child.children:
            result += "  " * sub_child.level + f"Node (Level {sub_child.level}): {sub_child.keywords}\n"
        graph.append(result)
    return graph

# ............................................................................

# RAG pipeline

def flatten_graph(node: Node, parent_summaries: list[str] = None) -> List[str]:
    docs = []
    parent_summaries = parent_summaries or []

    path = parent_summaries + [node.summary]

    text = " > ".join(path)

    metadata = {
        "level": node.level,
        "keywords": node.keywords,
        "url": node.url or ""
    }

    docs.append(Document(page_content=text, metadata=metadata))

    for child in node.children:
        docs.extend(flatten_graph(child, path))

    return docs

docs = None
doc_embeddings = None

def retrieve(query: str, top_k: int = 3) -> List[str]:
    # Encode query
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

    # Compute cosine similarity
    scores = np.dot(doc_embeddings, query_embedding)

    # Pick top-k indices
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    # Return top-k docs with scores
    return "\n".join([docs[i] for i in top_k_idx])

def get_narration_script(query: str, outline) -> Dict[str, Any]:

    filepath = get_filepath(query, "narration-script")

    if os.path.exists(filepath):
        print("Script already exists. Loading from file...")
        with open(filepath, "r", encoding="utf-8") as f:
                narration_script = json.load(f)
        return narration_script

    context = retrieve(query)
    sys_msg = f"Don't specify the title of the slide. Here is the context to answer the following query:\n{context}\nGenerate a narration script, giving an introduction to the video (without any concluding remarks) and briefly explaining the following news topic and also covering its affect on India if it is a global news, in no more than 300 characters:"
    
    response = get_structured_response(sys_msg, query, NarrationItem)

    narration_script = {"topic": query, "content": response["content"], "questions": [], "subtopics": []}

    for item in outline:
        
        sub_query = item["topic"]

        sub_context = retrieve(sub_query)
        sub_sys_msg = f"Don't specify the title of the slide. Here is the context to answer the following query:\n{sub_context}\nGenerate a narration script giving an overview of the following news topic enough to cover one slide, in no more than 400 characters:"
        
        sub_response = get_structured_response(sub_sys_msg, sub_query, NarrationItem)

        narration_sub_script = {"topic": sub_query, "content": sub_response["content"], "questions": [], "subtopics": []}

        for subtopic in item["subtopics"]:

            sub_sub_query = subtopic
            sub_sub_context = retrieve(sub_sub_query)
            sub_sub_sys_msg = f"Don't specify the title of the slide. Here is the context to answer the following query:\n{sub_sub_context}\nGenerate some questions that could be asked in UPSC related to the following topic to the viewers. Generate a narration script that explains the following subtopic in depth, covering every angle of it and also talking about its affect on India if it is a global issue, while also integrating the generated questions indirectly (such a question might appear in the UPSC exam) in the script, whenever you discuss a point which is related to that question, in no more than 600 characters :"
            
            sub_sub_response = get_structured_response(sub_sub_sys_msg, sub_sub_query, NarrationItem)
            
            narration_sub_script["subtopics"].append({"topic": sub_sub_query, "content": sub_sub_response["content"], "questions": sub_sub_response["questions"], "subtopics": []})

        narration_script["subtopics"].append(narration_sub_script)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(narration_script, f, ensure_ascii=False, indent=2)
    
    return narration_script

# ............................................................................

# Audio generation pipeline

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

def generate_audio(i: int, content: str):

    filepath = get_filepath(i.__str__(), "audio-json-dump")

    if os.path.exists(filepath):
        print("Specific audio already exists. Loading from file...")
        with open(filepath, "r", encoding="utf-8") as f:
            audio = json.load(f)
        return audio

    response = elevenlabs.text_to_speech.convert_with_timestamps(
        voice_id="2zRM7PkgwBPiau2jvVXc",
        text=content,
        model_id="eleven_turbo_v2"
    )

    output = {
        "audio_base_64": response.audio_base_64,
        "alignment": response.alignment.model_dump(),
        "normalized_alignment": response.normalized_alignment.model_dump(),
    }

    save_audio(response.audio_base_64, get_filepath(i.__str__(), "audio", extension="mp3"))

    audio_filepath = get_filepath(i.__str__(), "audio", extension="mp3")
    audio = AudioSegment.from_mp3(audio_filepath)
    duration_seconds = len(audio) / 1000.0

    output["length"] = duration_seconds

    with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output

def get_narration_audio(narration):
        
    filepath = get_filepath(narration["topic"], "narration-audio")

    if os.path.exists(filepath):
        print("Audio already exists. Loading from file...")
        with open(filepath, "r", encoding="utf-8") as f:
            narration = json.load(f)
        return narration
    i = 1
    narration["audio"] = generate_audio(i, f'{narration["topic"]}. {narration["content"]}')
    i += 1
    for subtopic in narration["subtopics"]:
        subtopic["audio"] = generate_audio(i, f'{subtopic["topic"]}. {subtopic["content"]}')
        i += 1
        for sub_subtopic in subtopic["subtopics"]:
            sub_subtopic["audio"] = generate_audio(i, f'{sub_subtopic["topic"]}. {sub_subtopic["content"]}')
            i += 1
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(narration, f, ensure_ascii=False, indent=2)

    return narration

# ............................................................................

# Generate Slide content

def get_slide_content(narration): # give the json with audio as param

    filepath = get_filepath(narration["topic"], "final-content")

    if os.path.exists(filepath):
        print("Final content already exists. Loading from file...")
        with open(filepath, "r", encoding="utf-8") as f:
            narration = json.load(f)
        return narration
    
    sys_msg = "You are a timestamping and summarization expert. You are provided with a mapping of narration to its timestamp in seconds for the slide and a list of questions. Generate me a list of points (complete sentences, not more than 5) that I can display on the slide along with their start time strictly according to the timestamps I have provided. Do not include the topic of the content in the points but do include an initial buffer time when providing the timestamping. Do not include any text which is present in both narration and questions and don't include any questions."
    
    prompt = f'Questions: {narration["questions"]}\nNarration with timestamps: {narration["audio"]["normalized_alignment"]}'
    narration["slide"] = get_structured_response(sys_msg, prompt, SlideContent)["slide"]

    for subtopic in narration["subtopics"]:
        sub_prompt = f'Questions: {subtopic["questions"]}\nNarration with timestamps: {subtopic["audio"]["normalized_alignment"]}'
        subtopic["slide"] = get_structured_response(sys_msg, sub_prompt, SlideContent)["slide"]

        for sub_subtopic in subtopic["subtopics"]:
            sub_sub_prompt = f'Questions: {sub_subtopic["questions"]}\nNarration with timestamps: {sub_subtopic["audio"]["normalized_alignment"]}'
            sub_subtopic["slide"] = get_structured_response(sys_msg, sub_sub_prompt, SlideContent)["slide"]

    else:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(narration, f, ensure_ascii=False, indent=2)

    return narration

# ............................................................................

def main(): 

    global docs
    global embedding_model
    global doc_embeddings
    global query

    knowledge_graph = get_knowledge_graph(query)

    outline = create_outline(query, traverse_graph(knowledge_graph.src))

    docs = flatten_graph(knowledge_graph.src)
    docs = [ f"{doc.page_content}\nMetadata: {doc.metadata}" for doc in docs]
    doc_embeddings = embedding_model.encode(docs, normalize_embeddings=True)

    narration_script = get_narration_script(query, outline)

    narration_audio = get_narration_audio(narration_script)

    final_content = get_slide_content(narration_audio)

# ............................................................................

if __name__ == "__main__":
    main()

# ............................................................................