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
from manim import *

# ............................................................................

query = "Russia leaves Nuclear Arms Treaty with US"

# ............................................................................

# Util methods

# Load environment variables
load_dotenv()

# Manim configuration (*can also be done using command line arguments)
config.output_file = 'sample_video.mp4'
config.media_dir = 'out'

# Helper methods

def safe_filename(query: str) -> str:
    # Replace non-alphanumeric characters with dashes
    return re.sub(r'[^a-zA-Z0-9]+', '-', query.lower()).strip('-')

def get_filepath(name: str, results_dir: str, extension: str = "json") -> str:
    results_dir = os.path.join(query, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{safe_filename(name)}.{extension}"
    filepath = os.path.join(results_dir, filename)
    
    return filepath

def save_audio(base64_str: str, filepath: str):
    audio_bytes = base64.b64decode(base64_str)
    with open(filepath, "wb") as f:
        f.write(audio_bytes)
    audio = AudioSegment.from_file(filepath, format="mp3")
    return audio.duration_seconds

# ............................................................................

# AI generation methods and classes for structured response

class Outline(BaseModel):
    topic: str
    subtopics: list[str]

class Outlines(BaseModel):
    outline: list[Outline]

class NarrationItem(BaseModel):
    content: str

def get_structured_response(sys_msg: str, prompt: str, format: str ,retry: int = 1):

    response = completion(
        model="azure/gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": format,
                "schema": Outlines.model_json_schema() if format == "Outline" else NarrationItem.model_json_schema()
            }
        }
    )

    if (response is None):
        if retry > 3:
            raise Exception("Failed to generate output after multiple retries")
        else:
            print(f"Retrying outline generation... Attempt {retry + 1}")
            return get_structured_response(sys_msg, prompt, format, retry + 1)
    
    return json.loads(response.choices[0].message.content)[format.lower()]

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

    sys_msg = "You are a pedagogy specialist. We want to create an informative video for UPSC aspirants. Create a structured outline by eliminiating redundancies and unrelated content as well as restructuring the order of topics (2) and subtopics (2 for each topic) for such a video on detailed analysis using the following topics:"
    narration_outline = get_structured_response(sys_msg, "\n".join(graph for graph in graph_summary), "Outline")

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

def retrieve(query: str, top_k: int = 1) -> List[str]:
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
    sys_msg = f"Here is the context to answer the following query:\n{context}\nGenerate a narration script for the following topic enough for just one slide only (there are other topics as well, therefore just focus on this particular topic only):"
    
    response = get_structured_response(sys_msg, query, "NarrationItem")

    narration_script = {"topic": query, "content": response["content"], "subtopics": []}
    for item in outline:
        
        sub_query = item["topic"]

        sub_context = retrieve(sub_query)
        sub_sys_msg = f"Here is the context to answer the following query:\n{sub_context}\nGenerate a narration script for the following topic enough for just one slide only (there are other topics as well, therefore just focus on this particular topic only):"
        
        sub_response = get_structured_response(sub_sys_msg, sub_query, "NarrationItem")

        narration_sub_script = {"topic": sub_query, "content": sub_response["content"], "subtopics": []}

        for subtopic in item["subtopics"]:

            sub_sub_topic_query = subtopic
            sub_sub_topic_context = retrieve(sub_sub_topic_query)
            sub_sub_sys_msg = f"Here is the context to answer the following query:\n{sub_sub_topic_context}\nGenerate a narration script for the following subtopic enough for just one slide only (there are other subtopics as well, therefore just focus on this particular subtopic only):"
            
            sub_sub_response = get_structured_response(sub_sub_sys_msg, sub_sub_topic_query, "NarrationItem")
            
            narration_sub_script["subtopics"].append({"topic": sub_sub_topic_query, "content": sub_sub_response["content"], "subtopics": []})

        narration_script["subtopics"].append(narration_sub_script)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(narration_script, f, ensure_ascii=False, indent=2)
    
    return narration_script

# ............................................................................

# Audio generation pipeline

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

def generate_audio(content: str):

    response = elevenlabs.text_to_speech.convert_with_timestamps(
        voice_id="6JsmTroalVewG1gA6Jmw",
        text=content
    )

    return {
        "audio_base_64": response.audio_base_64,
        "alignment": response.alignment.model_dump(),
        "normalized_alignment": response.normalized_alignment.model_dump(),
    }

def get_narration_audio(narration_script):

    narration = narration_script
        
    filepath = get_filepath(narration["topic"], "narration-audio")

    if os.path.exists(filepath):
        print("Audio already exists. Loading from file...")
        with open(filepath, "r", encoding="utf-8") as f:
            narration = json.load(f)
        return narration

    narration["audio"] = generate_audio(narration["content"])

    for subtopic in narration["subtopics"]:
        subtopic["audio"] = generate_audio(subtopic["content"])

        for sub_subtopic in subtopic["subtopics"]:
            sub_subtopic["audio"] = generate_audio(sub_subtopic["content"])

    else:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(narration, f, ensure_ascii=False, indent=2)

    return narration

# ............................................................................

# Manim class for Video Generation

class NarrationScene(Scene):

    def construct(self):

        with open(get_filepath(query, "narration-audio"), "r", encoding="utf-8") as f:
            narration_audio = json.load(f)
        
        self.play_slide(narration_audio)

    def play_slide(self, topic_dict):

        filepath = get_filepath(topic_dict["topic"], "sounds", extension = "mp3")

        if not os.path.exists(filepath):
            save_audio(topic_dict["audio"]["audio_base_64"], filepath)

        normalized_alignment = topic_dict["audio"]["normalized_alignment"]
        chars = normalized_alignment["characters"]
        starts = normalized_alignment["character_start_times_seconds"]
        ends = normalized_alignment["character_end_times_seconds"]
         
        # group into words
        words, word_starts, word_ends = [], [], []
        current_word, current_start = "", None

        for c, s, e in zip(chars, starts, ends):
            if c != " ":  # part of a word
                if current_word == "":
                    current_start = s
                current_word += c
                current_end = e
            else:  # space -> close word
                if current_word:
                    words.append(current_word)
                    word_starts.append(current_start)
                    word_ends.append(current_end)
                current_word, current_start = "", None

        # Last word (if not ended with space)
        if current_word:
            words.append(current_word)
            word_starts.append(current_start)
            word_ends.append(current_end)

        # Show topic
        topic_text = Text(topic_dict["topic"], font_size=36).to_edge(UP)
        self.add(topic_text)
        self.play(Write(topic_text))
        self.wait(0.5)

        # Play sound
        self.add_sound(filepath)

        # Animate words
        subtitle = Paragraph("", alignment="center", line_spacing=0.8, width=6).to_edge(DOWN)

        self.add(subtitle)

        current_text = ""
        for w, s, e in zip(words, word_starts, word_ends):
            current_text += " " + w
            new_subtitle = Paragraph(current_text, alignment="center", line_spacing=0.8, width=6).to_edge(DOWN)
            self.play(Transform(subtitle, new_subtitle), run_time=(e - s))

        self.wait(1)

        # Recursively handle subtopics
        for subtopic in topic_dict["subtopics"]:
            self.play(FadeOut(topic_text), FadeOut(subtitle))
            self.clear()
            self.play_slide(subtopic)

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

# ............................................................................

if __name__ == "__main__":
    main()

# ............................................................................
