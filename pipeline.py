from tavily import TavilyClient
import os
import json
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# ............................................................................

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
kw_model = KeyBERT(model=embedding_model)

def extract_keywords(text):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 4),
        stop_words="english",
        top_n=1  # extract more first
    )
    keywords = [kw for kw, score in keywords]
    return keywords[0]

# ............................................................................

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def summarize_text(text):
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

# We can scrape the content of these URLs and extract the relevant information.
def extract_main_content(url, chunk_size=800, overlap=50):

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

def safe_filename(query: str) -> str:
    # Replace non-alphanumeric characters with dashes
    return re.sub(r'[^a-zA-Z0-9]+', '-', query).strip('-')

# Function to perform web search using Tavily API
def web_search(query):
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Ensure directory exists
    results_dir = "web-search-results"
    os.makedirs(results_dir, exist_ok=True)

    # Filepath with dashed query
    filename = f"{safe_filename(query)}.json"
    filepath = os.path.join(results_dir, filename)

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

@dataclass
class Node:
    def __init__(self, content, summary = None, keywords = None, url = None, parent = None):
        self.summary = summary
        self.parent = parent
        self.keywords = keywords or []
        self.level = parent.level + 1 if parent else 1
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
    
# ............................................................................

class KnowledgeGraphBuilder:
    src = None
    def __init__(self, query, maxLevel=3):
        self.maxLevel = maxLevel
        self.src = Node(content=query, keywords=query, summary=query)
        self.build_graph(self.src)
    
    def build_graph(self,parent = None):
        if parent is None:
            parent = self.src
        
        urls = web_search(parent.keywords)

        for url in urls:

            contents = extract_main_content(url)
            print(len(contents), "contents found for", parent.keywords)

            for i in range(len(contents)):
                child = Node(content=contents[i], url=url, parent=parent)
                child.summary = summarize_text(contents[i])
                print("Summary for child:", child.summary)
                child.keywords = extract_keywords(child.summary)
                print("Keywords for child:", child.keywords)
                parent.children.append(child)
                if child.level < self.maxLevel:
                    self.build_graph(child)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.src.to_dict()

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

# ............................................................................

if __name__ == "__main__":

    # Enter the topic of the video here
    # we are doing it manually for now
    # but we can automate it later (consider it a black box for now)
    query = "Russia leaves Nuclear Arms Treaty with US"
    knowledge_graph = KnowledgeGraphBuilder(query=query)

    print("Knowledge Graph:", knowledge_graph)
    knowledge_graph.save_json("knowledge_graph.json")
    # print("Knowledge graph saved to knowledge_graph.json")