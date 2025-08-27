import os
import openai
import fitz  # PyMuPDF
from pinecone import Pinecone
# --- 1. Embed function ---
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("vihaan-chatbot")

MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 800  # chars (not tokens)

DOCS_DIR = "docs"  # your folder of PDFs and text files


def extract_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # Split into overlapping chunks
    chunks = []
    text = full_text.strip().replace("\n", " ")
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i+CHUNK_SIZE]
        if len(chunk) > 200:  # skip tiny junk
            chunks.append(chunk)
    return chunks


def extract_chunks_from_text(text_path):
    with open(text_path, 'r', encoding='utf-8') as file:
        full_text = file.read()
    
    # Split into overlapping chunks
    chunks = []
    text = full_text.strip().replace("\n", " ")
    for i in range(0, len(text), CHUNK_SIZE):
        chunk = text[i:i+CHUNK_SIZE]
        if len(chunk) > 200:  # skip tiny junk
            chunks.append(chunk)
    return chunks


def extract_chunks(filepath):
    """Extract chunks from either PDF or text files"""
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.pdf':
        return extract_chunks_from_pdf(filepath)
    elif file_ext in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
        return extract_chunks_from_text(filepath)
    else:
        print(f"⚠️  Unsupported file type: {file_ext} for {filepath}")
        return []


def embed_texts(texts):
    res = client.embeddings.create(
        model=MODEL,
        input=texts
    )
    return [record.embedding for record in res.data]


def process_file(filepath, doc_id_start):
    filename = os.path.basename(filepath)
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # Skip unsupported file types
    if file_ext not in ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
        print(f"⚠️  Skipping unsupported file: {filename}")
        return 0
    
    chunks = extract_chunks(filepath)
    if not chunks:
        print(f"⚠️  No content extracted from {filename}")
        return 0
    
    embeddings = embed_texts(chunks)
    vectors = []
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        uid = f"{filename}_{i + doc_id_start}"
        meta = {
            "source": filename,
            "content": chunk,
            "file_type": file_ext[1:]  # Remove the dot
        }
        vectors.append((uid, emb, meta))
    
    index.upsert(vectors=vectors)
    print(f"✅ Uploaded {len(vectors)} chunks from {filename}")
    return len(vectors)


if __name__ == "__main__":
    # Get all supported files (PDFs and text files)
    supported_extensions = ['.pdf', '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']
    files = [f for f in os.listdir(DOCS_DIR) 
             if any(f.lower().endswith(ext) for ext in supported_extensions)]
    
    if not files:
        print("⚠️  No supported files found in the docs directory.")
        print(f"Supported extensions: {', '.join(supported_extensions)}")
        exit()
    
    print(f"📁 Found {len(files)} files to process:")
    for f in files:
        print(f"   - {f}")
    print()
    
    doc_id = 10000  # avoid clashing with earlier upserts

    for file in files:
        full_path = os.path.join(DOCS_DIR, file)
        num_uploaded = process_file(full_path, doc_id)
        doc_id += num_uploaded

    print("🎉 All supported files processed and uploaded.")
