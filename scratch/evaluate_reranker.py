import httpx
import json

queries = [
    "What is Maaz’s CGPA?",
    "What programming languages does Maaz know?",
    "What internships has Maaz completed?",
    "What AWS services has Maaz worked with?"
]

url = "http://127.0.0.1:8000/api/rag/search"

print("Starting evaluation...")
for query in queries:
    print(f"\n==================================================")
    print(f"QUERY: '{query}'")
    print(f"==================================================")
    try:
        response = httpx.post(url, json={"query": query, "top_k": 5})
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            continue
            
        data = response.json()
        results = data.get("results", [])
        print(f"Found {len(results)} results (Rerank latency: {data.get('rerank_ms')} ms, Total search: {data.get('took_ms')} ms):")
        for i, hit in enumerate(results):
            meta = hit.get("metadata", {})
            print(f"\n{i+1}. Chunk ID: {hit.get('chunk_id')}")
            print(f"   Document: {meta.get('source')}")
            print(f"   vector_score: {meta.get('vector_score')}")
            print(f"   keyword_score: {meta.get('keyword_score')}")
            print(f"   rrf_score: {meta.get('rrf_score')}")
            print(f"   rerank_score: {meta.get('rerank_score')}")
            print(f"   Snippet: {hit.get('content')[:150]}...")
    except Exception as e:
        print(f"Failed to query search API: {e}")
