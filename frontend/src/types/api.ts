export interface DocumentSummary {
  id: string;
  filename: string;
  file_type?: string;
  source?: string;
  mime_type?: string | null;
  file_size?: number | null;
  status?: string;
  chunk_count: number;
  user_id: string | null;
  agent_id: string | null;
  conversation_id: string | null;
  metadata: Record<string, any>;
  created_at: string | null;
  updated_at: string | null;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  file_type: string;
  chunks: number;
  characters: number;
  pages: number | null;
  embedding_model: string;
  embedding_dim: number;
  took_ms: number;
  metadata: Record<string, any>;
}

export interface SearchRequest {
  query: string;
  top_k?: number | null;
  threshold?: number | null;
  user_id?: string | null;
  agent_id?: string | null;
  conversation_id?: string | null;
  document_ids?: string[] | null;
}

export interface SearchResult {
  chunk_id: string;
  document_id: string;
  content: string;
  similarity: number;
  vector_score?: number | null;
  keyword_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  metadata: Record<string, any>;
}

export interface SearchResponse {
  query: string;
  found: boolean;
  results: SearchResult[];
  threshold: number;
  top_k: number;
  took_ms: number;
  rerank_ms?: number | null;
  rrf_k?: number | null;
}

export interface ContextRequest extends SearchRequest {
  max_chars?: number | null;
}

export interface ContextSource {
  document_id: string;
  source: string;
  page: number | null;
  chunk_index: number | null;
  similarity: number;
}

export interface ContextResponse {
  query: string;
  found: boolean;
  context: string;
  sources: ContextSource[];
  chunks_used: number;
  took_ms: number;
}

export interface CitationSource {
  source_id: string;
  chunk_id: string;
  document_id: string;
  source: string;
  page: number | null;
  chunk_index: number | null;
}

export interface LatencyMetrics {
  retrieval_ms: number;
  rerank_ms: number;
  llm_ms: number;
  total_ms: number;
}

export interface RetrievalSummary {
  top_k: number;
  rrf_k?: number | null;
  chunks_found: number;
  chunks_used: number;
  results: SearchResult[];
}

export interface AskRequest {
  query: string;
  tenant_id?: string | null;
  user_id?: string | null;
  agent_id?: string | null;
  conversation_id?: string | null;
  document_ids?: string[] | null;
  top_k?: number | null;
  threshold?: number | null;
  max_context_chars?: number | null;
}

export interface AskResponse {
  query: string;
  answer: string;
  sources: CitationSource[];
  retrieval: RetrievalSummary;
  latency: LatencyMetrics;
}

export interface SearchLogEntry {
  query: string;
  timestamp: string;
  request_id?: string;
  document_ids?: string[];
  chunks_retrieved_count?: number;
  chunk_ids?: string[];
  similarity_scores?: number[];
  threshold?: number;
  top_k?: number;
  found?: boolean;
  took_ms?: number;
  model_name?: string;
  error?: string | null;
}

export interface DocumentListResponse {
  documents: DocumentSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface DeleteResponse {
  ok: boolean;
  document_id: string;
  chunks_deleted: number;
}

export interface RagHealthResponse {
  ok: boolean;
  configuration: Record<string, any>;
  database: {
    configured?: boolean;
    connected?: boolean;
    schema_ready?: boolean;
    column_dimension?: number;
    documents?: number;
    chunks?: number;
    error?: string;
    [key: string]: any;
  };
  embeddings: {
    model?: string;
    backend?: string;
    loaded?: boolean;
    dimension?: number;
    error?: string;
    [key: string]: any;
  };
}

export interface ServiceHealthResponse {
  status: string;
  service: string;
}
