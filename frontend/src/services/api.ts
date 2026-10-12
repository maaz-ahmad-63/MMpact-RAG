import type {
  ServiceHealthResponse,
  RagHealthResponse,
  DocumentListResponse,
  DocumentSummary,
  UploadResponse,
  SearchRequest,
  SearchResponse,
  ContextRequest,
  ContextResponse,
  AskRequest,
  AskResponse,
  DeleteResponse,
  SearchLogEntry,
} from "../types/api";

export interface ApiSettings {
  baseUrl: string;
  token: string;
}

const getInitialBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (envUrl !== undefined && envUrl.trim() !== "") {
    return envUrl.trim();
  }
  // In development, default to local backend server
  if (import.meta.env.DEV) {
    return "http://localhost:8000";
  }
  // In production, when served from same origin, use relative requests ("")
  return "";
};

// Preconfigured with local server and project's default token
const DEFAULT_SETTINGS: ApiSettings = {
  baseUrl: getInitialBaseUrl(),
  token: "H2X5HuDMwf7QaUhMhqnK6dgz1fNor_yl",
};

export const getApiSettings = (): ApiSettings => {
  const stored = localStorage.getItem("rag_api_settings");
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      let baseUrl = parsed.baseUrl;
      
      // If deployed in production on a remote domain and stored baseUrl is localhost,
      // fallback to same-origin relative requests so stale localStorage does not break requests
      const isRemote =
        typeof window !== "undefined" &&
        window.location.hostname !== "localhost" &&
        window.location.hostname !== "127.0.0.1";

      if (!import.meta.env.DEV && isRemote && (baseUrl === "http://localhost:8000" || !baseUrl)) {
        baseUrl = DEFAULT_SETTINGS.baseUrl;
      } else if (baseUrl === undefined || baseUrl === null) {
        baseUrl = DEFAULT_SETTINGS.baseUrl;
      }

      return {
        baseUrl,
        token: parsed.token !== undefined ? parsed.token : DEFAULT_SETTINGS.token,
      };
    } catch {
      return DEFAULT_SETTINGS;
    }
  }
  return DEFAULT_SETTINGS;
};

export const saveApiSettings = (settings: ApiSettings): void => {
  localStorage.setItem("rag_api_settings", JSON.stringify(settings));
};

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data: any) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const { baseUrl, token } = getApiSettings();
  const url = `${baseUrl.replace(/\/$/, "")}${path}`;

  const headers = new Headers(options.headers);
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorData: any;
    try {
      errorData = await response.json();
    } catch {
      try {
        errorData = await response.text();
      } catch {
        errorData = "Unknown error";
      }
    }
    throw new ApiError(
      `API Request Failed: ${response.status} ${response.statusText}`,
      response.status,
      errorData
    );
  }

  return response.json() as Promise<T>;
}

export const ragApi = {
  // Service health (unauthenticated)
  async getServiceHealth(): Promise<ServiceHealthResponse> {
    return request<ServiceHealthResponse>("/");
  },

  // RAG/database/embedding health
  async getRagHealth(): Promise<RagHealthResponse> {
    return request<RagHealthResponse>("/api/rag/health");
  },

  // List ingested documents (normalizes both array and object responses)
  async getDocuments(
    limit = 50,
    offset = 0,
    filters: {
      user_id?: string;
      agent_id?: string;
      conversation_id?: string;
    } = {}
  ): Promise<DocumentListResponse> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });
    if (filters.user_id) params.set("user_id", filters.user_id);
    if (filters.agent_id) params.set("agent_id", filters.agent_id);
    if (filters.conversation_id) {
      params.set("conversation_id", filters.conversation_id);
    }

    const raw = await request<any>(`/api/rag/documents?${params.toString()}`);
    
    // Normalize response shape safely
    if (Array.isArray(raw)) {
      const documents: DocumentSummary[] = raw.map((d: any) => ({
        id: d.id,
        filename: d.filename || d.source || "Untitled Document",
        file_type: d.file_type || d.mime_type || "txt",
        source: d.source || d.filename,
        mime_type: d.mime_type,
        file_size: d.file_size,
        status: d.status || "completed",
        chunk_count: d.chunk_count ?? d.metadata?.chunk_count ?? 0,
        user_id: d.user_id || d.tenant_id || null,
        agent_id: d.agent_id || null,
        conversation_id: d.conversation_id || null,
        metadata: d.metadata || {},
        created_at: d.created_at || null,
        updated_at: d.updated_at || null,
      }));
      return {
        documents,
        total: documents.length,
        limit,
        offset,
      };
    }

    if (raw && Array.isArray(raw.documents)) {
      const documents: DocumentSummary[] = raw.documents.map((d: any) => ({
        id: d.id,
        filename: d.filename || d.source || "Untitled Document",
        file_type: d.file_type || d.mime_type || "txt",
        source: d.source || d.filename,
        mime_type: d.mime_type,
        file_size: d.file_size,
        status: d.status || "completed",
        chunk_count: d.chunk_count ?? d.metadata?.chunk_count ?? 0,
        user_id: d.user_id || null,
        agent_id: d.agent_id || null,
        conversation_id: d.conversation_id || null,
        metadata: d.metadata || {},
        created_at: d.created_at || null,
        updated_at: d.updated_at || null,
      }));
      return {
        documents,
        total: raw.total ?? documents.length,
        limit: raw.limit ?? limit,
        offset: raw.offset ?? offset,
      };
    }

    return { documents: [], total: 0, limit, offset };
  },

  // Get details of one document
  async getDocument(documentId: string): Promise<DocumentSummary> {
    return request<DocumentSummary>(`/api/rag/documents/${encodeURIComponent(documentId)}`);
  },

  // Delete a document
  async deleteDocument(documentId: string): Promise<DeleteResponse> {
    return request<DeleteResponse>(
      `/api/rag/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" }
    );
  },

  // Vector similarity search
  async search(req: SearchRequest): Promise<SearchResponse> {
    return request<SearchResponse>("/api/rag/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  },

  // Context retrieval
  async getContext(req: ContextRequest): Promise<ContextResponse> {
    return request<ContextResponse>("/api/rag/context", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  },

  // Grounded Q&A with citations
  async ask(req: AskRequest): Promise<AskResponse> {
    return request<AskResponse>("/api/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  },

  // Search logs from MongoDB
  async getSearchLogs(limit = 50): Promise<SearchLogEntry[]> {
    return request<SearchLogEntry[]>(`/api/logs/searches?limit=${limit}`);
  },

  // Upload and ingest documents
  async upload(
    file: File,
    options: {
      document_id?: string;
      user_id?: string;
      agent_id?: string;
      conversation_id?: string;
      metadata?: Record<string, any>;
      onProgress?: (progress: number) => void;
    } = {}
  ): Promise<UploadResponse> {
    const { baseUrl, token } = getApiSettings();
    const url = `${baseUrl.replace(/\/$/, "")}/api/rag/upload`;

    return new Promise<UploadResponse>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url);

      if (token) {
        xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      }

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && options.onProgress) {
          const percent = Math.round((event.loaded / event.total) * 100);
          options.onProgress(percent);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const res = JSON.parse(xhr.responseText);
            resolve(res);
          } catch {
            reject(new Error("Invalid JSON response from server"));
          }
        } else {
          let errorData: any;
          try {
            errorData = JSON.parse(xhr.responseText);
          } catch {
            errorData = xhr.responseText || "Unknown upload error";
          }
          reject(
            new ApiError(
              `Upload failed: ${xhr.status} ${xhr.statusText}`,
              xhr.status,
              errorData
            )
          );
        }
      };

      xhr.onerror = () => {
        reject(new Error("Network error during file upload"));
      };

      const formData = new FormData();
      formData.append("file", file);
      if (options.document_id) formData.append("document_id", options.document_id);
      if (options.user_id) formData.append("user_id", options.user_id);
      if (options.agent_id) formData.append("agent_id", options.agent_id);
      if (options.conversation_id) {
        formData.append("conversation_id", options.conversation_id);
      }
      if (options.metadata) {
        formData.append("metadata", JSON.stringify(options.metadata));
      }

      xhr.send(formData);
    });
  },
};
