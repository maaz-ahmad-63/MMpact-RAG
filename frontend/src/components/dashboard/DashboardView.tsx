import React from "react";
import {
  FileText,
  Layers,
  Search,
  Plus,
  Cpu,
  Database,
  ArrowRight,
  ShieldCheck,
} from "lucide-react";
import type { DocumentSummary, RagHealthResponse } from "../../types/api";
import { FileTypeBadge } from "../ui/Badge";
import type { NavTab } from "../layout/Sidebar";

interface DashboardViewProps {
  documents: DocumentSummary[];
  ragHealth: RagHealthResponse | null;
  onNavigate: (tab: NavTab) => void;
  onOpenUpload: () => void;
  onViewDocDetails: (doc: DocumentSummary) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  documents,
  ragHealth,
  onNavigate,
  onOpenUpload,
  onViewDocDetails,
}) => {
  const db = ragHealth?.database || {};
  const config = ragHealth?.configuration || {};
  const embeddings = ragHealth?.embeddings || {};

  const totalDocuments = db.documents ?? documents.length;
  const totalChunks = db.chunks ?? documents.reduce((acc, d) => acc + (d.chunk_count || 0), 0);
  const dimension = db.column_dimension ?? config.embedding_dim ?? 1024;
  const isConnected = Boolean(db.connected && db.schema_ready);

  return (
    <div className="space-y-6">
      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Metric 1: Documents */}
        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-500 uppercase font-mono tracking-wider">
              Ingested Documents
            </span>
            <div className="p-1.5 rounded-md bg-slate-100 text-slate-600">
              <FileText className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
              {totalDocuments}
            </span>
            <span className="text-xs text-slate-500">records active</span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Library capacity</span>
            <button
              onClick={() => onNavigate("documents")}
              className="text-blue-600 hover:text-blue-700 font-medium flex items-center gap-0.5"
            >
              Browse
              <ArrowRight className="w-2.5 h-2.5" />
            </button>
          </div>
        </div>

        {/* Metric 2: Vector Chunks */}
        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-500 uppercase font-mono tracking-wider">
              Stored Chunks
            </span>
            <div className="p-1.5 rounded-md bg-slate-100 text-slate-600">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
              {totalChunks.toLocaleString()}
            </span>
            <span className="text-xs text-slate-500">dense vectors</span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Overlap: {config.chunk_overlap ?? 150} chars</span>
            <span className="font-mono text-slate-600">Size: {config.chunk_size ?? 800}</span>
          </div>
        </div>

        {/* Metric 3: Vector Dimension */}
        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-500 uppercase font-mono tracking-wider">
              Vector Dimension
            </span>
            <div className="p-1.5 rounded-md bg-slate-100 text-slate-600">
              <Cpu className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
              {dimension}-d
            </span>
            <span className="text-xs text-slate-500">BGE-M3</span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span>Distance metric</span>
            <span className="font-mono text-slate-700 font-medium">
              {config.distance_metric ?? "Cosine"}
            </span>
          </div>
        </div>

        {/* Metric 4: Index Status */}
        <div className="bg-white p-4 rounded-xl border border-slate-200/90 shadow-2xs">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-500 uppercase font-mono tracking-wider">
              pgvector Engine
            </span>
            <div className="p-1.5 rounded-md bg-slate-100 text-slate-600">
              <Database className="w-4 h-4" />
            </div>
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isConnected ? "bg-emerald-500" : "bg-rose-500"
              }`}
            />
            <span className="text-base font-semibold text-slate-900 font-mono">
              {isConnected ? "HNSW Ready" : "Disconnected"}
            </span>
          </div>
          <div className="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500 flex items-center justify-between">
            <span>PostgreSQL 12+</span>
            <span className="text-emerald-700 font-medium">Schema verified</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Architecture & Retrieval Stack + Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Ingest & Engine Pipeline Spec */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-100 pb-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Retrieval Pipeline Architecture</h2>
              <p className="text-xs text-slate-500">
                Hybrid dense vector retrieval fused with PostgreSQL full-text search and cross-encoder reranking.
              </p>
            </div>
            <button
              onClick={() => onNavigate("settings")}
              className="text-xs text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
            >
              Configure
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            {/* Step 1 */}
            <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/60 space-y-1.5">
              <div className="flex items-center gap-2 text-slate-800 font-semibold">
                <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-mono text-[11px]">
                  1
                </span>
                <span>Dual Candidate Search</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Fetches {config.vector_candidates ?? 20} candidates via pgvector cosine distance, and{" "}
                {config.keyword_candidates ?? 20} candidates via PostgreSQL English ts_rank_cd FTS.
              </p>
            </div>

            {/* Step 2 */}
            <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/60 space-y-1.5">
              <div className="flex items-center gap-2 text-slate-800 font-semibold">
                <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-mono text-[11px]">
                  2
                </span>
                <span>RRF Score Fusion</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                Applies Reciprocal Rank Fusion (k={config.rrf_k ?? 60}) to combine vector similarity and lexical
                rankings without score distortion.
              </p>
            </div>

            {/* Step 3 */}
            <div className="p-3 rounded-lg border border-slate-200/80 bg-slate-50/60 space-y-1.5">
              <div className="flex items-center gap-2 text-slate-800 font-semibold">
                <span className="w-5 h-5 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center font-mono text-[11px]">
                  3
                </span>
                <span>Cross-Encoder Rerank</span>
              </div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                {config.rerank_enabled ? (
                  <>Re-scores top candidates with {config.rerank_model || "BAAI/bge-reranker-base"} for optimal precision.</>
                ) : (
                  <>Reranker disabled in current configuration.</>
                )}
              </p>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-blue-50/60 border border-blue-100/80 flex items-center justify-between text-xs text-blue-900">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-blue-600 flex-shrink-0" />
              <span>Multi-tenant scopes (User ID, Agent ID, Thread ID) enforced at query time.</span>
            </div>
            <button
              onClick={() => onNavigate("search")}
              className="text-xs font-semibold text-blue-700 hover:text-blue-800 flex items-center gap-1"
            >
              Test Query
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Right 1 Col: Quick Actions & Status */}
        <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-4 flex flex-col justify-between">
          <div className="space-y-3">
            <h2 className="text-sm font-semibold text-slate-900">Quick Actions</h2>
            <div className="space-y-2">
              <button
                onClick={onOpenUpload}
                className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200/90 hover:border-blue-400 hover:bg-blue-50/30 transition-all text-left group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-md bg-blue-50 text-blue-600 group-hover:bg-blue-600 group-hover:text-white transition-colors">
                    <Plus className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-800">Upload Document</div>
                    <div className="text-[11px] text-slate-500">PDF, DOCX, TXT, or MD</div>
                  </div>
                </div>
                <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-blue-600 group-hover:translate-x-0.5 transition-all" />
              </button>

              <button
                onClick={() => onNavigate("search")}
                className="w-full flex items-center justify-between p-3 rounded-lg border border-slate-200/90 hover:border-slate-300 hover:bg-slate-50 transition-all text-left group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-md bg-slate-100 text-slate-600 group-hover:bg-slate-200 transition-colors">
                    <Search className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-slate-800">Search Knowledge Base</div>
                    <div className="text-[11px] text-slate-500">Test similarity & context</div>
                  </div>
                </div>
                <ArrowRight className="w-3.5 h-3.5 text-slate-400 group-hover:translate-x-0.5 transition-all" />
              </button>
            </div>
          </div>

          <div className="pt-3 border-t border-slate-100 text-[11px] text-slate-500 font-mono space-y-1">
            <div className="flex justify-between">
              <span>Embedding Engine</span>
              <span className="text-slate-800 font-semibold">{embeddings.model || config.embedding_model || "BAAI/bge-m3"}</span>
            </div>
            <div className="flex justify-between">
              <span>Backend Mode</span>
              <span className="text-slate-800 font-semibold capitalize">{embeddings.backend || config.embedding_backend || "local"}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Ingested Documents List */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-slate-900">Recent Documents</h2>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-600">
              {documents.length}
            </span>
          </div>
          <button
            onClick={() => onNavigate("documents")}
            className="text-xs text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
          >
            View all
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>

        {documents.length === 0 ? (
          <div className="p-12 text-center space-y-2">
            <p className="text-xs text-slate-500">No documents ingested in this workspace yet.</p>
            <button
              onClick={onOpenUpload}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md bg-blue-600 text-white text-xs font-medium"
            >
              <Plus className="w-3.5 h-3.5" />
              Upload First Document
            </button>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {documents.slice(0, 5).map((doc) => (
              <div
                key={doc.id}
                onClick={() => onViewDocDetails(doc)}
                className="px-5 py-3 flex items-center justify-between hover:bg-slate-50/70 cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-3 truncate max-w-md">
                  <div className="p-2 rounded-lg bg-slate-100 text-slate-500">
                    <FileText className="w-4 h-4" />
                  </div>
                  <div className="truncate">
                    <div className="text-xs font-semibold text-slate-900 truncate">
                      {doc.filename}
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono truncate">
                      {doc.id}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs font-mono">
                  <FileTypeBadge fileType={doc.file_type || doc.mime_type} />
                  <div className="text-slate-600 flex items-center gap-1">
                    <Layers className="w-3 h-3 text-slate-400" />
                    <span>{doc.chunk_count} chunks</span>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    {doc.created_at ? new Date(doc.created_at).toLocaleDateString() : "—"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
