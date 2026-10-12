import React, { useState } from "react";
import {
  Database,
  Cpu,
  RefreshCw,
  Eye,
  EyeOff,
  CheckCircle2,
  AlertCircle,
  Lock,
} from "lucide-react";
import { ragApi, saveApiSettings } from "../../services/api";
import type { ApiSettings } from "../../services/api";
import type { RagHealthResponse } from "../../types/api";

interface SettingsViewProps {
  settings: ApiSettings;
  onUpdateSettings: (newSettings: ApiSettings) => void;
  ragHealth: RagHealthResponse | null;
  onRefreshHealth: () => void;
  isCheckingHealth: boolean;
  onShowToast: (message: string, type: "success" | "error" | "info") => void;
}

export const SettingsView: React.FC<SettingsViewProps> = ({
  settings,
  onUpdateSettings,
  ragHealth,
  onRefreshHealth,
  isCheckingHealth,
  onShowToast,
}) => {
  const [baseUrl, setBaseUrl] = useState(settings.baseUrl);
  const [token, setToken] = useState(settings.token);
  const [showToken, setShowToken] = useState(false);
  const [testingConn, setTestingConn] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    const updated: ApiSettings = {
      baseUrl: baseUrl.trim(),
      token: token.trim(),
    };
    saveApiSettings(updated);
    onUpdateSettings(updated);
    onShowToast("Connection settings saved to local storage", "success");
    onRefreshHealth();
  };

  const handleTestConnection = async () => {
    setTestingConn(true);
    setTestResult(null);
    try {
      const s = await ragApi.getServiceHealth();
      setTestResult({
        ok: true,
        message: `Successfully connected to ${s.service} (Status: ${s.status})`,
      });
      onShowToast("Connection verified", "success");
    } catch (err: any) {
      setTestResult({
        ok: false,
        message: err.message || "Could not reach API server.",
      });
      onShowToast("Connection failed", "error");
    } finally {
      setTestingConn(false);
    }
  };

  const db = ragHealth?.database || {};
  const config = ragHealth?.configuration || {};
  const embeddings = ragHealth?.embeddings || {};

  return (
    <div className="space-y-6 max-w-4xl">
      {/* 1. Connection & Credentials Card */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-4">
        <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Workspace API Connection</h2>
            <p className="text-xs text-slate-500">
              Configure the client endpoint and bearer authorization token for this session.
            </p>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-slate-500 font-mono">
            <Lock className="w-3.5 h-3.5 text-slate-400" />
            <span>Client-side storage only</span>
          </div>
        </div>

        <form onSubmit={handleSave} className="space-y-4 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Endpoint */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1 font-mono">
                API BASE URL
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="http://localhost:8000"
                className="w-full text-xs px-3 py-2 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
              />
              <p className="text-[10px] text-slate-400 mt-1">
                FastAPI host location. Defaults to http://localhost:8000
              </p>
            </div>

            {/* Token */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1 font-mono">
                BEARER AUTHORIZATION TOKEN
              </label>
              <div className="relative flex items-center">
                <input
                  type={showToken ? "text" : "password"}
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="RAG_API_TOKEN secret"
                  className="w-full text-xs pl-3 pr-9 py-2 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
                />
                <button
                  type="button"
                  onClick={() => setShowToken(!showToken)}
                  className="absolute right-2 text-slate-400 hover:text-slate-600 p-1"
                  aria-label={showToken ? "Hide token" : "Show token"}
                >
                  {showToken ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                </button>
              </div>
              <p className="text-[10px] text-slate-400 mt-1">
                Protected RAG routes enforce Bearer authentication.
              </p>
            </div>
          </div>

          {testResult && (
            <div
              className={`p-3 rounded-lg border text-xs flex items-center gap-2 ${
                testResult.ok
                  ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                  : "bg-rose-50 border-rose-200 text-rose-800"
              }`}
            >
              {testResult.ok ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              ) : (
                <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
              )}
              <span className="font-mono">{testResult.message}</span>
            </div>
          )}

          <div className="pt-2 flex items-center gap-2.5">
            <button
              type="submit"
              className="px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors"
            >
              Save Configuration
            </button>
            <button
              type="button"
              onClick={handleTestConnection}
              disabled={testingConn}
              className="px-3 py-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium transition-colors disabled:opacity-50 flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${testingConn ? "animate-spin" : ""}`} />
              <span>Test Connection</span>
            </button>
          </div>
        </form>
      </div>

      {/* 2. Real System Diagnostics Card */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-4">
        <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Verified System Subsystems</h2>
            <p className="text-xs text-slate-500">
              Live status reported by /api/rag/health. Never displaying raw secrets or connection strings.
            </p>
          </div>
          <button
            onClick={onRefreshHealth}
            disabled={isCheckingHealth}
            className="flex items-center gap-1.5 text-xs text-slate-600 hover:text-slate-900 font-mono"
          >
            <RefreshCw className={`w-3 h-3 ${isCheckingHealth ? "animate-spin" : ""}`} />
            <span>Re-probe</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          {/* Subsystem 1: PostgreSQL & pgvector */}
          <div className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/50 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-900 font-semibold font-sans">
                <Database className="w-4 h-4 text-blue-600" />
                <span>PostgreSQL + pgvector</span>
              </div>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  db.connected
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-rose-100 text-rose-800"
                }`}
              >
                {db.connected ? "Connected" : "Disconnected"}
              </span>
            </div>

            <div className="space-y-1.5 text-[11px] text-slate-600">
              <div className="flex justify-between">
                <span>Schema State:</span>
                <span className="font-semibold text-slate-800">
                  {db.schema_ready ? "Ready (rag_documents, rag_chunks)" : "Pending"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Vector Dimension:</span>
                <span className="font-semibold text-slate-800">
                  {db.column_dimension ?? 1024}-d
                </span>
              </div>
              <div className="flex justify-between">
                <span>Distance Metric:</span>
                <span className="font-semibold text-slate-800 uppercase">
                  {config.distance_metric ?? "Cosine"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Stored Records:</span>
                <span className="font-semibold text-slate-800">
                  {db.documents ?? 0} docs / {db.chunks ?? 0} chunks
                </span>
              </div>
            </div>
          </div>

          {/* Subsystem 2: Embedding Engine */}
          <div className="p-4 rounded-xl border border-slate-200/80 bg-slate-50/50 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-900 font-semibold font-sans">
                <Cpu className="w-4 h-4 text-blue-600" />
                <span>Embedding Model</span>
              </div>
              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                  embeddings.loaded
                    ? "bg-emerald-100 text-emerald-800"
                    : "bg-amber-100 text-amber-800"
                }`}
              >
                {embeddings.loaded ? "Loaded" : "Backend Standby"}
              </span>
            </div>

            <div className="space-y-1.5 text-[11px] text-slate-600">
              <div className="flex justify-between">
                <span>Model Name:</span>
                <span className="font-semibold text-slate-800">
                  {embeddings.model || config.embedding_model || "BAAI/bge-m3"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Backend Provider:</span>
                <span className="font-semibold text-slate-800 capitalize">
                  {embeddings.backend || config.embedding_backend || "local"}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Chunk Boundary:</span>
                <span className="font-semibold text-slate-800">
                  {config.chunk_size ?? 800} chars (overlap: {config.chunk_overlap ?? 150})
                </span>
              </div>
              <div className="flex justify-between">
                <span>Reranker Subsystem:</span>
                <span className="font-semibold text-slate-800">
                  {config.rerank_enabled ? config.rerank_model || "bge-reranker-base" : "Disabled"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Subsystem 3: LLM Provider Flags */}
        <div className="p-3.5 rounded-lg border border-slate-200/80 bg-white flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="space-y-0.5">
            <div className="font-semibold text-slate-800 font-sans">Configured LLM Gateways</div>
            <div className="text-[11px] text-slate-500 font-sans">
              Upstream LLM connectors available for grounded question answering (/api/rag/ask).
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px]">
            <span
              className={`px-2 py-0.5 rounded border ${
                config.groq_configured
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                  : "bg-slate-100 border-slate-200 text-slate-400"
              }`}
            >
              Groq: {config.groq_configured ? "Active" : "Off"}
            </span>
            <span
              className={`px-2 py-0.5 rounded border ${
                config.gemini_configured
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                  : "bg-slate-100 border-slate-200 text-slate-400"
              }`}
            >
              Gemini: {config.gemini_configured ? "Active" : "Off"}
            </span>
            <span
              className={`px-2 py-0.5 rounded border ${
                config.openai_configured
                  ? "bg-emerald-50 border-emerald-200 text-emerald-700"
                  : "bg-slate-100 border-slate-200 text-slate-400"
              }`}
            >
              OpenAI: {config.openai_configured ? "Active" : "Off"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
