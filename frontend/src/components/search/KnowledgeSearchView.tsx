import React, { useState } from "react";
import {
  Search,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  FileText,
  Clock,
  Sparkles,
  Info,
  AlertCircle,
} from "lucide-react";
import { ragApi } from "../../services/api";
import type {
  SearchResponse,
  ContextResponse,
  AskResponse,
} from "../../types/api";
import { SimilarityBadge } from "../ui/Badge";

type SearchMode = "search" | "context" | "ask";

interface KnowledgeSearchViewProps {
  onShowToast: (message: string, type: "success" | "error" | "info") => void;
}

export const KnowledgeSearchView: React.FC<KnowledgeSearchViewProps> = ({
  onShowToast,
}) => {
  const [mode, setMode] = useState<SearchMode>("search");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [threshold, setThreshold] = useState(0.2);
  const [userId, setUserId] = useState("");
  const [documentIds, setDocumentIds] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  // Results & States
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Search Results
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [contextResponse, setContextResponse] = useState<ContextResponse | null>(null);
  const [askResponse, setAskResponse] = useState<AskResponse | null>(null);

  // Passage copy feedback
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedDetailsId, setExpandedDetailsId] = useState<string | null>(null);

  const handleCopyPassage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    onShowToast("Passage copied to clipboard", "success");
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleExecuteSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) {
      onShowToast("Please enter a search query", "error");
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setSearchResponse(null);
    setContextResponse(null);
    setAskResponse(null);

    const docIdList = documentIds
      .split(",")
      .map((d) => d.trim())
      .filter((d) => d.length > 0);

    try {
      if (mode === "search") {
        const res = await ragApi.search({
          query: query.trim(),
          top_k: topK,
          threshold: threshold,
          user_id: userId.trim() || undefined,
          document_ids: docIdList.length ? docIdList : undefined,
        });
        setSearchResponse(res);
        if (res.results.length === 0) {
          onShowToast("No passages met the similarity threshold", "info");
        } else {
          onShowToast(`Retrieved ${res.results.length} relevant chunks`, "success");
        }
      } else if (mode === "context") {
        const res = await ragApi.getContext({
          query: query.trim(),
          top_k: topK,
          threshold: threshold,
          user_id: userId.trim() || undefined,
          document_ids: docIdList.length ? docIdList : undefined,
        });
        setContextResponse(res);
        onShowToast(`Assembled context from ${res.chunks_used} chunks`, "success");
      } else if (mode === "ask") {
        const res = await ragApi.ask({
          query: query.trim(),
          top_k: topK,
          threshold: threshold,
          user_id: userId.trim() || undefined,
          document_ids: docIdList.length ? docIdList : undefined,
        });
        setAskResponse(res);
        onShowToast("Generated grounded answer", "success");
      }
    } catch (err: any) {
      setErrorMsg(err.message || "Retrieval request failed");
      onShowToast(err.message || "Retrieval error", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Search Header & Mode Selector */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-4">
        {/* Mode Selector Tabs */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg text-xs font-medium">
            <button
              onClick={() => setMode("search")}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                mode === "search"
                  ? "bg-white text-slate-900 shadow-2xs font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Hybrid Vector Search
            </button>
            <button
              onClick={() => setMode("context")}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                mode === "context"
                  ? "bg-white text-slate-900 shadow-2xs font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Context Assembly
            </button>
            <button
              onClick={() => setMode("ask")}
              className={`px-3 py-1.5 rounded-md transition-colors ${
                mode === "ask"
                  ? "bg-white text-slate-900 shadow-2xs font-semibold"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              Grounded Q&A (Ask)
            </button>
          </div>

          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-xs font-medium transition-colors ${
              showFilters
                ? "bg-blue-50 border-blue-200 text-blue-700 font-semibold"
                : "border-slate-200 hover:bg-slate-50 text-slate-600"
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>Parameters & Filters</span>
            {showFilters ? (
              <ChevronUp className="w-3 h-3 ml-0.5" />
            ) : (
              <ChevronDown className="w-3 h-3 ml-0.5" />
            )}
          </button>
        </div>

        {/* Query Input Bar */}
        <form onSubmit={handleExecuteSearch} className="relative">
          <div className="relative flex items-center">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question or enter keywords to retrieve passages..."
              className="w-full text-sm pl-10 pr-24 py-2.5 rounded-lg border border-slate-200 bg-slate-50/40 hover:bg-slate-50 focus:bg-white focus:border-blue-500 focus:outline-hidden transition-all placeholder:text-slate-400"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-1.5 px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors disabled:opacity-50 active:scale-[0.98] shadow-2xs"
            >
              {loading ? "Searching..." : "Retrieve"}
            </button>
          </div>
        </form>

        {/* Expandable Parameters Drawer */}
        {showFilters && (
          <div className="pt-3 border-t border-slate-100 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs animate-in fade-in duration-150">
            {/* Top-K */}
            <div>
              <div className="flex justify-between text-[11px] font-medium text-slate-600 mb-1">
                <span>Top-K Results</span>
                <span className="font-mono text-slate-900 font-semibold">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="20"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-blue-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-0.5">
                <span>1</span>
                <span>10</span>
                <span>20</span>
              </div>
            </div>

            {/* Threshold */}
            <div>
              <div className="flex justify-between text-[11px] font-medium text-slate-600 mb-1">
                <span>Similarity Threshold</span>
                <span className="font-mono text-slate-900 font-semibold">
                  {Math.round(threshold * 100)}%
                </span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-full accent-blue-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono mt-0.5">
                <span>0% (All)</span>
                <span>70% (Default)</span>
                <span>100%</span>
              </div>
            </div>

            {/* User / Agent Scope */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Tenant / User Scope
              </label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="e.g. org_32"
                className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
              />
            </div>

            {/* Document IDs Scope */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Restrict Document IDs
              </label>
              <input
                type="text"
                value={documentIds}
                onChange={(e) => setDocumentIds(e.target.value)}
                placeholder="Comma separated IDs"
                className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
              />
            </div>
          </div>
        )}
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2.5">
          <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold">Retrieval Pipeline Notice</div>
            <p className="text-[12px] text-rose-700">{errorMsg}</p>
          </div>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="bg-white p-5 rounded-xl border border-slate-200/90 shadow-2xs space-y-3 animate-pulse"
            >
              <div className="flex items-center justify-between">
                <div className="h-4 bg-slate-200 rounded w-1/4" />
                <div className="h-4 bg-slate-200 rounded w-16" />
              </div>
              <div className="space-y-2">
                <div className="h-3 bg-slate-100 rounded w-full" />
                <div className="h-3 bg-slate-100 rounded w-5/6" />
                <div className="h-3 bg-slate-100 rounded w-4/6" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* MODE 1: Search Results */}
      {!loading && mode === "search" && searchResponse && (
        <div className="space-y-4">
          {/* Summary Bar */}
          <div className="flex items-center justify-between text-xs text-slate-500 font-mono px-1">
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-800">
                {searchResponse.results.length} chunks retrieved
              </span>
              <span>•</span>
              <span>Threshold: {searchResponse.threshold}</span>
            </div>
            <div className="flex items-center gap-3">
              {searchResponse.rerank_ms !== undefined && searchResponse.rerank_ms !== null && (
                <span>Rerank: {searchResponse.rerank_ms}ms</span>
              )}
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3 text-slate-400" />
                {searchResponse.took_ms} ms total
              </span>
            </div>
          </div>

          {searchResponse.results.length === 0 ? (
            <div className="bg-white rounded-xl border border-slate-200 p-12 text-center space-y-2">
              <Info className="w-6 h-6 text-slate-400 mx-auto" />
              <div className="text-sm font-semibold text-slate-800">
                No matching passages found
              </div>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                No document chunks surpassed the {Math.round(searchResponse.threshold * 100)}%
                similarity threshold. Try lowering the threshold or refining your search query.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {searchResponse.results.map((result, idx) => {
                const meta = result.metadata || {};
                const sourceName = meta.source || result.document_id;
                const isDetailsOpen = expandedDetailsId === result.chunk_id;

                return (
                  <div
                    key={result.chunk_id || idx}
                    className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-3 hover:border-slate-300 transition-all"
                  >
                    {/* Result Header */}
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-2.5 truncate">
                        <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-600 flex items-center justify-center font-mono text-[11px] font-semibold flex-shrink-0">
                          {idx + 1}
                        </span>
                        <div className="p-1.5 rounded-md bg-blue-50 text-blue-600 flex-shrink-0">
                          <FileText className="w-3.5 h-3.5" />
                        </div>
                        <div className="truncate">
                          <span
                            className="text-xs font-semibold text-slate-900 truncate"
                            title={sourceName}
                          >
                            {sourceName}
                          </span>
                          <div className="text-[10px] font-mono text-slate-400 flex items-center gap-1.5">
                            {meta.page !== undefined && meta.page !== null && (
                              <span>Page {meta.page}</span>
                            )}
                            {meta.chunk_index !== undefined && meta.chunk_index !== null && (
                              <span>Chunk #{meta.chunk_index}</span>
                            )}
                            <span className="truncate max-w-[120px]">{result.document_id}</span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2 flex-shrink-0">
                        <SimilarityBadge score={result.similarity} />
                        <button
                          onClick={() => handleCopyPassage(result.chunk_id, result.content)}
                          className="p-1.5 rounded-md border border-slate-200 hover:bg-slate-50 text-slate-500 hover:text-slate-800 transition-colors"
                          title="Copy extracted passage"
                        >
                          {copiedId === result.chunk_id ? (
                            <Check className="w-3.5 h-3.5 text-emerald-600" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Extracted Text Content */}
                    <div className="text-xs text-slate-700 leading-relaxed p-3.5 rounded-lg bg-slate-50/70 border border-slate-100 font-sans whitespace-pre-wrap select-text">
                      {result.content}
                    </div>

                    {/* Technical Retrieval Scores Accordion */}
                    <div className="pt-1">
                      <button
                        onClick={() =>
                          setExpandedDetailsId(isDetailsOpen ? null : result.chunk_id)
                        }
                        className="text-[11px] font-mono text-slate-500 hover:text-slate-800 flex items-center gap-1 transition-colors"
                      >
                        {isDetailsOpen ? (
                          <ChevronUp className="w-3 h-3" />
                        ) : (
                          <ChevronDown className="w-3 h-3" />
                        )}
                        <span>Retrieval Scoring Breakdown</span>
                      </button>

                      {isDetailsOpen && (
                        <div className="mt-2.5 p-3 rounded-lg border border-slate-200/80 bg-slate-50 text-xs font-mono space-y-2 animate-in fade-in duration-100">
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                            <div className="p-2 rounded bg-white border border-slate-200/60">
                              <div className="text-[10px] text-slate-400">COSINE SIMILARITY</div>
                              <div className="font-semibold text-slate-800">
                                {result.similarity.toFixed(4)}
                              </div>
                            </div>

                            <div className="p-2 rounded bg-white border border-slate-200/60">
                              <div className="text-[10px] text-slate-400">VECTOR SCORE</div>
                              <div className="font-semibold text-slate-800">
                                {result.vector_score !== undefined && result.vector_score !== null
                                  ? result.vector_score.toFixed(4)
                                  : "—"}
                              </div>
                            </div>

                            <div className="p-2 rounded bg-white border border-slate-200/60">
                              <div className="text-[10px] text-slate-400">KEYWORD FTS SCORE</div>
                              <div className="font-semibold text-slate-800">
                                {result.keyword_score !== undefined && result.keyword_score !== null
                                  ? result.keyword_score.toFixed(4)
                                  : "—"}
                              </div>
                            </div>

                            <div className="p-2 rounded bg-white border border-slate-200/60">
                              <div className="text-[10px] text-slate-400">RERANK SCORE</div>
                              <div className="font-semibold text-slate-800">
                                {result.rerank_score !== undefined && result.rerank_score !== null
                                  ? result.rerank_score.toFixed(4)
                                  : "—"}
                              </div>
                            </div>
                          </div>

                          <div className="text-[10px] text-slate-500 truncate pt-1">
                            Chunk ID: {result.chunk_id}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* MODE 2: Context Assembly Response */}
      {!loading && mode === "context" && contextResponse && (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-500 font-mono px-1">
            <span>
              {contextResponse.chunks_used} chunks assembled ({contextResponse.context.length} chars)
            </span>
            <span>Took {contextResponse.took_ms} ms</span>
          </div>

          <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="text-xs font-semibold text-slate-900">
                Prompt-Ready Context Block
              </div>
              <button
                onClick={() => handleCopyPassage("context", contextResponse.context)}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-xs text-slate-700 font-medium transition-colors"
              >
                {copiedId === "context" ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Context</span>
                  </>
                )}
              </button>
            </div>

            <pre className="p-4 rounded-lg bg-slate-900 text-slate-100 text-xs font-mono overflow-x-auto max-h-96 whitespace-pre-wrap leading-relaxed select-text">
              {contextResponse.context || "No context assembled above the similarity threshold."}
            </pre>

            {contextResponse.sources.length > 0 && (
              <div className="space-y-2 pt-2">
                <div className="text-xs font-semibold text-slate-800">Citations Included</div>
                <div className="flex flex-wrap gap-2">
                  {contextResponse.sources.map((src, i) => (
                    <span
                      key={i}
                      className="px-2.5 py-1 rounded-md bg-slate-100 text-slate-700 text-xs font-mono flex items-center gap-1.5"
                    >
                      <FileText className="w-3 h-3 text-slate-400" />
                      <span>{src.source}</span>
                      <span className="text-slate-400">({Math.round(src.similarity * 100)}%)</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* MODE 3: Grounded Q&A (Ask) Response */}
      {!loading && mode === "ask" && askResponse && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2 text-slate-900 font-semibold text-sm">
                <Sparkles className="w-4 h-4 text-blue-600" />
                <span>Grounded Answer</span>
              </div>
              <div className="text-xs font-mono text-slate-500">
                Total: {askResponse.latency.total_ms} ms
              </div>
            </div>

            {/* Answer Text */}
            <div className="text-sm text-slate-800 leading-relaxed font-sans whitespace-pre-wrap select-text p-4 rounded-xl bg-slate-50/60 border border-slate-200/70">
              {askResponse.answer}
            </div>

            {/* Citations List */}
            {askResponse.sources.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-semibold text-slate-700 uppercase font-mono tracking-wider">
                  Referenced Citations
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {askResponse.sources.map((src, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded-lg border border-slate-200/80 bg-white flex items-center justify-between text-xs"
                    >
                      <div className="flex items-center gap-2 truncate">
                        <span className="font-mono font-semibold text-blue-600">
                          {src.source_id}
                        </span>
                        <span className="text-slate-800 truncate" title={src.source}>
                          {src.source}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-400">
                        {src.page ? `p. ${src.page}` : `chunk ${src.chunk_index}`}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Latency Breakdown Bar */}
            <div className="pt-2 border-t border-slate-100 grid grid-cols-4 gap-2 text-center text-xs font-mono">
              <div className="p-2 rounded bg-slate-50">
                <div className="text-[10px] text-slate-400">RETRIEVAL</div>
                <div className="font-semibold text-slate-800">
                  {askResponse.latency.retrieval_ms} ms
                </div>
              </div>
              <div className="p-2 rounded bg-slate-50">
                <div className="text-[10px] text-slate-400">RERANK</div>
                <div className="font-semibold text-slate-800">
                  {askResponse.latency.rerank_ms} ms
                </div>
              </div>
              <div className="p-2 rounded bg-slate-50">
                <div className="text-[10px] text-slate-400">LLM GENERATION</div>
                <div className="font-semibold text-slate-800">
                  {askResponse.latency.llm_ms} ms
                </div>
              </div>
              <div className="p-2 rounded bg-slate-50">
                <div className="text-[10px] text-slate-400">TOTAL</div>
                <div className="font-semibold text-blue-600">
                  {askResponse.latency.total_ms} ms
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Initial Empty State before querying */}
      {!loading && !searchResponse && !contextResponse && !askResponse && !errorMsg && (
        <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs p-16 text-center space-y-3">
          <div className="w-12 h-12 mx-auto rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center">
            <Search className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="text-sm font-semibold text-slate-800">Semantic & Hybrid Knowledge Search</h3>
            <p className="text-xs text-slate-500 max-w-md mx-auto leading-relaxed">
              Enter any natural language question to search stored documents across dense vector
              embeddings (pgvector) and keyword indices with real-time scoring.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
