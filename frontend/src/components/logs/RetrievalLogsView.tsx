import React, { useState, useEffect } from "react";
import {
  Activity,
  Search,
  Layers,
  RefreshCw,
} from "lucide-react";
import { ragApi } from "../../services/api";
import type { SearchLogEntry } from "../../types/api";

interface RetrievalLogsViewProps {
  onShowToast: (message: string, type: "success" | "error" | "info") => void;
  onSelectQuery?: (query: string) => void;
}

export const RetrievalLogsView: React.FC<RetrievalLogsViewProps> = ({
  onShowToast,
  onSelectQuery,
}) => {
  const [logs, setLogs] = useState<SearchLogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [filterText, setFilterText] = useState("");

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async (isManual = false) => {
    setLoading(true);
    try {
      const data = await ragApi.getSearchLogs(50);
      setLogs(Array.isArray(data) ? data : []);
      if (isManual) {
        onShowToast("Search audit logs refreshed", "success");
      }
    } catch {
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  const filteredLogs = logs.filter((log) =>
    log.query?.toLowerCase().includes(filterText.toLowerCase())
  );

  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return iso;
    }
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200/90 shadow-2xs flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative w-full">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter audit logs by query text..."
              className="w-full text-xs pl-8.5 pr-3 py-1.5 rounded-md border border-slate-200 bg-slate-50/50 hover:bg-slate-50 focus:bg-white focus:border-blue-500 focus:outline-hidden transition-colors"
            />
          </div>
        </div>

        <button
          onClick={() => fetchLogs(true)}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-200 hover:bg-slate-50 text-xs text-slate-700 transition-colors disabled:opacity-50 self-end sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>Refresh Logs</span>
        </button>
      </div>

      {/* Logs Table Surface */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs overflow-hidden">
        {loading && logs.length === 0 ? (
          <div className="p-12 text-center space-y-3">
            <RefreshCw className="w-6 h-6 text-slate-400 animate-spin mx-auto" />
            <p className="text-xs text-slate-500">Loading search logs from MongoDB...</p>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="p-16 text-center space-y-3">
            <div className="w-12 h-12 mx-auto rounded-xl bg-slate-100 flex items-center justify-center text-slate-400">
              <Activity className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-slate-800">
                {logs.length === 0 ? "No search logs recorded yet" : "No matching queries"}
              </h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                {logs.length === 0
                  ? "Queries submitted through /api/rag/search and /api/rag/ask are automatically audited in MongoDB."
                  : "Try clearing your search filter."}
              </p>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-semibold text-slate-500 uppercase tracking-wider font-mono">
                  <th className="py-2.5 px-4">Query</th>
                  <th className="py-2.5 px-3">Chunks Recalled</th>
                  <th className="py-2.5 px-3">Top Similarity</th>
                  <th className="py-2.5 px-3">Latency</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {filteredLogs.map((log, idx) => {
                  const maxScore =
                    log.similarity_scores && log.similarity_scores.length > 0
                      ? Math.max(...log.similarity_scores)
                      : null;

                  return (
                    <tr
                      key={log.request_id || idx}
                      className="hover:bg-slate-50/70 transition-colors"
                    >
                      <td className="py-3 px-4 font-sans max-w-xs truncate">
                        <div className="font-medium text-slate-900 truncate" title={log.query}>
                          {log.query}
                        </div>
                        <div className="text-[10px] text-slate-400 font-mono">
                          {log.request_id || "req_internal"}
                        </div>
                      </td>

                      <td className="py-3 px-3 text-slate-700">
                        <span className="flex items-center gap-1">
                          <Layers className="w-3 h-3 text-slate-400" />
                          {log.chunks_retrieved_count ?? 0}
                        </span>
                      </td>

                      <td className="py-3 px-3">
                        {maxScore !== null ? (
                          <span className="text-emerald-700 font-medium">
                            {Math.round(maxScore * 100)}%
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>

                      <td className="py-3 px-3 text-slate-600">
                        {log.took_ms !== undefined ? `${log.took_ms} ms` : "—"}
                      </td>

                      <td className="py-3 px-3 text-slate-500 text-[11px]">
                        {formatDate(log.timestamp)}
                      </td>

                      <td className="py-3 px-4 text-right">
                        {onSelectQuery && (
                          <button
                            onClick={() => onSelectQuery(log.query)}
                            className="text-xs font-sans text-blue-600 hover:text-blue-700 font-medium"
                          >
                            Re-run
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="px-4 py-3 border-t border-slate-100 bg-slate-50/40 flex items-center justify-between text-[11px] text-slate-500 font-mono">
          <span>MongoDB collection: universal_rag.rag_logs</span>
          <span>Showing {filteredLogs.length} events</span>
        </div>
      </div>
    </div>
  );
};
