import React, { useState, useMemo } from "react";
import {
  FileText,
  Search,
  Plus,
  Trash2,
  Eye,
  Layers,
  RefreshCw,
  FolderOpen,
} from "lucide-react";
import type { DocumentSummary } from "../../types/api";
import { FileTypeBadge } from "../ui/Badge";

interface DocumentsViewProps {
  documents: DocumentSummary[];
  loading: boolean;
  onRefresh: () => void;
  onOpenUpload: () => void;
  onViewDetails: (doc: DocumentSummary) => void;
  onDeleteClick: (doc: DocumentSummary) => void;
}

export const DocumentsView: React.FC<DocumentsViewProps> = ({
  documents,
  loading,
  onRefresh,
  onOpenUpload,
  onViewDetails,
  onDeleteClick,
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState<string>("all");

  const filteredDocs = useMemo(() => {
    return documents.filter((doc) => {
      const matchesSearch =
        doc.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
        doc.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (doc.user_id && doc.user_id.toLowerCase().includes(searchQuery.toLowerCase()));

      if (!matchesSearch) return false;

      if (selectedType === "all") return true;
      const type = (doc.file_type || doc.mime_type || "").toLowerCase();
      if (selectedType === "pdf") return type.includes("pdf");
      if (selectedType === "docx") return type.includes("doc") || type.includes("word");
      if (selectedType === "txt") return type.includes("txt") || type.includes("plain");
      if (selectedType === "md") return type.includes("md") || type.includes("markdown");
      return true;
    });
  }, [documents, searchQuery, selectedType]);

  const formatDate = (iso: string | null) => {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    } catch {
      return iso;
    }
  };

  return (
    <div className="space-y-4">
      {/* Search & Filter Toolbar */}
      <div className="bg-white p-3.5 rounded-xl border border-slate-200/90 shadow-2xs flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative w-full">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter by filename, ID, or tenant scope..."
              className="w-full text-xs pl-8.5 pr-3 py-1.5 rounded-md border border-slate-200 bg-slate-50/50 hover:bg-slate-50 focus:bg-white focus:border-blue-500 focus:outline-hidden transition-colors"
            />
          </div>
        </div>

        <div className="flex items-center gap-2 self-end sm:self-auto">
          {/* File type filter pills */}
          <div className="flex items-center bg-slate-100 p-0.5 rounded-lg text-[11px] font-medium text-slate-600">
            {["all", "pdf", "docx", "txt", "md"].map((type) => (
              <button
                key={type}
                onClick={() => setSelectedType(type)}
                className={`px-2.5 py-1 rounded-md capitalize transition-colors ${
                  selectedType === type
                    ? "bg-white text-slate-900 shadow-2xs font-semibold"
                    : "text-slate-500 hover:text-slate-900"
                }`}
              >
                {type}
              </button>
            ))}
          </div>

          <button
            onClick={onRefresh}
            disabled={loading}
            className="p-1.5 rounded-md border border-slate-200 hover:bg-slate-50 text-slate-600 transition-colors disabled:opacity-50"
            title="Refresh document list"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      {/* Documents Table / Card Surface */}
      <div className="bg-white rounded-xl border border-slate-200/90 shadow-2xs overflow-hidden">
        {loading && documents.length === 0 ? (
          <div className="p-12 text-center space-y-3">
            <RefreshCw className="w-6 h-6 text-slate-400 animate-spin mx-auto" />
            <p className="text-xs text-slate-500">Loading document library...</p>
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="p-16 text-center space-y-3">
            <div className="w-12 h-12 mx-auto rounded-xl bg-slate-100 flex items-center justify-center text-slate-400">
              <FolderOpen className="w-6 h-6" />
            </div>
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-slate-800">
                {documents.length === 0 ? "No documents ingested" : "No matching documents"}
              </h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto">
                {documents.length === 0
                  ? "Ingest PDFs, Word documents, or text files to begin semantic vector retrieval."
                  : "Try adjusting your search query or file type filter."}
              </p>
            </div>
            {documents.length === 0 && (
              <button
                onClick={onOpenUpload}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Upload First Document</span>
              </button>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-100 bg-slate-50/70 text-[11px] font-semibold text-slate-500 uppercase tracking-wider font-mono">
                  <th className="py-2.5 px-4">Document</th>
                  <th className="py-2.5 px-3">Type</th>
                  <th className="py-2.5 px-3">Chunks</th>
                  <th className="py-2.5 px-3">Tenant / Scope</th>
                  <th className="py-2.5 px-3">Ingested</th>
                  <th className="py-2.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredDocs.map((doc) => (
                  <tr
                    key={doc.id}
                    onClick={() => onViewDetails(doc)}
                    className="hover:bg-slate-50/70 transition-colors cursor-pointer group"
                  >
                    {/* Filename & ID */}
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-100 text-slate-500 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors">
                          <FileText className="w-4 h-4" />
                        </div>
                        <div className="truncate max-w-xs sm:max-w-md">
                          <div className="font-medium text-slate-900 truncate" title={doc.filename}>
                            {doc.filename}
                          </div>
                          <div className="text-[10px] font-mono text-slate-400 truncate">
                            {doc.id}
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* File Type Badge */}
                    <td className="py-3 px-3">
                      <FileTypeBadge fileType={doc.file_type || doc.mime_type} />
                    </td>

                    {/* Chunk Count */}
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-1.5 font-mono text-slate-700">
                        <Layers className="w-3 h-3 text-slate-400" />
                        <span>{doc.chunk_count}</span>
                      </div>
                    </td>

                    {/* Scope */}
                    <td className="py-3 px-3">
                      <span className="font-mono text-[11px] text-slate-500">
                        {doc.user_id ? (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700">
                            {doc.user_id}
                          </span>
                        ) : (
                          <span className="text-slate-400">Global</span>
                        )}
                      </span>
                    </td>

                    {/* Created Date */}
                    <td className="py-3 px-3 text-slate-500 font-mono text-[11px]">
                      {formatDate(doc.created_at)}
                    </td>

                    {/* Action Buttons */}
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={() => onViewDetails(doc)}
                          className="p-1.5 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
                          title="View Details"
                        >
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => onDeleteClick(doc)}
                          className="p-1.5 rounded text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                          title="Delete Document"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Footer summary */}
        <div className="px-4 py-3 border-t border-slate-100 bg-slate-50/40 flex items-center justify-between text-[11px] text-slate-500 font-mono">
          <span>Showing {filteredDocs.length} of {documents.length} documents</span>
          <span>PostgreSQL pgvector storage</span>
        </div>
      </div>
    </div>
  );
};
