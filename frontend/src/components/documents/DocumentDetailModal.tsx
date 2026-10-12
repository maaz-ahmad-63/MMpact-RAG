import React, { useState } from "react";
import { X, Copy, Check, FileText, Calendar, Layers } from "lucide-react";
import type { DocumentSummary } from "../../types/api";
import { FileTypeBadge } from "../ui/Badge";

interface DocumentDetailModalProps {
  document: DocumentSummary | null;
  isOpen: boolean;
  onClose: () => void;
  onDeleteClick: (doc: DocumentSummary) => void;
}

export const DocumentDetailModal: React.FC<DocumentDetailModalProps> = ({
  document,
  isOpen,
  onClose,
  onDeleteClick,
}) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen || !document) return null;

  const handleCopyId = () => {
    navigator.clipboard.writeText(document.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatDate = (iso: string | null) => {
    if (!iso) return "Unknown";
    try {
      return new Date(iso).toLocaleString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return iso;
    }
  };

  const formatFileSize = (bytes?: number | null) => {
    if (!bytes) return null;
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="bg-white rounded-xl border border-slate-200 shadow-xl max-w-lg w-full overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-blue-50 text-blue-600">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900 truncate max-w-xs" title={document.filename}>
                {document.filename}
              </h2>
              <div className="flex items-center gap-2 mt-0.5">
                <FileTypeBadge fileType={document.file_type || document.mime_type} />
                <span className="text-[11px] text-slate-500 font-mono">
                  {document.chunk_count} {document.chunk_count === 1 ? "chunk" : "chunks"}
                </span>
                {formatFileSize(document.file_size) && (
                  <>
                    <span className="text-slate-300">•</span>
                    <span className="text-[11px] text-slate-500 font-mono">
                      {formatFileSize(document.file_size)}
                    </span>
                  </>
                )}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-4 max-h-[70vh] overflow-y-auto">
          {/* Document ID with copy */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-between">
            <div>
              <div className="text-[10px] font-mono font-semibold uppercase text-slate-400">
                Document ID
              </div>
              <div className="text-xs font-mono text-slate-800 break-all select-all">
                {document.id}
              </div>
            </div>
            <button
              onClick={handleCopyId}
              className="p-1.5 rounded-md hover:bg-white text-slate-500 hover:text-slate-800 transition-all flex items-center gap-1 text-[11px] font-mono border border-transparent hover:border-slate-200 ml-2"
              title="Copy Document ID"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="text-emerald-600">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>

          {/* Properties Grid */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-2.5 rounded-lg border border-slate-200/80 bg-white">
              <div className="text-[10px] text-slate-400 font-mono uppercase mb-0.5">Ingested At</div>
              <div className="font-medium text-slate-700 flex items-center gap-1.5">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                <span>{formatDate(document.created_at)}</span>
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-200/80 bg-white">
              <div className="text-[10px] text-slate-400 font-mono uppercase mb-0.5">Stored Chunks</div>
              <div className="font-medium text-slate-700 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-slate-400" />
                <span>{document.chunk_count} partitions</span>
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-200/80 bg-white">
              <div className="text-[10px] text-slate-400 font-mono uppercase mb-0.5">User / Tenant Scope</div>
              <div className="font-mono text-slate-700 truncate">
                {document.user_id || "None (Global)"}
              </div>
            </div>

            <div className="p-2.5 rounded-lg border border-slate-200/80 bg-white">
              <div className="text-[10px] text-slate-400 font-mono uppercase mb-0.5">Agent / Thread Scope</div>
              <div className="font-mono text-slate-700 truncate">
                {document.agent_id || document.conversation_id || "None"}
              </div>
            </div>
          </div>

          {/* Metadata JSON */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-slate-700">Metadata Payload</span>
              <span className="text-[10px] text-slate-400 font-mono">
                {Object.keys(document.metadata || {}).length} keys
              </span>
            </div>
            <pre className="p-3 rounded-lg bg-slate-900 text-slate-100 text-[11px] font-mono overflow-x-auto max-h-48 leading-relaxed border border-slate-800">
              {JSON.stringify(document.metadata || {}, null, 2)}
            </pre>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/50 flex items-center justify-between">
          <button
            onClick={() => {
              onClose();
              onDeleteClick(document);
            }}
            className="text-xs text-rose-600 hover:text-rose-700 hover:underline font-medium"
          >
            Delete Document
          </button>
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
