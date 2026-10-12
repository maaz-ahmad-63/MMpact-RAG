import React from "react";
import { AlertTriangle, Trash2 } from "lucide-react";
import type { DocumentSummary } from "../../types/api";

interface DeleteConfirmModalProps {
  document: DocumentSummary | null;
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (doc: DocumentSummary) => void;
  deleting: boolean;
}

export const DeleteConfirmModal: React.FC<DeleteConfirmModalProps> = ({
  document,
  isOpen,
  onClose,
  onConfirm,
  deleting,
}) => {
  if (!isOpen || !document) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="bg-white rounded-xl border border-slate-200 shadow-xl max-w-md w-full overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="p-5">
          <div className="flex items-start gap-3.5">
            <div className="p-2.5 rounded-full bg-rose-50 text-rose-600 border border-rose-100 flex-shrink-0">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <h2 className="text-sm font-semibold text-slate-900">Delete Document</h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Are you sure you want to permanently delete{" "}
                <span className="font-semibold text-slate-700">{document.filename}</span>?
              </p>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-lg bg-slate-50 border border-slate-200/80 text-[11px] text-slate-600 space-y-1">
            <div className="font-medium text-slate-700">Cascade Deletion Warning:</div>
            <p>
              This will remove the document record and delete all{" "}
              <span className="font-semibold">{document.chunk_count} chunk partitions</span> and their vector
              embeddings from PostgreSQL in a single atomic transaction.
            </p>
          </div>
        </div>

        <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/50 flex items-center justify-end gap-2.5">
          <button
            onClick={onClose}
            disabled={deleting}
            className="px-3 py-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm(document)}
            disabled={deleting}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-rose-600 hover:bg-rose-700 text-white text-xs font-medium transition-colors disabled:opacity-50"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>{deleting ? "Deleting..." : "Delete Document"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
