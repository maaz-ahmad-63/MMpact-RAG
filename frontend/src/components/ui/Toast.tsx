import React from "react";
import { CheckCircle2, AlertCircle, Info, X } from "lucide-react";

export interface ToastMessage {
  id: number;
  message: string;
  type: "success" | "error" | "info";
}

interface ToastContainerProps {
  toasts: ToastMessage[];
  onDismiss: (id: number) => void;
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2.5 max-w-md w-full pointer-events-none">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-lg border shadow-sm text-sm transition-all duration-200 animate-in fade-in slide-in-from-bottom-2 ${
            toast.type === "success"
              ? "bg-white border-emerald-200 text-slate-800"
              : toast.type === "error"
              ? "bg-white border-rose-200 text-slate-800"
              : "bg-white border-slate-200 text-slate-800"
          }`}
        >
          <div className="mt-0.5 flex-shrink-0">
            {toast.type === "success" && (
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            )}
            {toast.type === "error" && (
              <AlertCircle className="w-4 h-4 text-rose-600" />
            )}
            {toast.type === "info" && (
              <Info className="w-4 h-4 text-blue-600" />
            )}
          </div>
          <div className="flex-1 text-[13px] leading-snug">{toast.message}</div>
          <button
            onClick={() => onDismiss(toast.id)}
            className="text-slate-400 hover:text-slate-600 transition-colors p-0.5 rounded"
            aria-label="Dismiss toast"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  );
};
