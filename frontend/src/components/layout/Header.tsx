import React from "react";
import { Menu, Plus, Search, RefreshCw, CheckCircle2, AlertCircle } from "lucide-react";
import type { NavTab } from "./Sidebar";

interface HeaderProps {
  activeTab: NavTab;
  onOpenMobileSidebar: () => void;
  onOpenUploadModal: () => void;
  onQuickSearchClick: () => void;
  onRefreshHealth: () => void;
  isHealthy: boolean;
  isCheckingHealth: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onOpenMobileSidebar,
  onOpenUploadModal,
  onQuickSearchClick,
  onRefreshHealth,
  isHealthy,
  isCheckingHealth,
}) => {
  const titles: Record<NavTab, { title: string; subtitle: string }> = {
    dashboard: {
      title: "Workspace Overview",
      subtitle: "Ingestion statistics, vector store metrics, and storage activity",
    },
    documents: {
      title: "Document Library",
      subtitle: "Manage ingested files, chunk partitions, and metadata scopes",
    },
    search: {
      title: "Knowledge Search",
      subtitle: "Hybrid semantic & keyword retrieval, prompt context, and grounded answers",
    },
    logs: {
      title: "Retrieval Logs",
      subtitle: "Audit log of search requests, candidate recall, and query latency",
    },
    settings: {
      title: "Settings & System",
      subtitle: "API endpoint configuration, pgvector verification, and model parameters",
    },
  };

  const current = titles[activeTab];

  return (
    <header className="h-16 px-6 bg-white border-b border-slate-200/90 flex items-center justify-between sticky top-0 z-20">
      {/* Left: Mobile hamburger + Page Title */}
      <div className="flex items-center gap-3">
        <button
          onClick={onOpenMobileSidebar}
          className="p-1.5 -ml-1 text-slate-500 hover:text-slate-800 md:hidden rounded-md hover:bg-slate-100"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div>
          <h1 className="text-base font-semibold text-slate-900 tracking-tight leading-tight">
            {current.title}
          </h1>
          <p className="text-[12px] text-slate-500 hidden sm:block leading-none mt-0.5">
            {current.subtitle}
          </p>
        </div>
      </div>

      {/* Right: Search shortcut, Upload Document primary action, Status indicator */}
      <div className="flex items-center gap-3">
        {/* Quick search button for non-search views */}
        {activeTab !== "search" && (
          <button
            onClick={onQuickSearchClick}
            className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-md border border-slate-200 bg-slate-50/80 hover:bg-slate-100 text-xs text-slate-500 transition-colors shadow-2xs"
          >
            <Search className="w-3.5 h-3.5 text-slate-400" />
            <span>Search documents...</span>
            <kbd className="text-[10px] font-mono text-slate-400 bg-white px-1.5 py-0.5 rounded border border-slate-200">
              ⌘K
            </kbd>
          </button>
        )}

        {/* System Health Heartbeat Pill */}
        <button
          onClick={onRefreshHealth}
          disabled={isCheckingHealth}
          title="Click to re-probe backend health"
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-slate-200 hover:bg-slate-50 text-[12px] font-mono text-slate-600 transition-colors cursor-pointer disabled:opacity-50"
        >
          {isCheckingHealth ? (
            <RefreshCw className="w-3.5 h-3.5 text-slate-400 animate-spin" />
          ) : isHealthy ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          ) : (
            <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
          )}
          <span className="hidden sm:inline">
            {isCheckingHealth ? "Probing..." : isHealthy ? "Online" : "Check Status"}
          </span>
        </button>

        {/* Primary Action: Upload Document */}
        <button
          onClick={onOpenUploadModal}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-[13px] font-medium transition-colors shadow-xs active:scale-[0.98]"
        >
          <Plus className="w-4 h-4" />
          <span>Upload Document</span>
        </button>
      </div>
    </header>
  );
};
