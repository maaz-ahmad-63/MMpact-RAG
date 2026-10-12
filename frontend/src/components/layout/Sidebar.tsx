import React from "react";
import {
  LayoutDashboard,
  FileText,
  Search,
  Activity,
  Settings,
  Layers,
  Database,
  ExternalLink,
  ChevronRight,
} from "lucide-react";

export type NavTab = "dashboard" | "documents" | "search" | "logs" | "settings";

interface SidebarProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  documentCount: number;
  isDbConnected: boolean;
  dimension: number;
  isOpenMobile: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  documentCount,
  isDbConnected,
  dimension,
  isOpenMobile,
  onCloseMobile,
}) => {
  const navItems = [
    {
      id: "dashboard" as NavTab,
      label: "Overview",
      icon: LayoutDashboard,
      badge: null,
    },
    {
      id: "documents" as NavTab,
      label: "Documents",
      icon: FileText,
      badge: documentCount > 0 ? documentCount : null,
    },
    {
      id: "search" as NavTab,
      label: "Knowledge Search",
      icon: Search,
      badge: null,
      hotkey: "⌘K",
    },
    {
      id: "logs" as NavTab,
      label: "Retrieval Logs",
      icon: Activity,
      badge: null,
    },
    {
      id: "settings" as NavTab,
      label: "Settings & System",
      icon: Settings,
      badge: null,
    },
  ];

  return (
    <>
      {/* Mobile backdrop */}
      {isOpenMobile && (
        <div
          className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-xs md:hidden"
          onClick={onCloseMobile}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 bg-white border-r border-slate-200/90 flex flex-col transition-transform duration-200 ease-in-out md:static md:translate-x-0 ${
          isOpenMobile ? "translate-x-0" : "-translate-x-0 -translate-x-full md:translate-x-0"
        }`}
      >
        {/* Brand / Workspace header */}
        <div className="h-14 px-4 border-b border-slate-200/80 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-xs">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-[13px] font-semibold text-slate-900 tracking-tight">MMpact RAG</span>
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-100 text-slate-500 font-medium">v2.0</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-none">Enterprise Knowledge</p>
            </div>
          </div>
        </div>

        {/* Navigation list */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <div className="px-3 pb-2 text-[10px] font-semibold tracking-wider text-slate-400 uppercase font-mono">
            Platform
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onSelectTab(item.id);
                  onCloseMobile();
                }}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-[13px] font-medium transition-colors ${
                  isActive
                    ? "bg-slate-100 text-slate-900 font-semibold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Icon
                    className={`w-4 h-4 ${
                      isActive ? "text-blue-600" : "text-slate-400"
                    }`}
                  />
                  <span>{item.label}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  {item.badge !== null && (
                    <span className="text-[11px] font-mono px-1.5 py-0.5 rounded-full bg-slate-200/80 text-slate-700 font-medium">
                      {item.badge}
                    </span>
                  )}
                  {item.hotkey && (
                    <kbd className="text-[10px] font-mono text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded border border-slate-200">
                      {item.hotkey}
                    </kbd>
                  )}
                </div>
              </button>
            );
          })}
        </nav>

        {/* System & Connection Status Footer */}
        <div className="p-3 border-t border-slate-200/80 bg-slate-50/60">
          <button
            onClick={() => {
              onSelectTab("settings");
              onCloseMobile();
            }}
            className="w-full text-left p-2.5 rounded-lg border border-slate-200/90 bg-white hover:border-slate-300 transition-all shadow-xs group"
          >
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-[11px] font-semibold text-slate-700">Vector Store</span>
              </div>
              <div className="flex items-center gap-1">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isDbConnected ? "bg-emerald-500" : "bg-rose-500"
                  }`}
                />
                <span className="text-[10px] text-slate-500 font-mono">
                  {isDbConnected ? "PostgreSQL" : "Offline"}
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
              <span>{dimension > 0 ? `${dimension}-d pgvector` : "Cosine HNSW"}</span>
              <ChevronRight className="w-3 h-3 text-slate-400 group-hover:translate-x-0.5 transition-transform" />
            </div>
          </button>

          <div className="mt-2.5 px-1 flex items-center justify-between text-[10px] text-slate-400">
            <span>FastAPI • BGE-M3 • RRF</span>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-0.5 hover:text-slate-600 transition-colors"
            >
              API Docs
              <ExternalLink className="w-2.5 h-2.5" />
            </a>
          </div>
        </div>
      </aside>
    </>
  );
};
