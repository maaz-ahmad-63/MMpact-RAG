import React, { useState, useEffect, useCallback } from "react";
import { Sidebar } from "./components/layout/Sidebar";
import type { NavTab } from "./components/layout/Sidebar";
import { Header } from "./components/layout/Header";
import { DashboardView } from "./components/dashboard/DashboardView";
import { DocumentsView } from "./components/documents/DocumentsView";
import { KnowledgeSearchView } from "./components/search/KnowledgeSearchView";
import { RetrievalLogsView } from "./components/logs/RetrievalLogsView";
import { SettingsView } from "./components/settings/SettingsView";
import { UploadModal } from "./components/documents/UploadModal";
import { DocumentDetailModal } from "./components/documents/DocumentDetailModal";
import { DeleteConfirmModal } from "./components/documents/DeleteConfirmModal";
import { ToastContainer } from "./components/ui/Toast";
import type { ToastMessage } from "./components/ui/Toast";
import { ragApi, getApiSettings } from "./services/api";
import type { ApiSettings } from "./services/api";
import type {
  DocumentSummary,
  RagHealthResponse,
  ServiceHealthResponse,
  UploadResponse,
} from "./types/api";

export default function App() {
  const [activeTab, setActiveTab] = useState<NavTab>("dashboard");
  const [apiSettings, setApiSettings] = useState<ApiSettings>(getApiSettings());
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  // Health states
  const [ragHealth, setRagHealth] = useState<RagHealthResponse | null>(null);
  const [serviceHealth, setServiceHealth] = useState<ServiceHealthResponse | null>(null);
  const [healthChecking, setHealthChecking] = useState(false);

  // Document states
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);

  // Modals
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<DocumentSummary | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [docToDelete, setDocToDelete] = useState<DocumentSummary | null>(null);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);

  // Toasts
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const toastIdRef = React.useRef(0);

  const showToast = useCallback(
    (message: string, type: "success" | "error" | "info" = "info") => {
      const id = toastIdRef.current++;
      setToasts((prev) => [...prev, { id, message, type }]);
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 3500);
    },
    []
  );

  const dismissToast = (id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  // Health probe
  const checkHealth = useCallback(async () => {
    setHealthChecking(true);
    try {
      const [service, rag] = await Promise.allSettled([
        ragApi.getServiceHealth(),
        ragApi.getRagHealth(),
      ]);

      if (service.status === "fulfilled") {
        setServiceHealth(service.value);
      } else {
        setServiceHealth(null);
      }

      if (rag.status === "fulfilled") {
        setRagHealth(rag.value);
      } else {
        setRagHealth(null);
      }
    } finally {
      setHealthChecking(false);
    }
  }, []);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    setDocsLoading(true);
    try {
      const res = await ragApi.getDocuments(50, 0);
      setDocuments(res.documents);
    } catch (err: any) {
      showToast(err.message || "Failed to load documents", "error");
    } finally {
      setDocsLoading(false);
    }
  }, [showToast]);

  // Initial load
  useEffect(() => {
    checkHealth();
    fetchDocuments();
  }, [checkHealth, fetchDocuments]);

  // Sync documents when switching to documents or dashboard
  useEffect(() => {
    if (activeTab === "documents" || activeTab === "dashboard") {
      fetchDocuments();
    }
  }, [activeTab, fetchDocuments]);

  // Keyboard shortcut: Cmd+K / Ctrl+K opens Knowledge Search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setActiveTab("search");
      }
      if (e.key === "Escape") {
        setIsUploadOpen(false);
        setIsDetailOpen(false);
        setIsDeleteOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Document deletion handler
  const handleConfirmDelete = async (doc: DocumentSummary) => {
    setDeleting(true);
    try {
      await ragApi.deleteDocument(doc.id);
      showToast(`Document "${doc.filename}" and its chunks deleted`, "success");
      setIsDeleteOpen(false);
      setDocToDelete(null);
      fetchDocuments();
      checkHealth();
    } catch (err: any) {
      showToast(err.message || "Deletion failed", "error");
    } finally {
      setDeleting(false);
    }
  };

  const handleUploadSuccess = (res: UploadResponse) => {
    showToast(`"${res.filename}" ingested (${res.chunks} chunks)`, "success");
    fetchDocuments();
    checkHealth();
  };

  const isHealthy = Boolean(
    serviceHealth?.status === "healthy" || ragHealth?.database?.connected
  );

  const dimension =
    ragHealth?.database?.column_dimension ??
    ragHealth?.configuration?.embedding_dim ??
    1024;

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#f8fafc]">
      {/* Toast Notifications */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />

      {/* Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        documentCount={documents.length}
        isDbConnected={Boolean(ragHealth?.database?.connected)}
        dimension={dimension}
        isOpenMobile={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header */}
        <Header
          activeTab={activeTab}
          onOpenMobileSidebar={() => setMobileSidebarOpen(true)}
          onOpenUploadModal={() => setIsUploadOpen(true)}
          onQuickSearchClick={() => setActiveTab("search")}
          onRefreshHealth={checkHealth}
          isHealthy={isHealthy}
          isCheckingHealth={healthChecking}
        />

        {/* Scrollable Page Body */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {activeTab === "dashboard" && (
            <DashboardView
              documents={documents}
              ragHealth={ragHealth}
              onNavigate={setActiveTab}
              onOpenUpload={() => setIsUploadOpen(true)}
              onViewDocDetails={(doc) => {
                setSelectedDoc(doc);
                setIsDetailOpen(true);
              }}
            />
          )}

          {activeTab === "documents" && (
            <DocumentsView
              documents={documents}
              loading={docsLoading}
              onRefresh={fetchDocuments}
              onOpenUpload={() => setIsUploadOpen(true)}
              onViewDetails={(doc) => {
                setSelectedDoc(doc);
                setIsDetailOpen(true);
              }}
              onDeleteClick={(doc) => {
                setDocToDelete(doc);
                setIsDeleteOpen(true);
              }}
            />
          )}

          {activeTab === "search" && (
            <KnowledgeSearchView onShowToast={showToast} />
          )}

          {activeTab === "logs" && (
            <RetrievalLogsView
              onShowToast={showToast}
              onSelectQuery={() => {
                setActiveTab("search");
              }}
            />
          )}

          {activeTab === "settings" && (
            <SettingsView
              settings={apiSettings}
              onUpdateSettings={setApiSettings}
              ragHealth={ragHealth}
              onRefreshHealth={checkHealth}
              isCheckingHealth={healthChecking}
              onShowToast={showToast}
            />
          )}
        </main>
      </div>

      {/* Modals */}
      <UploadModal
        isOpen={isUploadOpen}
        onClose={() => setIsUploadOpen(false)}
        onSuccess={handleUploadSuccess}
      />

      <DocumentDetailModal
        document={selectedDoc}
        isOpen={isDetailOpen}
        onClose={() => {
          setIsDetailOpen(false);
          setSelectedDoc(null);
        }}
        onDeleteClick={(doc) => {
          setIsDetailOpen(false);
          setDocToDelete(doc);
          setIsDeleteOpen(true);
        }}
      />

      <DeleteConfirmModal
        document={docToDelete}
        isOpen={isDeleteOpen}
        onClose={() => {
          setIsDeleteOpen(false);
          setDocToDelete(null);
        }}
        onConfirm={handleConfirmDelete}
        deleting={deleting}
      />
    </div>
  );
}
