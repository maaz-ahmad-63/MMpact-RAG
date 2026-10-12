import React, { useState, useRef } from "react";
import {
  X,
  UploadCloud,
  FileText,
  CheckCircle2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { ragApi } from "../../services/api";
import type { UploadResponse } from "../../types/api";

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (res: UploadResponse) => void;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [docId, setDocId] = useState("");
  const [userId, setUserId] = useState("");
  const [agentId, setAgentId] = useState("");
  const [convId, setConvId] = useState("");
  const [metadataStr, setMetadataStr] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (selectedFile: File) => {
    setErrorMsg(null);
    setUploadResult(null);

    // Validate size (25MB)
    if (selectedFile.size > 25 * 1024 * 1024) {
      setErrorMsg("File exceeds the 25 MB limit.");
      return;
    }

    // Validate type extension
    const name = selectedFile.name.toLowerCase();
    const valid = [".pdf", ".docx", ".txt", ".md"].some((ext) => name.endsWith(ext));
    if (!valid) {
      setErrorMsg("Unsupported file type. Please upload a PDF, DOCX, TXT, or MD file.");
      return;
    }

    setFile(selectedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      setErrorMsg("Please select a document to upload.");
      return;
    }

    let parsedMeta: Record<string, any> | undefined = undefined;
    if (metadataStr.trim()) {
      try {
        parsedMeta = JSON.parse(metadataStr);
      } catch {
        setErrorMsg("Metadata must be valid JSON.");
        return;
      }
    }

    setUploading(true);
    setProgress(0);
    setErrorMsg(null);

    try {
      const res = await ragApi.upload(file, {
        document_id: docId.trim() || undefined,
        user_id: userId.trim() || undefined,
        agent_id: agentId.trim() || undefined,
        conversation_id: convId.trim() || undefined,
        metadata: parsedMeta,
        onProgress: (p) => setProgress(p),
      });

      setUploadResult(res);
      onSuccess(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Upload and ingestion failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleResetAndClose = () => {
    setFile(null);
    setDocId("");
    setUserId("");
    setAgentId("");
    setConvId("");
    setMetadataStr("");
    setUploadResult(null);
    setErrorMsg(null);
    onClose();
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs">
      <div className="bg-white rounded-xl border border-slate-200 shadow-xl max-w-lg w-full overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-slate-900">Upload & Ingest Document</h2>
            <p className="text-xs text-slate-500">
              Extracts text, normalizes chunks, and generates vector embeddings.
            </p>
          </div>
          <button
            onClick={handleResetAndClose}
            className="p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-4 max-h-[75vh] overflow-y-auto">
          {errorMsg && (
            <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <div className="flex-1">{errorMsg}</div>
            </div>
          )}

          {uploadResult ? (
            <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-200 text-slate-800 space-y-3">
              <div className="flex items-center gap-2 text-emerald-800 font-semibold text-sm">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Document Ingested Successfully</span>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-white/80 p-2 rounded border border-emerald-100">
                  <div className="text-slate-500 text-[10px]">DOCUMENT ID</div>
                  <div className="font-semibold text-slate-800 truncate" title={uploadResult.document_id}>
                    {uploadResult.document_id}
                  </div>
                </div>
                <div className="bg-white/80 p-2 rounded border border-emerald-100">
                  <div className="text-slate-500 text-[10px]">CHUNKS GENERATED</div>
                  <div className="font-semibold text-slate-800">{uploadResult.chunks}</div>
                </div>
                <div className="bg-white/80 p-2 rounded border border-emerald-100">
                  <div className="text-slate-500 text-[10px]">CHARACTERS CLEANED</div>
                  <div className="font-semibold text-slate-800">{uploadResult.characters.toLocaleString()}</div>
                </div>
                <div className="bg-white/80 p-2 rounded border border-emerald-100">
                  <div className="text-slate-500 text-[10px]">INDEXING LATENCY</div>
                  <div className="font-semibold text-slate-800">{uploadResult.took_ms} ms</div>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* File Dropzone */}
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                  dragActive
                    ? "border-blue-500 bg-blue-50/50"
                    : file
                    ? "border-emerald-300 bg-emerald-50/30"
                    : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/50"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  onChange={handleFileChange}
                  className="hidden"
                />

                {file ? (
                  <div className="flex items-center justify-between p-2 rounded-lg bg-white border border-slate-200">
                    <div className="flex items-center gap-3 truncate">
                      <div className="p-2 rounded bg-blue-50 text-blue-600">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div className="text-left truncate">
                        <div className="text-xs font-semibold text-slate-800 truncate">
                          {file.name}
                        </div>
                        <div className="text-[11px] text-slate-500 font-mono">
                          {formatFileSize(file.size)}
                        </div>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setFile(null);
                      }}
                      className="text-xs text-slate-400 hover:text-slate-700 px-2 py-1 rounded"
                    >
                      Change
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="w-10 h-10 mx-auto rounded-full bg-slate-100 flex items-center justify-center text-slate-500">
                      <UploadCloud className="w-5 h-5" />
                    </div>
                    <div>
                      <span className="text-xs font-semibold text-blue-600 hover:text-blue-700">
                        Click to upload
                      </span>{" "}
                      <span className="text-xs text-slate-500">or drag and drop</span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      PDF, DOCX, TXT, or MD (up to 25 MB)
                    </p>
                  </div>
                )}
              </div>

              {/* Progress bar */}
              {uploading && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[11px] text-slate-500">
                    <span>Embedding & Ingesting...</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="w-full bg-slate-100 h-1.5 rounded-full overflow-hidden">
                    <div
                      className="bg-blue-600 h-full rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Advanced Metadata / Scopes Toggle */}
              <div>
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 transition-colors"
                >
                  {showAdvanced ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                  <span>Optional Scopes & Metadata</span>
                </button>

                {showAdvanced && (
                  <div className="mt-3 p-3.5 rounded-lg border border-slate-200/90 bg-slate-50/50 space-y-3 animate-in fade-in duration-150">
                    <div className="grid grid-cols-2 gap-2.5">
                      <div>
                        <label className="block text-[11px] font-medium text-slate-600 mb-1">
                          Custom Document ID
                        </label>
                        <input
                          type="text"
                          value={docId}
                          onChange={(e) => setDocId(e.target.value)}
                          placeholder="Autogenerated if empty"
                          className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-medium text-slate-600 mb-1">
                          User / Tenant ID Scope
                        </label>
                        <input
                          type="text"
                          value={userId}
                          onChange={(e) => setUserId(e.target.value)}
                          placeholder="e.g. user_101"
                          className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-medium text-slate-600 mb-1">
                          Agent ID Scope
                        </label>
                        <input
                          type="text"
                          value={agentId}
                          onChange={(e) => setAgentId(e.target.value)}
                          placeholder="e.g. support_agent"
                          className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-medium text-slate-600 mb-1">
                          Conversation ID Scope
                        </label>
                        <input
                          type="text"
                          value={convId}
                          onChange={(e) => setConvId(e.target.value)}
                          placeholder="e.g. thread_xyz"
                          className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-[11px] font-medium text-slate-600 mb-1">
                        Metadata JSON
                      </label>
                      <textarea
                        value={metadataStr}
                        onChange={(e) => setMetadataStr(e.target.value)}
                        placeholder='{"category": "policy", "version": "1.0"}'
                        rows={2}
                        className="w-full text-xs px-2.5 py-1.5 rounded-md border border-slate-200 bg-white font-mono focus:border-blue-500 focus:outline-hidden resize-none"
                      />
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-slate-100 bg-slate-50/50 flex items-center justify-end gap-2.5">
          {uploadResult ? (
            <button
              onClick={handleResetAndClose}
              className="px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors"
            >
              Done
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={handleResetAndClose}
                disabled={uploading}
                className="px-3 py-1.5 rounded-md border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!file || uploading}
                className="px-3.5 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium transition-colors disabled:opacity-50 active:scale-[0.98]"
              >
                {uploading ? "Ingesting..." : "Start Ingestion"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
