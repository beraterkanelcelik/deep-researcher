import { useState, useEffect, useCallback } from "react";
import {
  uploadDocument,
  listDocuments,
  deleteDocument,
} from "@/lib/chatApi";
import { Upload, FileText, Trash2, Loader2 } from "lucide-react";

interface DocumentItem {
  id: string;
  filename: string;
  chunks: number;
  created_at: string;
}

export function DocumentUpload() {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const loadDocuments = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocuments(data);
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  }, []);

  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await uploadDocument(file);
      }
      loadDocuments();
    } catch (err) {
      console.error("Failed to upload:", err);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      loadDocuments();
    } catch (err) {
      console.error("Failed to delete document:", err);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  };

  return (
    <div className="p-3 border-t border-zinc-800">
      <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-2 px-1">
        Documents
      </h3>

      {/* Upload area */}
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center justify-center gap-1 p-3 rounded-lg border border-dashed cursor-pointer transition-colors text-xs ${
          dragOver
            ? "border-violet-500 bg-violet-500/10 text-violet-400"
            : "border-zinc-700 hover:border-zinc-600 text-zinc-500 hover:text-zinc-400"
        }`}
      >
        {uploading ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <Upload className="h-4 w-4" />
        )}
        <span>{uploading ? "Uploading..." : "Drop files or click"}</span>
        <input
          type="file"
          className="hidden"
          multiple
          accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.ts,.html,.css"
          onChange={(e) => handleUpload(e.target.files)}
          disabled={uploading}
        />
      </label>

      {/* Document list */}
      {documents.length > 0 && (
        <div className="mt-2 space-y-1">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="group flex items-center gap-2 px-2 py-1.5 rounded text-xs text-zinc-400"
            >
              <FileText className="h-3 w-3 flex-shrink-0" />
              <span className="truncate flex-1" title={doc.filename}>
                {doc.filename}
              </span>
              <span className="text-zinc-600 flex-shrink-0">
                {doc.chunks}ch
              </span>
              <button
                onClick={() => handleDelete(doc.id)}
                className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-red-400 transition-all"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
