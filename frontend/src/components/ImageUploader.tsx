/**
 * Image Upload Component
 */

import { useCallback, useState } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  disabled?: boolean;
}

export function ImageUploader({ onImageSelect, disabled }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    onImageSelect(file);
  }, [onImageSelect]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const clearPreview = useCallback(() => {
    setPreview(null);
  }, []);

  return (
    <div className="image-uploader">
      {preview ? (
        <div className="preview-container">
          <img src={preview} alt="Preview" className="preview-image" />
          <button
            className="clear-button"
            onClick={clearPreview}
            disabled={disabled}
          >
            <X size={16} />
          </button>
        </div>
      ) : (
        <div
          className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            disabled={disabled}
            id="file-input"
            className="file-input"
          />
          <label htmlFor="file-input" className="drop-zone-content">
            <Upload size={48} className="upload-icon" />
            <p className="drop-text">
              Drag & drop a microscopy image here
            </p>
            <p className="drop-subtext">
              or click to browse
            </p>
            <p className="supported-formats">
              Supports: PNG, JPEG, TIFF
            </p>
          </label>
        </div>
      )}
    </div>
  );
}
