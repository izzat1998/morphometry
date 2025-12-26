/**
 * API Service for Morphometry Analysis
 */

import axios from 'axios';
import type { AnalysisResponse, HealthResponse, SegmentationModel, ImageModality } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for large images
});

export async function checkHealth(): Promise<HealthResponse> {
  const response = await api.get<HealthResponse>('/health');
  return response.data;
}

export async function analyzeImage(
  file: File,
  options: {
    modality?: string;
    model?: string;
    diameter?: number | null;
    pixel_size_um?: number;
    compute_texture?: boolean;
    generate_report?: boolean;
    report_format?: string;
  } = {}
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const params = new URLSearchParams();
  if (options.modality) params.append('modality', options.modality);
  if (options.model) params.append('model', options.model);
  if (options.diameter) params.append('diameter', options.diameter.toString());
  if (options.pixel_size_um) params.append('pixel_size_um', options.pixel_size_um.toString());
  if (options.compute_texture !== undefined) params.append('compute_texture', options.compute_texture.toString());
  if (options.generate_report !== undefined) params.append('generate_report', options.generate_report.toString());
  if (options.report_format) params.append('report_format', options.report_format);

  const response = await api.post<AnalysisResponse>(`/analyze?${params.toString()}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
}

export async function getModels(): Promise<{
  cellpose_available: boolean;
  gpu_available: boolean;
  models: SegmentationModel[];
}> {
  const response = await api.get('/models');
  return response.data;
}

export async function getModalities(): Promise<{ modalities: ImageModality[] }> {
  const response = await api.get('/modalities');
  return response.data;
}

export function getDownloadUrl(filename: string): string {
  return `${API_BASE_URL}/download/${filename}`;
}

export function getFullDownloadUrl(path: string): string {
  // Handle both relative and full paths
  if (path.startsWith('/download/')) {
    return `${API_BASE_URL}${path}`;
  }
  return `${API_BASE_URL}/download/${path}`;
}
