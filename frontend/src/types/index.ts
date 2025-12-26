/**
 * TypeScript types for Morphometry Analysis
 */

export interface CellMeasurement {
  cell_id: number;
  area: number;
  perimeter: number;
  circularity: number;
  eccentricity: number;
  solidity: number;
  aspect_ratio: number;
  centroid_x: number;
  centroid_y: number;
  intensity_mean: number | null;
}

export interface SummaryStatistics {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
}

export interface AnalysisSummary {
  total_cells: number;
  cell_density: number;
  area_stats: SummaryStatistics | null;
  circularity_stats: SummaryStatistics | null;
  processing_time_seconds: number;
}

export interface AnalysisResponse {
  success: boolean;
  message: string;
  analysis_id: string;
  summary: AnalysisSummary;
  measurements: CellMeasurement[];
  report_urls: Record<string, string> | null;
  mask_url: string | null;
  overlay_url: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  gpu_available: boolean;
  cellpose_available: boolean;
  timestamp: string;
}

export interface SegmentationModel {
  id: string;
  name: string;
  description: string;
  recommended: boolean;
}

export interface ImageModality {
  id: string;
  name: string;
  description: string;
}

export type AnalysisStatus = 'idle' | 'uploading' | 'processing' | 'complete' | 'error';
