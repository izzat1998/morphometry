/**
 * Web Report Component
 * Full, printable analysis report in web format
 */

import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell
} from 'recharts';
import {
  X, Printer, Microscope, Calendar, Hash, Clock,
  Activity, Target, CircleDot, Maximize2
} from 'lucide-react';
import type { AnalysisResponse, CellMeasurement, SummaryStatistics } from '../types';
import { getFullDownloadUrl } from '../services/api';

interface WebReportProps {
  results: AnalysisResponse;
  onClose: () => void;
}

// Chart colors
const CHART_COLORS = {
  primary: '#4f46e5',
  secondary: '#8b5cf6',
  tertiary: '#ec4899',
  grid: '#e2e8f0',
  axis: '#64748b',
};

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number; name?: string }>;
  label?: string;
}) => {
  if (active && payload && payload.length) {
    return (
      <div className="report-tooltip">
        <p className="tooltip-label">Value: {label}</p>
        <p className="tooltip-value">Count: {payload[0].value}</p>
      </div>
    );
  }
  return null;
};

// Scatter tooltip
const ScatterTooltip = ({ active, payload }: {
  active?: boolean;
  payload?: Array<{ payload: CellMeasurement }>;
}) => {
  if (active && payload && payload.length) {
    const cell = payload[0].payload;
    return (
      <div className="report-tooltip">
        <p className="tooltip-label">Cell #{cell.cell_id}</p>
        <p className="tooltip-value">Area: {cell.area.toFixed(1)} px²</p>
        <p className="tooltip-value">Circularity: {cell.circularity.toFixed(3)}</p>
      </div>
    );
  }
  return null;
};

export function WebReport({ results, onClose }: WebReportProps) {
  const [isPrinting, setIsPrinting] = useState(false);
  const { summary, measurements, analysis_id, mask_url, overlay_url } = results;

  // Create histogram data
  const createHistogram = (data: CellMeasurement[], field: keyof CellMeasurement, bins: number) => {
    const values = data.map(d => d[field] as number).filter(v => v != null);
    if (values.length === 0) return [];

    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins;

    return Array.from({ length: bins }, (_, i) => {
      const binStart = min + i * binWidth;
      const binEnd = min + (i + 1) * binWidth;
      const count = values.filter(v => v >= binStart && v < binEnd).length;
      return { bin: binStart.toFixed(1), count };
    });
  };

  const areaHistogram = createHistogram(measurements, 'area', 15);
  const circularityHistogram = createHistogram(measurements, 'circularity', 15);
  const eccentricityHistogram = createHistogram(measurements, 'eccentricity', 15);

  // Handle print
  const handlePrint = () => {
    setIsPrinting(true);
    setTimeout(() => {
      window.print();
      setIsPrinting(false);
    }, 100);
  };

  // Format date
  const formatDate = () => {
    return new Date().toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Render stat row
  const renderStatRow = (label: string, stats: SummaryStatistics | null) => {
    if (!stats) return null;
    return (
      <tr>
        <td>{label}</td>
        <td>{stats.mean.toFixed(3)}</td>
        <td>{stats.std.toFixed(3)}</td>
        <td>{stats.min.toFixed(3)}</td>
        <td>{stats.max.toFixed(3)}</td>
        <td>{stats.median.toFixed(3)}</td>
      </tr>
    );
  };

  return (
    <div className="web-report-overlay">
      <div className="web-report-container">
        {/* Header Actions */}
        <div className="report-actions no-print">
          <button className="action-btn print-btn" onClick={handlePrint} disabled={isPrinting}>
            <Printer size={18} />
            {isPrinting ? 'Preparing...' : 'Print Report'}
          </button>
          <button className="action-btn close-btn" onClick={onClose}>
            <X size={18} />
            Close
          </button>
        </div>

        {/* Report Content */}
        <div className="web-report">
          {/* Report Header */}
          <header className="report-header">
            <div className="report-logo">
              <Microscope size={40} />
            </div>
            <div className="report-title-section">
              <h1>Cell Morphometry Analysis Report</h1>
              <p className="report-subtitle">Automated Cell Segmentation & Measurement Analysis</p>
            </div>
            <div className="report-meta">
              <div className="meta-item">
                <Calendar size={14} />
                <span>{formatDate()}</span>
              </div>
              <div className="meta-item">
                <Hash size={14} />
                <span>ID: {analysis_id.slice(0, 8)}</span>
              </div>
            </div>
          </header>

          {/* Executive Summary */}
          <section className="report-section">
            <h2 className="section-title">
              <Target size={20} />
              Executive Summary
            </h2>
            <div className="summary-grid">
              <div className="summary-stat-card primary">
                <div className="stat-icon">
                  <Hash size={24} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">{summary.total_cells}</span>
                  <span className="stat-label">Total Cells Detected</span>
                </div>
              </div>
              <div className="summary-stat-card">
                <div className="stat-icon">
                  <Activity size={24} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">{summary.area_stats?.mean.toFixed(1) || 'N/A'}</span>
                  <span className="stat-label">Mean Area (px²)</span>
                </div>
              </div>
              <div className="summary-stat-card">
                <div className="stat-icon">
                  <CircleDot size={24} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">{summary.circularity_stats?.mean.toFixed(3) || 'N/A'}</span>
                  <span className="stat-label">Mean Circularity</span>
                </div>
              </div>
              <div className="summary-stat-card">
                <div className="stat-icon">
                  <Clock size={24} />
                </div>
                <div className="stat-content">
                  <span className="stat-value">{summary.processing_time_seconds.toFixed(2)}s</span>
                  <span className="stat-label">Processing Time</span>
                </div>
              </div>
            </div>
          </section>

          {/* Segmentation Visualization */}
          {(mask_url || overlay_url) && (
            <section className="report-section">
              <h2 className="section-title">
                <Maximize2 size={20} />
                Segmentation Results
              </h2>
              <div className="image-grid">
                {overlay_url && (
                  <div className="image-card">
                    <h4>Segmentation Overlay</h4>
                    <img
                      src={getFullDownloadUrl(overlay_url)}
                      alt="Segmentation Overlay"
                      className="result-image"
                    />
                    <p className="image-caption">Original image with detected cell boundaries</p>
                  </div>
                )}
                {mask_url && (
                  <div className="image-card">
                    <h4>Segmentation Mask</h4>
                    <img
                      src={getFullDownloadUrl(mask_url)}
                      alt="Segmentation Mask"
                      className="result-image"
                    />
                    <p className="image-caption">Binary mask showing identified cell regions</p>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* Distribution Analysis */}
          {measurements.length > 0 && (
            <section className="report-section">
              <h2 className="section-title">
                <Activity size={20} />
                Distribution Analysis
              </h2>

              <div className="charts-row">
                <div className="chart-card">
                  <h4>Cell Area Distribution</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={areaHistogram} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={CHART_COLORS.primary} stopOpacity={0.9} />
                          <stop offset="100%" stopColor={CHART_COLORS.primary} stopOpacity={0.4} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                      <XAxis dataKey="bin" tick={{ fontSize: 9, fill: CHART_COLORS.axis }} />
                      <YAxis tick={{ fontSize: 9, fill: CHART_COLORS.axis }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="url(#areaGrad)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-card">
                  <h4>Circularity Distribution</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={circularityHistogram} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="circGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={CHART_COLORS.secondary} stopOpacity={0.9} />
                          <stop offset="100%" stopColor={CHART_COLORS.secondary} stopOpacity={0.4} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                      <XAxis dataKey="bin" tick={{ fontSize: 9, fill: CHART_COLORS.axis }} />
                      <YAxis tick={{ fontSize: 9, fill: CHART_COLORS.axis }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="url(#circGrad)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="chart-card">
                  <h4>Eccentricity Distribution</h4>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={eccentricityHistogram} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                      <defs>
                        <linearGradient id="eccGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor={CHART_COLORS.tertiary} stopOpacity={0.9} />
                          <stop offset="100%" stopColor={CHART_COLORS.tertiary} stopOpacity={0.4} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} vertical={false} />
                      <XAxis dataKey="bin" tick={{ fontSize: 9, fill: CHART_COLORS.axis }} />
                      <YAxis tick={{ fontSize: 9, fill: CHART_COLORS.axis }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="count" fill="url(#eccGrad)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Scatter Plot: Area vs Circularity */}
              <div className="chart-card full-width">
                <h4>Cell Morphology Scatter Plot (Area vs Circularity)</h4>
                <ResponsiveContainer width="100%" height={280}>
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS.grid} />
                    <XAxis
                      type="number"
                      dataKey="area"
                      name="Area"
                      tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                      label={{ value: 'Area (px²)', position: 'bottom', fontSize: 11, fill: CHART_COLORS.axis }}
                    />
                    <YAxis
                      type="number"
                      dataKey="circularity"
                      name="Circularity"
                      tick={{ fontSize: 10, fill: CHART_COLORS.axis }}
                      label={{ value: 'Circularity', angle: -90, position: 'left', fontSize: 11, fill: CHART_COLORS.axis }}
                    />
                    <ZAxis type="number" dataKey="solidity" range={[20, 100]} />
                    <Tooltip content={<ScatterTooltip />} />
                    <Scatter data={measurements.slice(0, 200)} fill={CHART_COLORS.primary}>
                      {measurements.slice(0, 200).map((_, index) => (
                        <Cell key={`cell-${index}`} fillOpacity={0.6} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </section>
          )}

          {/* Statistical Summary */}
          <section className="report-section">
            <h2 className="section-title">
              <Activity size={20} />
              Statistical Summary
            </h2>
            <table className="report-table stats-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Mean</th>
                  <th>Std Dev</th>
                  <th>Min</th>
                  <th>Max</th>
                  <th>Median</th>
                </tr>
              </thead>
              <tbody>
                {renderStatRow('Area (px²)', summary.area_stats)}
                {renderStatRow('Circularity', summary.circularity_stats)}
              </tbody>
            </table>
          </section>

          {/* Individual Cell Measurements */}
          <section className="report-section page-break-before">
            <h2 className="section-title">
              <Hash size={20} />
              Individual Cell Measurements
            </h2>
            <p className="section-description">
              Detailed measurements for each detected cell. Showing {Math.min(measurements.length, 50)} of {measurements.length} cells.
            </p>
            <div className="table-wrapper">
              <table className="report-table measurements-table">
                <thead>
                  <tr>
                    <th>Cell ID</th>
                    <th>Area (px²)</th>
                    <th>Perimeter</th>
                    <th>Circularity</th>
                    <th>Eccentricity</th>
                    <th>Solidity</th>
                    <th>Aspect Ratio</th>
                  </tr>
                </thead>
                <tbody>
                  {measurements.slice(0, 50).map((cell) => (
                    <tr key={cell.cell_id}>
                      <td className="cell-id">{cell.cell_id}</td>
                      <td>{cell.area.toFixed(2)}</td>
                      <td>{cell.perimeter.toFixed(2)}</td>
                      <td>{cell.circularity.toFixed(4)}</td>
                      <td>{cell.eccentricity.toFixed(4)}</td>
                      <td>{cell.solidity.toFixed(4)}</td>
                      <td>{cell.aspect_ratio.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {measurements.length > 50 && (
              <p className="table-note">
                + {measurements.length - 50} additional cells. Download Excel/CSV for complete dataset.
              </p>
            )}
          </section>

          {/* Report Footer */}
          <footer className="report-footer">
            <div className="footer-line"></div>
            <div className="footer-content">
              <p className="footer-title">Morphometry Analysis System</p>
              <p className="footer-subtitle">
                Powered by Cellpose AI Segmentation | Report generated on {formatDate()}
              </p>
              <p className="footer-id">Analysis ID: {analysis_id}</p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}
