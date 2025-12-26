/**
 * Results Display Component
 * Scientific data visualization with professional light theme
 */

import { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Download, FileText, Table, FileJson, Activity, Clock, Hash, Globe } from 'lucide-react';
import type { AnalysisResponse, CellMeasurement } from '../types';
import { getFullDownloadUrl } from '../services/api';
import { WebReport } from './WebReport';

interface ResultsDisplayProps {
  results: AnalysisResponse;
}

// Custom tooltip for light theme
const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        padding: '10px 14px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
      }}>
        <p style={{
          margin: 0,
          fontFamily: "'Inter', sans-serif",
          fontSize: '11px',
          color: '#64748b',
        }}>
          Bin: {label}
        </p>
        <p style={{
          margin: '4px 0 0 0',
          fontFamily: "'Inter', sans-serif",
          fontSize: '13px',
          fontWeight: 600,
          color: '#0f172a',
        }}>
          Count: {payload[0].value}
        </p>
      </div>
    );
  }
  return null;
};

export function ResultsDisplay({ results }: ResultsDisplayProps) {
  const [showWebReport, setShowWebReport] = useState(false);
  const { summary, measurements, report_urls, mask_url } = results;

  // Prepare histogram data
  const areaHistogram = createHistogram(measurements, 'area', 20);
  const circularityHistogram = createHistogram(measurements, 'circularity', 20);

  // Chart styling constants - Light theme with indigo/violet accents
  const chartColors = {
    area: '#4f46e5',
    circularity: '#8b5cf6',
    grid: '#e2e8f0',
    axis: '#64748b',
  };

  return (
    <div className="results-display">
      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="summary-card">
          <Hash size={24} />
          <div className="card-content">
            <span className="card-value">{summary.total_cells}</span>
            <span className="card-label">Cells Detected</span>
          </div>
        </div>

        <div className="summary-card">
          <Clock size={24} />
          <div className="card-content">
            <span className="card-value">{summary.processing_time_seconds.toFixed(2)}s</span>
            <span className="card-label">Processing Time</span>
          </div>
        </div>

        {summary.area_stats && (
          <div className="summary-card">
            <Activity size={24} />
            <div className="card-content">
              <span className="card-value">{summary.area_stats.mean.toFixed(1)}</span>
              <span className="card-label">Mean Area (px²)</span>
            </div>
          </div>
        )}

        {summary.circularity_stats && (
          <div className="summary-card">
            <Activity size={24} />
            <div className="card-content">
              <span className="card-value">{summary.circularity_stats.mean.toFixed(3)}</span>
              <span className="card-label">Mean Circularity</span>
            </div>
          </div>
        )}
      </div>

      {/* Distribution Charts */}
      {measurements.length > 0 && (
        <div className="charts-section">
          <h3>Distribution Analysis</h3>
          <div className="charts-grid">
            <div className="chart-container">
              <h4>Cell Area Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={areaHistogram} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={chartColors.area} stopOpacity={0.9} />
                      <stop offset="100%" stopColor={chartColors.area} stopOpacity={0.4} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
                  <XAxis
                    dataKey="bin"
                    tick={{ fontSize: 10, fill: chartColors.axis }}
                    axisLine={{ stroke: chartColors.grid }}
                    tickLine={{ stroke: chartColors.grid }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: chartColors.axis }}
                    axisLine={{ stroke: chartColors.grid }}
                    tickLine={{ stroke: chartColors.grid }}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(79, 70, 229, 0.08)' }} />
                  <Bar dataKey="count" fill="url(#areaGradient)" radius={[4, 4, 0, 0]}>
                    {areaHistogram.map((_, index) => (
                      <Cell key={`cell-${index}`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-container">
              <h4>Circularity Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={circularityHistogram} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="circularityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={chartColors.circularity} stopOpacity={0.9} />
                      <stop offset="100%" stopColor={chartColors.circularity} stopOpacity={0.4} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
                  <XAxis
                    dataKey="bin"
                    tick={{ fontSize: 10, fill: chartColors.axis }}
                    axisLine={{ stroke: chartColors.grid }}
                    tickLine={{ stroke: chartColors.grid }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: chartColors.axis }}
                    axisLine={{ stroke: chartColors.grid }}
                    tickLine={{ stroke: chartColors.grid }}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(139, 92, 246, 0.08)' }} />
                  <Bar dataKey="count" fill="url(#circularityGradient)" radius={[4, 4, 0, 0]}>
                    {circularityHistogram.map((_, index) => (
                      <Cell key={`cell-${index}`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Statistics Table */}
      {summary.area_stats && (
        <div className="stats-section">
          <h3>Summary Statistics</h3>
          <table className="stats-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
              </tr>
            </thead>
            <tbody>
              {summary.area_stats && (
                <tr>
                  <td>Area (px²)</td>
                  <td>{summary.area_stats.mean.toFixed(2)}</td>
                  <td>{summary.area_stats.std.toFixed(2)}</td>
                  <td>{summary.area_stats.min.toFixed(2)}</td>
                  <td>{summary.area_stats.max.toFixed(2)}</td>
                  <td>{summary.area_stats.median.toFixed(2)}</td>
                </tr>
              )}
              {summary.circularity_stats && (
                <tr>
                  <td>Circularity</td>
                  <td>{summary.circularity_stats.mean.toFixed(4)}</td>
                  <td>{summary.circularity_stats.std.toFixed(4)}</td>
                  <td>{summary.circularity_stats.min.toFixed(4)}</td>
                  <td>{summary.circularity_stats.max.toFixed(4)}</td>
                  <td>{summary.circularity_stats.median.toFixed(4)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Download Section */}
      <div className="downloads-section">
        <h3>Download Results</h3>
        <div className="download-buttons">
          {/* Web Report Button */}
          <button
            className="download-button web-report"
            onClick={() => setShowWebReport(true)}
          >
            <Globe size={18} />
            View Web Report
          </button>

          {report_urls?.pdf && (
            <a
              href={getFullDownloadUrl(report_urls.pdf)}
              className="download-button pdf"
              download
            >
              <FileText size={18} />
              PDF Report
            </a>
          )}

          {report_urls?.excel && (
            <a
              href={getFullDownloadUrl(report_urls.excel)}
              className="download-button excel"
              download
            >
              <Table size={18} />
              Excel Data
            </a>
          )}

          {report_urls?.json && (
            <a
              href={getFullDownloadUrl(report_urls.json)}
              className="download-button json"
              download
            >
              <FileJson size={18} />
              JSON Data
            </a>
          )}

          {mask_url && (
            <a
              href={getFullDownloadUrl(mask_url)}
              className="download-button mask"
              download
            >
              <Download size={18} />
              Segmentation Mask
            </a>
          )}
        </div>
      </div>

      {/* Measurements Table (limited) */}
      {measurements.length > 0 && (
        <div className="measurements-section">
          <h3>Individual Cell Measurements (First 20)</h3>
          <div className="table-container">
            <table className="measurements-table">
              <thead>
                <tr>
                  <th>Cell ID</th>
                  <th>Area</th>
                  <th>Perimeter</th>
                  <th>Circularity</th>
                  <th>Eccentricity</th>
                  <th>Solidity</th>
                </tr>
              </thead>
              <tbody>
                {measurements.slice(0, 20).map((cell) => (
                  <tr key={cell.cell_id}>
                    <td>{cell.cell_id}</td>
                    <td>{cell.area.toFixed(2)}</td>
                    <td>{cell.perimeter.toFixed(2)}</td>
                    <td>{cell.circularity.toFixed(4)}</td>
                    <td>{cell.eccentricity.toFixed(4)}</td>
                    <td>{cell.solidity.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {measurements.length > 20 && (
            <p className="more-data-hint">
              ... and {measurements.length - 20} more cells. Download the Excel file for complete data.
            </p>
          )}
        </div>
      )}

      {/* Web Report Modal */}
      {showWebReport && (
        <WebReport results={results} onClose={() => setShowWebReport(false)} />
      )}
    </div>
  );
}

// Helper function to create histogram data
function createHistogram(
  data: CellMeasurement[],
  field: keyof CellMeasurement,
  bins: number
): { bin: string; count: number }[] {
  const values = data.map(d => d[field] as number).filter(v => v != null);

  if (values.length === 0) return [];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const binWidth = (max - min) / bins;

  const histogram: { bin: string; count: number }[] = [];

  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binWidth;
    const binEnd = min + (i + 1) * binWidth;
    const count = values.filter(v => v >= binStart && v < binEnd).length;

    histogram.push({
      bin: binStart.toFixed(1),
      count,
    });
  }

  return histogram;
}
