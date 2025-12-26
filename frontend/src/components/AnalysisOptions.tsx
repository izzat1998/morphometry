/**
 * Analysis Options Component
 */

import { useState, useEffect } from 'react';
import { Settings, Cpu, Microscope } from 'lucide-react';
import { getModels, getModalities } from '../services/api';
import type { SegmentationModel, ImageModality } from '../types';

export interface AnalysisConfig {
  modality: string;
  model: string;
  diameter: number | null;
  pixel_size_um: number;
  compute_texture: boolean;
  generate_report: boolean;
  report_format: string;
}

interface AnalysisOptionsProps {
  config: AnalysisConfig;
  onChange: (config: AnalysisConfig) => void;
  disabled?: boolean;
}

export function AnalysisOptions({ config, onChange, disabled }: AnalysisOptionsProps) {
  const [models, setModels] = useState<SegmentationModel[]>([]);
  const [modalities, setModalities] = useState<ImageModality[]>([]);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    // Load available models and modalities
    getModels().then(data => {
      setModels(data.models);
      setGpuAvailable(data.gpu_available);
    }).catch(console.error);

    getModalities().then(data => {
      setModalities(data.modalities);
    }).catch(console.error);
  }, []);

  const updateConfig = (key: keyof AnalysisConfig, value: unknown) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <div className="analysis-options">
      <div className="options-header">
        <Settings size={20} />
        <h3>Analysis Settings</h3>
        {gpuAvailable && (
          <span className="gpu-badge">
            <Cpu size={14} /> GPU
          </span>
        )}
      </div>

      <div className="options-grid">
        {/* Image Modality */}
        <div className="option-group">
          <label>
            <Microscope size={16} />
            Image Type
          </label>
          <select
            value={config.modality}
            onChange={(e) => updateConfig('modality', e.target.value)}
            disabled={disabled}
          >
            {modalities.map(mod => (
              <option key={mod.id} value={mod.id}>
                {mod.name}
              </option>
            ))}
          </select>
        </div>

        {/* Segmentation Model */}
        <div className="option-group">
          <label>Segmentation Model</label>
          <select
            value={config.model}
            onChange={(e) => updateConfig('model', e.target.value)}
            disabled={disabled}
          >
            {models.map(model => (
              <option key={model.id} value={model.id}>
                {model.name} {model.recommended && '(Recommended)'}
              </option>
            ))}
          </select>
        </div>

        {/* Pixel Size */}
        <div className="option-group">
          <label>Pixel Size (µm)</label>
          <input
            type="number"
            value={config.pixel_size_um}
            onChange={(e) => updateConfig('pixel_size_um', parseFloat(e.target.value) || 1)}
            min={0.001}
            max={100}
            step={0.1}
            disabled={disabled}
          />
        </div>

        {/* Report Format */}
        <div className="option-group">
          <label>Report Format</label>
          <select
            value={config.report_format}
            onChange={(e) => updateConfig('report_format', e.target.value)}
            disabled={disabled}
          >
            <option value="pdf">PDF</option>
            <option value="excel">Excel</option>
            <option value="json">JSON</option>
            <option value="all">All Formats</option>
          </select>
        </div>
      </div>

      {/* Advanced Options Toggle */}
      <button
        className="advanced-toggle"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </button>

      {showAdvanced && (
        <div className="advanced-options">
          {/* Cell Diameter */}
          <div className="option-group">
            <label>
              Cell Diameter (px)
              <span className="hint">Leave empty for auto-detection</span>
            </label>
            <input
              type="number"
              value={config.diameter || ''}
              onChange={(e) => updateConfig('diameter', e.target.value ? parseFloat(e.target.value) : null)}
              placeholder="Auto"
              min={1}
              max={500}
              disabled={disabled}
            />
          </div>

          {/* Texture Analysis */}
          <div className="option-group checkbox">
            <label>
              <input
                type="checkbox"
                checked={config.compute_texture}
                onChange={(e) => updateConfig('compute_texture', e.target.checked)}
                disabled={disabled}
              />
              Compute Texture Features
              <span className="hint">(Slower but more detailed)</span>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
