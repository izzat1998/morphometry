/**
 * Morphometry Analysis Application
 * Dashboard layout for scientific cell image analysis
 */

import { useState, useCallback } from 'react';
import { Microscope, AlertCircle, Loader2, FlaskConical } from 'lucide-react';
import { ImageUploader } from './components/ImageUploader';
import { AnalysisOptions, type AnalysisConfig } from './components/AnalysisOptions';
import { ResultsDisplay } from './components/ResultsDisplay';
import { analyzeImage } from './services/api';
import type { AnalysisResponse, AnalysisStatus } from './types';
import './App.css';

const DEFAULT_CONFIG: AnalysisConfig = {
  modality: 'brightfield',
  model: 'cyto3',
  diameter: null,
  pixel_size_um: 1.0,
  compute_texture: true,
  generate_report: true,
  report_format: 'pdf',
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [config, setConfig] = useState<AnalysisConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;

    setStatus('processing');
    setError(null);

    try {
      const response = await analyzeImage(selectedFile, config);
      setResults(response);
      setStatus('complete');
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setStatus('error');
    }
  }, [selectedFile, config]);

  const isProcessing = status === 'processing';

  return (
    <div className="app dashboard-layout">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <Microscope size={28} />
            <div className="header-text">
              <h1>Morphometry Analysis</h1>
              <p>Scientific cell image analysis with AI-powered segmentation</p>
            </div>
          </div>
          <div className="header-right">
            <span className="header-badge">
              <FlaskConical size={14} />
              Laboratory Dashboard
            </span>
          </div>
        </div>
      </header>

      {/* Dashboard Main Content */}
      <main className="dashboard-main">
        {/* Left Panel - Input */}
        <div className="dashboard-panel input-panel">
          <div className="panel-header">
            <h2>Input</h2>
            <span className="panel-badge">Step 1</span>
          </div>

          <div className="panel-content">
            {/* Image Upload Section */}
            <div className="panel-section">
              <h3>Microscopy Image</h3>
              <ImageUploader
                onImageSelect={handleImageSelect}
                disabled={isProcessing}
              />
            </div>

            {/* Analysis Options */}
            <div className="panel-section">
              <h3>Analysis Configuration</h3>
              <AnalysisOptions
                config={config}
                onChange={setConfig}
                disabled={isProcessing}
              />
            </div>

            {/* Analyze Button */}
            <div className="panel-action">
              <button
                className="analyze-button"
                onClick={handleAnalyze}
                disabled={!selectedFile || isProcessing}
              >
                {isProcessing ? (
                  <>
                    <Loader2 size={18} className="spinner" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Microscope size={18} />
                    Run Analysis
                  </>
                )}
              </button>
            </div>

            {/* Error Display */}
            {error && (
              <div className="error-message">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Output */}
        <div className="dashboard-panel output-panel">
          <div className="panel-header">
            <h2>Results</h2>
            <span className="panel-badge">Step 2</span>
          </div>

          <div className="panel-content">
            {results ? (
              <ResultsDisplay results={results} />
            ) : (
              <div className="empty-state">
                <div className="empty-icon">
                  <FlaskConical size={48} />
                </div>
                <h3>No Results Yet</h3>
                <p>
                  Upload a microscopy image and run analysis to see results here.
                </p>
                <div className="empty-steps">
                  <div className="step">
                    <span className="step-num">1</span>
                    <span>Upload image</span>
                  </div>
                  <div className="step">
                    <span className="step-num">2</span>
                    <span>Configure settings</span>
                  </div>
                  <div className="step">
                    <span className="step-num">3</span>
                    <span>Click "Run Analysis"</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Powered by Cellpose + scikit-image | Built for accurate scientific morphometry analysis
        </p>
      </footer>
    </div>
  );
}

export default App;
