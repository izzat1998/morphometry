# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Morphometry is a scientific cell image analysis application with AI-powered segmentation. It consists of a FastAPI backend for image processing and analysis, and a React/TypeScript frontend dashboard.

## Commands

### Backend Development
```bash
# Start backend server (from backend directory)
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
pytest -v tests/test_specific.py::test_function  # single test
```

### Frontend Development
```bash
# Start frontend dev server (from frontend directory)
cd frontend
npm run dev

# Build for production
npm run build

# Lint
npm run lint
```

### Environment Variables
Backend configuration uses `MORPH_` prefix (e.g., `MORPH_CELLPOSE_GPU=true`). See `backend/app/core/config.py` for all options.

## Architecture

### Backend (FastAPI + Python)
```
backend/app/
├── main.py              # FastAPI app with lifespan, CORS, static files
├── api/routes.py        # API endpoints: /analyze, /health, /models, /download
├── core/
│   ├── config.py        # Pydantic settings with env var support
│   ├── preprocessing.py # ImagePreprocessor with modality-specific pipelines
│   └── translations.py  # Uzbek language translations for reports
├── models/schemas.py    # Pydantic request/response models
└── services/
    ├── segmentation.py  # CellSegmenter: Cellpose, Cellpose-SAM, watershed, otsu
    ├── morphometry.py   # MorphometryAnalyzer: 30+ measurements per cell
    ├── report.py        # ReportGenerator: PDF, Excel, JSON output
    └── pdf/             # Modular PDF generation system
```

### Frontend (React + TypeScript + Vite)
```
frontend/src/
├── App.tsx              # Main dashboard layout
├── services/api.ts      # Axios client for backend API
├── types/index.ts       # TypeScript interfaces
└── components/
    ├── ImageUploader.tsx
    ├── AnalysisOptions.tsx
    ├── ResultsDisplay.tsx
    └── WebReport.tsx
```

## Key Processing Pipeline

1. **Preprocessing** (`core/preprocessing.py`): Modality-specific filters (brightfield, fluorescence, phase contrast, H&E stained)
2. **Segmentation** (`services/segmentation.py`): Cellpose v4+ with automatic GPU detection, fallback to classical methods
3. **Morphometry** (`services/morphometry.py`): Calculates area, perimeter, circularity, eccentricity, solidity, texture features (GLCM)
4. **Reports** (`services/report.py`): Publication-quality PDF with Uzbek translations, Excel, JSON

## API Endpoints

- `POST /api/v1/analyze` - Main analysis endpoint (file upload + options)
- `GET /api/v1/health` - Health check with GPU/Cellpose status
- `GET /api/v1/models` - Available segmentation models
- `GET /api/v1/modalities` - Supported image modalities
- `GET /api/v1/download/{filename}` - Download generated reports/masks

## Segmentation Models

| Model | Use Case |
|-------|----------|
| `cyto3` | Default, best for most cell types |
| `nuclei` | Nuclei-only detection |
| `cpsam` | Cellpose-SAM, zero-shot generalization |
| `watershed` | Classical fallback (no GPU) |
| `otsu` | Simple thresholding |

## Image Modalities

Each modality has a specific preprocessing pipeline in `ImagePreprocessor`:
- `brightfield`: Background subtraction + CLAHE + bilateral denoise
- `fluorescence`: NLM denoise + background subtraction + contrast stretching
- `phase_contrast`: Median filter + unsharp mask + CLAHE
- `he_stained`: Color deconvolution to extract hematoxylin channel

## Configuration

Key settings in `backend/app/core/config.py`:
- `cellpose_model`: Default model (cyto3)
- `cellpose_gpu`: Enable GPU acceleration
- `cellpose_diameter`: Auto-detect if None
- `upload_dir` / `results_dir`: File storage paths

## Internationalization

Reports support Uzbek language. Translations are in `backend/app/core/translations.py`. The PDF system uses modular templates in `backend/app/services/pdf/`.
