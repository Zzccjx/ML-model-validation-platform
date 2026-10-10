# AI/ML Model Validation & Assurance Platform

A comprehensive web-based platform that evaluates machine-learning models across multiple dimensions — performance, data quality, fairness, drift, robustness, explainability, and reliability — generating structured validation reports with actionable findings.

> **Note**: This platform provides validation indicators and identifies potential issues. It does not certify or guarantee model safety. Scores represent the platform's assessment based on configurable rules and should be interpreted alongside domain expertise.

## 🏗️ Architecture

The platform follows a modern decoupled architecture:
- **Backend**: FastAPI + SQLAlchemy 2.x (async) + PostgreSQL
- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS
- **ML Stack**: scikit-learn, SHAP, Fairlearn, Evidently
- **Charts**: Recharts
- **Auth**: JWT + bcrypt

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+ (or Docker)
- Git

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd AI_ML_Validation
cp .env.example .env
```

### 2. Start PostgreSQL

**Option A: Docker**
```bash
docker compose up -d
```

**Option B: Local PostgreSQL**
Create database `ai_validation` and update `DATABASE_URL` in `.env`.

### 3. Backend Setup

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

### 5. Generate Sample Data (Optional)

```bash
python scripts/create_sample_data.py
```

## 📊 How It Works

1. **Upload** a trained ML model (.pkl/.joblib) and test dataset (.csv)
2. **Configure** target column and optional sensitive attributes
3. **Run Validation** — the platform automatically:
   - Checks model-dataset compatibility
   - Runs registered validators (performance, data quality, fairness, etc.)
   - Generates structured findings with severity levels
   - Calculates category scores and overall Model Health Score
4. **Review** results in the dashboard
5. **Download** a comprehensive validation report

## 🔐 Security Notes

- Model files (.pkl/.joblib) use pickle/joblib deserialization which can execute arbitrary code
- **Never load untrusted models in production without sandboxing**
- The MVP loads models in the API process for development convenience
- Production deployment should isolate model execution in containers/workers
- See [docs/security.md](docs/security.md) for details

## 📁 Project Structure

```
AI_ML_Validation/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── core/                # Config, security, logging, database
│   │   ├── api/routes/          # REST API endpoints
│   │   ├── models/              # SQLAlchemy ORM models
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── services/            # Business logic
│   │   ├── validators/          # Validation modules (plugin architecture)
│   │   └── utils/               # Utilities
│   ├── tests/                   # Automated tests
│   ├── alembic/                 # Database migrations
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/          # Reusable UI components
│       ├── pages/               # Page components
│       ├── services/            # API client
│       ├── hooks/               # React hooks
│       └── types/               # TypeScript types
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── docker-compose.yml           # PostgreSQL dev setup
└── .env.example                 # Environment template
```

## 🧪 Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

## 📖 Documentation

- [Architecture](docs/architecture.md)
- [Validation Methodology](docs/validation-methodology.md)
- [Scoring Methodology](docs/scoring-methodology.md)
- [Security](docs/security.md)
- [API Reference](docs/api.md)

## 🗺️ Roadmap & Phase Completion Status

### Core Platform Implementation (Phases 1 – 11) ✅ COMPLETE
- **Phase 1: Architecture Foundation** ✅ (FastAPI async, SQLAlchemy 2.0 Async, Alembic, BaseValidator plugin registry, React 18 + TS + Vite + Tailwind shell).
- **Phase 2: Ingestion & Uploads** ✅ (Model & Dataset uploads, file inspection, Project isolation, run launcher).
- **Phase 3: Compatibility Engine** ✅ (Feature name ordering, NaN checking, continuous/categorical target validation).
- **Phase 4: Module 1 — Performance Validation** ✅ (Classification & Regression, confusion matrix, residual diagnostics, metrics calculation).
- **Phase 5: Module 2 — Data Quality Validation** ✅ (Missingness, deduplication, variance checks, outliers, class balance, target leakage).
- **Phase 6: Module 3 — Fairness & Bias Validation** ✅ (EEOC 80% Four-Fifths rule, Demographic Parity Diff, Equal Opportunity Diff, Equalized Odds, subgroup breakdown).
- **Phase 7: Module 4 — Data & Prediction Drift** ✅ (Two-sample KS tests, Population Stability Index 10-bin PSI, categorical PSI, concept shift).
- **Phase 8: Module 5 — Adversarial Robustness** ✅ (Calibrated Gaussian noise perturbation, prediction flip rate, feature masking resilience, stability ratings).
- **Phase 9: Module 6 — Explainability Validation** ✅ (Adaptive tree/linear/permutation attributions, normalized %, dominance detection $>60\%$, dead features).
- **Phase 10: Module 7 — Reliability & Calibration** ✅ (Brier score, Expected Calibration Error ECE, MCE, overconfident error identification, prediction intervals).
- **Phase 11: Validation Report Generation & Dashboard** ✅ (Standard 15-section audit schema, audit-grade PDF download via `fpdf2`, structured JSON download, responsive React UI cards).

### Future Extensibility Modules (Architecture-Ready)
The plugin architecture (`BaseValidator`, `ValidatorRegistry`, and dynamic orchestrator) is designed to incorporate these future dimensions without modifying existing engines:
1. **AI-Generated Code Validation**: Linting, security scanning, static analysis, and automated unit test execution for code produced by LLMs.
2. **Autonomous Software Validation Agent**: Continuous validation agent querying the API to perform iterative model/pipeline repairs.

## 🧪 Automated Testing & Verification
The platform includes 83 automated backend pytest unit/integration tests and automated end-to-end multi-module verification:
```bash
# Backend pytest suite
cd backend
python -m pytest tests/ -v

# Frontend production build check
cd frontend
npm run build

# End-to-end multi-pillar validation & report download
python scripts/verify_e2e.py
```

## 📜 License
MIT

