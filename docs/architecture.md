# System Architecture

## System Overview
The AI/ML Model Validation & Assurance Platform consists of a decoupled architecture with a Python-based backend, a React frontend, and a PostgreSQL database.

## Component Diagram
- **Frontend**: React SPA serving the user interface.
- **Backend API**: FastAPI application handling HTTP requests.
- **Database**: PostgreSQL storing users, projects, models, datasets, validation runs, and findings.
- **Validator Plugins**: A modular registry system where specific validation tasks (Performance, Fairness, Drift, etc.) can be registered and executed.
- **File Storage**: Local file system (or abstraction) for storing uploaded datasets and model weights.

## Data Flow for a Validation Run
1. User uploads a model file and dataset file.
2. User configures the validation run (target column, sensitive attributes, etc.).
3. Backend triggers the validation orchestration engine.
4. Orchestrator invokes registered validator plugins sequentially or concurrently.
5. Plugins produce findings and category scores.
6. Overall Model Health Score is computed.
7. Results are persisted to the database and returned to the frontend.

## Validator Plugin Architecture
The system employs a plugin architecture. Each validator inherits from a common `BaseValidator` interface with `validate()` method. This ensures that new validation dimensions (e.g., LLM-specific validations) can be easily added in the future.

## File Storage Abstraction
Currently, files are saved locally in the `uploads/` directory. The architecture is designed such that this can be easily replaced with cloud storage (e.g., AWS S3, Azure Blob Storage) via an interface abstraction.

## Security Model
- Uses JWT for stateless authentication.
- Passwords are encrypted using bcrypt.
- **Warning**: Deserialization of `.pkl` and `.joblib` files poses significant security risks. Currently, this is done directly in the backend process for development. In production, this must be isolated in sandboxed environments (e.g., gVisor, separate containers with strict resource limits).

## Future Extensibility
- **AI-Generated Code Validation**: Plugins to validate not just ML models, but code generation models.
- **Autonomous Software Validation Agent**: Extensibility points to allow agents to interact with the API to perform iterative validation and repairs.
