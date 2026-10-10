# Security Model

## Threat Model: ML Model Uploads
Uploading machine learning models introduces significant security risks.

### Pickle / Joblib Deserialization Risks
Python's `pickle` and `joblib` libraries are inherently insecure. Loading a malicious `.pkl` file can execute arbitrary code on the host machine.
If an attacker uploads a crafted model file, they could gain Remote Code Execution (RCE) on the server.

### Current MVP Approach (Development Only)
In this Phase 1 MVP, models are loaded directly within the FastAPI API process. **This is for development convenience only and is highly insecure.**

### Production Security Recommendations
For production deployments, model deserialization must be isolated:
1. **Containerized Execution**: Load models in ephemeral Docker containers with no network access.
2. **Sandboxing**: Use sandboxes like gVisor or Firecracker microVMs.
3. **Format Migration**: Migrate to safer formats like ONNX, PMML, or SafeTensors where possible.

## Authentication Security
- Passwords are automatically hashed using `bcrypt` prior to database storage.
- Stateless authentication is implemented via JSON Web Tokens (JWT).

## File Upload Security
- The system enforces a configurable `MAX_UPLOAD_SIZE_MB` to prevent denial-of-service via large files.
- Uploads should be scanned for malware in a production environment.

## Input Validation
All API inputs are strictly validated using Pydantic schemas to prevent injection attacks and ensure data integrity.
