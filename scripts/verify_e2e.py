import asyncio
import os
import sys

# Configure SQLite for local test run
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./e2e_test.db"
os.environ["SECRET_KEY"] = "e2e-secret-key-12345"
os.environ["JWT_SECRET"] = "e2e-jwt-secret-12345678901234567890"

sys.path.insert(0, os.path.abspath("backend"))

from httpx import AsyncClient, ASGITransport
from app.core.config import get_settings
get_settings.cache_clear()

from app.main import app
from app.core.database import init_db, close_db

async def run_e2e():
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        email = "validator_e2e_dq@example.com"
        pwd = "password12345"
        
        # 1. Register & Login
        await client.post("/api/auth/register", json={"email": email, "password": pwd, "full_name": "E2E DQ Tester"})
        login_res = await client.post("/api/auth/login", json={"email": email, "password": pwd})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("1. User authenticated successfully")

        # 2. Create Project
        proj_res = await client.post("/api/projects", json={"name": "Multi-Module Benchmark", "description": "Performance & Data Quality verification"}, headers=headers)
        proj_id = proj_res.json()["id"]
        print(f"2. Project created: {proj_id}")

        sample_dir = os.path.abspath("scripts/sample_data")
        
        # 3. Upload Classification Data & Model
        with open(os.path.join(sample_dir, "classification_test.csv"), "rb") as f:
            ds_res = await client.post(
                "/api/datasets/upload",
                data={
                    "project_id": proj_id,
                    "name": "Classification Benchmark Data",
                    "target_column": "target",
                    "sensitive_columns": '["gender", "region"]',
                },
                files={"file": ("classification_test.csv", f, "text/csv")},
                headers=headers
            )
        ds_id = ds_res.json()["id"]

        with open(os.path.join(sample_dir, "random_forest_classifier.pkl"), "rb") as f:
            m_res = await client.post(
                "/api/models/upload",
                data={"project_id": proj_id, "name": "RF Classifier"},
                files={"file": ("random_forest_classifier.pkl", f, "application/octet-stream")},
                headers=headers
            )
        m_id = m_res.json()["id"]

        # 4. Trigger Validation Run
        run_res = await client.post("/api/validation/runs", json={"model_id": m_id, "dataset_id": ds_id}, headers=headers)
        run_data = run_res.json()
        run_id = run_data["id"]
        print(f"3. Run #{run_data.get('run_number')} -> Status: {run_data.get('status')}, Health Score: {run_data.get('overall_score')}/100")
        print(f"   Category Scores: {run_data.get('category_scores_json')}")
        
        # 5. Verify Module 1, Module 2, Module 3, Module 4, Module 5 & Module 6 Metrics in config_json
        run_detail_res = await client.get(f"/api/validation/runs/{run_id}", headers=headers)
        metrics = run_detail_res.json().get("config_json", {}).get("metrics", {})
        print("4. Executed Validators in Run:")
        for v_name in metrics:
            print(f"   - {v_name}")
        assert "performance" in metrics, "Missing performance metrics"
        assert "data_quality" in metrics, "Missing data_quality metrics"
        assert "fairness" in metrics, "Missing fairness metrics"
        assert "gender" in metrics["fairness"]["attributes"], "Missing gender fairness attribute"
        assert "drift" in metrics, "Missing drift metrics"
        assert "robustness" in metrics, "Missing robustness metrics"
        assert "explainability" in metrics, "Missing explainability metrics"
        assert "reliability" in metrics, "Missing reliability metrics"
        print(f"   Drift Status: {metrics['drift'].get('overall_drift_status')}, Max PSI: {metrics['drift'].get('max_psi')}")
        print(f"   Robustness Rating: {metrics['robustness'].get('robustness_rating')}, 5% Flip Rate: {metrics['robustness'].get('flip_rate_5pct')}")
        print(f"   Explainability Method: {metrics['explainability'].get('attribution_method')}, Top Feature: {metrics['explainability'].get('top_feature_name')} ({metrics['explainability'].get('top_feature_share_percentage')}%)")
        print(f"   Reliability Mode: {metrics['reliability'].get('classification_mode')}, ECE: {metrics['reliability'].get('expected_calibration_error')}, Brier: {metrics['reliability'].get('brier_score')}")

        # 6. Verify Phase 11 Report Downloads (JSON & PDF)
        json_report_res = await client.get(f"/api/reports/{run_id}/json", headers=headers)
        assert json_report_res.status_code == 200, f"JSON report failed: {json_report_res.status_code}"
        report_json = json_report_res.json()
        assert "section_1_executive_summary" in report_json, "Missing section 1 in exported JSON"
        assert "section_11_reliability" in report_json, "Missing section 11 in exported JSON"
        assert "section_15_limitations" in report_json, "Missing section 15 in exported JSON"
        print(f"6. JSON Report generated successfully: {len(report_json)} top-level sections.")

        pdf_report_res = await client.get(f"/api/reports/{run_id}/pdf", headers=headers)
        assert pdf_report_res.status_code == 200, f"PDF report failed: {pdf_report_res.status_code}"
        pdf_bytes = pdf_report_res.content
        assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF signature"
        print(f"7. Audit-Grade PDF Report downloaded successfully: {len(pdf_bytes)} bytes.")

        # 7. Test with Missing Values Dataset
        with open(os.path.join(sample_dir, "missing_values_test.csv"), "rb") as f:
            mv_ds_res = await client.post(
                "/api/datasets/upload",
                data={"project_id": proj_id, "name": "Missing Values Test Data", "target_column": "target"},
                files={"file": ("missing_values_test.csv", f, "text/csv")},
                headers=headers
            )
        mv_ds_id = mv_ds_res.json()["id"]

        mv_run_res = await client.post("/api/validation/runs", json={"model_id": m_id, "dataset_id": mv_ds_id}, headers=headers)
        mv_run_data = mv_run_res.json()
        print(f"8. Missing Values Run #{mv_run_data.get('run_number')} -> Status: {mv_run_data.get('status')}, Health Score: {mv_run_data.get('overall_score')}/100")
        print(f"   Category Scores: {mv_run_data.get('category_scores_json')}")

        mv_findings_res = await client.get(f"/api/validation/runs/{mv_run_data.get('id')}/findings", headers=headers)
        mv_findings = mv_findings_res.json().get("findings", [])
        print(f"   Findings emitted: {len(mv_findings)}")
        for f in mv_findings:
            print(f"   - [{f.get('severity').upper()}] ({f.get('category')}) {f.get('title')}")

        print("\nAll End-to-End validation operations for Phase 10 & 11 (All 7 Pillars + JSON/PDF Reporting) verified successfully!")

    await close_db()
    if os.path.exists("e2e_test.db"):
        try:
            os.remove("e2e_test.db")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(run_e2e())

