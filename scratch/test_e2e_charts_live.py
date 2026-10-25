"""End-to-End live test for Visual Analytics engine and chart endpoints."""
import os
import sys

# Ensure backend is on sys.path and DATABASE_URL points to SQLite
sys.path.insert(0, os.path.abspath("backend"))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./backend/ai_validation_dev.db"

import uuid
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.database import get_db, init_db, Base, engine
from app.core.security import create_access_token
from app.models.user import User, UserRole
from app.models.project import Project
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.validation_run import ValidationRun, RunStatus
from app.services.validation_service import execute_validation


@pytest.mark.asyncio
async def test_live_validation_charts_flow():
    # 1. Create a dummy model and dataset on disk
    os.makedirs("scratch/test_live", exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    
    # Add dummy sensitive column
    df["demographic_cohort"] = np.random.choice(["cohort_a", "cohort_b"], size=len(df))
    dataset_path = "scratch/test_live/iris_live.csv"
    df.to_csv(dataset_path, index=False)

    feature_cols = [c for c in df.columns if c not in ("target", "demographic_cohort")]
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(df[feature_cols], df["target"])
    model_path = "scratch/test_live/iris_live.joblib"
    joblib.dump(rf, model_path)

    # 2. Database setup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    user_email = f"tester_{uuid.uuid4().hex[:6]}@example.com"
    user_id = uuid.uuid4()
    token = create_access_token({"sub": user_email})
    headers = {"Authorization": f"Bearer {token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create user & auth token directly using DB session
        async for session in get_db():
            user = User(
                id=user_id,
                email=user_email,
                hashed_password="hash",
                full_name="Test User",
                role=UserRole.ADMIN,
                is_active=True,
            )
            session.add(user)
            project = Project(
                id=uuid.uuid4(),
                name="Visual Analytics Test Project",
                owner_id=user_id,
            )
            session.add(project)
            model_rec = MLModel(
                id=uuid.uuid4(),
                project_id=project.id,
                name="Live Test Iris RF",
                filename="iris_live.joblib",
                file_path=model_path,
                file_size=os.path.getsize(model_path),
                model_type="classification",
                framework="scikit-learn",
                feature_count=len(feature_cols),
                class_count=3,
            )
            session.add(model_rec)
            dataset_rec = Dataset(
                id=uuid.uuid4(),
                project_id=project.id,
                name="Live Test Iris Dataset",
                filename="iris_live.csv",
                file_path=dataset_path,
                file_size=os.path.getsize(dataset_path),
                row_count=len(df),
                column_count=len(df.columns),
                target_column="target",
                feature_columns_json=feature_cols,
                sensitive_columns_json=["demographic_cohort"],
            )
            session.add(dataset_rec)
            run = ValidationRun(
                id=uuid.uuid4(),
                run_number=f"RUN-LIVE-{uuid.uuid4().hex[:6].upper()}",
                project_id=project.id,
                model_id=model_rec.id,
                dataset_id=dataset_rec.id,
                status=RunStatus.PENDING,
                config_json={
                    "target_column": "target",
                    "sensitive_columns": ["demographic_cohort"],
                },
            )
            session.add(run)
            await session.commit()

            # Execute validation run
            await execute_validation(session, run)
            run_id = str(run.id)
            break

        # 3. Test GET /api/validation/runs/{run_id}/charts
        res = await client.get(f"/api/validation/runs/{run_id}/charts", headers=headers)
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["run_id"] == run_id
        assert data["total_charts"] > 0
        charts = data["charts"]
        chart_ids = [c["chart_id"] for c in charts]
        print(f"\nGenerated charts ({len(charts)}): {chart_ids}")

        assert "perf_confusion_matrix" in chart_ids
        assert "perf_per_class_metrics" in chart_ids
        assert "dq_missing_values" in chart_ids
        assert "drift_feature_psi" in chart_ids
        assert "explainability_feature_importance" in chart_ids
        assert "robustness_perturbation_curve" in chart_ids
        assert "reliability_calibration_curve" in chart_ids

        # 4. Test GET /api/validation/runs/{run_id}/charts/{chart_id}/export-csv
        csv_res = await client.get(f"/api/validation/runs/{run_id}/charts/drift_feature_psi/export-csv", headers=headers)
        assert csv_res.status_code == 200
        assert "text/csv" in csv_res.headers.get("content-type", "")
        csv_text = csv_res.text
        assert "feature" in csv_text
        assert "psi" in csv_text
        print("CSV Export verified successfully!")

        # 5. Test JSON report contains visual_analytics
        rep_res = await client.get(f"/api/reports/{run_id}/json", headers=headers)
        assert rep_res.status_code == 200
        rep_data = rep_res.json()
        assert "visual_analytics" in rep_data
        assert rep_data["visual_analytics"]["total_charts"] == len(charts)
        print("JSON report with embedded visual_analytics verified!")

        # 6. Test PDF report generation
        pdf_res = await client.get(f"/api/reports/{run_id}/pdf", headers=headers)
        assert pdf_res.status_code == 200
        assert pdf_res.content.startswith(b"%PDF")
        print("PDF report generation verified!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_live_validation_charts_flow())
