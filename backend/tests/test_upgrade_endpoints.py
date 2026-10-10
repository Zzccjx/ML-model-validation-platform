import pytest
import uuid
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.project import Project
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.validation_run import ValidationRun, RunStatus
from app.models.finding import Finding, Severity

@pytest.mark.asyncio
async def test_preview_and_inspect_endpoints(db_session, test_user, auth_headers, tmp_path):
    project = Project(name="Upgrade Test Project", owner_id=test_user.id)
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)

    df = pd.DataFrame({
        "age": [25, 30, 45, 50, 22],
        "income": [50000, 60000, 90000, 110000, 48000],
        "gender": ["F", "M", "M", "F", "F"],
        "approved": [1, 1, 0, 0, 1]
    })
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)

    dataset = Dataset(
        project_id=project.id,
        name="Test Data",
        filename="test_data.csv",
        file_path=str(csv_path),
        file_size=1024,
        row_count=5,
        column_count=4,
        target_column="approved",
        feature_columns_json=["age", "income", "gender"],
        sensitive_columns_json=["gender"]
    )
    db_session.add(dataset)

    X = np.array([[25, 50000], [30, 60000], [45, 90000], [50, 110000], [22, 48000]])
    y = np.array([1, 1, 0, 0, 1])
    clf = LogisticRegression()
    clf.fit(X, y)
    model_path = tmp_path / "model.pkl"
    joblib.dump(clf, model_path)

    ml_model = MLModel(
        project_id=project.id,
        name="Test Logistic Model",
        filename="model.pkl",
        file_path=str(model_path),
        file_size=2048,
        model_type="classification",
        framework="scikit-learn"
    )
    db_session.add(ml_model)
    await db_session.commit()
    await db_session.refresh(dataset)
    await db_session.refresh(ml_model)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_preview = await client.get(f"/api/datasets/{dataset.id}/preview", headers=auth_headers)
        assert res_preview.status_code == 200
        preview_data = res_preview.json()
        assert preview_data["total_rows"] == 5
        assert len(preview_data["preview_rows"]) == 5
        assert "income" in preview_data["numeric_columns"]
        assert "gender" in preview_data["categorical_columns"]

        res_inspect = await client.get(f"/api/models/{ml_model.id}/inspect", headers=auth_headers)
        assert res_inspect.status_code == 200
        inspect_data = res_inspect.json()
        assert inspect_data["model_type"] == "classification"
        assert inspect_data["has_predict_proba"] is True

        compat_payload = {
            "model_id": str(ml_model.id),
            "dataset_id": str(dataset.id),
            "target_column": "approved",
            "sensitive_column": "gender"
        }
        res_compat = await client.post("/api/validation/compatibility-check", json=compat_payload, headers=auth_headers)
        assert res_compat.status_code == 200
        compat_data = res_compat.json()
        assert "validator_readiness" in compat_data
        fairness_item = next(v for v in compat_data["validator_readiness"] if v["name"] == "fairness")
        assert fairness_item["ready"] is True

        res_reports = await client.get("/api/reports", headers=auth_headers)
        assert res_reports.status_code == 200
        assert isinstance(res_reports.json(), list)
