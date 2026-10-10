"""Seeds initial demo account, sample assets, and completed validation run for platform exploration."""
import asyncio
import os
import httpx

API_BASE = "http://127.0.0.1:8000/api"

async def seed():
    async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client:
        # Check health
        health = await client.get("/health")
        print(f"Health Check: {health.json()}")

        # Register demo user
        user_data = {
            "email": "demo@example.com",
            "password": "Password123!",
            "full_name": "Senior ML Validation Engineer",
        }
        reg_res = await client.post("/auth/register", json=user_data)
        if reg_res.status_code == 200:
            print("Demo user registered successfully.")
        else:
            print(f"User registration response: {reg_res.status_code} (User may already exist)")

        # Login
        login_res = await client.post("/auth/login", json={"email": user_data["email"], "password": user_data["password"]})
        if login_res.status_code != 200:
            print(f"Login failed: {login_res.text}")
            return
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("Logged in successfully. Token acquired.")

        # Create Project
        proj_res = await client.post(
            "/projects",
            json={
                "name": "Credit Default Risk Assessment 2026",
                "description": "Comprehensive multi-dimensional validation of retail loan underwriting classifier across accuracy, EEOC fairness, drift, perturbation robustness, feature dominance, and probability calibration.",
            },
            headers=headers,
        )
        proj_id = proj_res.json()["id"]
        print(f"Project created with ID: {proj_id}")

        sample_dir = os.path.abspath("scripts/sample_data")

        # Upload Dataset
        with open(os.path.join(sample_dir, "classification_test.csv"), "rb") as f:
            ds_res = await client.post(
                "/datasets/upload",
                data={
                    "project_id": proj_id,
                    "name": "Underwriting Test Sample (Holdout)",
                    "target_column": "target",
                    "sensitive_columns": '["gender", "region"]',
                },
                files={"file": ("classification_test.csv", f, "text/csv")},
                headers=headers,
            )
        ds_id = ds_res.json()["id"]
        print(f"Dataset uploaded with ID: {ds_id}")

        # Upload Model
        with open(os.path.join(sample_dir, "random_forest_classifier.pkl"), "rb") as f:
            m_res = await client.post(
                "/models/upload",
                data={"project_id": proj_id, "name": "Random Forest Loan Underwriter v1.2"},
                files={"file": ("random_forest_classifier.pkl", f, "application/octet-stream")},
                headers=headers,
            )
        m_id = m_res.json()["id"]
        print(f"Model uploaded with ID: {m_id}")

        # Trigger Validation Run
        run_res = await client.post(
            "/validation/runs",
            json={"model_id": m_id, "dataset_id": ds_id},
            headers=headers,
        )
        run_data = run_res.json()
        print(f"Validation Run initiated: #{run_data.get('run_number')} -> Status: {run_data.get('status')}")
        print(f"Overall Model Health Score: {run_data.get('overall_score')}/100")
        print(f"Category Scores: {run_data.get('category_scores_json')}")
        print("\nSeed completed successfully! Platform is populated and ready for immediate dashboard exploration.")

if __name__ == "__main__":
    asyncio.run(seed())
