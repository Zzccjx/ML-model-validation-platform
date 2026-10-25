"""Tests for model endpoints."""
import pytest
from httpx import AsyncClient
import uuid

@pytest.mark.asyncio
async def test_upload_model(client: AsyncClient, auth_headers: dict, test_project, sample_model_path):
    with open(sample_model_path, "rb") as f:
        files = {"file": ("test_model.pkl", f, "application/octet-stream")}
        data = {"project_id": str(test_project.id), "name": "Test Model"}
        
        response = await client.post(
            "/api/models/upload",
            data=data,
            files=files,
            headers=auth_headers
        )
    assert response.status_code in (200, 201)
    res_data = response.json()
    assert res_data["name"] == "Test Model"
    assert "id" in res_data

@pytest.mark.asyncio
async def test_upload_invalid_extension(client: AsyncClient, auth_headers: dict, test_project, tmp_path):
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("dummy content")
    
    with open(txt_path, "rb") as f:
        files = {"file": ("test.txt", f, "text/plain")}
        data = {"project_id": str(test_project.id), "name": "Invalid Model"}
        
        response = await client.post(
            "/api/models/upload",
            data=data,
            files=files,
            headers=auth_headers
        )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_list_models(client: AsyncClient, auth_headers: dict):
    response = await client.get("/api/models", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_get_model(client: AsyncClient, auth_headers: dict, test_project, sample_model_path):
    with open(sample_model_path, "rb") as f:
        files = {"file": ("test_model.pkl", f, "application/octet-stream")}
        data = {"project_id": str(test_project.id), "name": "Test Model"}
        upload_resp = await client.post("/api/models/upload", data=data, files=files, headers=auth_headers)
        
    model_id = upload_resp.json()["id"]
    
    response = await client.get(f"/api/models/{model_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"] == model_id

@pytest.mark.asyncio
async def test_delete_model(client: AsyncClient, auth_headers: dict, test_project, sample_model_path):
    with open(sample_model_path, "rb") as f:
        files = {"file": ("test_model.pkl", f, "application/octet-stream")}
        data = {"project_id": str(test_project.id), "name": "Test Model"}
        upload_resp = await client.post("/api/models/upload", data=data, files=files, headers=auth_headers)
        
    model_id = upload_resp.json()["id"]
    
    response = await client.delete(f"/api/models/{model_id}", headers=auth_headers)
    assert response.status_code == 204
    
    get_resp = await client.get(f"/api/models/{model_id}", headers=auth_headers)
    assert get_resp.status_code == 404

@pytest.mark.asyncio
async def test_inspect_model_non_neural_parameters_null(client: AsyncClient, auth_headers: dict, test_project, sample_model_path):
    with open(sample_model_path, "rb") as f:
        files = {"file": ("test_model.pkl", f, "application/octet-stream")}
        data = {"project_id": str(test_project.id), "name": "Test Model For Inspect"}
        upload_resp = await client.post("/api/models/upload", data=data, files=files, headers=auth_headers)
    assert upload_resp.status_code in (200, 201)
    model_id = upload_resp.json()["id"]

    inspect_resp = await client.get(f"/api/models/{model_id}/inspect", headers=auth_headers)
    assert inspect_resp.status_code == 200
    inspect_data = inspect_resp.json()
    assert inspect_data["id"] == model_id
    assert inspect_data["framework"] == "scikit-learn"
    assert inspect_data["total_parameters"] is None
    assert inspect_data["layer_count"] is None

