"""Tests for dataset endpoints."""
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_upload_dataset(client: AsyncClient, auth_headers: dict, test_project, sample_dataset_path):
    with open(sample_dataset_path, "rb") as f:
        files = {"file": ("test_data.csv", f, "text/csv")}
        data = {"project_id": str(test_project.id), "name": "Test Dataset", "target_column": "target"}
        
        response = await client.post(
            "/api/datasets/upload",
            data=data,
            files=files,
            headers=auth_headers
        )
    assert response.status_code in (200, 201)
    res_data = response.json()
    assert res_data["name"] == "Test Dataset"

@pytest.mark.asyncio
async def test_upload_invalid_extension(client: AsyncClient, auth_headers: dict, test_project, tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text("{}")
    
    with open(json_path, "rb") as f:
        files = {"file": ("data.json", f, "application/json")}
        data = {"project_id": str(test_project.id), "name": "Invalid Dataset", "target_column": "target"}
        
        response = await client.post(
            "/api/datasets/upload",
            data=data,
            files=files,
            headers=auth_headers
        )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_upload_wrong_target(client: AsyncClient, auth_headers: dict, test_project, sample_dataset_path):
    with open(sample_dataset_path, "rb") as f:
        files = {"file": ("test_data.csv", f, "text/csv")}
        data = {"project_id": str(test_project.id), "name": "Test Dataset", "target_column": "nonexistent"}
        
        response = await client.post(
            "/api/datasets/upload",
            data=data,
            files=files,
            headers=auth_headers
        )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_list_datasets(client: AsyncClient, auth_headers: dict):
    response = await client.get("/api/datasets", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_get_dataset(client: AsyncClient, auth_headers: dict, test_project, sample_dataset_path):
    with open(sample_dataset_path, "rb") as f:
        files = {"file": ("test_data.csv", f, "text/csv")}
        data = {"project_id": str(test_project.id), "name": "Test Dataset", "target_column": "target"}
        upload_resp = await client.post("/api/datasets/upload", data=data, files=files, headers=auth_headers)
        
    dataset_id = upload_resp.json()["id"]
    
    response = await client.get(f"/api/datasets/{dataset_id}", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["id"] == dataset_id
