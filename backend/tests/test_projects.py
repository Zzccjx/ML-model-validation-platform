"""Tests for project endpoints."""
import pytest
import uuid
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_project(client: AsyncClient, auth_headers: dict):
    response = await client.post(
        "/api/projects",
        json={"name": "New Project", "description": "Project description"},
        headers=auth_headers
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "New Project"
    assert "id" in data

@pytest.mark.asyncio
async def test_list_projects(client: AsyncClient, auth_headers: dict, test_project):
    response = await client.get("/api/projects", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert any(p["id"] == str(test_project.id) for p in data)

@pytest.mark.asyncio
async def test_get_project(client: AsyncClient, auth_headers: dict, test_project):
    response = await client.get(f"/api/projects/{test_project.id}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == str(test_project.id)
    assert data["name"] == test_project.name

@pytest.mark.asyncio
async def test_update_project(client: AsyncClient, auth_headers: dict, test_project):
    response = await client.put(
        f"/api/projects/{test_project.id}",
        json={"name": "Updated Project", "description": "Updated description"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Project"

@pytest.mark.asyncio
async def test_delete_project(client: AsyncClient, auth_headers: dict, test_project):
    response = await client.delete(f"/api/projects/{test_project.id}", headers=auth_headers)
    assert response.status_code == 204
    
    get_resp = await client.get(f"/api/projects/{test_project.id}", headers=auth_headers)
    assert get_resp.status_code == 404

@pytest.mark.asyncio
async def test_project_not_found(client: AsyncClient, auth_headers: dict):
    random_id = str(uuid.uuid4())
    response = await client.get(f"/api/projects/{random_id}", headers=auth_headers)
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_create_project_no_auth(client: AsyncClient):
    response = await client.post(
        "/api/projects",
        json={"name": "New Project", "description": "Project description"}
    )
    assert response.status_code == 401
