"""Tests for authentication endpoints."""
import pytest
from httpx import AsyncClient
from app.models.user import User
from app.core.security import verify_password
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

@pytest.mark.asyncio
async def test_register_success(client: AsyncClient, db_session: AsyncSession):
    response = await client.post(
        "/api/auth/register",
        json={"email": "newuser@example.com", "password": "password123", "full_name": "New User"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    
    # Verify user in db
    result = await db_session.execute(select(User).where(User.email == "newuser@example.com"))
    user = result.scalars().first()
    assert user is not None
    assert user.full_name == "New User"
    assert verify_password("password123", user.hashed_password)

@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, test_user: User):
    response = await client.post(
        "/api/auth/register",
        json={"email": test_user.email, "password": "password123", "full_name": "Duplicate User"}
    )
    assert response.status_code == 409

@pytest.mark.asyncio
async def test_register_short_password(client: AsyncClient):
    response = await client.post(
        "/api/auth/register",
        json={"email": "short@example.com", "password": "short", "full_name": "Short Password"}
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_login_success(client: AsyncClient, test_user: User):
    response = await client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "testpass123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, test_user: User):
    response = await client.post(
        "/api/auth/login",
        json={"email": test_user.email, "password": "wrongpassword"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_login_nonexistent_user(client: AsyncClient):
    response = await client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "password"}
    )
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, auth_headers: dict, test_user: User):
    response = await client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email
    assert data["full_name"] == test_user.full_name

@pytest.mark.asyncio
async def test_get_me_no_token(client: AsyncClient):
    response = await client.get("/api/auth/me")
    assert response.status_code == 401
