"""Test fixtures for the AI/ML Validation Platform."""
import asyncio
import os
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# Set test env vars BEFORE importing app modules
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///'
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['JWT_SECRET'] = 'test-jwt-secret'
os.environ['UPLOAD_DIRECTORY'] = './test_uploads'
os.environ['LOG_LEVEL'] = 'WARNING'

from app.core.database import Base, get_db
from app.core.security import hash_password, create_access_token
from app.core.config import get_settings
from app.models.user import User
from app.models.project import Project
from app.main import app

# Clear cached settings so test env vars take effect
get_settings.cache_clear()

# Test database engine (SQLite async)
test_engine = create_async_engine('sqlite+aiosqlite:///', echo=False)
test_session_maker = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_maker() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope='session')
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Create tables before each test and drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with test_session_maker() as session:
        yield session


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email='test@example.com',
        hashed_password=hash_password('testpass123'),
        full_name='Test User',
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def auth_headers(test_user: User) -> dict:
    """Get authorization headers for the test user."""
    token = create_access_token(data={'sub': test_user.email})
    return {'Authorization': f'Bearer {token}'}


@pytest_asyncio.fixture
async def test_project(db_session: AsyncSession, test_user: User) -> Project:
    """Create a test project."""
    project = Project(
        name='Test Project',
        description='A test project',
        owner_id=test_user.id,
    )
    db_session.add(project)
    await db_session.commit()
    await db_session.refresh(project)
    return project


@pytest.fixture
def sample_model_path(tmp_path):
    """Create a sample sklearn model file."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    X = np.random.rand(100, 5)
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    
    path = tmp_path / 'test_model.pkl'
    joblib.dump(model, path)
    return str(path)


@pytest.fixture
def sample_dataset_path(tmp_path):
    """Create a sample CSV dataset."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    df = pd.DataFrame({
        **{f'feature_{i}': np.random.rand(100) for i in range(5)},
        'target': np.random.choice([0, 1], 100),
    })
    path = tmp_path / 'test_data.csv'
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_regression_model_path(tmp_path):
    """Create a sample regression model."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    X = np.random.rand(100, 4)
    y = X[:, 0] * 10 + X[:, 1] * 5 + np.random.normal(0, 1, 100)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    path = tmp_path / 'test_regressor.pkl'
    joblib.dump(model, path)
    return str(path)


@pytest.fixture
def sample_regression_dataset_path(tmp_path):
    """Create a sample regression dataset."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    df = pd.DataFrame({
        **{f'feat_{i}': np.random.rand(80) for i in range(4)},
        'price': np.random.rand(80) * 100,
    })
    path = tmp_path / 'test_regression_data.csv'
    df.to_csv(path, index=False)
    return str(path)
