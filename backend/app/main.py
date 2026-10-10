"""AI/ML Model Validation & Assurance Platform - FastAPI Application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import init_db, close_db
from app.core.logging import setup_logging
from app.api.routes import auth, projects, models, datasets, validation, reports

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    settings = get_settings()
    logger.info('Starting %s v%s', settings.APP_NAME, settings.APP_VERSION)
    await init_db()
    logger.info('Database initialized')
    yield
    await close_db()
    logger.info('Application shutdown complete')


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            'A comprehensive platform that evaluates machine-learning models '
            'across multiple dimensions — performance, data quality, fairness, '
            'drift, robustness, explainability, and reliability.'
        ),
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    
    # Register routers
    app.include_router(auth.router, prefix='/api')
    app.include_router(projects.router, prefix='/api')
    app.include_router(models.router, prefix='/api')
    app.include_router(datasets.router, prefix='/api')
    app.include_router(validation.router, prefix='/api')
    app.include_router(reports.router, prefix='/api')
    
    @app.get('/api/health', tags=['Health'])
    async def health_check():
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'version': settings.APP_VERSION,
            'app': settings.APP_NAME,
        }
    
    return app


app = create_app()
