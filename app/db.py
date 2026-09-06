from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

# Base is SQLAlchemy's marker class — any model that inherits from it
# becomes a real database table.
class Base(DeclarativeBase):
    pass


_settings = get_settings()
# The actual connection pool to Postgres
engine = create_async_engine(_settings.database_url, echo=False)
# A factory for creating new sessions
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

# FastAPI dependency (see app/auth.py for how Depends() uses this)
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


async def init_db() -> None:
    import app.models  # noqa: F401 — importing this registers User on Base.metadata
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
