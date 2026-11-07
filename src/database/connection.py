import os
from pathlib import Path
from collections.abc import Iterator

from sqlalchemy import URL, create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "outputs" / "receipts.db"


def _create_engine() -> Engine:
    configured_url = os.getenv("DATABASE_URL")
    if configured_url:
        database_url = make_url(configured_url)
        if database_url.drivername.startswith("sqlite") and database_url.database not in (None, ":memory:"):
            Path(database_url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    else:
        database_path = Path(os.getenv("DATABASE_PATH", str(DEFAULT_DATABASE_PATH))).expanduser().resolve()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = URL.create("sqlite", database=str(database_path))

    connect_args = {"check_same_thread": False, "timeout": 30} if database_url.drivername.startswith("sqlite") else {}
    database_engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)

    if database_url.drivername.startswith("sqlite"):
        @event.listens_for(database_engine, "connect")
        def _enable_sqlite_constraints(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return database_engine


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def create_db_and_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session