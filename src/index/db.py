import sqlite3
from pathlib import Path

import sqlite_vec


def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Открывает (или создаёт) БД, загружает sqlite-vec, применяет PRAGMA."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous  = NORMAL;
        PRAGMA cache_size   = -64000;
        PRAGMA temp_store   = MEMORY;
        PRAGMA foreign_keys = ON;
    """)

    return conn


def apply_schema(conn: sqlite3.Connection, schema_path: str | Path | None = None) -> None:
    """Применяет DDL из schema.sql к открытому соединению."""
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"
    sql = Path(schema_path).read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
