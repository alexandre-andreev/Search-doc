-- Метаданные индекса
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Книги
CREATE TABLE IF NOT EXISTS books (
    id              INTEGER PRIMARY KEY,
    catalog_id      INTEGER UNIQUE,

    title           TEXT NOT NULL,
    author          TEXT,
    year            INTEGER,
    publisher       TEXT,
    category        TEXT,
    section         TEXT,
    subsection      TEXT,
    file_format     TEXT,
    file_size_mb    REAL,
    filename        TEXT,
    folder          TEXT,
    summary         TEXT,
    xlsx_row_hash   TEXT NOT NULL,

    text_simhash    INTEGER,
    duplicate_of    INTEGER REFERENCES books(id),

    status          TEXT NOT NULL,
    indexed_at      REAL
);

CREATE INDEX IF NOT EXISTS idx_books_catalog_id ON books(catalog_id);
CREATE INDEX IF NOT EXISTS idx_books_section    ON books(section);
CREATE INDEX IF NOT EXISTS idx_books_simhash    ON books(text_simhash);
CREATE INDEX IF NOT EXISTS idx_books_dup        ON books(duplicate_of);

-- Чанки
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    book_id     INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chunk_kind  TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    text_hash   TEXT    NOT NULL,
    char_count  INTEGER NOT NULL,
    UNIQUE(book_id, chunk_kind, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(chunk_kind);

-- Векторы (sqlite-vec)
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
    chunk_id  INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);

-- FTS5 для полнотекстового поиска
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

-- История импортов
CREATE TABLE IF NOT EXISTS import_runs (
    id              INTEGER PRIMARY KEY,
    started_at      REAL    NOT NULL,
    finished_at     REAL,
    catalog_path    TEXT    NOT NULL,
    books_added     INTEGER DEFAULT 0,
    books_updated   INTEGER DEFAULT 0,
    books_skipped   INTEGER DEFAULT 0,
    books_failed    INTEGER DEFAULT 0,
    chunks_created  INTEGER DEFAULT 0,
    status          TEXT    NOT NULL
);
