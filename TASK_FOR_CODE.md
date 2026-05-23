# TASK_FOR_CODE.md — Постановка задачи для Claude Code

**Проект:** search-doc — семантический поиск по локальной библиотеке книг
**Версия документа:** 1.0 FINAL
**Целевая среда:** Windows 11, Python 3.13, GTX 1650 Ti (4GB VRAM), 16GB RAM
**Рабочий каталог:** `D:\_project\Search-doc`
**Исполнитель:** Claude Code 2.1.140

---

## 0. Краткая постановка

Реализовать локальный CLI-инструмент для семантического поиска по библиотеке из ~3180 технических книг. Интерфейс взаимодействия — локальный Codex GUI, который вызывает CLI с JSON-выводом.

**Главное требование к качеству поиска:** находить документы по смыслу запроса, даже если ключевых слов нет в названиях/саммари книг напрямую. Пользователь должен получать ответ в стиле «что могло бы быть полезным по теме вопроса», а не только точные текстовые совпадения.

---

## 1. Контекст: что уже сделано и проверено

Этот проект — НЕ старт с нуля. Перед написанием кода Claude Code обязан изучить существующие активы.

### 1.1 Уже работающий прототип

В корне проекта лежит `pre_mvp_eval.py` — рабочий скрипт, который:

- Читает `data/catalog.xlsx` (3180 книг с метаданными и AI-саммари).
- Строит embedding'и для названий и саммари через `intfloat/multilingual-e5-small`.
- Кэширует embedding'и в `cache/pre_mvp_embeddings.npy`.
- Выполняет семантический поиск через NumPy (cosine similarity).
- Прогоняет eval-набор из `eval/eval_queries_v2.yaml`.

**Этот код проверен на реальных данных.** Результат последнего прогона:
- R@5 = 0.800, R@10 = 0.690, MRR = 0.763
- 27/35 запросов работают приемлемо, 15/35 идеально
- Время одного поиска < 50 мс

**Большая часть логики из `pre_mvp_eval.py` переиспользуется** в финальной системе. НЕ переписывать с нуля. Конкретно:
- Функции `load_catalog`, `build_chunks`, `safe_int`, `safe_str` — перенести как есть в `src/catalog_import/`.
- Логику embedding-индексации — перенести в `src/embedder/` + `src/index/`.
- Функции `evaluate_query`, `print_eval_report` — перенести в `eval/run_eval.py`.

### 1.2 Eval-набор

`eval/eval_queries_v2.yaml` — 35 запросов с реальными именами книг из каталога. Используется как regression-тест и приёмочный критерий.

### 1.3 Бенчмарк и инвентаризация

В `benchmark.md` зафиксированы результаты тестирования модели на целевой GPU: 384 чанка/сек, ~480 MB VRAM при batch_size=32. Полная индексация ~6500 чанков занимает 21 секунду. НЕ нужно повторно проводить benchmark.

`library_inventory.csv` содержит 2943 файла поддерживаемых форматов. НЕ нужно сканировать заново — использовать как есть.

### 1.4 Каталог книг (главный источник метаданных)

`data/catalog.xlsx`, лист «Каталог». **Колонки** (порядок и имена обязательны):

| Колонка | Тип | Описание |
|---|---|---|
| № | int | Уникальный ID |
| Название | str | Заголовок книги (обязательно) |
| Автор | str / NULL | Автор |
| Год | int / "2016-2017" / NULL | Год издания (может быть диапазоном!) |
| Издательство | str / NULL | |
| Категория | str | Формат "Раздел/Подраздел", напр. "Программирование/Python" |
| Раздел | str | Дублирует первую часть Категории |
| Подраздел | str | Дублирует вторую часть Категории |
| Формат | str | pdf, epub, fb2, docx, txt, mobi, djvu |
| Размер МБ | float | |
| Статус | str | Технический статус классификации, нам не важен |
| Имя файла | str | Имя файла на диске (обязательно) |
| Папка | str | Полный путь к папке с файлом |
| Ссылка на файл | str | Обычно дублирует «Имя файла» |
| Саммари | str / NULL | AI-сгенерированное описание книги, 500 символов в среднем |

**Статистика:** 3180 книг, 3179 с саммари (99.97%), 2881 с автором (90.6%).

**Парсинг года требует aware-логики:** в данных встречаются значения "2016-2017" и подобные. Используется функция `safe_int` из существующего pre_mvp_eval.py — она извлекает первое 4-значное число.

---

## 2. Архитектура финальной системы

### 2.1 Поток данных

```
catalog.xlsx ──► catalog_import ──► SQLite ◄── library files
                                       │       (через extractor — опционально)
                                       ▼
                              embedder (e5-small) ──► chunk_vectors (sqlite-vec)
                                       │
                                       ▼
                              FTS5 индекс ──► чанки полнотекстово

User Query ──► CLI search ──► semantic + FTS5 + related ──► JSON ──► Codex
```

### 2.2 Структура проекта

```
D:\_project\Search-doc\
├── search-doc.py                # CLI entry point (главный файл)
├── pyproject.toml
├── README.md
├── config/
│   └── default.yaml
├── data/
│   ├── catalog.xlsx             # уже есть
│   └── taxonomy.xlsx            # уже есть, опциональный
├── cache/                       # БД и логи
│   └── semantic_index.sqlite
├── src/
│   ├── __init__.py
│   ├── catalog_import/
│   │   ├── __init__.py
│   │   ├── xlsx_loader.py       # перенос из pre_mvp_eval.load_catalog
│   │   └── matcher.py           # сопоставление xlsx ↔ inventory
│   ├── embedder/
│   │   ├── __init__.py
│   │   └── e5_small.py          # обёртка над sentence-transformers
│   ├── index/
│   │   ├── __init__.py
│   │   ├── schema.sql
│   │   └── db.py                # SQLite + sqlite-vec + FTS5
│   ├── search/
│   │   ├── __init__.py
│   │   ├── semantic.py          # cosine через sqlite-vec
│   │   ├── fts.py               # BM25 через FTS5
│   │   ├── ranker.py            # RRF + адаптивные веса
│   │   └── related.py           # ★ "соседние книги"
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── indexer.py
│   └── util/
│       ├── __init__.py
│       └── logging.py
├── eval/
│   ├── eval_queries_v2.yaml     # уже есть
│   └── run_eval.py              # перенос из pre_mvp_eval с адаптацией
└── tests/
    ├── fixtures/
    └── test_*.py
```

**ВАЖНО:** существующее виртуальное окружение `venv-embed-test\` уже содержит все нужные зависимости. НЕ создавать новое окружение. НЕ переустанавливать torch, sentence-transformers, pandas, openpyxl, pyyaml. Только доустановить: `sqlite-vec`, `click`, `tqdm`, `pytest`.

---

## 3. Этапы реализации

**Дисциплина выполнения:** реализовывать строго по этапам. Каждый этап завершается рабочим артефактом, который можно проверить. Не переходить к следующему этапу, пока предыдущий не работает.

### Этап 1: Каркас + БД + миграция данных каталога

**Цель:** к концу этапа в SQLite лежат 3180 книг с метаданными и саммари из catalog.xlsx, плюс их embedding'и.

**Реализовать:**

- `src/index/schema.sql` — DDL (см. раздел 4).
- `src/index/db.py` — обёртка SQLite с PRAGMA (WAL, sync=NORMAL, cache=64MB), загрузка `sqlite-vec`.
- `src/catalog_import/xlsx_loader.py` — перенос `load_catalog` из pre_mvp_eval, без изменений в логике парсинга.
- `src/embedder/e5_small.py` — обёртка над `SentenceTransformer`, с обязательной проверкой `cuda_available` (падать с ошибкой, если CUDA нет).
- CLI-команда `search-doc.py import` — читает catalog.xlsx, строит чанки (title + summary), считает embedding'и, кладёт в БД.

**Чанки в БД:**
- Для каждой книги создаётся 1-2 записи в таблице `chunks`:
  - `chunk_kind='title'`, text = "Название. Автор"
  - `chunk_kind='summary'`, text = саммари (если есть и длина ≥ 100)
- Для каждого чанка считается embedding (e5-small, normalize=True, префикс "passage: ") и кладётся в `chunk_vectors`.
- Параллельно текст чанка идёт в FTS5 (через триггер).

**Acceptance:**
```powershell
python search-doc.py import
# Ожидаемый вывод: "Импортировано 3180 книг, создано ~6300 чанков, время ~25 сек"

python search-doc.py status
# Ожидаемый вывод: "Книг: 3180, чанков: 6267 (title: 3180, summary: 3087), модель: e5-small, dim: 384"
```

### Этап 2: Базовый семантический поиск

**Цель:** воспроизвести качество `pre_mvp_eval.py` (R@5=0.800, MRR=0.763), но через SQLite вместо NumPy.

**Реализовать:**

- `src/search/semantic.py` — `semantic_search(query, k=50) -> [(chunk_id, score)]`. Использует `sqlite-vec` для поиска k ближайших векторов.
- `src/search/ranker.py` — функция `aggregate_to_books(hits)` агрегирует чанки в книги:
  - `book_score = max(chunk_scores * kind_bonus) + log(1 + N_matched) * 0.1`
  - `kind_bonus`: title=1.5, summary=1.3, body=1.0
- CLI-команда `search-doc.py search "<query>" --format json` — выполняет поиск, возвращает JSON по схеме из раздела 5.

**Acceptance:**
- Прогон `eval/run_eval.py` (адаптированный из `pre_mvp_eval.py`) даёт метрики не хуже текущих: R@5 ≥ 0.78, MRR ≥ 0.74.
- Время поиска < 200 мс на запрос.

### Этап 3: Гибридный поиск (semantic + FTS5)

**Цель:** добавить FTS5 как второй источник кандидатов; объединить через RRF с **адаптивными весами**.

**Реализовать:**

- `src/search/fts.py` — `fts_search(query, k=50) -> [(chunk_id, bm25_score)]`. Использует FTS5 MATCH.
- В `src/search/ranker.py` добавить:
  - Детектор «технического запроса»: содержит ли запрос короткие технические токены типа `pytest`, `Go`, `LLM`, `React`, `K8s` и т.п. Эвристика: токен длиной ≤ 8 символов, в верхнем регистре или CamelCase, или входит в whitelist `["pytest", "react", "vue", "go", "rust", "llm", "gpt", "tdd", "ddd", "sql", "css", "k8s", "rag", ...]` (см. project_spec.yaml).
  - Если запрос технический → RRF веса: semantic=0.4, fts=0.6.
  - Если запрос описательный → RRF веса: semantic=0.85, fts=0.15.
  - RRF: `score = w_sem * 1/(k_sem + rank_sem) + w_fts * 1/(k_fts + rank_fts)`, где `k_sem=60`, `k_fts=200` (асимметричные константы дают приоритет semantic).

**Acceptance:**
- На eval-наборе среди НАСТОЯЩИХ провалов pre_mvp (`pytest`, `golang`, `llm_abbreviation`) — все три должны теперь находить нужные книги в топ-3.
- Описательные запросы (`semantic_purely`, `cross_lingual`) НЕ должны деградировать: R@10 ≥ 0.9 на них.

### Этап 4: Соседние книги (related_books)

**Цель:** реализовать главную фичу — «что могло бы быть полезно по теме».

**Реализовать:**

- `src/search/related.py` — `find_related(book_id, k=10) -> [(book_id, score)]`. Берёт embedding саммари указанной книги, ищет k ближайших (через `sqlite-vec`), исключает саму книгу и её дубликаты.
- В CLI `search-doc.py search` добавить ВТОРОЙ ШАГ ПОИСКА после основного:
  - Если основной поиск нашёл хотя бы один результат → берём book_id топ-1 → запускаем `find_related(book_id, k=10)` → фильтруем книги, которые уже есть в основной выдаче → берём топ-5 оставшихся.
- В JSON-выводе появляется поле `related_books` параллельно `results`.

**Acceptance:**
- Запрос «как развиваться программисту и расти в карьере» → топ-1 это книга типа «The Pragmatic Programmer» → related_books содержит другие книги по soft skills/карьерному росту, не дубликаты топ-10.
- Время поиска (основной + related) < 250 мс суммарно.

### Этап 5: Дедупликация

**Цель:** не показывать в выдаче 5 копий одной и той же книги (PDF/EPUB одного издания, копии в разных папках).

**Реализовать:**

- `src/index/dedup.py` — функция `compute_simhash(text)` (через библиотеку `simhash` или собственная реализация на 64 битах).
- На этапе импорта: для каждой книги считается simhash первых 50 KB саммари (или title если саммари нет). Сохраняется в `books.text_simhash`.
- После импорта: команда `search-doc.py dedup` — проходит по всем книгам, для каждой ищет другие с Hamming distance ≤ 3, помечает как `duplicate_of`. Канонической становится: epub/fb2 > docx > pdf > mobi > djvu > txt; при равенстве — большая по размеру файла.
- В выдаче `search` — книги с `duplicate_of != NULL` пропускаются (но их пути добавляются в поле `duplicates` канонической записи).

**Acceptance:**
- Команда `dedup` выполняется без ошибок, рапортует количество найденных дубликатов.
- Запросы с очевидными дублями в каталоге (например, есть и «PostgreSQL 14» pdf, и epub) показывают одну запись в выдаче.

### Этап 6: Инкрементальный переиндекс

**Цель:** при еженедельном пополнении каталога не пересчитывать всё с нуля.

**Реализовать:**

- В таблице `books` поле `xlsx_row_hash` — sha256 от строки catalog.xlsx (название + автор + саммари + год).
- Команда `search-doc.py import` сравнивает хеш строки в xlsx с хешем в БД:
  - Если книги нет в БД → добавить + посчитать embedding'и.
  - Если хеш не изменился → пропустить.
  - Если хеш изменился → удалить старые чанки и embedding'и, посчитать новые.
- Опция `--rebuild` для принудительной полной переиндексации.

**Acceptance:**
- Первый запуск `import` индексирует 3180 книг (~25 сек).
- Повторный запуск без изменений в catalog.xlsx — 0 новых, 0 обновлённых, ~1 сек.
- Если изменить саммари у одной книги в xlsx → следующий import пересчитает только её.

### Этап 7: Полировка CLI

**Цель:** довести интерфейс до production-качества.

**Реализовать:**

- `search-doc.py categories` — список разделов/подразделов из таксономии.
- `search-doc.py book <id>` — детали книги в JSON.
- `search-doc.py open <id>` — открыть файл системным просмотрщиком (`os.startfile` на Windows).
- Фильтры для `search`: `--section "Программирование"`, `--year-from 2020`, `--format pdf,epub`, `--semantic-only`, `--fts-only`.
- Подробные сообщения об ошибках, понятные коды возврата (0/1/2 для разных случаев).
- Поддержка `--verbose` и логирование в `cache/search-doc.log`.

**Acceptance:**
- Все команды работают, JSON-выводы валидны.
- При повреждённой БД или отсутствующих файлах — внятная диагностика, не stacktrace.

---

## 4. Схема SQLite

```sql
-- Метаданные индекса
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Обязательные ключи: schema_version, embedding_model, embedding_dim,
-- created_at, last_indexed_at, catalog_imported_at

-- Книги
CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    catalog_id INTEGER UNIQUE,                  -- № из catalog.xlsx

    -- Метаданные из xlsx
    title TEXT NOT NULL,
    author TEXT,
    year INTEGER,                               -- может быть NULL (или взято из диапазона)
    publisher TEXT,
    category TEXT,                              -- "Программирование/Python"
    section TEXT,
    subsection TEXT,
    file_format TEXT,                           -- pdf, epub, ...
    file_size_mb REAL,
    filename TEXT,
    folder TEXT,
    summary TEXT,                               -- AI-саммари из xlsx
    xlsx_row_hash TEXT NOT NULL,                -- для инкрементального переиндекса

    -- Дедупликация
    text_simhash INTEGER,
    duplicate_of INTEGER REFERENCES books(id),

    -- Состояние
    status TEXT NOT NULL,                       -- 'imported' | 'failed'
    indexed_at REAL
);

CREATE INDEX idx_books_catalog_id ON books(catalog_id);
CREATE INDEX idx_books_section ON books(section);
CREATE INDEX idx_books_simhash ON books(text_simhash);
CREATE INDEX idx_books_dup ON books(duplicate_of);

-- Чанки
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    chunk_kind TEXT NOT NULL,                   -- 'title' | 'summary' | (в будущем 'body')
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    char_count INTEGER NOT NULL,
    UNIQUE(book_id, chunk_kind, chunk_index)
);

CREATE INDEX idx_chunks_book ON chunks(book_id);
CREATE INDEX idx_chunks_kind ON chunks(chunk_kind);

-- Векторы (sqlite-vec)
CREATE VIRTUAL TABLE chunk_vectors USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[384]
);

-- FTS5 для полнотекстового поиска
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

-- История импортов
CREATE TABLE import_runs (
    id INTEGER PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    catalog_path TEXT NOT NULL,
    books_added INTEGER DEFAULT 0,
    books_updated INTEGER DEFAULT 0,
    books_skipped INTEGER DEFAULT 0,
    books_failed INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    status TEXT NOT NULL                        -- 'completed' | 'interrupted' | 'failed'
);
```

**Обязательные PRAGMA при открытии БД:**
```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;     -- 64 MB
PRAGMA temp_store = MEMORY;
PRAGMA foreign_keys = ON;
```

---

## 5. CLI и JSON-контракт

### 5.1 Команды

```
search-doc.py import [--catalog data/catalog.xlsx] [--rebuild]
search-doc.py search "<query>" [--top 10] [--format json|text] [--section "..."] [--year-from N] [--semantic-only] [--fts-only] [--no-related]
search-doc.py status [--format json|text]
search-doc.py book <id> [--format json|text]
search-doc.py open <id>
search-doc.py categories [--format json|text]
search-doc.py dedup
```

### 5.2 JSON-схема ответа `search`

ЭТО КОНТРАКТ С CODEX. Менять только при bump'е `schema_version`.

```json
{
  "schema_version": "1.0",
  "query": "как стать лучше как программист",
  "search_strategy": "hybrid",
  "weights": {"semantic": 0.85, "fts": 0.15},
  "embedding_model": "intfloat/multilingual-e5-small",
  "filters_applied": {"section": null, "year_from": null, "format": null},
  "search_time_ms": 87,

  "results": [
    {
      "rank": 1,
      "book_id": 1247,
      "catalog_id": 42,
      "title": "The Pragmatic Programmer",
      "author": "Andrew Hunt, David Thomas",
      "year": 2019,
      "publisher": "Addison-Wesley",
      "category": "Программирование/Общие вопросы",
      "section": "Программирование",
      "subsection": "Общие вопросы",
      "file_format": "epub",
      "file_path": "C:/Users/alexa/YandexDisk/Книги/Общие вопросы/The Pragmatic Programmer.epub",
      "score": 0.847,
      "matched_in": ["title", "summary"],
      "summary": "Классическая книга о профессиональном развитии программистов...",
      "matched_chunks": [
        {
          "kind": "summary",
          "text": "Книга описывает практики, привычки и философию...",
          "semantic_score": 0.91,
          "fts_score": null
        }
      ],
      "duplicates": []
    }
  ],

  "related_books": [
    {
      "rank": 1,
      "book_id": 892,
      "title": "Soft Skills: The Software Developer's Life Manual",
      "author": "John Sonmez",
      "year": 2020,
      "category": "Саморазвитие",
      "file_format": "pdf",
      "file_path": "...",
      "similarity_to_top_result": 0.81,
      "summary": "Книга о soft skills..."
    }
  ]
}
```

**Поведение по полям:**
- `results` — топ-N книг по гибридному скорингу.
- `related_books` — топ-5 семантически близких к топ-1 из results, БЕЗ пересечения с results.
- Если `--no-related` → `related_books` пустой массив.
- Если запрос ничего не нашёл → `results` пустой, `related_books` пустой.
- `matched_chunks[].text` обрезается до 500 символов в JSON.
- `score` округляется до 3 знаков.

### 5.3 Текстовый формат (для отладки)

При `--format text` — человекочитаемый вывод, по одной книге на блок, с разделителями. Подробности на усмотрение реализации, но обязательно: заголовок, автор, score, кратко matched chunks, отдельная секция «Related».

### 5.4 Коды возврата

```
0  — найдены результаты (search) / операция успешна
1  — поиск выполнен, ничего не найдено
2  — ошибка конфигурации (нет БД, нет CUDA, отсутствует sqlite-vec)
3  — частичный успех (есть failed-записи, но команда отработала)
130 — прерывание (Ctrl+C)
```

---

## 6. Зависимости

Доустановить в существующее `venv-embed-test`:

```powershell
pip install sqlite-vec click tqdm pytest
```

Уже установлено и НЕ ТРОГАТЬ: `torch`, `sentence-transformers`, `pandas`, `openpyxl`, `pyyaml`, `numpy`, `psutil`.

`pyproject.toml` должен включать все используемые пакеты, но `torch` указать с примечанием в README, что для CUDA устанавливается отдельно:

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 7. Тестирование

### 7.1 Что обязательно покрыть unit-тестами

- `safe_int` — все edge cases (None, "2016-2017", "2016", 2016.0, "abc").
- `load_catalog` — загрузка из xlsx, корректная обработка NULL.
- `compute_simhash` — стабильность, одинаковый текст → одинаковый хеш; чуть изменённый → малое расстояние Хэмминга.
- `rrf_fuse` — корректное слияние, асимметричные веса.
- `find_related` — исключение себя и дубликатов из выдачи.

### 7.2 E2E-тест

`tests/test_e2e.py` — берёт фикстуру из 5 книг (можно использовать первые 5 строк catalog.xlsx), индексирует во временную БД, прогоняет 3 запроса, проверяет структуру JSON-ответа.

### 7.3 Регрессионный тест (eval)

`eval/run_eval.py` — переносится из `pre_mvp_eval.py` с минимальной адаптацией под новый источник (читает из БД, а не из NumPy). После каждого этапа прогоняется eval. Метрики **не должны деградировать** относительно baseline (Этап 2: R@5 ≥ 0.78, MRR ≥ 0.74).

---

## 8. Критерии приёмки финального MVP

**Функциональные:**

1. `search-doc.py import` создаёт БД с 3180 книгами и 6200+ чанками за ≤ 60 сек на холодном старте.
2. `search-doc.py search "...."  --format json` возвращает валидный JSON по схеме раздела 5.2.
3. Время поиска: 95-перцентиль ≤ 300 мс (включая related_books).
4. Прогон `eval/run_eval.py` даёт **R@5 ≥ 0.85, R@10 ≥ 0.80, MRR ≥ 0.80** на eval_queries_v2.yaml.
   - Это выше baseline pre_mvp потому что добавлен FTS5 для технических запросов.
5. Все 3 настоящих провала pre_mvp (`pytest`, `golang`, `llm_abbreviation`) находят целевые книги в топ-3.
6. Запрос «как развиваться программисту и расти в карьере» возвращает результаты + related_books, в которых есть книги по soft skills/карьере, не дубли результатов.
7. Повторный `import` без изменений в catalog.xlsx занимает < 5 сек.

**Качественные:**

8. Код покрыт unit-тестами на ключевые функции (см. 7.1).
9. `README.md` содержит инструкцию: установка → import → search → eval.
10. Все ошибки логируются в `cache/search-doc.log`, понятные сообщения пользователю.

---

## 9. Что НЕ делать в MVP

Чтобы не размывать фокус:

- ❌ **Body-extractor для PDF/EPUB/DOCX/FB2/TXT.** Это Этап 8 (после MVP). MVP полностью работает на title+summary из catalog.xlsx.
- ❌ **OCR для сканированных PDF.**
- ❌ **GUI-вкладка.** Codex GUI — это и есть пользовательский интерфейс.
- ❌ **Реклассификация книг через LLM.** Используем то, что уже есть в catalog.xlsx.
- ❌ **Альтернативные embedding-провайдеры** (OpenAI, Gemini). Только e5-small локально.
- ❌ **Cross-encoder reranking.** Это потенциальная следующая итерация.
- ❌ **Web/REST API.** Только CLI.
- ❌ **Многопроцессность.** Для 6000 чанков на одной GPU не нужна.

---

## 10. Технические детали, на которые часто ошибаются

### 10.1 e5-small требует префиксы

- При индексации (passages): `"passage: " + text`
- При поиске (query): `"query: " + text`

Без префиксов качество падает на 10-20%. Это требование модели, не наша прихоть.

### 10.2 sentence-transformers и normalize_embeddings

```python
model.encode(texts, normalize_embeddings=True)
```

Обязательно `True`. Это даёт unit-vectors, после чего cosine = dot product, и sqlite-vec работает корректно.

### 10.3 sqlite-vec на Windows

Загрузка расширения:
```python
import sqlite_vec
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)
```

DLL ставится автоматически с pip-пакетом. Не нужно вручную добавлять в PATH.

### 10.4 FTS5 и русский язык

Токенизатор `unicode61 remove_diacritics 2` корректно работает с кириллицей. НЕ использовать `porter` (только английский).

### 10.5 Проверка CUDA

При инициализации embedder ОБЯЗАТЕЛЬНО проверять:
```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA недоступна. Установите PyTorch с CUDA-поддержкой.")
```

Тихий fallback на CPU превратит 25-секундный импорт в 5-минутный.

### 10.6 Кэш модели HuggingFace

Модель `intfloat/multilingual-e5-small` уже скачана в `C:\Users\<user>\.cache\huggingface\hub\`. Использовать её, не загружать повторно. `SentenceTransformer(name)` сам найдёт.

---

## 11. Порядок работы с этим документом

Claude Code не должен пытаться реализовать всё за один сеанс. Рекомендованный workflow:

1. **Перед началом:** прочитать этот документ + `project_spec.yaml` + просмотреть `pre_mvp_eval.py` целиком.
2. **Реализовывать по этапам** (1-7). Каждый этап — отдельная сессия.
3. После каждого этапа — прогонять unit-тесты + eval (после этапа 2 и далее).
4. Если этап не проходит критерии приёмки — НЕ переходить дальше, разбираться.
5. Использовать `pre_mvp_eval.py` как референс корректности — он работает, его логика верна.

---

## 12. Контактные точки с пользователем

Случаи, когда Claude Code должен **остановиться и спросить**, а не действовать сам:

- Любая необходимость изменить JSON-схему ответа (раздел 5.2) — это контракт с Codex.
- Любое решение поменять модель embedding'ов или её параметры.
- Изменение целевых метрик eval (раздел 8 пункт 4).
- Конфликты PRAGMA SQLite или непонятное поведение `sqlite-vec`.
- Просьба «пропустить тестирование, чтобы быстрее» — нет, тестирование не пропускается.

Случаи, когда можно действовать самостоятельно:

- Имена внутренних функций, переменных.
- Структура отдельных модулей, если общая архитектура из раздела 2.2 соблюдена.
- Стиль форматирования вывода `--format text`.
- Конкретный whitelist «технических токенов» для адаптивных весов (минимально нужный список в project_spec.yaml).
- Выбор библиотеки для simhash (или собственная реализация).
