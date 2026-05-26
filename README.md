# Search-doc — Семантический поиск по личной библиотеке книг

Локальный CLI-инструмент: ищет книги **по смыслу запроса**, а не только по ключевым словам. Работает офлайн на собственном железе. Возвращает JSON — интегрируется с любым LLM-клиентом (Codex, Cherry Studio, OpenWebUI и др.).

**Пример:** запрос «как развиваться программисту» находит «The Pragmatic Programmer», «Путь программиста» Сонмеза, «Программист-фанатик» — даже если в заголовках нет слов «развиваться» или «карьера».

---

## Как это работает

Search-doc — вторая ступень двухэтапного конвейера:

```
[Ваша папка с книгами]
         │
         ▼
┌──────────────────────┐
│      catalog-doc     │  GUI-приложение: сканирует архив,
│  (отдельный проект)  │  генерирует AI-саммари, экспортирует
└──────────────────────┘  в catalog.xlsx
         │  catalog.xlsx
         ▼
┌──────────────────────┐
│      search-doc      │  CLI: строит векторный индекс,
│    (этот проект)     │  отвечает на запросы, JSON API
└──────────────────────┘
         │
         ▼
   Результаты в терминале / JSON / LLM-клиент
```

1. **catalog-doc** сканирует библиотеку, извлекает метаданные, генерирует AI-саммари (DeepSeek / OpenAI / Groq / Gemini) и экспортирует в `catalog.xlsx`.
2. **search-doc** читает каталог, строит векторный индекс и отвечает на запросы естественным языком.

Саммари — ключевой фактор качества: без них поиск работает только по заголовкам, точность резко падает.

---

## Метрики качества

Eval-набор: 35 типичных пользовательских запросов.

| Метрика | Значение |
|---|---|
| Recall@5 | 0.929 |
| Recall@10 | 0.814 |
| MRR | 0.981 |
| Время поиска (тёплый старт) | ~50 мс |
| Время холодного старта CLI | ~8-10 сек |

---

## Что внутри

- **Embedding-модель:** `intfloat/multilingual-e5-small` — многоязычная, 384-мерные векторы, ~480 МБ VRAM, ~300-400 чанков/сек на GTX 1650 Ti
- **Хранилище:** SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) для векторов + FTS5 для полнотекстового поиска
- **Гибридный поиск:** semantic + BM25 через Reciprocal Rank Fusion с адаптивными весами (короткие технические термины → FTS-приоритет, описательные запросы → семантика)
- **Related books:** при поиске дополнительно возвращаются книги, семантически близкие к топ-результату — раздел «также по теме»
- **Дедупликация:** через simhash от саммари — копии одной книги в разных форматах группируются в одну запись
- **Инкрементальный импорт:** при пополнении каталога переиндексируются только изменённые записи

---

## Требования

### Железо

| Компонент | Минимум | Тестировалось |
|---|---|---|
| GPU | Любой CUDA-совместимый | GTX 1650 Ti (4 GB VRAM) |
| RAM | 8 GB | 16 GB |
| Диск | 5 GB | 5 GB модель + БД |

> **Без GPU:** модель работает на CPU, но в 10-15× медленнее. Для небольших каталогов (до ~500 книг) и разового использования это приемлемо.

### Программное обеспечение

- Python 3.11–3.13
- CUDA Toolkit 12.x (для GPU-ускорения)
- NVIDIA-драйверы, совместимые с CUDA

### Данные

- Файл `catalog.xlsx` из проекта **catalog-doc** с колонками: `Название`, `Автор`, `Год`, `Категория`, `Имя файла`, `Папка`, `Саммари`
- Сами файлы книг на диске — опционально, нужны только для команды `open`

---

## Шаг 0: Подготовить catalog.xlsx через catalog-doc

Если `catalog.xlsx` у вас уже есть — переходите к установке.

**catalog-doc** — GUI-приложение для Windows, которое делает предварительную работу:

- Рекурсивно сканирует папку с книгами (PDF, EPUB, DJVU, FB2, DOCX и др.)
- Извлекает метаданные из файлов: название, автор, год, издательство
- Генерирует AI-саммари на русском через DeepSeek, OpenAI, Groq или Gemini
- Классифицирует книги по иерархической таксономии
- Находит дубликаты (одна книга в нескольких форматах)
- Экспортирует всё в `catalog.xlsx`

Репозиторий: **[catalog-doc](https://github.com/alexandre-andreev/catalog-doc)**

Быстрый старт с нуля:
```
1. Клонируйте catalog-doc и запустите python main.py
2. В GUI выберите папку с книгами → запустите сканирование
3. На вкладке AI-саммари введите API-ключ (DeepSeek дешевле всего) → запустите генерацию
4. Экспортируйте catalog.xlsx
5. Скопируйте catalog.xlsx в data/ этого проекта → переходите к установке ниже
```

---

## Установка

### 1. Клонировать репозиторий

```powershell
git clone https://github.com/alexandre-andreev/Search-doc.git
cd Search-doc
```

### 2. Создать виртуальное окружение

```powershell
python -m venv venv-embed
.\venv-embed\Scripts\Activate.ps1
```

Если PowerShell ругается на политику выполнения:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. Установить PyTorch с CUDA-поддержкой

> Это критический шаг. Обычный `pip install torch` ставит CPU-версию — в 10-15× медленнее.

```powershell
# CUDA 12.4 (Python 3.13, актуальный вариант)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1 (Python 3.11–3.12)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Проверка:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
# Ожидаемый вывод: CUDA: True | NVIDIA GeForce ...
```

Если вывод `CUDA: False` — переустановите PyTorch с правильным CUDA-индексом.

### 4. Установить остальные зависимости


Создать requirements.txt или ссылки на pyproject.toml. Это даёт воспроизводимую установку с зафиксированными версиями

```
powershell
# Если есть requirements.txt:
pip install -r requirements.txt

# Или (если есть pyproject.toml):
pip install -e .
```

Или выполнить "pip install" + список_пакетов_подряд.

```powershell
pip install sentence-transformers sqlite-vec pandas openpyxl pyyaml click tqdm simhash numpy psutil pytest
```

Проверка установки

```powershell
python -c "import sentence_transformers, sqlite_vec, click; print('OK')"
```

### 5. Подготовить данные

```powershell
mkdir data
copy <путь_к_вашему_catalog.xlsx> data\catalog.xlsx
```

### 6. Построить индекс

```powershell
python search-doc.py import
```

При первом запуске:
- Скачивается модель `multilingual-e5-small` (~470 МБ) в HuggingFace-кэш — **один раз**
- Индексация ~3000 книг занимает 25-30 секунд на GPU

Результат — файл `cache/semantic_index.sqlite`.

### 7. Проверить работу

```powershell
python search-doc.py status
python search-doc.py search "TDD на Python" --top 5
```

Если второй запрос вернул список с релевантными книгами — установка успешна.

---

## Использование

### CLI-команды

| Команда | Что делает |
|---|---|
| `import [--catalog path] [--rebuild]` | Импорт каталога и построение/обновление индекса |
| `search "<запрос>" [опции]` | Гибридный поиск |
| `status [--format text\|json]` | Статус индекса: книги, чанки, модель, дата индексации |
| `book <id> [--format json\|text]` | Детали конкретной книги |
| `open <id>` | Открыть файл книги системным просмотрщиком |
| `categories [--format text\|json]` | Список разделов таксономии |
| `dedup` | Дедупликация (обычно вызывается автоматически при import) |

### Опции команды `search`

```
--top N              Количество книг в выдаче (по умолчанию 10)
--format json|text   Формат вывода (по умолчанию json)
--section "..."      Фильтр по разделу таксономии
--year-from YYYY     Только книги от указанного года
--file-format EXT    Фильтр по формату файла: pdf, epub, ...
--semantic-only      Только семантический поиск (без FTS)
--fts-only           Только FTS5 (ключевые слова, без эмбеддингов)
--no-related         Без раздела "также по теме"
```

### Примеры

```powershell
# Базовый поиск
python search-doc.py search "архитектура микросервисов"

# JSON-вывод (для интеграции с LLM)
python search-doc.py search "TDD на Python" --format json

# Только в разделе "Программирование"
python search-doc.py search "тестирование" --section "Программирование"

# Только книги последних лет
python search-doc.py search "kubernetes" --year-from 2022

# Открыть книгу по ID из результатов поиска
python search-doc.py open 42
```

---

## Интеграция с LLM-клиентами

### Codex GUI (через Skill)

Создайте файл `C:\Users\<your-user>\.codex\skills\local-book-search\SKILL.md`:

```markdown
---
name: local-book-search
description: Use this skill when the user asks about books, learning materials, or technical topics they want to read about. Searches local library and returns curated recommendations with paths to files.
---

# Local Book Search

## When to use

ALWAYS invoke when the user asks to find, recommend, or compare books, or says what they want to learn.
DO NOT skip — the user has a curated personal library and wants results from it specifically.

## How to invoke

Working directory: `<path-to-Search-doc>`

```powershell
$env:PYTHONIOENCODING='utf-8'
& '<python.exe path>' '<path-to-Search-doc>\search-doc.py' search '<query>' --top 10 --format json
```

Pass the user's query as-is. Do NOT rephrase it.

## JSON response fields

- `results` — top books by relevance; `related_books` — semantically close extras.
- Use per book: `book_id`, `title`, `author`, `year`, `category`, `file_path`, `summary`, `matched_in`.
- Ignore: `score`, `semantic_score`, `fts_score`, `search_time_ms`, `weights`.

## Presenting results

Show top 5 results: bold title, one sentence of relevance, one clickable link (convert `\` to `/` in path).
Add "Также по теме:" section with top 3 from `related_books` if non-empty.

## Opening books

Always run `search-doc.py open <book_id>` — never open by file path directly.

## Honesty

If empty results: "В вашей библиотеке ничего не найдено."
Never mention books absent from the JSON response.
```

Подставьте свои пути к Python и к папке проекта. Перезапустите Codex.

### Другие клиенты (Cherry Studio, AnythingLLM, OpenWebUI)

Зарегистрируйте кастомный инструмент:
- **Имя:** `search_library`
- **Команда:** `python <path>\search-doc.py search "{query}" --format json`
- **Параметр:** `query` (string)

Промпт по аналогии с SKILL.md выше.

### Без интеграции

```powershell
python search-doc.py search "ваш запрос"
```

Текстовый вывод (`--format text`) читаем без дополнительной обвязки.

---

## Структура проекта

```
Search-doc/
├── search-doc.py                # CLI entry point
├── data/
│   ├── catalog.xlsx             # из catalog-doc, обязателен
│   └── taxonomy.xlsx            # опционально
├── cache/
│   ├── semantic_index.sqlite    # БД с векторным индексом
│   ├── codex_calls.log          # лог вызовов (отладка интеграции)
│   └── search-doc.log
├── src/
│   ├── catalog_import/          # чтение xlsx
│   ├── embedder/                # multilingual-e5-small wrapper
│   ├── index/                   # SQLite + sqlite-vec + dedup
│   ├── search/                  # semantic + FTS5 + ranker + related
│   ├── pipeline/                # оркестрация импорта
│   └── util/
├── tests/
└── eval/
    ├── eval_queries_v2.yaml     # 35 эталонных запросов
    └── run_eval.py
```

---

## Диагностика

### `status` падает с ошибкой sqlite-vec

```
sqlite3.OperationalError: not authorized
```

```powershell
pip show sqlite-vec  # убедитесь, что пакет установлен
```

### CUDA не используется (медленный поиск)

```powershell
python -c "import torch; print(torch.cuda.is_available())"
# False → неправильный вариант PyTorch, переустановите по шагу 3
```

### Холодный старт ~8-10 секунд

Норма: Python каждый раз загружает torch и модель на GPU. При интеграции с LLM-клиентом это «один медленный первый запрос»; далее в той же сессии клиент обычно держит процесс живым.

### 0 результатов поиска

```powershell
python search-doc.py status   # books_count должен быть > 0
python search-doc.py import   # если books_count = 0 — запустите импорт
```

### Поиск возвращает нерелевантные результаты

Откройте `cache/codex_calls.log` — там точный query, с которым работал поиск. Если LLM-клиент перефразировал запрос — попробуйте вызвать search-doc напрямую с исходным текстом.

---

## Лицензия и атрибуция

Search-doc построен на:
- [sentence-transformers](https://www.sbert.net/) + [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- [sqlite-vec](https://github.com/asg017/sqlite-vec)
- [FTS5](https://www.sqlite.org/fts5.html) (встроен в SQLite)

Реализация: Александр Андреев, при участии Claude (Anthropic) для проектирования архитектуры и Claude Code для написания кода.

---

## Обратная связь

Issues и PR: [github.com/alexandre-andreev/Search-doc](https://github.com/alexandre-andreev/Search-doc)

Проект каталогизации: [catalog-doc](https://github.com/alexandre-andreev/catalog-doc)
