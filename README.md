# Search-doc — семантический поиск по личной библиотеке книг

Локальный CLI-инструмент для поиска книг в персональной библиотеке **по смыслу запроса**, а не только по ключевым словам. Работает офлайн на собственном железе, интегрируется с любым LLM-клиентом, поддерживающим вызов внешних команд (например, Codex).

Пример: на запрос «как развиваться программисту» система находит «The Pragmatic Programmer», «Путь программиста» Сонмеза, «Программист-фанатик» Фаулера — даже если в самих заголовках нет слов «развиваться» или «карьера».

## Связанные проекты

Search-doc использует результаты предварительной каталогизации книг с AI-сгенерированными саммари, полученные через отдельный проект:

**[catalog-doc](https://github.com/alexandre-andreev/catalog-doc)** — каталогизация локальной библиотеки с AI-классификацией и генерацией саммари через DeepSeek API.

Без `catalog.xlsx` из catalog-doc этот проект работать не будет — саммари критически важны для качества семантического поиска.

## Что внутри

- **Embedding-модель:** `intfloat/multilingual-e5-small` (384-мерные векторы, ~480 MB VRAM, ~300-400 чанков/сек на GTX 1650 Ti).
- **Хранилище:** SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec) для векторов + FTS5 для полнотекстового поиска.
- **Гибридный поиск:** semantic + BM25 через Reciprocal Rank Fusion с адаптивными весами (для коротких технических терминов FTS5 получает приоритет, для описательных запросов — semantic).
- **Related books:** при основном поиске дополнительно возвращаются книги, семантически близкие к топ-результату — раздел «также по теме».
- **Дедупликация:** через simhash от саммари — копии одной книги в разных форматах группируются в одну запись.
- **Инкрементальный импорт:** при еженедельном пополнении каталога переиндексируются только изменённые записи.
- **JSON-вывод** для интеграции с LLM-клиентами.

## Метрики качества

На eval-наборе из 35 типичных пользовательских запросов:

| Метрика | Значение |
|---|---|
| Recall@5 | 0.929 |
| Recall@10 | 0.814 |
| MRR | 0.981 |
| Время поиска (тёплый кэш) | ~50 мс |
| Время холодного старта CLI | ~8-10 сек |

---

## Требования

### Железо

- **GPU с CUDA поддержкой** (рекомендуется ≥ 4 GB VRAM). Тестировалось на GTX 1650 Ti (4 GB).
- **RAM**: 16 GB достаточно.
- **Диск**: ~5 GB на модель + БД (зависит от размера каталога).
- ОС: Windows 11 (тестировалось). Должно работать на Linux/macOS с минимальными правками путей.

### Программное обеспечение

- **Python 3.11–3.13.**
- **CUDA Toolkit 12.x** (для GPU-ускорения). Если CUDA нет — модель будет работать на CPU, но в 10-15× медленнее.
- **NVIDIA драйверы**, поддерживающие выбранную версию CUDA.

### Данные

- Файл `catalog.xlsx` из проекта [catalog-doc](https://github.com/alexandre-andreev/catalog-doc) с колонками: `№`, `Название`, `Автор`, `Год`, `Категория`, `Имя файла`, `Папка`, `Саммари` и др.
- Файлы книг физически на диске по путям, указанным в каталоге (опционально — нужны для функции `open`).

---

## Установка

### 1. Клонировать репозиторий

```powershell
cd D:\
git clone https://github.com/alexandre-andreev/Search-doc.git
cd Search-doc
```

(Путь `D:\` — пример, можно любой.)

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

**Это критический шаг.** Обычный `pip install torch` ставит CPU-версию, которая в 10-15× медленнее.

Для CUDA 12.4 (актуально для Python 3.13):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Для CUDA 12.1 (для Python 3.11-3.12):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Проверка, что CUDA работает:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

Должно вывести `CUDA: True | NVIDIA GeForce ...`. Если `False` — переустановите PyTorch с правильным CUDA-индексом.

### 4. Установить остальные зависимости

```powershell
pip install -r requirements.txt
```

Если `requirements.txt` отсутствует — установить вручную:
```powershell
pip install sentence-transformers sqlite-vec pandas openpyxl pyyaml click tqdm simhash numpy psutil pytest
```

### 5. Подготовить данные каталога

Скопируйте `catalog.xlsx` из проекта catalog-doc в папку проекта:

```powershell
mkdir data
copy <путь>\catalog.xlsx data\catalog.xlsx
```

Если у вас нет catalog.xlsx — сначала запустите проект [catalog-doc](https://github.com/alexandre-andreev/catalog-doc) для каталогизации вашей библиотеки.

### 6. Построить индекс

```powershell
python search-doc.py import
```

При первом запуске:
- Будет скачана embedding-модель `multilingual-e5-small` (~470 MB) в HuggingFace-кэш — это **разово**.
- Индексация ~3000 книг занимает 25-30 секунд на GPU.

Результат — файл `cache/semantic_index.sqlite`.

### 7. Проверить работу

```powershell
python search-doc.py status
python search-doc.py search "TDD на Python" --top 5
```

Если второй запрос вернул список книг с релевантными названиями — установка прошла успешно.

---

## Использование

### CLI-команды

```
python search-doc.py import [--catalog data/catalog.xlsx] [--rebuild]
    # Импорт каталога и построение индекса.
    # --rebuild — полная переиндексация (удаление старой БД).

python search-doc.py search "<query>" [опции]
    # Гибридный поиск (semantic + FTS5 + related).

python search-doc.py status [--format text|json]
    # Состояние индекса: количество книг, чанков, модель, время последнего импорта.

python search-doc.py book <id> [--format text|json]
    # Подробная информация по книге.

python search-doc.py open <id>
    # Открыть файл книги системным просмотрщиком.

python search-doc.py categories [--format text|json]
    # Список категорий из таксономии.

python search-doc.py dedup
    # Запуск дедупликации (обычно вызывается автоматически при import).
```

### Опции команды `search`

```
--top N                  Сколько книг вернуть (по умолчанию 10).
--format json|text       Формат вывода (по умолчанию text).
--section "..."          Фильтр по разделу таксономии.
--year-from YYYY         Только книги от указанного года.
--format-filter EXT      Фильтр по формату файла (pdf, epub, ...).
--semantic-only          Только семантический поиск.
--fts-only               Только FTS5 (по словам).
--no-related             Не вычислять related_books.
```

### Примеры

```powershell
# Базовый поиск
python search-doc.py search "архитектура микросервисов"

# JSON-вывод (для интеграции с LLM)
python search-doc.py search "TDD на Python" --format json

# Поиск только в разделе "Программирование"
python search-doc.py search "тестирование" --section "Программирование"

# Только недавние книги
python search-doc.py search "kubernetes" --year-from 2022
```

---

## Интеграция с Codex (или другим LLM-клиентом)

Если ваш LLM-клиент поддерживает Skills/Tools/Plugins, которые могут запускать локальные скрипты — можно интегрировать search-doc как «инструмент поиска книг».

### Для Codex GUI (через Skill)

Создайте файл `C:\Users\<your-user>\.codex\skills\local-book-search\SKILL.md`:

```
---
name: local-book-search
description: Use this skill when the user asks about books, learning materials, technical topics they want to read about, or asks "what to read about X". Searches local YandexDisk library and returns curated recommendations with paths to files.
---

# Local Book Search

## When to use this skill

ALWAYS invoke this skill when the user:
- Asks to find, recommend, select, suggest, or compare books or documents.
- Mentions a topic and asks what to read about it ("книги про X", "что почитать про X", "материалы по теме X").
- Says they want to learn or understand a topic ("хочу разобраться в X", "хочу освоить X").
- Asks for sources, references, or learning materials on any technical subject.
- Mentions any specific technology, methodology, or concept and the context suggests they may want references.

DO NOT skip this skill thinking you can answer from general knowledge. The user has a curated personal library and wants results from it specifically.

## How to invoke

Run from working directory: `D:\_project\Search-doc`

Command:

```powershell
$env:PYTHONPATH='D:\_project\Search-doc\venv-embed-test\Lib\site-packages'
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\alexa\AppData\Local\Programs\Python\Python313\python.exe' 'D:\_project\Search-doc\search-doc.py' search '<query>' --top 10 --format json


Pass the user's query as-is. Do NOT rephrase it before passing to the tool.

## Source of truth

The only trusted source for catalog recommendations is the JSON returned by:
search-doc.py search '<query>' --top 10 --format json
The only trusted way to open a book is:
search-doc.py open <book_id>
Never infer, add, recommend, validate, rank, or open catalog books from UI previews, generated attachment cards, filesystem autocomplete, browser previews, internet links, or any source other than the JSON response.
Internet links or UI-generated previews may appear in the client interface, but they must never be treated as catalog evidence, recommendations, validation, or opening targets. Only search-doc.py output is authoritative.

## How to interpret the JSON response

The response has two arrays you MUST use:
results — top books matching the query, sorted by relevance.
related_books — books semantically close to the top result, "also useful by topic".

For each book, use these fields:
book_id — trusted identifier for opening the book.
title — book title.
author — author, may be null.
year — publication year, may be null.
category — section/subsection.
file_path — full path on disk.
summary — description; use it to explain relevance.
matched_in — where the match happened, such as title/summary.
duplicates — duplicate copies; mention only briefly if non-empty.
Ignore technical fields unless needed for debugging: score, semantic_score, fts_score, catalog_id, schema_version, search_time_ms, weights, embedding_model.

##How to present results to the user

Return the answer in Markdown.
The answer must contain recommendation sections only. Do not add a second bare list of links, attachment list, bibliography paths, repeated links, or a simplified duplicate block after the recommendations.

##Section 1: Top matches

Use top 5 from results.
For each book, on its own block:
1. Bold title — author, year, if known.
2. One sentence explaining why this book matches the user's query, based on summary and matched_in.
3. One clickable Markdown link using file_path, but first convert all Windows backslashes \ to forward slashes /:
[Открыть файл](<converted_file_path>)

Do not show the file path anywhere else.
If duplicates is non-empty, add a quiet note: (есть ещё N копий в других папках) without listing duplicate paths.

##Section 2: Also useful by topic

Use top 3 from related_books.
Only include this section if related_books is non-empty and does not merely repeat the same books from results.
Introduce the section as:
Также может быть полезно по теме:

For each related book, on its own block:
1. Bold title — author, year, if known.
2. One sentence explaining why this book is useful by topic, based on summary.
3. One clickable Markdown link using file_path, but first convert all Windows backslashes \ to forward slashes /:
[Открыть файл](<converted_file_path>)

Do not show the file path anywhere else.

##Link formatting rule

When creating clickable links from file_path, always convert Windows backslashes \ to forward slashes /.
Good:
Открыть файл <C:/Users/alexa/YandexDisk/Книги/Программирование/Python/book.pdf>

Bad:
Открыть файл <C:\Users\alexa\YandexDisk\Книги\Программирование\Python\book.pdf>

Never print raw local file paths.
Never use local file paths as visible link labels.
Never wrap the Markdown link itself in backticks in the final answer.
Each book may have exactly one local file link in the recommendation block.

##External links rule

External internet links may be included only as clearly separate, optional convenience links, never as catalog evidence.
If external links are shown, label them clearly as external:
Внешние ссылки для поиска/справки:
Do not mix internet links with catalog file links.
Do not use internet links to add new books to the recommendation list.
Do not use internet links to validate, replace, open, or rank catalog books.
If no reliable external links are available, omit them silently.

##Follow-up opening rule

When the user asks to open a book, always use the book_id from the last trusted JSON search result and run:
powershell

& 'C:\Users\alexa\AppData\Local\Programs\Python\Python313\python.exe' 'D:\_project\Search-doc\search-doc.py' open <book_id>

Do not open files by clicking Markdown links, UI previews, attachment cards, internet links, or inferred paths.
Briefly confirm:
Открываю "<title>".


##Handling follow-up questions

If the user says "открой N-ю" / "хочу посмотреть первую" / "open book N":
Get book_id from the corresponding result in the last trusted JSON response.
Run search-doc.py open <book_id>.
Briefly confirm: Открываю "<title>".

If the user asks for more details on a specific book ("расскажи подробнее про вторую"):
Use the summary field from the JSON you already have.
If the summary is not enough, run:

& 'C:\Users\alexa\AppData\Local\Programs\Python\Python313\python.exe' 'D:\_project\Search-doc\search-doc.py' book <book_id> --format json

If the user refines the query ("а теперь только русскоязычные" / "только последние пять лет"):
Re-run search with the new query.
Do not pretend to filter the previous results — call the tool again.

##Honesty rules

If the tool returns empty results, say honestly: "В вашей библиотеке ничего не найдено по этому запросу".
Never mention catalog books not present in the JSON response.
Never treat UI-generated previews, file cards, internet links, or attachment suggestions as part of the catalog result.
If the tool errors out with a non-zero exit code, show the error briefly and suggest retrying or checking the catalog.
```

Подставьте свои пути в команде. Перезапустите Codex.

### Для других клиентов (Cherry Studio, AnythingLLM, OpenWebUI)

Зарегистрируйте кастомный инструмент:
- **Имя:** `search_library`
- **Команда:** `python D:\Search-doc\search-doc.py search "{query}" --format json`
- **Параметр:** `query` (string).
- **Промпт:** как в SKILL.md выше.

### Без интеграции

Можно использовать как обычную утилиту в PowerShell:

```powershell
python search-doc.py search "ваш запрос"
```

Вывод в текстовом формате уже читаем человеком — без JSON-обвязки.

---

## Структура проекта

```
Search-doc/
├── search-doc.py                # CLI entry point
├── requirements.txt
├── data/
│   ├── catalog.xlsx             # из catalog-doc, обязателен
│   └── taxonomy.xlsx            # опционально
├── cache/
│   ├── semantic_index.sqlite    # БД с индексом, создаётся при import
│   ├── codex_calls.log          # лог вызовов (для отладки интеграции)
│   └── search-doc.log
├── src/
│   ├── catalog_import/          # импорт из xlsx
│   ├── embedder/                # обёртка sentence-transformers
│   ├── index/                   # SQLite + sqlite-vec
│   ├── search/                  # semantic + FTS5 + ranker + related
│   ├── pipeline/                # оркестрация импорта
│   └── util/
├── tests/                       # unit и e2e тесты
└── eval/
    ├── eval_queries_v2.yaml     # эталонные запросы
    └── run_eval.py
```

---

## Диагностика

### `python search-doc.py status` падает с ошибкой про sqlite-vec

```
sqlite3.OperationalError: not authorized
```

Не загружено расширение sqlite-vec. Проверьте, что пакет установлен:
```powershell
pip show sqlite-vec
```

### CUDA не используется (поиск медленный)

Проверьте:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Если `False` — установлен CPU-вариант PyTorch. Переустановите по инструкции из шага 3.

### Холодный старт CLI ~8-10 секунд

Это нормально — каждый запуск Python заново импортирует torch и грузит модель в GPU. Для интеграции через LLM это «один долгий первый запрос», далее в той же сессии может быть быстрее (зависит от клиента).

Для скриптов, делающих много запросов подряд, можно реализовать daemon-режим (HTTP-сервер с предзагруженной моделью). В текущей версии не реализовано.

### Поиск возвращает 0 результатов

```powershell
python search-doc.py status
```

Проверьте, что в БД есть книги (`books_count > 0`) и чанки. Если нет — запустите `python search-doc.py import`.

### Поиск возвращает «не те» книги

См. лог `cache/codex_calls.log` — там точный query, с которым работал поиск. Если query был перефразирован LLM-клиентом — попробуйте вызвать search-doc напрямую с вашим оригинальным текстом.

---

## Лицензия и атрибуция

Search-doc написан на стыке готовых инструментов:
- [sentence-transformers](https://www.sbert.net/) и модель [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) — embeddings.
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — векторный поиск в SQLite.
- [FTS5](https://www.sqlite.org/fts5.html) — полнотекстовый поиск, встроен в SQLite.

Реализация: Александр Андреев, при участии Claude (Anthropic) для проектирования архитектуры и Claude Code для написания кода.

---

## Контакты и обратная связь

Issues и PR — в [GitHub-репозиторий](https://github.com/alexandre-andreev/Search-doc).

Связанный проект каталогизации: [catalog-doc](https://github.com/alexandre-andreev/catalog-doc).
