"""
Инвентаризация библиотеки.

Проходит по корневой папке, собирает список всех файлов поддерживаемых форматов,
выгружает в CSV для последующего составления eval-набора.

Никакой обработки содержимого. Никаких внешних зависимостей кроме стандартной библиотеки.
Работает быстро даже на больших каталогах.

Использование:
    python inventory.py "C:\\Users\\alexa\\YandexDisk\\Книги"
    python inventory.py "C:\\Users\\alexa\\YandexDisk\\Книги" --output library_inventory.csv

Что в CSV:
    relative_path  — путь от корня библиотеки (для приватности логов)
    filename       — имя файла
    extension      — расширение в нижнем регистре
    size_mb        — размер в мегабайтах
    folder_depth   — глубина вложенности папок
    parent_folder  — имя ближайшей родительской папки (часто это категория)
"""

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".fb2", ".docx", ".txt",
                        ".djvu", ".mobi", ".azw3"}  # вне MVP, но интересно знать сколько их

# Расширения внутри MVP (для выделения в статистике)
MVP_EXTENSIONS = {".pdf", ".epub", ".fb2", ".docx", ".txt"}


def is_yandex_placeholder(path: Path) -> bool:
    """
    Грубая эвристика: Yandex Disk "облачные" файлы часто имеют размер 0 байт
    или специальные атрибуты. Это приближённая проверка, не идеальная,
    но помогает оценить масштаб проблемы.
    """
    try:
        stat = path.stat()
        # Файл с подозрительно малым размером для книги
        if stat.st_size < 1024 and path.suffix.lower() in MVP_EXTENSIONS:
            return True
    except OSError:
        return True
    return False


def scan_library(root: Path, verbose: bool = False) -> tuple[list[dict], dict]:
    """Возвращает (записи_для_csv, статистика)."""
    records = []
    stats = {
        "total_files_scanned": 0,
        "by_extension": Counter(),
        "unsupported_extensions": Counter(),
        "suspected_yandex_placeholders": 0,
        "total_size_mb": 0.0,
        "errors": 0,
        "deepest_folder": 0,
    }

    for dirpath, dirnames, filenames in os.walk(root):
        # Пропускаем скрытые папки
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            stats["total_files_scanned"] += 1
            path = Path(dirpath) / filename
            ext = path.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                stats["unsupported_extensions"][ext] += 1
                continue

            try:
                stat = path.stat()
                size_mb = stat.st_size / (1024 * 1024)
            except OSError as e:
                stats["errors"] += 1
                if verbose:
                    print(f"  ошибка чтения {path}: {e}", file=sys.stderr)
                continue

            stats["by_extension"][ext] += 1
            stats["total_size_mb"] += size_mb

            placeholder = is_yandex_placeholder(path)
            if placeholder:
                stats["suspected_yandex_placeholders"] += 1

            try:
                rel_path = path.relative_to(root)
            except ValueError:
                rel_path = path

            depth = len(rel_path.parts) - 1  # минус сам файл
            stats["deepest_folder"] = max(stats["deepest_folder"], depth)

            records.append({
                "relative_path": str(rel_path).replace("\\", "/"),
                "filename": filename,
                "extension": ext,
                "size_mb": round(size_mb, 2),
                "folder_depth": depth,
                "parent_folder": path.parent.name,
                "suspected_placeholder": "yes" if placeholder else "no",
            })

    return records, stats


def write_csv(records: list[dict], output: Path):
    if not records:
        print("  нет файлов для записи в CSV")
        return
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def print_stats(stats: dict, records: list[dict]):
    print()
    print("=" * 60)
    print("  СВОДКА")
    print("=" * 60)
    print(f"  Всего просканировано файлов: {stats['total_files_scanned']}")
    print(f"  Из них в поддерживаемых форматах: {sum(stats['by_extension'].values())}")
    print(f"  Суммарный размер: {stats['total_size_mb']/1024:.2f} GB")
    print(f"  Максимальная глубина папок: {stats['deepest_folder']}")
    if stats["errors"]:
        print(f"  ⚠ Ошибки чтения: {stats['errors']}")

    print()
    print("  Расширения в библиотеке:")
    mvp_count = 0
    for ext, count in stats["by_extension"].most_common():
        mark = "✓" if ext in MVP_EXTENSIONS else "⊘ (вне MVP)"
        print(f"    {ext:8} {count:>5}   {mark}")
        if ext in MVP_EXTENSIONS:
            mvp_count += count

    if stats["unsupported_extensions"]:
        print()
        print("  Неподдерживаемые расширения (топ-10):")
        for ext, count in stats["unsupported_extensions"].most_common(10):
            print(f"    {ext or '(нет)':8} {count:>5}")

    print()
    print(f"  Итого для индексации MVP: {mvp_count} файлов")

    if stats["suspected_yandex_placeholders"] > 0:
        pct = 100 * stats["suspected_yandex_placeholders"] / max(1, mvp_count)
        print()
        print(f"  ⚠ Подозрение на 'облачные' файлы Яндекс.Диска: "
              f"{stats['suspected_yandex_placeholders']} ({pct:.1f}%)")
        print(f"    Это файлы со странно малым размером (<1 KB).")
        print(f"    Если их много — проверьте настройки Яндекс.Диска,")
        print(f"    перед индексацией нужно скачать всё локально.")

    # Топ папок по количеству книг — поможет составить eval
    folder_counts = Counter(r["parent_folder"] for r in records)
    print()
    print("  Топ-15 папок по количеству книг (это будущие категории):")
    for folder, count in folder_counts.most_common(15):
        print(f"    {count:>4}  {folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Инвентаризация локальной библиотеки книг",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("library_path",
                        help="Корневая папка библиотеки")
    parser.add_argument("--output", "-o", default="library_inventory.csv",
                        help="Путь к выходному CSV (default: library_inventory.csv)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Подробный вывод (ошибки чтения и т.п.)")
    args = parser.parse_args()

    root = Path(args.library_path)
    if not root.exists():
        print(f"  ✗ Путь не существует: {root}", file=sys.stderr)
        sys.exit(1)
    if not root.is_dir():
        print(f"  ✗ Не папка: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"  Сканируем: {root}")
    print()

    records, stats = scan_library(root, verbose=args.verbose)

    output = Path(args.output)
    write_csv(records, output)

    print_stats(stats, records)

    print()
    print(f"  Результат записан: {output.resolve()}")
    print(f"  Пришлите этот CSV для составления eval-набора.")


if __name__ == "__main__":
    main()
