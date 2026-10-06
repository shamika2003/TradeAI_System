from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(
    r"C:\Users\user\Documents\MULTI-LANG-PROJECT\TradeAI_System"
)

OUTPUT_FILE = ROOT_DIR / "TRADEAI_SYSTEM_FULL_CODE.md"


# ============================================================
# FILE TYPES WHOSE CONTENT WILL BE EXPORTED
# ============================================================

TEXT_EXTENSIONS = {
    # Python
    ".py",

    # Data / configuration
    ".json",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",

    # Documentation
    ".md",

    # PowerShell / Windows scripts
    ".ps1",
    ".psm1",
    ".bat",
    ".cmd",

    # Web
    ".html",
    ".htm",
    ".css",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",

    # SQL / shell
    ".sql",
    ".sh",

    # Java / C / C++ / C#
    ".java",
    ".cs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",

    # XML
    ".xml",
}


# Files without a normal extension that are still text files
SPECIAL_TEXT_FILES = {
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    "dockerfile",
    "makefile",
    "license",
}


# ============================================================
# OPTIONAL DIRECTORY CONTENT EXCLUSIONS
# ============================================================
#
# IMPORTANT:
#
# These directories STILL appear in the TREE.
#
# Their contents are simply not dumped into the source-code
# section because they are normally generated/cache folders.
#
# Remove anything from this set if you want its files dumped too.
# ============================================================

SKIP_CONTENT_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "env",
}


# ============================================================
# FILES WE MUST NOT READ BACK INTO THE EXPORT
# ============================================================

SKIP_CONTENT_FILES = {
    OUTPUT_FILE.name,
}


# ============================================================
# LANGUAGE NAMES FOR MARKDOWN CODE BLOCKS
# ============================================================

LANGUAGE_MAP = {
    ".py": "python",
    ".json": "json",
    ".txt": "text",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",

    ".ps1": "powershell",
    ".psm1": "powershell",
    ".bat": "bat",
    ".cmd": "bat",

    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".tsx": "tsx",

    ".sql": "sql",
    ".sh": "bash",

    ".java": "java",
    ".cs": "csharp",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",

    ".xml": "xml",
}


# ============================================================
# HELPERS
# ============================================================

def sort_entries(entries: list[Path]) -> list[Path]:
    """
    Sort directories first, then files, alphabetically.
    """

    return sorted(
        entries,
        key=lambda p: (
            not p.is_dir(),
            p.name.lower(),
        ),
    )


def is_text_file(path: Path) -> bool:
    """
    Return True if this file should have its content dumped
    into the Markdown document.
    """

    if path.name == OUTPUT_FILE.name:
        return False

    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True

    if path.name.lower() in SPECIAL_TEXT_FILES:
        return True

    return False


def should_skip_content(path: Path) -> bool:
    """
    Determine whether file content should be skipped because
    the file lives inside a generated/cache directory.
    """

    try:
        relative = path.relative_to(ROOT_DIR)
    except ValueError:
        return True

    for part in relative.parts[:-1]:
        if part in SKIP_CONTENT_DIRECTORIES:
            return True

    return False


def safe_read_text(path: Path) -> str:
    """
    Read a text file while handling common encodings.
    """

    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "cp1252",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as exc:
            return f"[ERROR READING FILE: {exc}]"

    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return f"[ERROR READING FILE: {exc}]"


def markdown_language(path: Path) -> str:
    """
    Get Markdown syntax-highlighting language.
    """

    if path.name.lower() == ".gitignore":
        return "text"

    return LANGUAGE_MAP.get(
        path.suffix.lower(),
        "text",
    )


def relative_path(path: Path) -> str:
    """
    Return a clean project-relative path.
    """

    return path.relative_to(ROOT_DIR).as_posix()


# ============================================================
# COMPLETE PROJECT TREE
# ============================================================

def build_tree(directory: Path) -> list[str]:
    """
    Build a complete visual project tree.

    Every directory/file is shown, including binary files,
    model files, images, datasets, artifacts, etc.

    The generated output Markdown itself is excluded because
    otherwise every export would contain its previous copy.
    """

    lines: list[str] = []

    lines.append(f"{ROOT_DIR.name}/")

    def walk(current: Path, prefix: str = "") -> None:

        try:
            entries = [
                p
                for p in current.iterdir()
                if p.resolve() != OUTPUT_FILE.resolve()
            ]
        except PermissionError:
            lines.append(
                prefix + "└── [PERMISSION DENIED]"
            )
            return

        entries = sort_entries(entries)

        for index, entry in enumerate(entries):

            is_last = index == len(entries) - 1

            connector = (
                "└── "
                if is_last
                else "├── "
            )

            if entry.is_dir():
                lines.append(
                    prefix
                    + connector
                    + entry.name
                    + "/"
                )
            else:
                lines.append(
                    prefix
                    + connector
                    + entry.name
                )

            if entry.is_dir():

                child_prefix = (
                    prefix
                    + (
                        "    "
                        if is_last
                        else "│   "
                    )
                )

                walk(
                    entry,
                    child_prefix,
                )

    walk(directory)

    return lines


# ============================================================
# FIND SOURCE FILES
# ============================================================

def collect_source_files() -> list[Path]:

    files: list[Path] = []

    for root, dirs, filenames in os.walk(ROOT_DIR):

        root_path = Path(root)

        for filename in filenames:

            path = root_path / filename

            if path.resolve() == OUTPUT_FILE.resolve():
                continue

            if filename in SKIP_CONTENT_FILES:
                continue

            if should_skip_content(path):
                continue

            if is_text_file(path):
                files.append(path)

    files.sort(
        key=lambda p: relative_path(p).lower()
    )

    return files


# ============================================================
# MARKDOWN FENCE
# ============================================================

def make_code_fence(content: str) -> str:
    """
    Create a Markdown fence long enough that code containing
    ``` does not accidentally close our Markdown block.
    """

    longest = 3

    current = 0

    for char in content:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return "`" * (longest + 1)


# ============================================================
# EXPORT
# ============================================================

def export_project() -> None:

    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"Project folder not found:\n{ROOT_DIR}"
        )

    print("=" * 75)
    print("TRADEAI SYSTEM - FULL PROJECT EXPORT")
    print("=" * 75)

    print()
    print(f"Project : {ROOT_DIR}")
    print(f"Output  : {OUTPUT_FILE}")
    print()

    # --------------------------------------------------------
    # BUILD PROJECT TREE
    # --------------------------------------------------------

    print("Building complete project tree...")

    tree_lines = build_tree(ROOT_DIR)

    print(
        f"Tree generated: {len(tree_lines):,} entries"
    )

    # --------------------------------------------------------
    # FIND SOURCE FILES
    # --------------------------------------------------------

    print("Finding source/text files...")

    source_files = collect_source_files()

    print(
        f"Source files found: {len(source_files):,}"
    )

    # --------------------------------------------------------
    # WRITE MARKDOWN
    # --------------------------------------------------------

    print("Writing Markdown export...")

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as md:

        # ====================================================
        # DOCUMENT HEADER
        # ====================================================

        md.write("# TradeAI System - Full Project Export\n\n")

        md.write(
            f"**Project:** `{ROOT_DIR.name}`\n\n"
        )

        md.write(
            "**Root:** "
            f"`{ROOT_DIR}`\n\n"
        )

        md.write(
            "**Generated:** "
            f"`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
        )

        md.write(
            f"**Exported source/text files:** "
            f"`{len(source_files)}`\n\n"
        )

        md.write("---\n\n")

        # ====================================================
        # PROJECT TREE
        # ====================================================

        md.write("# Project File Tree\n\n")

        md.write(
            "The tree below shows the complete project "
            "structure. Binary files are shown here even when "
            "their contents are not exported.\n\n"
        )

        md.write("```text\n")

        for line in tree_lines:
            md.write(line)
            md.write("\n")

        md.write("```\n\n")

        md.write("---\n\n")

        # ====================================================
        # SOURCE CONTENTS
        # ====================================================

        md.write("# Source Files\n\n")

        md.write(
            "The following sections contain the actual "
            "contents of supported source, configuration, "
            "script and text files.\n\n"
        )

        # Table of contents
        md.write("## Source File Index\n\n")

        for number, path in enumerate(
            source_files,
            start=1,
        ):
            md.write(
                f"{number}. `{relative_path(path)}`\n"
            )

        md.write("\n---\n\n")

        # Actual files
        for number, path in enumerate(
            source_files,
            start=1,
        ):

            rel = relative_path(path)

            print(
                f"[{number:04d}/{len(source_files):04d}] "
                f"{rel}"
            )

            content = safe_read_text(path)

            language = markdown_language(path)

            fence = make_code_fence(content)

            md.write(
                f"## {number:04d}. `{rel}`\n\n"
            )

            try:
                size = path.stat().st_size
            except OSError:
                size = 0

            md.write(
                f"**File:** `{rel}`  \n"
            )

            md.write(
                f"**Size:** `{size:,} bytes`  \n"
            )

            md.write(
                f"**Type:** `{path.suffix or 'no extension'}`\n\n"
            )

            md.write(
                f"{fence}{language}\n"
            )

            md.write(content)

            if content and not content.endswith("\n"):
                md.write("\n")

            md.write(
                f"{fence}\n\n"
            )

            md.write("---\n\n")

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    output_size = OUTPUT_FILE.stat().st_size

    print()
    print("=" * 75)
    print("EXPORT COMPLETE")
    print("=" * 75)
    print()

    print(
        f"Output file : {OUTPUT_FILE}"
    )

    print(
        f"Source files: {len(source_files):,}"
    )

    print(
        f"Output size : "
        f"{output_size / 1024 / 1024:.2f} MB"
    )

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    export_project()