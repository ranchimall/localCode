"""
Memory (Step 6 of the multi-agent architecture) — standalone module
=====================================================================
Stores CONFIRMED fixes (a bug + its verified working fix) and lets the
rest of the pipeline look up similar past cases before the Coder writes
new code or the Critic reviews it -- "have we seen something like this
before, and what actually fixed it."

DESIGN INTENT (read before wiring this into the pipeline):
  - This store only ever holds VERIFIED fixes -- i.e. records written
    only after a Tester run (or a human verifier) confirmed the fix
    actually worked. This module does not enforce that itself (it just
    takes whatever record you hand it via store()) -- enforcing "only
    write confirmed fixes" is the calling pipeline's job, same way
    plan_fixer.py doesn't call an LLM itself, it just trusts its
    caller. Writing unconfirmed guesses in here will get them
    retrieved later and quietly reinforce wrong fixes -- don't do it.
  - Retrieval here is APPROACH 1 ONLY: structured records + heuristic
    keyword/stem overlap. No embeddings, no vector store, no LLM calls
    at all for retrieval or storage. This is a deliberate first stage
    -- see the module docstring notes below on where v2 (embeddings)
    would slot in later without needing a schema migration.
  - Records are deduplicated on write: a new record that's a close
    keyword-overlap match to an existing one (same language+kind) just
    bumps that existing row's frequency/last_seen_at instead of adding
    a near-duplicate row. This is what keeps the table from growing
    unbounded with N copies of "the same off-by-one bug."
  - Scoped by `language` (+ optional free-text `kind`) from the start,
    even though today there's only one project using it -- retrofitting
    a scope column onto existing rows later is more painful than just
    having it from day one.

DELIBERATELY INDEPENDENT of coding_agent.py / planner.py / plan_fixer.py
/ critic.py right now, same pattern as those modules: no imports from
them, nothing here assumes how it's called. Wiring it in (Coder/Critic
querying retrieve() before acting, pipeline calling store() after a
Tester-confirmed fix) is a later step.

Flow (this module only):
  store(record)    -> MemoryRecord written (or an existing near-dup bumped)
  retrieve(query)   -> ranked list of (MemoryRecord, score) heuristic matches

Where v2 (embeddings) would slot in later:
  The `keywords` column here is precomputed at write time specifically
  so a future embedding-based retriever can coexist with this one
  without a migration -- add a `embedding` BLOB column, populate it
  alongside `keywords` on write, and have retrieve() blend both scores.
  Nothing about the current schema needs to change for that.
"""

import os
import re
import sqlite3
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Tuple


DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.db")


# ----------------------------------------------------------------------
# Record
# ----------------------------------------------------------------------
@dataclass
class MemoryRecord:
    language: str                  # "python", "javascript", "bash", "ffmpeg", ... -- required, primary scope
    fix_description: str           # the confirmed fix, in words -- required, this is the payload
    kind: str = "other"            # "bug" | "logic" | "syntax" | "security" | "performance" |
                                    # "portability" | "error_mismatch" | "style" | "other"
    task_description: str = ""     # what the code was supposed to do (optional but improves retrieval)
    error_log: str = ""            # the error/traceback this fix resolved (optional)
    buggy_code: str = ""           # the code as it was before the fix (optional)
    root_cause: str = ""           # why it broke, in words (optional but valuable for retrieval)
    fixed_code: str = ""           # the code after the fix, if worth keeping (optional)
    id: Optional[int] = None
    frequency: int = 1              # how many times a near-duplicate of this has been confirmed
    created_at: str = ""
    last_seen_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Keyword extraction -- same crude stem-and-filter approach used
# elsewhere in this codebase (plan_fixer.py's scope_drift check), kept
# consistent on purpose so retrieval behavior is predictable.
# ----------------------------------------------------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "it", "its", "this", "that", "these", "those",
    "with", "from", "into", "and", "for", "of", "to", "in", "on", "at",
    "was", "were", "are", "be", "been", "being", "as", "by", "or", "but",
    "if", "then", "than", "so", "not", "no", "do", "does", "did", "has",
    "have", "had", "will", "would", "should", "could", "can", "may", "might",
}


def _keywords(*texts: str) -> set:
    """Extract a set of lowercase 5-char stems from the given text(s),
    dropping stopwords and very short words. Same stem-length (5) as
    plan_fixer.py's verb-stem matching, for consistency across the
    codebase."""
    words = set()
    for text in texts:
        if not text:
            continue
        for w in re.findall(r"[a-zA-Z]+", text.lower()):
            if len(w) < 4 or w in _STOPWORDS:
                continue
            words.add(w[:5])
    return words


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------
class MemoryStore:
    """SQLite-backed store of confirmed fixes, with heuristic keyword
    retrieval and similarity-based dedup on write."""

    # near-duplicate threshold on write: a new record whose keyword
    # jaccard overlap with an existing same-language+kind record is at
    # or above this just bumps that record's frequency instead of
    # inserting a new row.
    DEDUP_THRESHOLD = 0.6

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                language        TEXT NOT NULL,
                kind            TEXT NOT NULL DEFAULT 'other',
                task_description TEXT NOT NULL DEFAULT '',
                error_log       TEXT NOT NULL DEFAULT '',
                buggy_code      TEXT NOT NULL DEFAULT '',
                root_cause      TEXT NOT NULL DEFAULT '',
                fix_description TEXT NOT NULL,
                fixed_code      TEXT NOT NULL DEFAULT '',
                keywords        TEXT NOT NULL DEFAULT '',
                frequency       INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL,
                last_seen_at    TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_lang ON memories(language)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_lang_kind ON memories(language, kind)")
        self._conn.commit()

    def close(self):
        self._conn.close()

    # -- write path --------------------------------------------------
    def store(self, record: MemoryRecord) -> Tuple[MemoryRecord, bool]:
        """Write a confirmed fix. Returns (stored_record, is_new).
        If a close keyword-match already exists for this language+kind,
        bumps its frequency/last_seen_at instead of inserting a
        near-duplicate row, and returns that existing record with
        is_new=False."""
        if not record.language:
            raise ValueError("MemoryRecord.language is required")
        if not record.fix_description:
            raise ValueError("MemoryRecord.fix_description is required "
                              "(this store is for CONFIRMED fixes only)")

        kw = _keywords(record.task_description, record.error_log,
                        record.root_cause, record.buggy_code)
        kw_str = " ".join(sorted(kw))

        existing = self._find_near_duplicate(record.language, record.kind, kw)
        if existing is not None:
            now = _now()
            self._conn.execute(
                "UPDATE memories SET frequency = frequency + 1, last_seen_at = ? WHERE id = ?",
                (now, existing["id"]),
            )
            self._conn.commit()
            existing = dict(existing)
            existing["frequency"] += 1
            existing["last_seen_at"] = now
            return self._row_to_record(existing), False

        now = _now()
        cur = self._conn.execute(
            """INSERT INTO memories
               (language, kind, task_description, error_log, buggy_code,
                root_cause, fix_description, fixed_code, keywords,
                frequency, created_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (record.language, record.kind, record.task_description,
             record.error_log, record.buggy_code, record.root_cause,
             record.fix_description, record.fixed_code, kw_str, now, now),
        )
        self._conn.commit()
        record.id = cur.lastrowid
        record.created_at = now
        record.last_seen_at = now
        record.frequency = 1
        return record, True

    def _find_near_duplicate(self, language: str, kind: str, kw: set):
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE language = ? AND kind = ?",
            (language, kind),
        ).fetchall()
        best, best_score = None, 0.0
        for row in rows:
            existing_kw = set(row["keywords"].split()) if row["keywords"] else set()
            score = _jaccard(kw, existing_kw)
            if score > best_score:
                best, best_score = row, score
        if best is not None and best_score >= self.DEDUP_THRESHOLD:
            return best
        return None

    # -- read path -----------------------------------------------------
    def retrieve(self, language: str, task_description: str = "",
                 error_log: str = "", code: str = "", kind: Optional[str] = None,
                 top_k: int = 3, min_score: float = 0.05) -> List[Tuple[MemoryRecord, float]]:
        """Return up to top_k (MemoryRecord, score) pairs for the given
        query context, filtered to `language` (required -- this store
        is scoped by language) and optionally `kind`, ranked by keyword
        jaccard overlap against each record's precomputed keywords.
        Records below min_score are dropped entirely rather than
        padding out the results with noise."""
        query_kw = _keywords(task_description, error_log, code)
        if not query_kw:
            return []

        if kind:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE language = ? AND kind = ?",
                (language, kind),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE language = ?", (language,)
            ).fetchall()

        scored = []
        for row in rows:
            existing_kw = set(row["keywords"].split()) if row["keywords"] else set()
            score = _jaccard(query_kw, existing_kw)
            if score >= min_score:
                scored.append((self._row_to_record(row), score))

        scored.sort(key=lambda pair: (pair[1], pair[0].frequency), reverse=True)
        return scored[:top_k]

    # -- inspection ------------------------------------------------------
    def list_all(self, language: Optional[str] = None) -> List[MemoryRecord]:
        if language:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE language = ? ORDER BY last_seen_at DESC", (language,)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY last_seen_at DESC"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        by_lang = self._conn.execute(
            "SELECT language, COUNT(*) AS c, SUM(frequency) AS seen FROM memories GROUP BY language"
        ).fetchall()
        by_kind = self._conn.execute(
            "SELECT kind, COUNT(*) AS c FROM memories GROUP BY kind"
        ).fetchall()
        return {
            "total_records": total,
            "by_language": {r["language"]: {"records": r["c"], "times_seen": r["seen"]} for r in by_lang},
            "by_kind": {r["kind"]: r["c"] for r in by_kind},
        }

    def delete(self, record_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def _row_to_record(self, row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"], language=row["language"], kind=row["kind"],
            task_description=row["task_description"], error_log=row["error_log"],
            buggy_code=row["buggy_code"], root_cause=row["root_cause"],
            fix_description=row["fix_description"], fixed_code=row["fixed_code"],
            frequency=row["frequency"], created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
        )


# ----------------------------------------------------------------------
# CLI -- subcommands, since this is a small local database tool, not a
# single-pipeline-stage script like planner.py/critic.py.
# ----------------------------------------------------------------------
def _print_record(record: MemoryRecord, score: Optional[float] = None):
    header = f"[id={record.id}] {record.language}/{record.kind}  (seen {record.frequency}x)"
    if score is not None:
        header += f"  score={score:.2f}"
    print(header)
    if record.task_description:
        print(f"    task: {record.task_description[:100]}")
    if record.root_cause:
        print(f"    root cause: {record.root_cause}")
    print(f"    fix: {record.fix_description}")
    print()


def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(description="Standalone Memory: store/retrieve confirmed fixes")
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to the SQLite memory database")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("store", help="Store a confirmed fix")
    s.add_argument("--language", required=True)
    s.add_argument("--kind", default="other")
    s.add_argument("--task", default="")
    s.add_argument("--task-file")
    s.add_argument("--error-log", default="")
    s.add_argument("--error-log-file")
    s.add_argument("--code", default="", help="The buggy code (before the fix)")
    s.add_argument("--code-file")
    s.add_argument("--root-cause", default="")
    s.add_argument("--fix", required=True, help="Description of the confirmed fix")
    s.add_argument("--fixed-code", default="")
    s.add_argument("--fixed-code-file")

    r = sub.add_parser("retrieve", help="Look up similar past fixes")
    r.add_argument("--language", required=True)
    r.add_argument("--kind", default=None)
    r.add_argument("--task", default="")
    r.add_argument("--task-file")
    r.add_argument("--error-log", default="")
    r.add_argument("--error-log-file")
    r.add_argument("--code", default="")
    r.add_argument("--code-file")
    r.add_argument("--top-k", type=int, default=3)

    l = sub.add_parser("list", help="List stored records")
    l.add_argument("--language", default=None)

    sub.add_parser("stats", help="Show summary stats")

    d = sub.add_parser("delete", help="Delete a record by id")
    d.add_argument("--id", type=int, required=True)

    return p


def _read_maybe_file(value: str, file_path: Optional[str]) -> str:
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return value


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    store = MemoryStore(args.db)

    if args.command == "store":
        record = MemoryRecord(
            language=args.language,
            kind=args.kind,
            task_description=_read_maybe_file(args.task, args.task_file),
            error_log=_read_maybe_file(args.error_log, args.error_log_file),
            buggy_code=_read_maybe_file(args.code, args.code_file),
            root_cause=args.root_cause,
            fix_description=args.fix,
            fixed_code=_read_maybe_file(args.fixed_code, args.fixed_code_file),
        )
        stored, is_new = store.store(record)
        if is_new:
            print(f"Stored new record [id={stored.id}].")
        else:
            print(f"Matched an existing record [id={stored.id}] "
                  f"(now seen {stored.frequency}x) -- bumped instead of duplicating.")

    elif args.command == "retrieve":
        results = store.retrieve(
            language=args.language,
            kind=args.kind,
            task_description=_read_maybe_file(args.task, args.task_file),
            error_log=_read_maybe_file(args.error_log, args.error_log_file),
            code=_read_maybe_file(args.code, args.code_file),
            top_k=args.top_k,
        )
        if not results:
            print("No matching records found.")
        else:
            print(f"=== {len(results)} match(es) ===\n")
            for record, score in results:
                _print_record(record, score)

    elif args.command == "list":
        records = store.list_all(language=args.language)
        if not records:
            print("(no records)")
        else:
            for record in records:
                _print_record(record)

    elif args.command == "stats":
        print(json.dumps(store.stats(), indent=2))

    elif args.command == "delete":
        ok = store.delete(args.id)
        print(f"Deleted [id={args.id}]." if ok else f"No record with id={args.id}.")

    store.close()
