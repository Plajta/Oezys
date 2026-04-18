import sqlite3
from pathlib import Path

from ..processing.metrics import MetricData

DB_PATH = Path(__file__).parent.parent.parent / "data" / "radbrecim.db"


class Repository:
    def __init__(self, db_path: str | Path = DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime    TEXT    NOT NULL,
                filename    TEXT    NOT NULL,
                filepath    TEXT    NOT NULL DEFAULT '',
                p_healthy   REAL    NOT NULL,
                p_diabetes  REAL    NOT NULL,
                p_dry_eye   REAL    NOT NULL,
                p_ms        REAL    NOT NULL,
                p_glaucoma  REAL    NOT NULL
            )
        """)
        # migrate older DBs that lack the filepath column
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(metrics)")}
        if "filepath" not in cols:
            self._conn.execute("ALTER TABLE metrics ADD COLUMN filepath TEXT NOT NULL DEFAULT ''")
        self._conn.commit()

    def save(self, data: MetricData) -> int:
        probs = data.Probabilities or [0.0] * 5
        cursor = self._conn.execute(
            """
            INSERT INTO metrics (datetime, filename, filepath, p_healthy, p_diabetes, p_dry_eye, p_ms, p_glaucoma)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (data.Datetime, data.FileName, data.FilePath or "", *probs),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_all(self) -> list[MetricData]:
        rows = self._conn.execute(
            "SELECT datetime, filename, filepath, p_healthy, p_diabetes, p_dry_eye, p_ms, p_glaucoma FROM metrics ORDER BY id DESC"
        ).fetchall()
        results = []
        for row in rows:
            m = MetricData(datetime=row[0], filename=row[1], probabilities=list(row[3:]))
            m.FilePath = row[2]
            results.append(m)
        return results

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
