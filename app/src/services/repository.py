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
            
        """)
        self._conn.commit()

    def save(self, data: MetricData) -> int:
        probs = data.Probabilities or [0.0] * 5
        cursor = self._conn.execute(
            """
            
            """,
            (data.Datetime, data.FileName, *probs),
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_all(self) -> list[MetricData]:
        rows = self._conn.execute(
        ).fetchall()
        results = []
        for row in rows:
            m = MetricData(
                datetime=row[0],
                filename=row[1],
                probabilities=list(row[2:]),
            )
            results.append(m)
        return results

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
