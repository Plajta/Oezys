import sys
from pathlib import Path
from PyQt6.QtGui import QFontDatabase, QIcon
from PyQt6.QtWidgets import QApplication

from src.pipeline import Pipeline
from src.ui.main_window import MainWindow
from src.processing import Model, Preprocessor, Metrics
from src.services.repository import Repository, DB_PATH


def _clear_database():
    if not DB_PATH.exists():
        print("Database does not exist, nothing to clear.")
        return
    with Repository() as repo:
        repo._conn.execute("DELETE FROM metrics")
        repo._conn.commit()
    print(f"Database cleared: {DB_PATH}")


def main():
    if "-d" in sys.argv:
        _clear_database()
        return

    app = QApplication(sys.argv)
    app.setApplicationName("name")
    

    font_path = Path(__file__).parent / "src" / "assets" / "fonts" / "OpenSans.ttf"
    QFontDatabase.addApplicationFont(str(font_path))

    logo_path = Path(__file__).parent / "src" / "assets" / "images" / "logo.png"
    app.setWindowIcon(QIcon(str(logo_path)))
    
    
    model = Model()
    preprocessor = Preprocessor()
    metrics = Metrics()
    

    pipeline = Pipeline(metrics=metrics, model=model, preprocessor=preprocessor)
    window = MainWindow(pipeline=pipeline)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
