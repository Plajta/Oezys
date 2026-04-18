import sys
from pathlib import Path
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtWidgets import QApplication

from src.pipeline import Pipeline
from src.ui.main_window import MainWindow

from src.processing import Model, Preprocessor, Metrics


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("name")

    font_path = Path(__file__).parent / "src" / "assets" / "fonts" / "OpenSans.ttf"
    QFontDatabase.addApplicationFont(str(font_path))
    
    
    model = Model()
    preprocessor = Preprocessor()
    metrics = Metrics()
    

    pipeline = Pipeline(model, preprocessor, metrics)  # swap stubs for real implementations here
    window = MainWindow(pipeline=pipeline)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
