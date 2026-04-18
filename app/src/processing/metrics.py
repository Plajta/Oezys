from datetime import datetime
from .model import ModelData
from .preprocessing import PreprocessorData
import random

class MetricData:
    def __init__(self, datetime=None, filename=None, probabilities=None):
        '''
        Datetime: 2024-06-17 20:00:00
        FileName: somedata.bmp
        Probabilities: [int, int, int, int, int] 
        
        1 - Healthy
        2 - Diabetes
        3 - Dry Eye Disease
        4 - Multiple Sclerosis
        5 - Primary Open-Angle Glaucoma
        '''
        self.Datetime = datetime
        self.FileName = filename
        self.FilePath = None
        self.Probabilities = probabilities
        self.labels = ["Healthy", "Diabetes", "Dry Eye Disease", "Multiple Sclerosis", "Primary Open-Angle Glaucoma"]

class Metrics:
    def __init__(self):
        pass
    
    def run(self, preprocessed: PreprocessorData, prediction: ModelData, path: str = "") -> MetricData:
        from pathlib import Path
        metrics = MetricData()
        now = datetime.now()
        metrics.Datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        stem = Path(path).stem
        suffix = Path(path).suffix
        ms = int(now.timestamp() * 1000)
        metrics.FileName = f"{stem}_{ms}{suffix}.bmp"
        metrics.FilePath = path
        if prediction.Probabilities:
            raw = prediction.Probabilities
        else:
            vals = [random.random() for _ in range(5)]
            total = sum(vals)
            raw = [v / total for v in vals]
            print(f"[Metrics] No model output — using fallback probabilities: {raw}")
        metrics.Probabilities = [round(p * 100, 2) for p in raw]
        return metrics