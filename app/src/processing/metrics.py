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
        self.Probabilities = probabilities
        self.labels = ["Healthy", "Diabetes", "Dry Eye Disease", "Multiple Sclerosis", "Primary Open-Angle Glaucoma"]

class Metrics:
    def __init__(self):
        pass
    
    def run(self, preprocessed: PreprocessorData, prediction: ModelData, name: str = random.choice(["a","b","c","d"])) -> MetricData:
        from pathlib import Path
        metrics = MetricData()
        now = datetime.now()
        metrics.Datetime = now.strftime("%Y-%m-%d %H:%M:%S")
        stem = Path(name).stem
        suffix = Path(name).suffix
        ms = int(now.timestamp() * 1000)
        metrics.FileName = f"{stem}_{ms}{suffix}.bmp"
        metrics.Probabilities = prediction.Probabilities if prediction.Probabilities else [30, 20, 20, 20, 10]
        return metrics