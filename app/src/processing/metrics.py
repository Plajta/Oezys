import time

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

class Metrics:
    def __init__(self):
        pass
    
    def run(self, name: str, preprocessed: PreprocessorData, prediction: ModelData) -> MetricData:
        metrics = MetricData()
        metrics.Datetime = time.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics.FileName = name
        metrics.Probabilities = prediction.Probabilities
        return metrics