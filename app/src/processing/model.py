
class ModelData:
    def __init__(self, label=None, probabilities=None):
        '''
        Label: Healthy, Diabetes, Dry Eye Disease, Multiple Sclerosis, Primary Open-Angle Glaucoma
        Probabilities: [int, int, int, int, int] 
        
        1 - Healthy
        2 - Diabetes
        3 - Dry Eye Disease
        4 - Multiple Sclerosis
        5 - Primary Open-Angle Glaucoma
        '''
        
        self.Label = ["Healthy", "Diabetes", "Dry Eye Disease", "Multiple Sclerosis", "Primary Open-Angle Glaucoma"]
        self.Probabilities = probabilities

class Model:
    def __init__(self):
        pass
    
    def run(self, preprocessed: PreprocessorData) -> ModelData:
        modelData = ModelData()
        
        return modelData