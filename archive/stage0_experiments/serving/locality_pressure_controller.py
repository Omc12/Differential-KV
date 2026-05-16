"""
Locality Pressure Controller.
"""
class LocalityPressureController:
    def __init__(self):
        self.pressure = 0.0
        
    def update_pressure(self, metrics):
        self.pressure = 0.4
        return self.pressure
