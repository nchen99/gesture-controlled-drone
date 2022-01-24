import threading

class Shared:
    def __init__(self, default):
        self.value = default
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.value
    
    def set(self, newVal):
        with self.lock:
            self.value = newVal

