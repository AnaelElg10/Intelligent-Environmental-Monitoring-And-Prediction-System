# Placeholder module
class PrioritizedExperienceReplay:
    def __init__(self, capacity): self.memory = []
    def push(self, experience): self.memory.append(experience)
    def sample(self, batch_size): return self.memory[:batch_size]
    def __len__(self): return len(self.memory)
