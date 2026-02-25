import numpy as np
from typing import Tuple

class TestDataLogic:
    def __init__(self):
        self.idx_per_set = (8*7) // 2
        self.next_set = False
        self.running_max = 8
        self.running_id = 1
        self.data = np.arange(16)
    
    def get_item(self, idx: int) -> Tuple[int, int]:
        current_set = idx // self.idx_per_set
        if idx == current_set * self.idx_per_set:
            self.next_set = True
            print(f"Next set triggered at idx {idx} (current_set: {current_set})")
        if self.next_set:
            self.running_id = 1
            self.running_max = 8
            self.next_set = False
        if self.running_id == self.running_max:
            print(f"Running max reached at idx {idx} (current_set: {current_set}, running_id: {self.running_id})")
            self.running_id = 1
            self.running_max -= 1
        diff = 8 - self.running_max
        template_idx = (diff + self.running_id + current_set * 8)
        self.running_id += 1

        impression_idx = current_set * 8 + 8 - self.running_max

        return impression_idx, template_idx
    
if __name__ == "__main__":
    test_logic = TestDataLogic()
    matrix = np.zeros((16, 16), dtype=int)
    for idx in range(56):
        impression, template = test_logic.get_item(idx)
        matrix[impression, template] = idx + 1
    print(matrix)