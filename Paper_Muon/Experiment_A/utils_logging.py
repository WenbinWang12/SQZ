import csv
import os
import json
import time
from collections import defaultdict

class CSVLogger:
    def __init__(self, out_dir, optimizer_name):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{optimizer_name}.csv")
        self.f = open(self.path, 'w', newline='')
        self.writer = None
        self.fieldnames = None

    def log_step(self, step_dict):
        if self.writer is None:
            self.fieldnames = list(step_dict.keys())
            self.writer = csv.DictWriter(self.f, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow(step_dict)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

class SelectionRecorder:
    def __init__(self):
        self.count = defaultdict(int)
        self.total_steps = 0

    def record(self, selected_names):
        self.total_steps += 1
        for n in selected_names:
            self.count[n] += 1

    def selection_heatmap(self, save_path, param_order=None):
        # produce a simple CSV of frequencies sorted by param_order if provided
        items = list(self.count.items())
        if param_order is not None:
            items = sorted(items, key=lambda x: param_order.index(x[0]) if x[0] in param_order else len(param_order))
        rows = [(name, cnt, cnt/self.total_steps if self.total_steps>0 else 0.0) for name,cnt in items]
        with open(save_path, 'w') as f:
            f.write('param_name,count,frequency\n')
            for name,cnt,freq in rows:
                f.write(f"{name},{cnt},{freq}\n")
        return rows