import json
from pprint import pprint

print("Loading evaluation results...")

with open("results/results.json") as f:
    data = json.load(f)

print("\n===== ORIGINAL YOLO METRICS =====")
pprint(data["original_yolo"])

print("\n===== IMPROVED YOLO METRICS =====")
pprint(data["improved_yolo"])

print("\n===== Confusion Matrix Dimensions =====")
print("Original:", len(data["confusion_matrices"]["original"]), "x", len(data["confusion_matrices"]["original"][0]))
print("Improved:", len(data["confusion_matrices"]["improved"]), "x", len(data["confusion_matrices"]["improved"][0]))

print("\nFigures and detailed metrics are stored in: results/")
