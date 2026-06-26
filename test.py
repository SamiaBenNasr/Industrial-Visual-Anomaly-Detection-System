from pathlib import Path

root = "data/mvtec/bottle"

print("Train good:", len(list(Path(root + "/train/good").glob("*"))))
print("Test good:", len(list(Path(root + "/test/good").glob("*"))))