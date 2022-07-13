import json
import math
from pathlib import Path
import random
import sys

if __name__ == "__main__":
    json_path = Path(sys.argv[1])
    with json_path.open() as f:
        data = json.load(f)
    random.shuffle(data)

    train_size = math.ceil(len(data) * 0.75)
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_data = sorted(train_data, key=lambda x: x["id"])
    val_data = sorted(val_data, key=lambda x: x["id"])

    train_data_name = sys.argv[2] + ".json"
    train_data_path = json_path.parent / train_data_name

    valid_data_name = sys.argv[3] + ".json"
    valid_data_path = json_path.parent / valid_data_name

    with train_data_path.open("w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with valid_data_path.open("w") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
