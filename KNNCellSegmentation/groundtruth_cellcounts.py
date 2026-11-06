from pathlib import Path
import json
from collections import Counter
import pandas as pd

ANNOTATION_FILE = Path(
    r"C:\Users\abreun01\OneDrive - NYU Langone Health\MSCI\FALL 2026\Machine Learning\Project\LIVECell\Annotations\livecell_coco_train.json"
)

OUTPUT_CSV = ANNOTATION_FILE.parent / "livecell_ground_truth_cell_counts.csv"

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# One annotation corresponds to one annotated cell.
counts_by_image_id = Counter(
    annotation["image_id"]
    for annotation in data["annotations"]
)

rows = []

for image in data["images"]:
    image_id = image["id"]

    rows.append({
        "image_id": image_id,
        "image_file": image["file_name"],
        "cell_line": image["file_name"].split("_")[0],
        "image_width": image.get("width"),
        "image_height": image.get("height"),
        "ground_truth_cell_count": counts_by_image_id.get(image_id, 0)
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print(df.head())
print(f"\nSaved ground-truth cell counts to:\n{OUTPUT_CSV}")
print(f"\nTotal annotated cells: {df['ground_truth_cell_count'].sum():,}")