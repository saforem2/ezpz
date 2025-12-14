"""
Utility to download a subset of the OpenImages dataset into an ImageFolder layout.
"""

import csv
import concurrent.futures
from pathlib import Path
from urllib.request import urlretrieve
from collections import defaultdict

OPENIMAGES_BASE = "https://storage.googleapis.com/cvdf-datasets/oid/"
# https://storage.googleapis.com/openimages/2018_04/"
# open-images-dataset-train0.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train1.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train2.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train3.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train4.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train5.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train6.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train7.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train8.tsv
# https://storage.googleapis.com/cvdf-datasets/oid/open-images-dataset-train9.tsv
ANNOTATIONS_URLS = {
    "train": [
        f"{OPENIMAGES_BASE}/open-images-dataset-train{i}.tsv"
        for i in range(10)
    ],
    "validation": f"{OPENIMAGES_BASE}/open-images-dataset-validation.tsv",
    "test": f"{OPENIMAGES_BASE}/open-images-dataset-test.tsv",
}

# ANNOTATIONS_URLS = {
#     "train": OPENIMAGES_BASE
#     + "train/metadata/train-annotations-human-imagelabels.csv",
#     "validation": OPENIMAGES_BASE
#     + "validation/metadata/validation-annotations-human-imagelabels.csv",
#     "class-descriptions": OPENIMAGES_BASE + "class-descriptions.csv",
# }

IMAGES_BASE = "https://storage.googleapis.com/openimages/"


def download_openimages_subset(
    outdir: str | Path,
    split: str = "train",
    max_classes: int = 50,
    num_workers: int = 32,
) -> None:
    """
    Download an OpenImages subset (single split) into an ImageFolder layout.

    root/
        train/
            CLASS_ID/
                image1.jpg
                image2.jpg
                ...
        val/
            CLASS_ID/
                ...

    Parameters
    ----------
    outdir : Path
        Destination directory.
    split : {"train", "validation"}
        Which OpenImages split to download.
    max_classes : int
        How many classes to download. OpenImages is huge;
        limiting classes makes a sane mid-size dataset.
    num_workers : int
        Parallel download workers.
    """

    outdir = Path(outdir)
    split = split.lower()
    assert split in {"train", "validation"}

    # ---------------------------------------------------------------------
    # STEP 1 — download class descriptions
    # ---------------------------------------------------------------------
    class_csv_path = outdir / "class-descriptions.csv"
    if not class_csv_path.exists():
        class_csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading class descriptions → {class_csv_path}")
        urlretrieve(ANNOTATIONS_URLS["class-descriptions"], class_csv_path)

    class_map = {}
    with open(class_csv_path, "r") as f:
        reader = csv.reader(f)
        for cid, cname in reader:
            class_map[cid] = cname.replace(" ", "_")

    # ---------------------------------------------------------------------
    # STEP 2 — download annotation CSV for this split
    # ---------------------------------------------------------------------
    ann_path = outdir / f"{split}-annotations.csv"
    if not ann_path.exists():
        print(f"Downloading annotations for {split} → {ann_path}")
        urlretrieve(ANNOTATIONS_URLS[split], ann_path)

    # ---------------------------------------------------------------------
    # STEP 3 — collect image → class assignments
    # ---------------------------------------------------------------------
    print(f"Parsing annotations: {ann_path}")
    image_to_classes = defaultdict(list)

    with ann_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Confidence"] == "1":
                cid = row["LabelName"]
                if cid in class_map:
                    image_to_classes[row["ImageID"]].append(cid)

    # Keep only a subset for a manageable dataset size
    print("Collecting top classes...")
    class_counts = defaultdict(int)
    for classes in image_to_classes.values():
        for cid in classes:
            class_counts[cid] += 1

    # Select top frequent classes
    top_classes = sorted(class_counts, key=class_counts.get, reverse=True)[
        :max_classes
    ]
    top_classes = set(top_classes)

    # Filter: keep only images that contain at least one top class
    filtered = {
        img: [cid for cid in classes if cid in top_classes]
        for img, classes in image_to_classes.items()
        if any(cid in top_classes for cid in classes)
    }

    print(
        f"Selected {len(filtered)} images across {len(top_classes)} classes."
    )

    # ---------------------------------------------------------------------
    # STEP 4 — Create output directories
    # ---------------------------------------------------------------------
    for cid in top_classes:
        cname = class_map[cid]
        (outdir / split / cname).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # STEP 5 — define download worker
    # ---------------------------------------------------------------------
    def download_image(image_id: str, cids: list[str]):
        # All OpenImages JPEGs follow this pattern
        url = IMAGES_BASE + f"{split}/{image_id}.jpg"

        # Save to first class directory only (multi-label ignored)
        class_id = cids[0]
        class_name = class_map[class_id]

        dst = outdir / split / class_name / f"{image_id}.jpg"
        if dst.exists():
            return

        try:
            urlretrieve(url, dst)
        except Exception as e:
            print(f"Failed {image_id}: {e}")

    # ---------------------------------------------------------------------
    # STEP 6 — parallel download
    # ---------------------------------------------------------------------
    print("Downloading images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(download_image, img_id, cids)
            for img_id, cids in filtered.items()
        ]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            if i % 500 == 0:
                print(f"Downloaded {i} images...")

    print("Finished!")


# -------------------------------------------------------------------------
# Example usage:
# -------------------------------------------------------------------------
if __name__ == "__main__":
    download_openimages_subset(
        outdir="./data/openimages_subset",
        split="train",
        max_classes=50,
        num_workers=64,
    )
