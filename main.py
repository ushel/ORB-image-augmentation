import os
import hashlib
from PIL import Image
import imagehash
import cv2
import numpy as np

# -----------------------------
# HASH FUNCTIONS
# -----------------------------
def compute_sha256(image_path):
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def compute_phash(image_path):
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(img))

def phash_prefix(phash, bits=16):
    return bin(int(phash, 16))[2:].zfill(64)[:bits]

def image_metadata(image_path):
    img = Image.open(image_path)
    w, h = img.size
    return w / h


# -----------------------------
# ORB VERIFICATION
# -----------------------------
def orb_verify(img1_path, img2_path,
               min_good_matches=15,
               min_inliers=10):

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return False

    orb = cv2.ORB_create(nfeatures=3000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < min_good_matches:
        return False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return False

    return int(mask.sum()) >= min_inliers


# -----------------------------
# BUILD FOLDER INDEX
# -----------------------------
def build_image_index(folder_path):
    records = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder_path, file)

        try:
            records.append({
                "image_id": file,
                "image_path": path,
                "sha256": compute_sha256(path),
                "phash_prefix": phash_prefix(compute_phash(path)),
                "aspect_ratio": image_metadata(path)
            })
        except Exception:
            continue

    return records


# -----------------------------
# MAIN COMPARISON FUNCTION
# -----------------------------
def compare_image_to_folder(input_image_path, folder_records):
    sha256 = compute_sha256(input_image_path)

    # 1️⃣ Exact duplicate
    for r in folder_records:
        if r["sha256"] == sha256:
            return True, r["image_id"], "exact_duplicate"

    # 2️⃣ Hash + metadata filtering
    phash = compute_phash(input_image_path)
    prefix = phash_prefix(phash)
    aspect_ratio = image_metadata(input_image_path)

    candidates = [
        r for r in folder_records
        if r["phash_prefix"] == prefix
        and abs(r["aspect_ratio"] - aspect_ratio) <= 0.1
    ]

    # 3️⃣ ORB verification
    for r in candidates:
        if orb_verify(input_image_path, r["image_path"]):
            return True, r["image_id"], "augmented_duplicate"

    return False, None, "new_image"


# -----------------------------
# RUN SCRIPT
# -----------------------------
if __name__ == "__main__":
    folder_path = input("Enter path to images folder: ").strip()
    input_image = input("Enter path to input image: ").strip()

    print("🔍 Indexing images...")
    folder_records = build_image_index(folder_path)

    print("🧠 Comparing image...")
    is_dup, image_id, reason = compare_image_to_folder(
        input_image, folder_records
    )

    if is_dup:
        print(f"🚫 Duplicate found ({reason}) → {image_id}")
    else:
        print("✅ New image")