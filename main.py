import cv2
import numpy as np

def is_same_marketing_image(
    img1_path,
    img2_path,
    min_good_matches=15,
    min_inliers=10
):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("❌ Error: Could not read one or both images")
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

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return False

    return int(mask.sum()) >= min_inliers


# -------------------------------
# USER INPUT
# -------------------------------
img1_path = input("Enter path of FIRST image: ").strip()
img2_path = input("Enter path of SECOND image: ").strip()

if is_same_marketing_image(img1_path, img2_path):
    print("✅ Same image (possibly augmented)")
else:
    print("🆕 New image")
