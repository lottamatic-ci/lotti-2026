import cv2
import numpy as np
from pathlib import Path
import json


class ImageSet:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.images = sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
        self.data = []
        
    def load(self):
        for img_path in self.images:
            img = cv2.imread(str(img_path))
            self.data.append((str(img_path), img))
    
    def __len__(self):
        return len(self.data)


class FeatureDetector:
    def __init__(self, method="sift"):
        if method == "sift":
            self.detector = cv2.SIFT_create()
        elif method == "orb":
            self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
    
    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.detector.detectAndCompute(gray, None)
        return kp, desc


class FeatureMatcher:
    def __init__(self, method="sift"):
        self.method = method
        if method == "sift":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def match(self, desc1, desc2, ratio=0.7):
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches


class PoseEstimator:
    def __init__(self, K):
        self.K = K
    
    def find_essential_matrix(self, pts1, pts2):
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        return E, pts1, pts2
    
    def recover_pose(self, E, pts1, pts2):
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=None)
        pts1 = pts1[mask.ravel() > 0]
        pts2 = pts2[mask.ravel() > 0]
        return R, t, pts1, pts2
    
    def triangulate(self, P1, P2, pts1, pts2):
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T


class Reconstruction:
    def __init__(self, K):
        self.K = K
        self.point_cloud = []
        self.camera_poses = []
        self.detector = FeatureDetector("sift")
        self.matcher = FeatureMatcher("sift")
        self.pose_estimator = PoseEstimator(K)
    
    def add_image(self, image):
        kp, desc = self.detector.detect(image)
        return kp, desc
    
    def process_image_pair(self, img1, img2):
        kp1, desc1 = self.detector.detect(img1)
        kp2, desc2 = self.detector.detect(img2)
        
        if desc1 is None or desc2 is None:
            return None
        
        matches = self.matcher.match(desc1, desc2)
        
        if len(matches) < 8:
            return None
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, pts1, pts2 = self.pose_estimator.find_essential_matrix(pts1, pts2)
        R, t, pts1, pts2 = self.pose_estimator.recover_pose(E, pts1, pts2)
        
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t])
        
        points_3d = self.pose_estimator.triangulate(P1, P2, pts1, pts2)
        
        return points_3d, R, t


def main():
    image_dir = "images"
    imageset = ImageSet(image_dir)
    imageset.load()
    
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=float)
    
    reconstruction = Reconstruction(K)
    
    if len(imageset) < 2:
        print("Need at least 2 images")
        return
    
    all_points = []
    
    for i in range(len(imageset) - 1):
        _, img1 = imageset.data[i]
        _, img2 = imageset.data[i + 1]
        
        result = reconstruction.process_image_pair(img1, img2)
        
        if result is None:
            print(f"Pair {i} - {i+1}: insufficient matches")
            continue
        
        points_3d, R, t = result
        all_points.append(points_3d)
        print(f"Pair {i} - {i+1}: {len(points_3d)} points")
    
    if all_points:
        final_cloud = np.vstack(all_points)
        output = {
            "points": final_cloud.tolist(),
            "num_points": len(final_cloud)
        }
        
        with open("point_cloud.json", "w") as f:
            json.dump(output, f)
        
        print(f"Total points: {len(final_cloud)}")
        print("Saved to point_cloud.json")


if __name__ == "__main__":
    main()
