"""
Professional Video Stabilization Script
Uses optical flow-based stabilization similar to DaVinci Resolve
"""

import cv2
import numpy as np

class VideoStabilizer:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        
        # Feature detection params
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )
        
        # Optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Smoothing params
        self.smoothing_window = 45
        
        self.transforms = []
        self.smoothed_transforms = []
        
    def estimate_motion(self, prev_gray, curr_gray, prev_points):
        """Estimate motion between frames using optical flow"""
        if prev_points is None or len(prev_points) < 20:
            prev_points = cv2.goodFeaturesToTrack(
                prev_gray, mask=None, **self.feature_params
            )
        
        if prev_points is None or len(prev_points) < 20:
            return np.eye(2, 3, dtype=np.float32), None
        
        # Ensure points are in correct format
        prev_points = np.float32(prev_points).reshape(-1, 1, 2)
        
        # Track features with optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params
        )
        
        # Filter valid points
        status = status.flatten()
        valid_prev = prev_points[status == 1].reshape(-1, 2)
        valid_curr = curr_points[status == 1].reshape(-1, 2)
        
        if len(valid_prev) < 20:
            return np.eye(2, 3, dtype=np.float32), None
        
        # Estimate affine transform
        transform, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
        
        if transform is None:
            return np.eye(2, 3, dtype=np.float32), None
        
        return transform, valid_curr
    
    def smooth_transforms(self):
        """Apply Gaussian smoothing to transforms"""
        dx = [t[0, 2] for t in self.transforms]
        dy = [t[1, 2] for t in self.transforms]
        da = [np.arctan2(t[1, 0], t[0, 0]) for t in self.transforms]
        
        kernel_size = min(self.smoothing_window, len(self.transforms))
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = kernel_size // 3
        
        dx_arr = np.array(dx, dtype=np.float32)
        dy_arr = np.array(dy, dtype=np.float32)
        da_arr = np.array(da, dtype=np.float32)
        
        dx_smooth = cv2.GaussianBlur(dx_arr.reshape(1, -1), (1, kernel_size), sigma)[0]
        dy_smooth = cv2.GaussianBlur(dy_arr.reshape(1, -1), (1, kernel_size), sigma)[0]
        da_smooth = cv2.GaussianBlur(da_arr.reshape(1, -1), (1, kernel_size), sigma)[0]
        
        self.smoothed_transforms = []
        for i in range(len(self.transforms)):
            sx = dx[i] - dx_smooth[i]
            sy = dy[i] - dy_smooth[i]
            da_s = da_smooth[i]
            
            transform = np.array([
                [np.cos(da_s), -np.sin(da_s), sx],
                [np.sin(da_s),  np.cos(da_s), sy]
            ], dtype=np.float32)
            
            self.smoothed_transforms.append(transform)
    
    def stabilize(self):
        """Main stabilization pipeline"""
        print("=" * 60)
        print("VIDEO STABILIZATION - OPTICAL FLOW BASED")
        print("=" * 60)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Input: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            return False
        
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_points = None
        
        print("Phase 1: Analyzing motion...")
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            transform, tracked_points = self.estimate_motion(prev_gray, curr_gray, prev_points)
            self.transforms.append(transform)
            
            if tracked_points is not None and len(tracked_points) >= 20:
                prev_points = tracked_points
            else:
                prev_points = None
            
            prev_gray = curr_gray
            
            if frame_num % 300 == 0:
                print(f"  Analyzed {frame_num}/{total_frames} frames...")
        
        cap.release()
        
        while len(self.transforms) < total_frames:
            self.transforms.append(np.eye(2, 3, dtype=np.float32))
        
        print(f"  Motion analysis complete: {len(self.transforms)} transforms")
        
        print("Phase 2: Smoothing transforms...")
        self.smooth_transforms()
        print("  Smoothing complete")
        
        print("Phase 3: Applying stabilization...")
        cap = cv2.VideoCapture(self.video_path)
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            idx = min(frame_num - 1, len(self.smoothed_transforms) - 1)
            transform = self.smoothed_transforms[idx] if idx >= 0 else np.eye(2, 3, dtype=np.float32)
            
            stabilized = cv2.warpAffine(frame, transform, (width, height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            
            out.write(stabilized)
            
            if frame_num % 300 == 0:
                print(f"  Stabilized {frame_num}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        print("=" * 60)
        print("STABILIZATION COMPLETE")
        print(f"Output: {self.output_path}")
        print("=" * 60)
        
        return True


def main():
    input_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/PXL_20260410_024928909.mp4"
    output_path = "/Users/kaarthikabhinav/Documents/SprindPOC_eyetracking/data/stabilized_output.mp4"
    
    stabilizer = VideoStabilizer(input_path, output_path)
    stabilizer.stabilize()


if __name__ == "__main__":
    main()
