import numpy as np
import cv2

class SwingDetector:
    """Detects swing start and end frames from motion data"""
    
    def __init__(self, motion_threshold=0.05, min_swing_frames=10):
        self.motion_threshold = motion_threshold
        self.min_swing_frames = min_swing_frames
    
    def detect_swing_boundaries(self, motion_history, frames):
        """
        Detect swing start and end frames
        
        Args:
            motion_history: List of motion metrics
            frames: List of video frames
        
        Returns:
            (start_frame, end_frame) indices
        """
        if not motion_history or len(motion_history) < self.min_swing_frames:
            print("[WARNING] Insufficient motion data")
            return 0, len(frames) - 1
        
        # Extract overall motion magnitude
        motion_magnitudes = [m.get('overall_motion', 0) for m in motion_history]
        
        # Smooth motion signal
        smoothed_motion = self._smooth_signal(motion_magnitudes, window=5)
        
        # Find peaks (swing start and end)
        swing_indices = np.where(np.array(smoothed_motion) > self.motion_threshold)
        
        if len(swing_indices) == 0:
            print("[WARNING] No swing detected, using entire video")
            return 0, len(frames) - 1
        
        # Group consecutive indices
        swing_groups = self._group_consecutive(swing_indices)
        
        if not swing_groups:
            return 0, len(frames) - 1
        
        # Find largest contiguous swing motion
        largest_group = max(swing_groups, key=lambda x: len(x))
        
        start_frame = largest_group
        end_frame = largest_group[-1]
        
        print(f"[SWING_DETECTOR] Detected swing: {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
        
        return start_frame, end_frame
    
    def _smooth_signal(self, signal, window=5):
        """Apply moving average smoothing"""
        if len(signal) < window:
            return signal
        
        smoothed = []
        for i in range(len(signal)):
            start = max(0, i - window // 2)
            end = min(len(signal), i + window // 2 + 1)
            smoothed.append(np.mean(signal[start:end]))
        
        return smoothed
    
    def _group_consecutive(self, indices):
        """Group consecutive indices"""
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices]
        
        for i in range(1, len(indices)):
            if indices[i] - indices[i-1] <= 1:
                current_group.append(indices[i])
            else:
                if len(current_group) >= self.min_swing_frames:
                    groups.append(current_group)
                current_group = [indices[i]]
        
        if len(current_group) >= self.min_swing_frames:
            groups.append(current_group)
        
        return groups
