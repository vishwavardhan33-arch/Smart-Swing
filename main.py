import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
from swing_detector import SwingDetector
from biomechanics import BiomechanicsAnalyzer
from performance_scoring import PerformanceScorer
from visualization import SwingVisualizer
from data_manager import DataManager


class SmartSwing:
    """Main SmartSwing application class"""
    
    def __init__(self):
        self.swing_detector = SwingDetector()
        self.biomechanics = BiomechanicsAnalyzer()
        self.scorer = PerformanceScorer()
        self.visualizer = SwingVisualizer()
        self.data_manager = DataManager()
    
    def analyze_swing(self, video_path, output_path="output_swing.mp4", session_name=None):
        """Complete swing analysis pipeline"""
        video_path = str(video_path).strip()
        
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return None
        
        print(f"[INFO] Loading video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("[ERROR] Could not open video file")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frames = []
        frame_count = 0
        all_keypoints = []
        motion_history = []
        
        print("[INFO] Extracting poses from video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            keypoints, pose_landmarks = self.biomechanics.extract_keypoints(frame)
            all_keypoints.append(keypoints)
            
            if len(all_keypoints) > 1:
                motion = self.biomechanics.calculate_motion_metrics(all_keypoints[-2], all_keypoints[-1])
                motion_history.append(motion)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        print(f"[INFO] Total frames processed: {frame_count}")
        
        print("[INFO] Detecting swing boundaries...")
        swing_start, swing_end = self.swing_detector.detect_swing_boundaries(motion_history, frames)
        
        print(f"[INFO] Swing detected: Frame {swing_start} to {swing_end}")
        
        print("[INFO] Analyzing swing mechanics...")
        swing_keypoints = all_keypoints[swing_start:swing_end]
        swing_frames = frames[swing_start:swing_end]
        
        backswing_plane, downswing_alignment, rotation_stability = self.biomechanics.analyze_swing_mechanics(swing_keypoints)
        
        print("[INFO] Calculating performance score...")
        score, errors = self.scorer.calculate_swing_score(backswing_plane, downswing_alignment, rotation_stability)
        
        print(f"\n{'='*50}")
        print(f"SWING ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Swing Score: {score:.1f}/100")
        print(f"Backswing Plane Consistency: {backswing_plane:.1f}%")
        print(f"Downswing Alignment: {downswing_alignment:.1f}%")
        print(f"Rotational Stability: {rotation_stability:.1f}%")
        print(f"\nIdentified Errors:")
        for error in errors:
            print(f"  • {error}")
        print(f"{'='*50}\n")
        
        print("[INFO] Creating visualized output...")
        for i, frame in enumerate(swing_frames):
            annotated_frame = self.visualizer.overlay_swing_analysis(frame, swing_keypoints[i], backswing_plane, downswing_alignment, errors, i)
            out.write(annotated_frame)
        
        out.release()
        print(f"[INFO] Output video saved: {output_path}")
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'session_name': session_name or f"Swing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'score': score,
            'backswing_plane': backswing_plane,
            'downswing_alignment': downswing_alignment,
            'rotation_stability': rotation_stability,
            'errors': errors,
            'swing_duration_frames': swing_end - swing_start
        }
        
        self.data_manager.save_session(session_data)
        
        return {'score': score, 'backswing_plane': backswing_plane, 'downswing_alignment': downswing_alignment, 'rotation_stability': rotation_stability, 'errors': errors, 'output_video': output_path, 'session_data': session_data}
    
    def get_progress_report(self):
        """Generate progress tracking report"""
        print("\n[INFO] Generating progress report...")
        sessions = self.data_manager.load_all_sessions()
        if not sessions:
            print("[WARNING] No sessions found")
            return None
        self.visualizer.plot_progress_chart(sessions)
        return sessions


def main():
    """Main execution function"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [session_name]")
        print("\nExample:")
        print("  python main.py golf_swing.mp4")
        print("  python main.py golf_swing.mp4 'My First Practice Session'")
        return None
    
    video_path = sys.argv[1]
    session_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"[DEBUG] Video path: {video_path}")
    print(f"[DEBUG] Session name: {session_name}")
    
    app = SmartSwing()
    result = app.analyze_swing(video_path, session_name=session_name)
    
    if result:
        print("[SUCCESS] Analysis complete!")
        print(f"Final Score: {result['score']:.1f}/100")
    else:
        print("[ERROR] Analysis failed - check video path and try again")
    
    return result


if __name__ == "__main__":
    main()
