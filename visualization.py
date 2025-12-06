import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class SwingVisualizer:
    """Creates visual feedback and progress charts"""
    
    def __init__(self):
        self.colors = {
            'shoulder': (255, 0, 0),
            'hip': (0, 255, 0),
            'wrist': (0, 0, 255),
            'plane': (255, 255, 0),
            'error': (0, 0, 255)
        }
    
    def overlay_swing_analysis(self, frame, keypoints, backswing_plane, 
                               downswing_alignment, errors, frame_num):
        """
        Overlay analysis information on video frame
        
        Args:
            frame: Input frame
            keypoints: Detected keypoints
            backswing_plane: Backswing score
            downswing_alignment: Downswing score
            errors: List of identified errors
            frame_num: Frame number in swing
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw skeleton
        annotated = self._draw_skeleton(annotated, keypoints)
        
        # Draw swing plane reference line
        annotated = self._draw_swing_plane(annotated, keypoints)
        
        # Add score info
        annotated = self._add_score_overlay(
            annotated, backswing_plane, downswing_alignment, frame_num
        )
        
        # Add error messages
        annotated = self._add_error_overlay(annotated, errors)
        
        return annotated
    
    def _draw_skeleton(self, frame, keypoints):
        """Draw pose skeleton on frame"""
        if not keypoints:
            return frame
        
        # Define connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
        ]
        
        # Draw connections
        for start, end in connections:
            if start in keypoints and end in keypoints:
                pt1 = tuple(keypoints[start].astype(int))
                pt2 = tuple(keypoints[end].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for key, point in keypoints.items():
            pt = tuple(point.astype(int))
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
            
            # FIX: Convert tuple elements correctly
            try:
                text_x = int(pt[0] + 5)
                text_y = int(pt[1] - 5)
                cv2.putText(frame, key, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            except:
                pass
        
        return frame
    
    def _draw_swing_plane(self, frame, keypoints):
        """Draw reference swing plane on frame"""
        if not keypoints or 'left_shoulder' not in keypoints:
            return frame
        
        h, w, _ = frame.shape
        
        # Draw vertical reference line (ideal swing plane)
        center_x = w // 2
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 0), 2)
        cv2.putText(frame, 'Ideal Plane', (center_x + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw shoulder line
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            pt1 = tuple(keypoints['left_shoulder'].astype(int))
            pt2 = tuple(keypoints['right_shoulder'].astype(int))
            cv2.line(frame, pt1, pt2, (255, 0, 255), 3)
        
        return frame
    
    def _add_score_overlay(self, frame, backswing, downswing, frame_num):
        """Add score information overlay"""
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Text
        cv2.putText(frame, f'Frame: {frame_num}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Backswing Plane: {backswing:.1f}%', (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Downswing Alignment: {downswing:.1f}%', (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _add_error_overlay(self, frame, errors):
        """Add error messages overlay"""
        h, w, _ = frame.shape
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 150), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        y_offset = h - 130
        for i, error in enumerate(errors[:3]):  # Show top 3 errors
            color = (0, 255, 0) if '✓' in error else (0, 0, 255)
            try:
                cv2.putText(frame, error[:50], (20, y_offset + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except:
                pass
        
        return frame
    
    def plot_progress_chart(self, sessions):
        """Plot progress over multiple sessions"""
        if not sessions:
            return
        
        dates = [s['timestamp'][:10] for s in sessions]
        scores = [s['score'] for s in sessions]
        backswing = [s['backswing_plane'] for s in sessions]
        downswing = [s['downswing_alignment'] for s in sessions]
        rotation = [s['rotation_stability'] for s in sessions]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('SmartSwing Progress Tracking', fontsize=16, fontweight='bold')
        
        # Overall Score
        axes[0, 0].plot(dates, scores, marker='o', linewidth=2, markersize=8, color='#2ecc71')
        axes[0, 0].set_title('Overall Swing Score')
        axes[0, 0].set_ylabel('Score (0-100)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 100])
        
        # Backswing Plane
        axes[0, 1].plot(dates, backswing, marker='s', linewidth=2, markersize=8, color='#3498db')
        axes[0, 1].set_title('Backswing Plane Consistency')
        axes[0, 1].set_ylabel('Consistency %')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])
        
        # Downswing Alignment
        axes[1, 0].plot(dates, downswing, marker='^', linewidth=2, markersize=8, color='#e74c3c')
        axes[1, 0].set_title('Downswing Alignment')
        axes[1, 0].set_ylabel('Alignment %')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 100])
        
        # Rotation Stability
        axes[1, 1].plot(dates, rotation, marker='D', linewidth=2, markersize=8, color='#f39c12')
        axes[1, 1].set_title('Rotational Stability')
        axes[1, 1].set_ylabel('Stability %')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 100])
        
        # Rotate x labels
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('swing_progress.png', dpi=150, bbox_inches='tight')
        print("[INFO] Progress chart saved: swing_progress.png")
        plt.show()
