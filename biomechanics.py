import mediapipe as mp
import numpy as np
import cv2

class BiomechanicsAnalyzer:
    """Extracts and analyzes biomechanical metrics from poses"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_keypoints(self, frame):
        """
        Extract pose keypoints from frame
        
        Returns:
            (keypoints_dict, pose_landmarks)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        keypoints = {}
        pose_landmarks = results.pose_landmarks
        
        if results.pose_landmarks:
            # Extract key body parts for golf swing analysis
            landmarks = results.pose_landmarks.landmark
            
            # FIX: Pass individual landmark objects, not the entire list
            keypoints = {
                'left_shoulder': self._get_coord(landmarks[11], frame),
                'right_shoulder': self._get_coord(landmarks[12], frame),
                'left_hip': self._get_coord(landmarks[23], frame),
                'right_hip': self._get_coord(landmarks[24], frame),
                'left_wrist': self._get_coord(landmarks[9], frame),
                'right_wrist': self._get_coord(landmarks[10], frame),
                'left_elbow': self._get_coord(landmarks[7], frame),
                'right_elbow': self._get_coord(landmarks[8], frame),
                'head': self._get_coord(landmarks[0], frame),
                'left_knee': self._get_coord(landmarks[25], frame),
                'right_knee': self._get_coord(landmarks[26], frame),
            }
        
        return keypoints, pose_landmarks

    def _get_coord(self, landmark, frame):
        """Convert normalized landmark to pixel coordinates"""
        h, w, _ = frame.shape
        
        # FIX: Ensure landmark has x and y attributes
        if landmark is None or not hasattr(landmark, 'x') or not hasattr(landmark, 'y'):
            return np.array([0, 0])
        
        try:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            return np.array([x, y])
        except Exception as e:
            print(f"[WARNING] Error converting coordinate: {e}")
            return np.array([0, 0])

    
    def calculate_motion_metrics(self, prev_keypoints, curr_keypoints):
        """Calculate motion between two frames"""
        if not prev_keypoints or not curr_keypoints:
            return {'overall_motion': 0}
        
        motion_vectors = []
        for key in prev_keypoints:
            if key in curr_keypoints:
                diff = np.linalg.norm(curr_keypoints[key] - prev_keypoints[key])
                motion_vectors.append(diff)
        
        overall_motion = np.mean(motion_vectors) if motion_vectors else 0
        
        return {
            'overall_motion': overall_motion,
            'motion_vectors': motion_vectors
        }
    
    def analyze_swing_mechanics(self, swing_keypoints):
        """
        Analyze swing mechanics across frames
        
        Returns:
            (backswing_plane, downswing_alignment, rotation_stability)
        """
        if not swing_keypoints:
            return 0, 0, 0
        
        # Analyze backswing plane consistency
        backswing_plane = self._analyze_backswing_plane(swing_keypoints)
        
        # Analyze downswing alignment
        downswing_alignment = self._analyze_downswing_alignment(swing_keypoints)
        
        # Analyze rotational stability
        rotation_stability = self._analyze_rotation_stability(swing_keypoints)
        
        return backswing_plane, downswing_alignment, rotation_stability
    
    def _analyze_backswing_plane(self, swing_keypoints):
        """Analyze consistency of backswing plane"""
        if len(swing_keypoints) < 2:
            return 50
        
        # Calculate shoulder-hip plane angles
        plane_angles = []
        for kp in swing_keypoints[:len(swing_keypoints)//2]:  # Backswing phase
            angle = self._calculate_plane_angle(kp)
            if angle is not None:
                plane_angles.append(angle)
        
        if not plane_angles:
            return 50
        
        # Score based on consistency (low variance = high score)
        variance = np.var(plane_angles)
        consistency = max(0, 100 - variance * 2)
        
        return min(100, consistency)
    
    def _analyze_downswing_alignment(self, swing_keypoints):
        """Analyze downswing alignment"""
        if len(swing_keypoints) < 2:
            return 50
        
        # Analyze second half (downswing)
        downswing_kps = swing_keypoints[len(swing_keypoints)//2:]
        
        alignment_scores = []
        for kp in downswing_kps:
            # Check if arms follow optimal path
            alignment = self._calculate_arm_alignment(kp)
            if alignment is not None:
                alignment_scores.append(alignment)
        
        if not alignment_scores:
            return 50
        
        return np.mean(alignment_scores)
    
    def _analyze_rotation_stability(self, swing_keypoints):
        """Analyze hip and shoulder rotation stability"""
        if len(swing_keypoints) < 2:
            return 50
        
        rotation_angles = []
        for kp in swing_keypoints:
            rotation = self._calculate_rotation_angle(kp)
            if rotation is not None:
                rotation_angles.append(rotation)
        
        if not rotation_angles:
            return 50
        
        # Score based on smooth rotation (low variance = high score)
        variance = np.var(rotation_angles)
        stability = max(0, 100 - variance)
        
        return min(100, stability)
    
    def _calculate_plane_angle(self, keypoints):
        """Calculate swing plane angle"""
        try:
            left_shoulder = keypoints.get('left_shoulder')
            right_shoulder = keypoints.get('right_shoulder')
            left_hip = keypoints.get('left_hip')
            right_hip = keypoints.get('right_hip')
            
            if any(v is None for v in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return None
            
            # Calculate angle between shoulder line and hip line
            shoulder_vec = right_shoulder - left_shoulder
            hip_vec = right_hip - left_hip
            
            cos_angle = np.dot(shoulder_vec, hip_vec) / (
                np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec) + 1e-6
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            return angle
        except:
            return None
    
    def _calculate_arm_alignment(self, keypoints):
        """Calculate if arms follow good downswing path (0-100 score)"""
        try:
            left_wrist = keypoints.get('left_wrist')
            right_wrist = keypoints.get('right_wrist')
            left_elbow = keypoints.get('left_elbow')
            right_elbow = keypoints.get('right_elbow')
            
            if any(v is None for v in [left_wrist, right_wrist, left_elbow, right_elbow]):
                return None
            
            # Check if wrists are below elbows (good position)
            left_alignment = 1 if left_wrist > left_elbow else 0
            right_alignment = 1 if right_wrist > right_elbow else 0
            
            alignment_score = (left_alignment + right_alignment) / 2 * 100
            return alignment_score
        except:
            return None
    
    def _calculate_rotation_angle(self, keypoints):
        """Calculate total body rotation angle"""
        try:
            left_shoulder = keypoints.get('left_shoulder')
            right_shoulder = keypoints.get('right_shoulder')
            left_hip = keypoints.get('left_hip')
            right_hip = keypoints.get('right_hip')
            
            if any(v is None for v in [left_shoulder, right_shoulder, left_hip, right_hip]):
                return None
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            # Calculate rotation angle from center line
            spine_vec = shoulder_center - hip_center
            angle = np.degrees(np.arctan2(spine_vec, spine_vec))
            
            return angle
        except:
            return None
