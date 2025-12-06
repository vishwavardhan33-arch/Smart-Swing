import numpy as np

class PerformanceScorer:
    """Calculates swing accuracy score and identifies errors"""
    
    def __init__(self):
        self.backswing_weight = 0.35
        self.downswing_weight = 0.40
        self.rotation_weight = 0.25
    
    def calculate_swing_score(self, backswing_plane, downswing_alignment, rotation_stability):
        """
        Calculate overall swing score (0-100)
        
        Args:
            backswing_plane: Consistency score (0-100)
            downswing_alignment: Alignment score (0-100)
            rotation_stability: Stability score (0-100)
        
        Returns:
            (overall_score, list_of_errors)
        """
        # Weighted combination
        overall_score = (
            backswing_plane * self.backswing_weight +
            downswing_alignment * self.downswing_weight +
            rotation_stability * self.rotation_weight
        )
        
        # Identify errors
        errors = self._identify_errors(
            backswing_plane, downswing_alignment, rotation_stability
        )
        
        return overall_score, errors
    
    def _identify_errors(self, backswing_plane, downswing_alignment, rotation_stability):
        """Identify specific swing errors"""
        errors = []
        
        # Backswing errors
        if backswing_plane < 60:
            errors.append("Inconsistent Backswing Plane - Work on maintaining plane consistency")
        if backswing_plane < 40:
            errors.append("CRITICAL: Severe Backswing Plane Issues - Seek coaching")
        
        # Downswing errors
        if downswing_alignment < 55:
            errors.append("Over-the-Top Downswing - Sequence issue detected")
        if downswing_alignment < 45:
            errors.append("Early Extension - Check hip and spine alignment")
        if downswing_alignment < 35:
            errors.append("CRITICAL: Poor Downswing Path - Fundamentals need work")
        
        # Rotation errors
        if rotation_stability < 50:
            errors.append("Unstable Rotation - Practice core stability exercises")
        if rotation_stability < 35:
            errors.append("CRITICAL: Severe Rotation Issues - Body mechanics need correction")
        
        # Positive feedback
        if all([backswing_plane > 75, downswing_alignment > 75, rotation_stability > 75]):
            errors.append("✓ Excellent swing mechanics!")
        elif all([backswing_plane > 65, downswing_alignment > 65, rotation_stability > 65]):
            errors.append("✓ Good swing with minor adjustments needed")
        
        return errors if errors else ["Keep practicing - minor improvements needed"]
