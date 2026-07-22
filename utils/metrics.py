from scipy.spatial import distance as dist

def calculate_ear(eye):
    """
    Calculates the Eye Aspect Ratio (EAR)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth):
    """
    Calculates the Mouth Aspect Ratio (MAR) for Yawn Detection
    """
    A = dist.euclidean(mouth[2], mouth[10]) 
    B = dist.euclidean(mouth[4], mouth[8])  
    
    C = dist.euclidean(mouth[0], mouth[6])  
    
    # Compute the Mouth Aspect Ratio
    mar = (A + B) / (2.0 * C)
    return mar
