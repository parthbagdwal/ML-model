# RiskAssessmentFrame.msg
# Comprehensive frame-level risk assessment payload containing all obstacles and highest threat
std_msgs/Header header
string frame_id
int32 objects_detected
int32 objects_evaluated
minerakshak_risk/ObjectRiskAssessment[] assessments
minerakshak_risk/ObjectRiskAssessment highest_threat
string highest_threat_level
string highest_threat_object_id
float64 total_frame_latency_ms
string[] processing_errors
