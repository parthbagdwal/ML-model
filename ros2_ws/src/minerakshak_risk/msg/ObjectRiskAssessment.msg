# ObjectRiskAssessment.msg
# Machine learning risk classification and probabilities for an individual obstacle
string object_id
string risk_level
string predicted_risk_level
int32 class_id
int32 risk_code
float64 confidence
float64 prob_safe
float64 prob_caution
float64 prob_warning
float64 prob_critical
string model
string status
string recommendation
string recommended_action
float64 inference_latency_ms
string inference_timestamp
