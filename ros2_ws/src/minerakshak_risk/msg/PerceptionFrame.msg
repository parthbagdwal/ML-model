# PerceptionFrame.msg
# Multi-object perception frame from 3D LiDAR perception node
std_msgs/Header header
string frame_id
int64 timestamp_ns
float64 truck_speed_kmph
minerakshak_risk/DetectedObject[] objects
