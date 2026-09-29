# DetectedObject.msg
# Input object/cluster representation from upstream 3D LiDAR perception
string object_id
float64 distance_m
float64 object_x_m
float64 object_y_m
float64 object_z_m
float64 object_width_m
float64 object_height_m
float64 object_length_m
int32 point_count
float64 relative_velocity_mps
float64 truck_speed_kmph
float64 time_to_collision_s
string object_type
