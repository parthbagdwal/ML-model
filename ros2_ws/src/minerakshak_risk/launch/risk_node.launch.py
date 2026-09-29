"""ROS 2 Launch file for MineRakshak Risk Assessment Node."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='minerakshak_risk',
            executable='minerakshak_risk_node',
            name='minerakshak_risk_node',
            output='screen',
            parameters=[{
                'input_topic': '/minerakshak/object_observations',
                'output_topic': '/minerakshak/risk_prediction',
                'threat_topic': '/minerakshak/highest_threat',
                'perception_topic': '',
                'risk_topic': '',
                'model_path': 'models/hgb_model.joblib',
                'preprocessor_path': 'models/preprocessor.joblib',
                'feature_schema_path': 'models/feature_schema.json',
                'label_mapping_path': 'models/label_mapping.json',
                'enable_secondary_voter': False,
                'log_latency': True,
            }],
        )
    ])
