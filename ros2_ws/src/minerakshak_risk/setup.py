from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'minerakshak_risk'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mineRakshak AI Safety Team',
    maintainer_email='safety@minerakshak.ai',
    description='ROS 2 Perception-to-Risk Assessment Node for haul truck safety.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'risk_node = minerakshak_risk.risk_node:main',
            'minerakshak_risk_node = minerakshak_risk.risk_node:main',
            'minerakshak_inference_node = minerakshak_risk.risk_node:main',
        ],
    },
)
