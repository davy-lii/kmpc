from setuptools import find_packages, setup
import os

package_name = 'whole_body_kmpc'


def _share(*parts: str) -> str:
    return os.path.join('share', package_name, *parts)


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (_share('rviz'), [
            'rviz/kmpc_sim.rviz',
        ]),
        (_share('config'), [
            'config/kmpc_params.yaml',
            'config/joint_limits.yaml',
            'config/joint_map.yaml',
            'config/sim_initial_joints.yaml',
        ]),
        (_share('urdf'), [
            'urdf/Light-duty wheeled robot.urdf',
            'urdf/Light-duty wheeled robot.urdf.xacro',
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'pyyaml'],
    zip_safe=True,
    maintainer='davy',
    maintainer_email='davy@todo.todo',
    description='Whole-body kinematic MPC for waist + dual-arm stream control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'p0_check = kmpc.p0_check:main',
            'send_ee_target = kmpc.send_ee_target:main',
            'calibrate_then_grasp = kmpc.calibrate_then_grasp:main',
            'pose_gui = kmpc.pose_gui:main',
            'reach_check = kmpc.reach_check:main',
        ],
    },
)
