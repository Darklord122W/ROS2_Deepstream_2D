from pathlib import Path

from setuptools import find_packages, setup

package_name = 'deepstream_ssh'
config_files = [str(path) for path in sorted(Path('config').glob('*')) if path.is_file()]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', config_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='autodrive',
    maintainer_email='2531403062@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ds_rosbag = deepstream_ssh.ds_rosbag:main',
        ],
    },
)
