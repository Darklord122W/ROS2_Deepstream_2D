from setuptools import find_packages, setup

package_name = 'deepstream_bro'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # REQUIRED for ROS2 package indexing
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # package.xml install
        ('share/' + package_name,
            ['package.xml']),

        # DeepStream config + model files
        ('share/' + package_name + '/config', [
            'config/dstest1_pgie_config.txt',
            'config/labels.txt',
            'config/resnet18_trafficcamnet_pruned.onnx',
            'config/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_int8.engine',
            'config/config_infer_primary_yolo11.txt',
            'config/labels2.txt',
            'config/barebest.onnx_b1_gpu0_fp32.engine',
            'config/barebest.onnx',
            'config/libnvdsinfer_custom_impl_Yolo.so',
            'config/classificationconfig.txt',
            'config/labels_imagenet_1k.txt',    
            'config/yolo26n-cls.onnx',
            'config/yolo26n-cls.onnx_b1_gpu0_fp32.engine',
            'config/maindetector_demo.txt',
            'config/secondclassifier_demo.txt',
            'config/yolo26n-clsbatch16.onnx',
            'config/yolo26n-clsbatch16.onnx_b16_gpu0_fp32.engine',
            'config/terminalbatch1imgsz224.onnx',
            'config/terminalbatch1imgsz224.onnx_b16_gpu0_fp32.engine'
        ]),
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
            "deepstream_ros_node = deepstream_bro.ds_ros_node:main",
            "deepstream_detection_to_classifiers_node=deepstream_bro.ds_DtoC_demo:main",
            "deepstream_detection_to_classifiers_new_node=deepstream_bro.ds_DtoC_new:main",
            "deepstream_ros_classifier_node=deepstream_bro.classificationmodel:main",
        ],
    },
    
)
