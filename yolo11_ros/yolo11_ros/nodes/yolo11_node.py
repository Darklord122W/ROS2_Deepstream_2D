#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class Yolo11Node(Node):
    def __init__(self):
        super().__init__('yolo11_node')

        # Params (override via --ros-args -p ...)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model_path', 'yolo11s.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('imgsz', 640)

        self.image_topic = self.get_parameter('image_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.device = self.get_parameter('device').value
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.bridge = CvBridge()
        self.det_pub = self.create_publisher(Detection2DArray, '/yolo11/detections', 10)
        self.image_pub = self.create_publisher(Image, '/yolo11/annotated_image', 10)
        self.get_logger().info(f'Loading model: {self.model_path} device={self.device}')
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

        self.sub = self.create_subscription(Image, self.image_topic, self.cb, 10)
        self.get_logger().info("using gpu" if self.device.startswith("cuda") else "using cpu")
        self.get_logger().info(f'Subscribed to: {self.image_topic}')
        self.get_logger().info('Publishing: /yolo11/detections')

        

    def cb(self, msg: Image):
        # Your camera is rgb8
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        results = self.model.predict(
            source=rgb,
            device=self.device,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            verbose=False
        )
        
        out = Detection2DArray()
        out.header = msg.header

        if len(results) == 0 or results[0].boxes is None:
            self.det_pub.publish(out)
            return
        annotated = rgb.copy()
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            conf = float(confs[i])
            cls_id = int(clss[i])
            class_name = self.class_names.get(cls_id, str(cls_id))

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            label = f"{class_name}: {conf:.2f}"
            cv2.putText(
                annotated,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )


            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            det = Detection2D()
            det.header = msg.header

            bbox = BoundingBox2D()
            # center can be geometry_msgs/Pose2D (x,y,theta) OR a Pose2D-like with position.x/y
            if hasattr(bbox.center, "x"):
                bbox.center.x = float(cx)
                bbox.center.y = float(cy)
                if hasattr(bbox.center, "theta"):
                    bbox.center.theta = 0.0
            elif hasattr(bbox.center, "position"):
                bbox.center.position.x = float(cx)
                bbox.center.position.y = float(cy)
                if hasattr(bbox.center, "theta"):
                    bbox.center.theta = 0.0
            else:
                raise AttributeError(f"Unsupported bbox.center type: {type(bbox.center)} slots={getattr(bbox.center,'__slots__',None)}")

            bbox.size_x = float(w)
            bbox.size_y = float(h)
            det.bbox = bbox
            hyp = ObjectHypothesisWithPose()
            class_name = self.class_names.get(cls_id, str(cls_id))
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = conf
            det.results.append(hyp)

            out.detections.append(det)

        self.det_pub.publish(out)
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='rgb8')
        annotated_msg.header = msg.header
        self.image_pub.publish(annotated_msg)


def main():
    rclpy.init()
    node = Yolo11Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
