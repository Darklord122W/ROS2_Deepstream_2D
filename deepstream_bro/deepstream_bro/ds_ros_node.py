#!/usr/bin/env python3
import threading
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds

from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw


MUXER_BATCH_TIMEOUT_USEC = 33000


def _make_bbox(cx, cy, w, h):
    bbox = BoundingBox2D()
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
        raise AttributeError(f"Unsupported bbox.center type: {type(bbox.center)}")

    bbox.size_x = float(w)
    bbox.size_y = float(h)
    return bbox


class DeepStreamUsbNode(Node):
    def __init__(self):
        super().__init__("deepstream_ros_image_node")

        # ROS topics
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("detections_topic", "/deepstream/detections")
        self.declare_parameter("frame_id", "default_cam")

        # Image expectations (must match the incoming ROS image for easiest path)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        # DeepStream streammux output size (can be different than camera)
        self.declare_parameter("streammux_width",640)
        self.declare_parameter("streammux_height", 480)

        # Inference config
        self.declare_parameter("pgie_config", "config_infer_primary_yolo11.txt")
        self.declare_parameter("sync", False)

        self.image_topic = self.get_parameter("image_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.streammux_width = int(self.get_parameter("streammux_width").value)
        self.streammux_height = int(self.get_parameter("streammux_height").value)

        pgie_cfg_param = self.get_parameter("pgie_config").value
        self.pgie_config = self._resolve_pkg_path(pgie_cfg_param)
        self.get_logger().info(f"Using PGIE config: {self.pgie_config}")

        self.sync = bool(self.get_parameter("sync").value)

        det_topic = self.get_parameter("detections_topic").value
        self.det_pub = self.create_publisher(Detection2DArray, det_topic, 10)
        self.get_logger().info(f"Publishing detections: {det_topic}")
        self.get_logger().info(f"Subscribing to images: {self.image_topic}")
        # Annotated image publisher
        self.declare_parameter("annotated_topic", "/deepstream/annotated")
        annotated_topic = self.get_parameter("annotated_topic").value
        self.ann_pub = self.create_publisher(Image, annotated_topic, 10)
        self.get_logger().info(f"Publishing annotated images: {annotated_topic}")

        # Shared detection cache for annotation
        self._det_lock = threading.Lock()
        self._last_dets = []  # list of tuples: (x1, y1, x2, y2, class_id, score)

        # GStreamer init
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)

        self.pipeline = self._build_pipeline()
        self._attach_probe()

        # Start pipeline + GLib loop
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop_thread.start()

        # ROS image subscription (sensor QoS)
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info("Image subscriber created (sensor QoS).")

        self._img_count = 0

                # Class labels (one label per line, index = class_id)
        self.declare_parameter("labels_file", "labels.txt")
        labels_param = self.get_parameter("labels_file").value
        self.labels_file = self._resolve_pkg_path(labels_param)

        self.class_names = self._load_labels(self.labels_file)
        self.get_logger().info(f"Loaded {len(self.class_names)} labels from: {self.labels_file}")

    def _load_labels(self, path: str):
        # Returns list where index is class_id
        names = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        names.append(s)
        except Exception as e:
            self.get_logger().warn(f"Could not read labels file '{path}': {e}")
        return names

    def _class_name(self, cid: int) -> str:
        if 0 <= cid < len(self.class_names):
            return self.class_names[cid]
        return f"class_{cid}"


    def _resolve_pkg_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        pkg_share = get_package_share_directory("deepstream_bro")
        cfg_path = os.path.join(pkg_share, "config", path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"PGIE config not found: '{cfg_path}'. "
                f"Installed config dir: {os.path.join(pkg_share, 'config')}"
            )
        return cfg_path

    def _build_pipeline(self):
        pipeline = Gst.Pipeline.new("ds-pipeline")
        if not pipeline:
            raise RuntimeError("Unable to create pipeline")

        # Replace v4l2src with appsrc (we will push ROS frames into appsrc)
        appsrc = Gst.ElementFactory.make("appsrc", "ros_appsrc")
        vidconvsrc = Gst.ElementFactory.make("videoconvert", "convertor_src1")
        nvvidconvsrc = Gst.ElementFactory.make("nvvideoconvert", "convertor_src2")
        # Force GPU for RGB conversions (VIC can't do RGB/BGR -> NV12)
        nvvidconvsrc.set_property("compute-hw", 1)   # 0=VIC (default), 1=GPU
        nvvidconvsrc.set_property("gpu-id", 0)
        caps_vidconvsrc = Gst.ElementFactory.make("capsfilter", "nvmm_caps")

        streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        streammux.set_property("live-source", 1)
        pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
        nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("fakesink", "sink")

        elems = [
            appsrc, vidconvsrc, nvvidconvsrc, caps_vidconvsrc,
            streammux, pgie, nvvidconv, nvosd, sink
        ]
        if any(e is None for e in elems):
            raise RuntimeError("Failed to create one or more GStreamer elements")

        # --- appsrc caps: matches ROS rgb8 frames ---
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("stream-type", 0)  # stream

        # Keep internal queue small to avoid building latency
        appsrc.set_property("block", False)
        appsrc.set_property("max-bytes", self.width * self.height * 3 * 2)  # ~2 frames

        appsrc_caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps}/1"
        )
        appsrc.set_property("caps", appsrc_caps)

        # Convert to NVMM+NV12 for DeepStream
        caps_vidconvsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.fps}/1"
            )
)

        # DeepStream settings
        streammux.set_property("width", self.streammux_width)
        streammux.set_property("height", self.streammux_height)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

        pgie.set_property("config-file-path", self.pgie_config)
        sink.set_property("sync", self.sync)

        # Add to pipeline
        for e in elems:
            pipeline.add(e)

        # Link appsrc -> videoconvert -> nvvideoconvert -> capsfilter
        if not appsrc.link(vidconvsrc):
            raise RuntimeError("link failed: appsrc -> videoconvert")
        if not vidconvsrc.link(nvvidconvsrc):
            raise RuntimeError("link failed: videoconvert -> nvvideoconvert")
        if not nvvidconvsrc.link(caps_vidconvsrc):
            raise RuntimeError("link failed: nvvideoconvert -> caps_vidconvsrc")

        # caps_vidconvsrc -> streammux.sink_0 (request pad)
        sinkpad = streammux.request_pad_simple("sink_0")
        srcpad = caps_vidconvsrc.get_static_pad("src")
        if sinkpad is None or srcpad is None:
            raise RuntimeError("Unable to get pads for streammux linking")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link caps_vidconvsrc -> streammux.sink_0")

        # Downstream
        if not streammux.link(pgie):
            raise RuntimeError("link failed: streammux -> pgie")
        if not pgie.link(nvvidconv):
            raise RuntimeError("link failed: pgie -> nvvidconv")
        if not nvvidconv.link(nvosd):
            raise RuntimeError("link failed: nvvidconv -> nvosd")
        if not nvosd.link(sink):
            raise RuntimeError("link failed: nvosd -> sink")

        # Bus watch
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, None)

        # Save handles
        self._nvosd = nvosd
        self._appsrc = appsrc

        return pipeline

    def _on_image(self, msg: Image):
        # Expect exactly what you showed: rgb8, step = width*3
        if msg.encoding.lower() != "rgb8":
            self.get_logger().warn(f"Expected rgb8, got {msg.encoding}")
            return

        if msg.width != self.width or msg.height != self.height:
            self.get_logger().warn(
                f"Image size {msg.width}x{msg.height} != expected {self.width}x{self.height}"
            )
            return

        expected_step = self.width * 3
        if msg.step != expected_step:
            self.get_logger().warn(f"Unexpected step={msg.step}, expected {expected_step}")
            return

        if not hasattr(self, "_appsrc") or self._appsrc is None:
            return

        data = bytes(msg.data)

        # Allocate and fill Gst.Buffer
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = int(1e9 / max(self.fps, 1))

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self.get_logger().warn(f"push-buffer returned {ret}")

        # Publish annotated image using latest detections
        self._publish_annotated(msg)

    def _on_bus_message(self, bus, message, _):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f"GstError: {err} debug={debug}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().info("Gst EOS")

    def _attach_probe(self):
        osdsinkpad = self._nvosd.get_static_pad("sink")
        if not osdsinkpad:
            raise RuntimeError("Unable to get sink pad of nvosd")
        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, self._osd_probe, None)

    # _osd_probe and destroy_node stay exactly the same as your current version
    def _osd_probe(self, pad, info, _):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.frame_id

        # Collect bboxes for annotated publishing
        det_list = []

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)

                x1 = float(obj.rect_params.left)
                y1 = float(obj.rect_params.top)
                w = float(obj.rect_params.width)
                h = float(obj.rect_params.height)

                x2 = x1 + w
                y2 = y1 + h

                cx = x1 + 0.5 * w
                cy = y1 + 0.5 * h

                det = Detection2D()
                det.header = out.header
                det.bbox = _make_bbox(cx, cy, w, h)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(int(obj.class_id))
                hyp.hypothesis.score = float(obj.confidence)
                det.results.append(hyp)

                out.detections.append(det)

                # Save for annotation
                det_list.append((x1, y1, x2, y2, int(obj.class_id), float(obj.confidence)))

                l_obj = l_obj.next

            l_frame = l_frame.next

        # Publish detections
        self.det_pub.publish(out)

        # Update cache for annotated publishing
        with self._det_lock:
            self._last_dets = det_list

        return Gst.PadProbeReturn.OK

    def _publish_annotated(self, msg: Image):
        # Pull latest detections
        with self._det_lock:
            dets = list(self._last_dets)

        if not dets:
            # If you still want to publish raw images when no dets, you can.
            # For now, publish nothing if empty.
            return

        # Convert ROS msg.data (rgb8) to numpy image (H,W,3)
        img = np.frombuffer(msg.data, dtype=np.uint8)
        if img.size != msg.height * msg.width * 3:
            return
        img = img.reshape((msg.height, msg.width, 3))

        pil = PILImage.fromarray(img, mode="RGB")
        draw = ImageDraw.Draw(pil)

        # Draw rectangles + simple labels
        for (x1, y1, x2, y2, cid, score) in dets:
            # Clamp to image bounds
            x1i = int(max(0, min(msg.width - 1, round(x1))))
            y1i = int(max(0, min(msg.height - 1, round(y1))))
            x2i = int(max(0, min(msg.width - 1, round(x2))))
            y2i = int(max(0, min(msg.height - 1, round(y2))))

            # Rectangle (default color; Pillow uses a default if not specified, but we set one)
            draw.rectangle([x1i, y1i, x2i, y2i], outline=(255, 0, 0), width=2)
            cname = self._class_name(cid)
            label = f"id:{cid} {cname} score:{score:.2f}"
            # Put text slightly above box if possible
            ty = y1i - 12 if y1i >= 12 else y1i + 2
            draw.text((x1i + 2, ty), label, fill=(255, 0, 0))

        out = Image()
        out.header = msg.header  # keep same timestamp/frame_id as input
        out.height = msg.height
        out.width = msg.width
        out.encoding = "rgb8"
        out.is_bigendian = msg.is_bigendian
        out.step = msg.width * 3
        out.data = pil.tobytes()

        self.ann_pub.publish(out)


    def destroy_node(self):
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                # Tell appsrc no more buffers (optional but clean)
                if hasattr(self, "_appsrc") and self._appsrc is not None:
                    try:
                        self._appsrc.emit("end-of-stream")
                    except Exception:
                        pass
                self.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

        try:
            if hasattr(self, "loop") and self.loop is not None:
                self.loop.quit()
        except Exception:
            pass

        super().destroy_node()



def main():
    rclpy.init()
    node = DeepStreamUsbNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
