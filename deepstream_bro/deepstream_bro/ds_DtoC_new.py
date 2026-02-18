#!/usr/bin/env python3
import os
import threading

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from PIL import Image as PILImage
from PIL import ImageDraw
MUXER_BATCH_TIMEOUT_USEC = 33000




def make_bbox(cx, cy, w, h):
    bbox = BoundingBox2D()

    # vision_msgs has two possible layouts depending on version
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


class DeepStreamPgieSgiePublisher(Node):
    def __init__(self):
        super().__init__("deepstream_pgie_sgie_pub_node")

        self._gst_initialized = False
        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        self._appsrc = None

        self._declare_params()
        self._read_params()
        self._resolve_paths()

        self.det_pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.ann_pub = self.create_publisher(Image, self.annotated_topic, 10)
        self._probe_extract_error_logged = False


        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._push_image_to_appsrc, qos_profile_sensor_data
        )

        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"Publishing:  {self.detections_topic}")
        self.get_logger().info(f"Publishing:  {self.annotated_topic}")
        self.get_logger().info(f"PGIE cfg:    {self.pgie_config}")
        self.get_logger().info(f"SGIE cfg:    {self.sgie_config}")
        self.get_logger().info(f"Expecting:   rgb8 {self.width}x{self.height} @ {self.fps} fps")

        self._setup_gstreamer()
        self.start_pipeline()

    # -------------------------
    # ROS params
    # -------------------------
    def _declare_params(self):
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("detections_topic", "/deepstream/detections")
        self.declare_parameter("annotated_topic", "/deepstream/annotated")
        self.declare_parameter("frame_id", "camera")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        self.declare_parameter("streammux_width", 640)
        self.declare_parameter("streammux_height", 480)

        self.declare_parameter("pgie_config", "maindetector_demo.txt")
        self.declare_parameter("sgie_config", "secondclassifier_demo.txt")


        self.declare_parameter("PGIElabel", "labels.txt")
        self.declare_parameter("SGIElabel", "labels_imagenet_1k.txt")

        self.declare_parameter("sync", False)

        # which SGIE unique-id you expect (must match sgie config gie-unique-id)
        self.declare_parameter("sgie_unique_id", 2)

        # if your detector rescales in streammux and you want to publish bboxes in original input resolution
        self.declare_parameter("publish_in_input_resolution", True)



    def _read_params(self):
        self.image_topic = self.get_parameter("image_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.streammux_width = int(self.get_parameter("streammux_width").value)
        self.streammux_height = int(self.get_parameter("streammux_height").value)

        self.pgie_config_param = self.get_parameter("pgie_config").value
        self.sgie_config_param = self.get_parameter("sgie_config").value

        self.sync = bool(self.get_parameter("sync").value)
        self.sgie_unique_id = int(self.get_parameter("sgie_unique_id").value)

        self.publish_in_input_resolution = bool(
            self.get_parameter("publish_in_input_resolution").value
        )
        self.pgie_label_param = self.get_parameter("PGIElabel").value
        self.sgie_label_param = self.get_parameter("SGIElabel").value

    def _resolve_paths(self):
        self.pgie_config = self._resolve_pkg_path(self.pgie_config_param)
        self.sgie_config = self._resolve_pkg_path(self.sgie_config_param)

        self.pgie_label_path = self._resolve_label_path(self.pgie_label_param)
        self.sgie_label_path = self._resolve_label_path(self.sgie_label_param)

        self.pgie_labels = self._load_labels(self.pgie_label_path)
        self.sgie_labels = self._load_labels(self.sgie_label_path)

        self.get_logger().info(f"PGIE labels: {self.pgie_label_path} ({len(self.pgie_labels)})")
        self.get_logger().info(f"SGIE labels: {self.sgie_label_path} ({len(self.sgie_labels)})")

    def _resolve_pkg_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        pkg_share = get_package_share_directory("deepstream_bro")
        cfg_path = os.path.join(pkg_share, "config", path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        return cfg_path
    def _resolve_label_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        pkg_share = get_package_share_directory("deepstream_bro")
        # assuming your labels are also in <pkg_share>/config/
        p = os.path.join(pkg_share, "config", path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Label file not found: {p}")
        return p

    def _load_labels(self, path: str):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # -------------------------
    # GStreamer pipeline
    # -------------------------
    def _setup_gstreamer(self):
        if not self._gst_initialized:
            Gst.init(None)
            self._gst_initialized = True

        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)

        self.pipeline = Gst.Pipeline.new("ds-pgie-sgie-pub")
        if not self.pipeline:
            raise RuntimeError("Unable to create pipeline")

        # Elements
        self.appsrc = Gst.ElementFactory.make("appsrc", "ros_appsrc")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "vcpu_convert")
        self.nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nv_convert")
        self.caps_nvmm_nv12 = Gst.ElementFactory.make("capsfilter", "caps_nvmm_nv12")

        self.streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        self.pgie = Gst.ElementFactory.make("nvinfer", "pgie")
        self.sgie = Gst.ElementFactory.make("nvinfer", "sgie")

        self.sink = Gst.ElementFactory.make("fakesink", "sink")

        elems = [
            self.appsrc, self.videoconvert, self.nvvideoconvert, self.caps_nvmm_nv12,
            self.streammux, self.pgie, self.sgie, self.sink
        ]
        if any(e is None for e in elems):
            raise RuntimeError("Failed to create one or more GStreamer elements")

        # appsrc caps: expect rgb8 ROS images
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("do-timestamp", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("stream-type", 0)
        self.appsrc.set_property("block", False)
        self.appsrc.set_property("max-bytes", self.width * self.height * 3 * 2)
        self.appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps}/1"
            ),
        )

        # convert to NVMM NV12
        self.nvvideoconvert.set_property("compute-hw", 1)
        self.nvvideoconvert.set_property("gpu-id", 0)

        self.caps_nvmm_nv12.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.fps}/1"
            ),
        )

        # streammux
        self.streammux.set_property("live-source", 1)
        self.streammux.set_property("width", self.streammux_width)
        self.streammux.set_property("height", self.streammux_height)
        self.streammux.set_property("batch-size", 1)
        self.streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

        # nvinfer configs
        self.pgie.set_property("config-file-path", self.pgie_config)
        self.sgie.set_property("config-file-path", self.sgie_config)

        self.sink.set_property("sync", self.sync)

        # Add to pipeline
        for e in elems:
            self.pipeline.add(e)

        # Link CPU path
        if not self.appsrc.link(self.videoconvert):
            raise RuntimeError("link failed: appsrc to videoconvert")
        if not self.videoconvert.link(self.nvvideoconvert):
            raise RuntimeError("link failed: videoconvert to nvvideoconvert")
        if not self.nvvideoconvert.link(self.caps_nvmm_nv12):
            raise RuntimeError("link failed: nvvideoconvert to caps_nvmm_nv12")

        # caps -> streammux sink_0 (request pad)
        sinkpad = self.streammux.request_pad_simple("sink_0")
        srcpad = self.caps_nvmm_nv12.get_static_pad("src")
        if sinkpad is None or srcpad is None:
            raise RuntimeError("Unable to get pads for streammux linking")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link caps_nvmm_nv12 to streammux sink_0")

        # streammux -> pgie -> sgie -> sink
        if not self.streammux.link(self.pgie):
            raise RuntimeError("link failed: streammux to pgie")
        if not self.pgie.link(self.sgie):
            raise RuntimeError("link failed: pgie to sgie")
        if not self.sgie.link(self.sink):
            raise RuntimeError("link failed: sgie to sink")

        self._appsrc = self.appsrc

        # Bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, None)

        # Probe on SGIE src pad (most important change)
        sgie_src_pad = self.sgie.get_static_pad("src")
        if not sgie_src_pad:
            raise RuntimeError("Unable to get SGIE src pad")
        sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, None)

    # -------------------------
    # ROS image callback
    # -------------------------
    def _push_image_to_appsrc(self, msg: Image):
        if self._appsrc is None:
            return

        if msg.encoding.lower() != "rgb8":
            self.get_logger().warn(f"Expected rgb8, got {msg.encoding}")
            return

        if msg.width != self.width or msg.height != self.height:
            self.get_logger().warn(
                f"Image {msg.width}x{msg.height} does not match expected {self.width}x{self.height}"
            )
            return

        expected_step = self.width * 3
        if msg.step != expected_step:
            self.get_logger().warn(f"Unexpected step={msg.step}, expected {expected_step}")
            return

        data = bytes(msg.data)
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = int(1e9 / max(self.fps, 1))

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self.get_logger().warn(f"push-buffer returned {ret}")

    def _draw_and_publish_annotated(self, rgb, ann_dets, stamp, is_bigendian):
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        h, w, _ = rgb.shape
        pil = PILImage.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(pil)

        sx_draw = float(w) / float(self.streammux_width) if self.streammux_width > 0 else 1.0
        sy_draw = float(h) / float(self.streammux_height) if self.streammux_height > 0 else 1.0

        for (x1, y1, x2, y2, det_label, det_score, cls_label, cls_score) in ann_dets:
            x1i = int(max(0, min(w - 1, round(x1 * sx_draw))))
            y1i = int(max(0, min(h - 1, round(y1 * sy_draw))))
            x2i = int(max(0, min(w - 1, round(x2 * sx_draw))))
            y2i = int(max(0, min(h - 1, round(y2 * sy_draw))))

            draw.rectangle([x1i, y1i, x2i, y2i], outline=(255, 0, 0), width=2)

            det_text = f"Det: {det_label} ({det_score:.2f})"
            if cls_label:
                cls_text = f"Cls: {cls_label} ({cls_score:.2f})"
            else:
                cls_text = "Cls: n/a"
            text = f"{det_text} | {cls_text}"

            ty = y1i - 14 if y1i >= 14 else y1i + 2
            draw.text((x1i + 2, ty), text, fill=(255, 255, 0))

        ann_msg = Image()
        ann_msg.header.stamp = stamp
        ann_msg.header.frame_id = self.frame_id
        ann_msg.height = h
        ann_msg.width = w
        ann_msg.encoding = "rgb8"
        ann_msg.is_bigendian = int(is_bigendian)
        ann_msg.step = w * 3
        ann_msg.data = pil.tobytes()
        self.ann_pub.publish(ann_msg)

    def _publish_annotated_from_probe(self, gst_buffer, frame_meta, ann_dets, stamp):
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            rgba = np.array(n_frame, copy=True, order="C")
        except Exception as e:
            if not self._probe_extract_error_logged:
                self.get_logger().warn(f"Failed to extract frame from probe buffer: {e}")
                self._probe_extract_error_logged = True
            return

        if rgba.ndim != 3 or rgba.shape[2] < 3:
            if not self._probe_extract_error_logged:
                self.get_logger().warn(
                    f"Unexpected probe frame shape for annotation: {getattr(rgba, 'shape', None)}"
                )
                self._probe_extract_error_logged = True
            return

        rgb = rgba[:, :, :3]
        self._probe_extract_error_logged = False
        self._draw_and_publish_annotated(rgb, ann_dets, stamp, is_bigendian=0)
    # -------------------------
    # Probe: publish detections + SGIE top1 per object
    # -------------------------
    def _on_buffer_probe(self, pad, info, _):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        out = Detection2DArray()
        stamp = self.get_clock().now().to_msg()
        out.header.stamp = stamp
        out.header.frame_id = self.frame_id
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            ann_dets = []

            # If streammux resizes, obj.rect_params are in mux space
            # We optionally scale back to input resolution you subscribed.
            if self.publish_in_input_resolution:
                mux_w = float(self.streammux_width)
                mux_h = float(self.streammux_height)
                sx = float(self.width) / mux_w if mux_w > 0 else 1.0
                sy = float(self.height) / mux_h if mux_h > 0 else 1.0
            else:
                sx = 1.0
                sy = 1.0

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                # Debug: check whether SGIE attached classifier meta to this object
                self._log_classifier_meta(obj, prefix="[SGIE CHECK] ")
                # print occasionally to avoid spam
                if int(frame_meta.frame_num) % 30 == 0:
                    self._log_obj_meta_state(obj)


                raw_x1 = float(obj.rect_params.left)
                raw_y1 = float(obj.rect_params.top)
                raw_w = float(obj.rect_params.width)
                raw_h = float(obj.rect_params.height)

                x1 = raw_x1 * sx
                y1 = raw_y1 * sy
                w = raw_w * sx
                h = raw_h * sy

                cx = x1 + 0.5 * w
                cy = y1 + 0.5 * h

                det = Detection2D()
                det.header = out.header
                det.bbox = make_bbox(cx, cy, w, h)

                # detection hypothesis
                hyp_det = ObjectHypothesisWithPose()
                pgie_id = int(obj.class_id)
                pgie_label = self.pgie_labels[pgie_id] if 0 <= pgie_id < len(self.pgie_labels) else str(pgie_id)
                hyp_det.hypothesis.class_id = f"PGIE:{pgie_label}"
                hyp_det.hypothesis.score = float(obj.confidence)
                det.results.append(hyp_det)

                # classification hypothesis (top1 from the SGIE with matching unique_component_id)
                sgie_label_text = ""
                sgie_prob = 0.0
                cls = self._get_sgie_top1(obj, sgie_uid=self.sgie_unique_id)
                if cls is not None:
                    label, cid, prob = cls
                    self.get_logger().info(f"[SGIE] class_name='{label}' class_id={cid} prob={prob:.3f}")
                    hyp_cls = ObjectHypothesisWithPose()
                    if (not label) and (0 <= int(cid) < len(self.sgie_labels)):
                        label = self.sgie_labels[int(cid)]

                    sgie_label_text = label if label else str(int(cid))
                    sgie_prob = float(prob)
                    hyp_cls.hypothesis.class_id = f"SGIE:{label}" if label else f"SGIE:{int(cid)}"
                    hyp_cls.hypothesis.score = float(prob)
                    det.results.append(hyp_cls)

                x2 = raw_x1 + raw_w
                y2 = raw_y1 + raw_h

                ann_dets.append(
                    (
                        raw_x1,
                        raw_y1,
                        x2,
                        y2,
                        pgie_label,
                        float(obj.confidence),
                        sgie_label_text,
                        sgie_prob,
                    )
                )

                out.detections.append(det)
                l_obj = l_obj.next

            self._publish_annotated_from_probe(gst_buffer, frame_meta, ann_dets, stamp)
            l_frame = l_frame.next

        self.det_pub.publish(out)
        return Gst.PadProbeReturn.OK

    def _log_classifier_meta(self, obj_meta, prefix=""):
        """
        Debug helper: prints whether classifier_meta_list exists,
        and if it exists, prints all classifier unique_component_id values.
        """
        l = obj_meta.classifier_meta_list
        if l is None:
            self.get_logger().warn(
                f"{prefix}NO classifier meta for det class_id={int(obj_meta.class_id)}"
            )
            return

        # If list exists, you can have multiple classifier metas (multiple SGIEs, or multiple heads)
        uids = []
        try:
            while l is not None:
                cmeta = pyds.NvDsClassifierMeta.cast(l.data)
                uids.append(int(getattr(cmeta, "unique_component_id", -1)))
                l = l.next
            self.get_logger().info(
                f"{prefix}HAS classifier meta uids={uids} for det class_id={int(obj_meta.class_id)}"
            )
        except Exception as e:
            self.get_logger().warn(f"{prefix}Error reading classifier meta: {e}")
        
    def _obj_has_tensor_meta(self, obj_meta) -> bool:
        l = obj_meta.obj_user_meta_list
        while l is not None:
            um = pyds.NvDsUserMeta.cast(l.data)
            if um and um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                return True
            l = l.next
        return False

    def _log_obj_meta_state(self, obj_meta):
        has_cls = (obj_meta.classifier_meta_list is not None)
        has_tensor = self._obj_has_tensor_meta(obj_meta)
        self.get_logger().warn(
            f"[SGIE STATE] det class_id={int(obj_meta.class_id)} "
            f"cls_meta={has_cls} tensor_meta={has_tensor} "
            f"obj_id={int(getattr(obj_meta, 'object_id', -1))} "
            f"uid={int(getattr(obj_meta, 'unique_component_id', -1))}"
        )



    def _get_sgie_top1(self, obj_meta, sgie_uid: int):
        try:
            l = obj_meta.classifier_meta_list
            while l is not None:
                cmeta = pyds.NvDsClassifierMeta.cast(l.data)
                uid = int(getattr(cmeta, "unique_component_id", -1))

                # only accept the SGIE we expect
                if uid != int(sgie_uid):
                    l = l.next
                    continue

                li = cmeta.label_info_list
                best_label = ""
                best_cid = -1
                best_prob = -1.0

                while li is not None:
                    info = pyds.NvDsLabelInfo.cast(li.data)
                    prob = float(getattr(info, "result_prob", 0.0))
                    cid = int(getattr(info, "result_class_id", -1))
                    label = (getattr(info, "result_label", "") or "").strip()

                    if prob > best_prob:
                        best_prob = prob
                        best_cid = cid
                        best_label = label

                    li = li.next

                if best_cid >= 0:
                    return best_label, best_cid, best_prob
                return None

            return None
        except Exception:
            return None

    # -------------------------
    # bus + lifecycle
    # -------------------------
    def _on_bus_message(self, bus, message, _):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f"GstError: {err} debug={debug}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().info("Gst EOS")

    def start_pipeline(self):
        self.get_logger().info("Starting DeepStream pipeline")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop_thread.start()

    def stop_pipeline(self):
        try:
            if self._appsrc is not None:
                try:
                    self._appsrc.emit("end-of-stream")
                except Exception:
                    pass
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
        finally:
            if self.loop is not None:
                try:
                    self.loop.quit()
                except Exception:
                    pass

    def destroy_node(self):
        self.stop_pipeline()
        super().destroy_node()


def main():
    rclpy.init()
    node = DeepStreamPgieSgiePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
