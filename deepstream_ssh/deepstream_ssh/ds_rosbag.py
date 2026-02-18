#!/usr/bin/env python3
import os
import threading

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds

MUXER_BATCH_TIMEOUT_USEC = 33000


class DeepStreamRosbagAnnotator(Node):
    def __init__(self):
        super().__init__("deepstream_rosbag_annotator")

        self._gst_initialized = False
        self.pipeline = None
        self.loop = None
        self.loop_thread = None
        self._appsrc = None
        self._probe_extract_error_logged = False

        self.bridge = CvBridge()

        self.video_writer = None
        self._video_writer_failed = False
        self._video_size_mismatch_logged = False
        self._video_frame_size = None

        self._declare_params()
        self._read_params()
        self._resolve_paths()

        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._push_image_to_appsrc, qos_profile_sensor_data
        )

        self.get_logger().info(f"Subscribing: {self.image_topic}")
        self.get_logger().info(f"Output MP4:  {self.output_mp4_path}")
        self.get_logger().info(f"PGIE cfg:    {self.pgie_config}")
        self.get_logger().info(f"SGIE cfg:    {self.sgie_config}")
        self.get_logger().info(
            f"Expecting:   rgb8/bgr8/mono8/Bayer8 {self.width}x{self.height} @ {self.fps} fps"
        )
        if self.width > 0 and self.height > 0:
            if self.streammux_width <= 0:
                self.streammux_width = self.width
            if self.streammux_height <= 0:
                self.streammux_height = self.height
            self._setup_gstreamer()
            self.start_pipeline()
        else:
            self.get_logger().info(
                "Waiting for first image to auto-detect width/height before starting pipeline"
            )

    # -------------------------
    # ROS params
    # -------------------------
    def _declare_params(self):
        self.declare_parameter("image_topic", "/blackfly_0/image_raw")

        # Set width/height to 0 for rosbag auto-detection from first frame.
        self.declare_parameter("width", 0)
        self.declare_parameter("height", 0)
        self.declare_parameter("fps", 30)

        # Set streammux_* to 0 to follow input resolution by default.
        self.declare_parameter("streammux_width", 0)
        self.declare_parameter("streammux_height", 0)

        self.declare_parameter("pgie_config", "maindetector_demo.txt")
        self.declare_parameter("sgie_config", "secondclassifier_demo.txt")
        self.declare_parameter("sgie_unique_id", 2)

        self.declare_parameter("sync", False)

        self.declare_parameter("output_mp4_path", "/tmp/deepstream_annotated.mp4")
        self.declare_parameter("output_mp4_codec", "mp4v")
        self.declare_parameter("output_mp4_fps", 0.0)

    def _read_params(self):
        self.image_topic = self.get_parameter("image_topic").value

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.streammux_width = int(self.get_parameter("streammux_width").value)
        self.streammux_height = int(self.get_parameter("streammux_height").value)

        self.pgie_config_param = self.get_parameter("pgie_config").value
        self.sgie_config_param = self.get_parameter("sgie_config").value
        self.sgie_unique_id = int(self.get_parameter("sgie_unique_id").value)

        self.sync = bool(self.get_parameter("sync").value)

        self.output_mp4_path = self.get_parameter("output_mp4_path").value
        self.output_mp4_codec = str(self.get_parameter("output_mp4_codec").value)
        self.output_mp4_fps = float(self.get_parameter("output_mp4_fps").value)

    def _resolve_paths(self):
        self.pgie_config = self._resolve_config_path(self.pgie_config_param)
        self.sgie_config = self._resolve_config_path(self.sgie_config_param)

    def _resolve_config_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path

        pkg_share = get_package_share_directory("deepstream_ssh")
        cfg_path = os.path.join(pkg_share, "config", path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        return cfg_path

    # -------------------------
    # Video writer
    # -------------------------
    def _ensure_video_writer(self, width: int, height: int) -> bool:
        if self.video_writer is not None:
            return True
        if self._video_writer_failed:
            return False

        codec = self.output_mp4_codec if len(self.output_mp4_codec) == 4 else "mp4v"
        if codec != self.output_mp4_codec:
            self.get_logger().warn(
                f"output_mp4_codec='{self.output_mp4_codec}' is invalid, using 'mp4v'"
            )

        out_dir = os.path.dirname(self.output_mp4_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out_fps = self.output_mp4_fps if self.output_mp4_fps > 0.0 else float(max(self.fps, 1))
        fourcc = cv2.VideoWriter_fourcc(*codec)

        self.video_writer = cv2.VideoWriter(
            self.output_mp4_path,
            fourcc,
            out_fps,
            (int(width), int(height)),
        )

        if not self.video_writer.isOpened():
            self.video_writer = None
            self._video_writer_failed = True
            self.get_logger().error(
                f"Failed to open MP4 writer at '{self.output_mp4_path}'."
            )
            return False

        self._video_frame_size = (int(width), int(height))
        self.get_logger().info(
            f"Writing annotated MP4: {self.output_mp4_path} "
            f"({self._video_frame_size[0]}x{self._video_frame_size[1]} @ {out_fps:.2f} fps)"
        )
        return True

    def _write_annotated_frame(self, frame_bgr: np.ndarray):
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return

        h, w = frame_bgr.shape[:2]
        if not self._ensure_video_writer(w, h):
            return

        if self._video_frame_size != (w, h):
            if not self._video_size_mismatch_logged:
                self.get_logger().warn(
                    f"Annotated frame size changed to {w}x{h}; expected "
                    f"{self._video_frame_size[0]}x{self._video_frame_size[1]}."
                )
                self._video_size_mismatch_logged = True
            return

        self.video_writer.write(frame_bgr)

    def _close_video_writer(self):
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
            self.video_writer = None
            self.get_logger().info(f"Saved annotated MP4: {self.output_mp4_path}")

    # -------------------------
    # GStreamer pipeline
    # -------------------------
    def _setup_gstreamer(self):
        if not self._gst_initialized:
            Gst.init(None)
            self._gst_initialized = True

        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)

        self.pipeline = Gst.Pipeline.new("ds-rosbag-annotator")
        if not self.pipeline:
            raise RuntimeError("Unable to create pipeline")

        self.appsrc = Gst.ElementFactory.make("appsrc", "ros_appsrc")
        self.videoconvert = Gst.ElementFactory.make("videoconvert", "vcpu_convert")
        self.nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "nv_convert")
        self.caps_nvmm_nv12 = Gst.ElementFactory.make("capsfilter", "caps_nvmm_nv12")
        self.post_sgie_convert = Gst.ElementFactory.make("nvvideoconvert", "post_sgie_convert")
        self.caps_post_sgie_rgba = Gst.ElementFactory.make("capsfilter", "caps_post_sgie_rgba")

        self.streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        self.pgie = Gst.ElementFactory.make("nvinfer", "pgie")
        self.sgie = Gst.ElementFactory.make("nvinfer", "sgie")
        self.sink = Gst.ElementFactory.make("fakesink", "sink")

        elems = [
            self.appsrc,
            self.videoconvert,
            self.nvvideoconvert,
            self.caps_nvmm_nv12,
            self.streammux,
            self.pgie,
            self.sgie,
            self.post_sgie_convert,
            self.caps_post_sgie_rgba,
            self.sink,
        ]
        if any(e is None for e in elems):
            raise RuntimeError("Failed to create one or more GStreamer elements")

        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("do-timestamp", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("stream-type", 0)
        # For rosbag playback, block when downstream is busy to avoid dropping frames.
        self.appsrc.set_property("block", True)
        self.appsrc.set_property("max-bytes", self.width * self.height * 3 * 2)
        self.appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps}/1"
            ),
        )

        self.nvvideoconvert.set_property("compute-hw", 1)
        self.nvvideoconvert.set_property("gpu-id", 0)
        self.post_sgie_convert.set_property("compute-hw", 1)
        self.post_sgie_convert.set_property("gpu-id", 0)

        self.caps_nvmm_nv12.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.fps}/1"
            ),
        )
        self.caps_post_sgie_rgba.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=RGBA,width={self.streammux_width},height={self.streammux_height}"
            ),
        )

        self.streammux.set_property("live-source", 0)
        self.streammux.set_property("width", self.streammux_width)
        self.streammux.set_property("height", self.streammux_height)
        self.streammux.set_property("batch-size", 1)
        self.streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

        self.pgie.set_property("config-file-path", self.pgie_config)
        self.sgie.set_property("config-file-path", self.sgie_config)

        self.sink.set_property("sync", self.sync)

        for e in elems:
            self.pipeline.add(e)

        if not self.appsrc.link(self.videoconvert):
            raise RuntimeError("link failed: appsrc to videoconvert")
        if not self.videoconvert.link(self.nvvideoconvert):
            raise RuntimeError("link failed: videoconvert to nvvideoconvert")
        if not self.nvvideoconvert.link(self.caps_nvmm_nv12):
            raise RuntimeError("link failed: nvvideoconvert to caps_nvmm_nv12")

        sinkpad = self.streammux.request_pad_simple("sink_0")
        srcpad = self.caps_nvmm_nv12.get_static_pad("src")
        if sinkpad is None or srcpad is None:
            raise RuntimeError("Unable to get pads for streammux linking")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link caps_nvmm_nv12 to streammux sink_0")

        if not self.streammux.link(self.pgie):
            raise RuntimeError("link failed: streammux to pgie")
        if not self.pgie.link(self.sgie):
            raise RuntimeError("link failed: pgie to sgie")
        if not self.sgie.link(self.post_sgie_convert):
            raise RuntimeError("link failed: sgie to post_sgie_convert")
        if not self.post_sgie_convert.link(self.caps_post_sgie_rgba):
            raise RuntimeError("link failed: post_sgie_convert to caps_post_sgie_rgba")
        if not self.caps_post_sgie_rgba.link(self.sink):
            raise RuntimeError("link failed: caps_post_sgie_rgba to sink")

        self._appsrc = self.appsrc

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, None)

        probe_src_pad = self.caps_post_sgie_rgba.get_static_pad("src")
        if not probe_src_pad:
            raise RuntimeError("Unable to get RGBA probe src pad")
        probe_src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer_probe, None)

    # -------------------------
    # ROS image input
    # -------------------------
    def _extract_packed(self, msg: Image, channels: int):
        expected_step = self.width * channels
        if msg.step < expected_step:
            self.get_logger().warn(f"Unexpected step={msg.step}, expected at least {expected_step}")
            return None

        expected_bytes = int(msg.step) * int(self.height)
        if len(msg.data) < expected_bytes:
            self.get_logger().warn(
                f"Unexpected data length={len(msg.data)}, expected at least {expected_bytes}"
            )
            return None

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8, count=expected_bytes)
            rows = raw.reshape(self.height, msg.step)[:, :expected_step]
            packed = rows.reshape(self.height, self.width, channels)
            return packed
        except Exception as e:
            self.get_logger().warn(f"Failed to reshape packed image: {e}")
            return None

    def _bayer8_to_rgb(self, msg: Image, pattern: str):
        code_map = {
            "bggr": cv2.COLOR_BayerBG2BGR,
            "rggb": cv2.COLOR_BayerRG2BGR,
            "gbrg": cv2.COLOR_BayerGB2BGR,
            "grbg": cv2.COLOR_BayerGR2BGR,
        }
        code = code_map.get(pattern)
        if code is None:
            self.get_logger().warn(f"Unsupported Bayer pattern in encoding '{msg.encoding}'")
            return None

        if msg.step < msg.width:
            self.get_logger().warn(f"Unexpected Bayer step={msg.step}, expected at least {msg.width}")
            return None

        expected_bytes = int(msg.step) * int(msg.height)
        if len(msg.data) < expected_bytes:
            self.get_logger().warn(
                f"Unexpected Bayer data length={len(msg.data)}, expected at least {expected_bytes}"
            )
            return None

        try:
            raw = np.frombuffer(msg.data, dtype=np.uint8, count=expected_bytes)
            raw2d = raw.reshape(msg.height, msg.step)[:, : msg.width]
            rgb = cv2.cvtColor(raw2d, code)
        except Exception as e:
            self.get_logger().warn(f"Failed Bayer conversion for {msg.encoding}: {e}")
            return None

        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            self.get_logger().warn(
                f"Converted Bayer image has unexpected shape/dtype: {rgb.shape}, {rgb.dtype}"
            )
            return None

        return np.ascontiguousarray(rgb).tobytes()

    def _to_rgb8_bytes(self, msg: Image):
        encoding = (msg.encoding or "").lower()

        if encoding == "rgb8":
            rgb = self._extract_packed(msg, channels=3)
            if rgb is None:
                return None
            return np.ascontiguousarray(rgb).tobytes()

        if encoding == "bgr8":
            bgr = self._extract_packed(msg, channels=3)
            if bgr is None:
                return None
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.get_logger().warn(f"Failed to convert bgr8 to rgb8: {e}")
                return None
            return np.ascontiguousarray(rgb).tobytes()

        if encoding == "mono8":
            if msg.step < self.width:
                self.get_logger().warn(f"Unexpected mono8 step={msg.step}, expected at least {self.width}")
                return None
            expected_bytes = int(msg.step) * int(self.height)
            if len(msg.data) < expected_bytes:
                self.get_logger().warn(
                    f"Unexpected mono8 data length={len(msg.data)}, expected at least {expected_bytes}"
                )
                return None
            try:
                gray = np.frombuffer(msg.data, dtype=np.uint8, count=expected_bytes)
                gray = gray.reshape(self.height, msg.step)[:, : self.width]
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            except Exception as e:
                self.get_logger().warn(f"Failed to convert mono8 to rgb8: {e}")
                return None
            return np.ascontiguousarray(rgb).tobytes()

        if encoding.startswith("bayer_") and encoding.endswith("8"):
            pattern = encoding[len("bayer_") : -1]
            direct_rgb = self._bayer8_to_rgb(msg, pattern)
            if direct_rgb is not None:
                return direct_rgb
            # Fallback to cv_bridge if the incoming encoding string is unusual
            try:
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                return np.ascontiguousarray(rgb).tobytes()
            except Exception as e:
                self.get_logger().warn(f"Failed to convert {msg.encoding} to rgb8: {e}")
                return None

        self.get_logger().warn(
            f"Unsupported encoding '{msg.encoding}'. Expected rgb8/bgr8/mono8/Bayer8 encoding."
        )
        return None

    def _push_image_to_appsrc(self, msg: Image):
        if self._appsrc is None:
            if msg.width <= 0 or msg.height <= 0:
                self.get_logger().warn(
                    f"Invalid incoming image size {msg.width}x{msg.height}; cannot start pipeline."
                )
                return
            self.width = int(msg.width)
            self.height = int(msg.height)
            if self.streammux_width <= 0:
                self.streammux_width = self.width
            if self.streammux_height <= 0:
                self.streammux_height = self.height
            self.get_logger().info(
                f"Auto-detected input size from first frame: "
                f"{self.width}x{self.height}, streammux={self.streammux_width}x{self.streammux_height}"
            )
            try:
                self._setup_gstreamer()
                self.start_pipeline()
            except Exception as e:
                self.get_logger().error(f"Failed to start pipeline after auto-detect: {e}")
                return

        if msg.width != self.width or msg.height != self.height:
            self.get_logger().warn(
                f"Image {msg.width}x{msg.height} does not match expected {self.width}x{self.height}"
            )
            return

        data = self._to_rgb8_bytes(msg)
        if data is None:
            return

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = int(1e9 / max(self.fps, 1))

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self.get_logger().warn(f"push-buffer returned {ret}")

    # -------------------------
    # Annotation
    # -------------------------
    def _annotate_and_write(self, rgb: np.ndarray, ann_dets):
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        for x1, y1, x2, y2, det_label, det_score, cls_label, cls_score in ann_dets:
            x1i = int(max(0, min(w - 1, round(x1))))
            y1i = int(max(0, min(h - 1, round(y1))))
            x2i = int(max(0, min(w - 1, round(x2))))
            y2i = int(max(0, min(h - 1, round(y2))))

            cv2.rectangle(bgr, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)

            det_text = f"Det:{det_label} {det_score:.2f}"
            cls_text = f"Cls:{cls_label} {cls_score:.2f}" if cls_label else "Cls:n/a"
            text = f"{det_text} | {cls_text}"

            ty = y1i - 8 if y1i >= 16 else y1i + 14
            cv2.putText(
                bgr,
                text,
                (x1i + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        self._write_annotated_frame(bgr)

    def _annotate_from_probe(self, gst_buffer, frame_meta):
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            rgba = np.array(n_frame, copy=True, order="C")
        except Exception as e:
            if not self._probe_extract_error_logged:
                self.get_logger().warn(f"Failed to extract frame from probe buffer: {e}")
                self._probe_extract_error_logged = True
            return

        if rgba.ndim != 3 or rgba.shape[2] != 4:
            if not self._probe_extract_error_logged:
                self.get_logger().warn(
                    f"Unexpected probe frame shape for RGBA annotation: {getattr(rgba, 'shape', None)}"
                )
                self._probe_extract_error_logged = True
            return

        try:
            rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        except Exception as e:
            if not self._probe_extract_error_logged:
                self.get_logger().warn(f"Failed RGBA->RGB conversion: {e}")
                self._probe_extract_error_logged = True
            return
        self._probe_extract_error_logged = False

        ann_dets = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)

            x1 = float(obj.rect_params.left)
            y1 = float(obj.rect_params.top)
            w = float(obj.rect_params.width)
            h = float(obj.rect_params.height)

            det_label = (getattr(obj, "obj_label", "") or "").strip()
            if not det_label:
                det_label = str(int(obj.class_id))

            cls_label = ""
            cls_prob = 0.0
            cls = self._get_sgie_top1(obj, self.sgie_unique_id)
            if cls is not None:
                label, cid, prob = cls
                cls_label = label if label else str(int(cid))
                cls_prob = float(prob)

            ann_dets.append(
                (
                    x1,
                    y1,
                    x1 + w,
                    y1 + h,
                    det_label,
                    float(obj.confidence),
                    cls_label,
                    cls_prob,
                )
            )

            l_obj = l_obj.next

        self._annotate_and_write(rgb, ann_dets)

    def _get_sgie_top1(self, obj_meta, sgie_uid: int):
        try:
            l = obj_meta.classifier_meta_list
            while l is not None:
                cmeta = pyds.NvDsClassifierMeta.cast(l.data)
                uid = int(getattr(cmeta, "unique_component_id", -1))
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

    def _on_buffer_probe(self, pad, info, _):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            self._annotate_from_probe(gst_buffer, frame_meta)
            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    # -------------------------
    # lifecycle
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
            self._close_video_writer()

    def destroy_node(self):
        self.stop_pipeline()
        super().destroy_node()


def main():
    rclpy.init()
    node = DeepStreamRosbagAnnotator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
