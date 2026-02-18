#!/usr/bin/env python3
import os
import threading
import ctypes

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String


import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds
from ament_index_python.packages import get_package_share_directory

MUXER_BATCH_TIMEOUT_USEC = 33000





def _infer_dims_to_shape(layer):
    """
    Convert NvDsInferDims to a python tuple.
    Works for most DeepStream python bindings.
    """
    d = layer.inferDims
    # d.numDims and d.d[] exist in DS python bindings
    shape = []
    for i in range(int(d.numDims)):
        shape.append(int(d.d[i]))
    # Some outputs are [C] or [1,C] etc; return as tuple
    return tuple(shape)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


class DeepStreamClassifierNode(Node):
    def __init__(self):
        super().__init__("deepstream_ros_classifier_node")

        # --- ROS params ---
        self.declare_parameter("image_topic", "/image_raw")

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)

        # streammux output (can differ from input)
        self.declare_parameter("streammux_width", 640)
        self.declare_parameter("streammux_height", 480)

        # nvinfer classifier config
        self.declare_parameter("gie_config", "classificationconfig.txt")

        # labels file (one label per line)
        self.declare_parameter("labels_file", "labels_imagenet_1k.txt")

        # publish options
        self.declare_parameter("topk", 5)

        # sink sync
        self.declare_parameter("sync", False)

        self.image_topic = self.get_parameter("image_topic").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.streammux_width = int(self.get_parameter("streammux_width").value)
        self.streammux_height = int(self.get_parameter("streammux_height").value)

        self.sync = bool(self.get_parameter("sync").value)

        self.topk = int(self.get_parameter("topk").value)

        gie_cfg_param = self.get_parameter("gie_config").value
        self.gie_config = self._resolve_pkg_config_path(gie_cfg_param)
        self.get_logger().info(f"Using GIE config: {self.gie_config}")

        labels_param = self.get_parameter("labels_file").value
        self.labels_file = self._resolve_pkg_config_path(labels_param)
        self.class_names = self._load_labels(self.labels_file)
        self.get_logger().info(f"Loaded {len(self.class_names)} labels from: {self.labels_file}")

        # pubs
        self.cls_pub = self.create_publisher(String, "/deepstream/classification", 10)

        # --- GStreamer init ---
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)

        self.pipeline = self._build_pipeline()
        self._attach_probe()

        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop_thread.start()

        # subscriber
        self.img_sub = self.create_subscription(
            Image, self.image_topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info("DeepStream classifier pipeline started.")

    def _resolve_pkg_config_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        pkg_share = get_package_share_directory("deepstream_bro")
        cfg_path = os.path.join(pkg_share, "config", path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"File not found: '{cfg_path}'. "
                f"Checked: {os.path.join(pkg_share, 'config')}"
            )
        return cfg_path

    def _load_labels(self, path: str):
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

    def _build_pipeline(self):
        pipeline = Gst.Pipeline.new("ds-cls-pipeline")
        if not pipeline:
            raise RuntimeError("Unable to create pipeline")

        # appsrc -> videoconvert -> nvvideoconvert -> capsfilter(NVMM/NV12) -> streammux -> nvinfer -> fakesink
        appsrc = Gst.ElementFactory.make("appsrc", "ros_appsrc")
        vidconv = Gst.ElementFactory.make("videoconvert", "convert_cpu")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convert_gpu")
        capsfilter = Gst.ElementFactory.make("capsfilter", "nvmm_caps")

        streammux = Gst.ElementFactory.make("nvstreammux", "streammux")
        gie = Gst.ElementFactory.make("nvinfer", "primary-classifier")
        sink = Gst.ElementFactory.make("fakesink", "sink")

        elems = [appsrc, vidconv, nvvidconv, capsfilter, streammux, gie, sink]
        if any(e is None for e in elems):
            raise RuntimeError("Failed to create one or more GStreamer elements")

        # --- appsrc caps: expect ROS rgb8 frames ---
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", True)
        appsrc.set_property("format", Gst.Format.TIME)
        appsrc.set_property("stream-type", 0)

        # keep queue small
        appsrc.set_property("block", False)
        appsrc.set_property("max-bytes", self.width * self.height * 3 * 2)

        appsrc_caps = Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={self.width},height={self.height},framerate={self.fps}/1"
        )
        appsrc.set_property("caps", appsrc_caps)

        # Force GPU for RGB conversions (VIC can’t do RGB/BGR to NV12 on Jetson reliably)
        nvvidconv.set_property("compute-hw", 1)  # 1=GPU
        nvvidconv.set_property("gpu-id", 0)

        capsfilter.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate={self.fps}/1"
            )
        )

        # streammux
        streammux.set_property("live-source", 1)
        streammux.set_property("width", self.streammux_width)
        streammux.set_property("height", self.streammux_height)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC)

        # nvinfer
        gie.set_property("config-file-path", self.gie_config)

        # sink
        sink.set_property("sync", self.sync)

        for e in elems:
            pipeline.add(e)

        if not appsrc.link(vidconv):
            raise RuntimeError("link failed: appsrc -> videoconvert")
        if not vidconv.link(nvvidconv):
            raise RuntimeError("link failed: videoconvert -> nvvideoconvert")
        if not nvvidconv.link(capsfilter):
            raise RuntimeError("link failed: nvvideoconvert -> capsfilter")

        # capsfilter -> streammux.sink_0
        sinkpad = streammux.request_pad_simple("sink_0")
        srcpad = capsfilter.get_static_pad("src")
        if sinkpad is None or srcpad is None:
            raise RuntimeError("Unable to get pads for streammux linking")
        if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("Failed to link capsfilter -> streammux.sink_0")

        if not streammux.link(gie):
            raise RuntimeError("link failed: streammux -> nvinfer")
        if not gie.link(sink):
            raise RuntimeError("link failed: nvinfer -> sink")

        # bus watch
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, None)

        self._appsrc = appsrc
        self._gie = gie
        return pipeline

    def _attach_probe(self):
        # Probe downstream of nvinfer (src pad) so tensor meta is already attached
        srcpad = self._gie.get_static_pad("src")
        if not srcpad:
            raise RuntimeError("Unable to get src pad of nvinfer")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._gie_probe, None)

    def _on_bus_message(self, bus, message, _):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f"GstError: {err} debug={debug}")
        elif t == Gst.MessageType.EOS:
            self.get_logger().info("Gst EOS")

    def _on_image(self, msg: Image):
        if msg.encoding.lower() != "rgb8":
            self.get_logger().warn(f"Expected rgb8, got {msg.encoding}")
            return
        if msg.width != self.width or msg.height != self.height:
            self.get_logger().warn(
                f"Image {msg.width}x{msg.height} != expected {self.width}x{self.height}"
            )
            return
        if msg.step != self.width * 3:
            self.get_logger().warn(f"Unexpected step={msg.step}, expected {self.width * 3}")
            return
        if self._appsrc is None:
            return

        data = bytes(msg.data)

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = int(1e9 / max(self.fps, 1))

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self.get_logger().warn(f"push-buffer returned {ret}")

    def _gie_probe(self, pad, info, _):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)

            # For primary(full-frame) inference, tensor meta is attached to frame_user_meta_list
            l_user = frame_meta.frame_user_meta_list
            probs = None

            while l_user is not None:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

                    # Take output layer 0 as "classification logits/probabilities"
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                    shape = _infer_dims_to_shape(layer)
                    # Flatten to 1D class vector as best-effort
                    n = 1
                    for s in shape:
                        n *= max(int(s), 1)

                    # DeepStream returns FP32 by default for tensor meta (common case).
                    # If your model outputs FP16/INT8, you must adjust dtype.
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    arr = np.ctypeslib.as_array(ptr, shape=(n,))
                    probs = arr.copy()  # copy out of DS buffer immediately
                    break

                l_user = l_user.next

            if probs is not None and probs.size > 0:
                # If it doesn't already look like probabilities, apply softmax.
                # (This is a safe-ish heuristic for classifier heads that output logits.)
                if np.min(probs) < 0.0 or np.max(probs) > 1.0 or abs(np.sum(probs) - 1.0) > 0.2:
                    probs = _softmax(probs)

                k = max(1, self.topk)
                top_idx = np.argsort(-probs)[:k]
                top = [(int(i), float(probs[i])) for i in top_idx]

                # Publish string
                msg = String()
                parts = []
                for cid, score in top:
                    parts.append(f"{cid}:{self._class_name(cid)}={score:.3f}")
                msg.data = " | ".join(parts)
                self.cls_pub.publish(msg)


            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK

    def destroy_node(self):
        try:
            if hasattr(self, "_appsrc") and self._appsrc is not None:
                try:
                    self._appsrc.emit("end-of-stream")
                except Exception:
                    pass
            if hasattr(self, "pipeline") and self.pipeline is not None:
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
    node = DeepStreamClassifierNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
