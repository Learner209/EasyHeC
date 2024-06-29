import cv2

import loguru
import numpy as np
import pyrealsense2 as rs
import sys

sys.path.append(".")
from easyhec.config import cfg_franka
from easyhec.utils.pn_utils import to_array
from easyhec.utils import render_api


class RealSenseAPI:
    pipeline, profile, align = None, None, None

    @staticmethod
    def setup_realsense():
        loguru.logger.info("Setting up RealSense")
        pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        # Start streaming
        profile = pipeline.start(config)
        profile.get_device().sensors[1].set_option(rs.option.exposure, 86)
        align_to = rs.stream.color
        align = rs.align(align_to)
        for _ in range(10):  # wait for white balance to stabilize
            frames = pipeline.wait_for_frames()
        return pipeline, profile, align

    @staticmethod
    def capture_data():
        if RealSenseAPI.pipeline is None or RealSenseAPI.profile is None or RealSenseAPI.align is None:
            RealSenseAPI.pipeline, RealSenseAPI.profile, RealSenseAPI.align = RealSenseAPI.setup_realsense()

        pipeline, profile, align = RealSenseAPI.pipeline, RealSenseAPI.profile, RealSenseAPI.align
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]])
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            break
        return rgb, K

    @staticmethod
    def capture_phoneic_data(
        cfg, qpos, fake_img_data=None, fake_camera_intrinsics=None
    ):
        """
        Capture_phoneic_data
        return parameter is: rgb(`(1080, 1920)` `bool`)
        """
        if fake_img_data is not None or fake_camera_intrinsics is not None:
            return fake_img_data, fake_camera_intrinsics

        trans_noise = np.random.normal(0, 0.002, (3))
        rotation_noise = np.random.normal(0, 0.0001, (3, 3))
        cam_pose = np.array(
            [
                [-0.99548723, -0.09197257, -0.02337417, 0.02048038],
                [0.01794234, 0.05944901, -0.9980701, 0.50420089],
                [0.09318464, -0.99398534, -0.05753053, 0.80130991],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).astype(np.float32)
        cam_pose[:3, :3] += rotation_noise
        cam_pose[:3, 3] += trans_noise.T
        cfg = cfg.model.space_explorer
        K = np.array(
            [
                [1.35220691e03, 0.00000000e00, 963.346497],
                [0.00000000e00, 1.35242883e03, 529.396179],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        height = 1080
        width = 1920
        rendered_mask = render_api.nvdiffrast_parallel_render_xarm_api(
            cfg.urdf_path,
            cam_pose,
            qpos[:7] + [0.1, 0.1],
            height,
            width,
            to_array(K),
            robot_type=0,
            return_ndarray=True,
        )
        rendered_mask = cv2.cvtColor(
            rendered_mask.astype(np.uint8) * 255, cv2.COLOR_BGR2RGB
        )
        return rendered_mask, K


if __name__ == "__main__":
    import os.path as osp

    cfg = cfg_franka
    config_file = "configs/franka/example_franka.yaml"
    cfg.merge_from_file(config_file)
    realSenseAPI = RealSenseAPI()
    print(cfg.model.rbsolver_iter.start_qpos)
    # rgb, K = realSenseAPI.capture_phoneic_data(cfg=cfg, qpos = cfg.model.rbsolver_iter.start_qpos)
    rgb, K = realSenseAPI.capture_data()
    print(f"Our K is {K}")
    cv2.imshow("photo", (rgb.astype(int) * 255).astype(np.uint8))
    cv2.waitKey(300)
