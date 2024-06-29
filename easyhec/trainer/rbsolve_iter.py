import io
import os
import os.path as osp
import subprocess
import sys
import time
import cv2

import imageio
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image
from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler
from dl_ext.timer import EvalTime
from loguru import logger
from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from easyhec.data import make_data_loader
from easyhec.modeling.build import build_model
from easyhec.modeling.models.rb_solve.space_explorer import SpaceExplorer
from easyhec.solver.build import make_optimizer, make_lr_scheduler
from easyhec.trainer.base import BaseTrainer
from easyhec.trainer.utils import *
from easyhec.utils import plt_utils
from easyhec.utils.os_utils import archive_runs
from easyhec.utils.point_drawer import PointDrawer
from easyhec.utils.realsense_api import RealSenseAPI
from easyhec.utils.vis3d_ext import Vis3D
from easyhec.utils.utils_3d import se3_log_map, se3_exp_map

from easycalib.utils.utilities import overlay_mask_on_img, render_mask


class RBSolverIterTrainer(BaseTrainer):
    def __init__(self, cfg):
        self.output_dir = cfg.output_dir
        self.num_epochs = cfg.solver.num_epochs
        self.begin_epoch = 0
        self.max_lr = cfg.solver.max_lr
        self.save_every = cfg.solver.save_every
        self.save_mode = cfg.solver.save_mode
        self.save_freq = cfg.solver.save_freq

        self.epoch_time_am = AverageMeter()
        self.cfg = cfg
        self._tb_writer = None
        self.state = TrainerState.BASE
        self.global_steps = 0
        self.best_val_loss = 100000
        self.val_loss = 100000
        self.qposes = np.array(self.cfg.model.rbsolver_iter.start_qpos)[None]
        if self.cfg.model.rbsolver_iter.use_realarm.enable is True and self.cfg.use_xarm is True:
            ip = self.cfg.model.rbsolver_iter.use_realarm.ip
            from xarm import XArmAPI
            arm = XArmAPI(ip)
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(state=0)
            self.arm = arm
        elif (
            self.cfg.use_xarm is False
            and self.cfg.model.rbsolver_iter.use_realarm.enable is True
        ):
            from easyhec.frankaAPI import MoveGroupPythonInterfaceTutorial

            self.arm = MoveGroupPythonInterfaceTutorial()
            self.arm.go_to_rest_pose()


    def train(self, epoch):
        loss_meter = AverageMeter()
        self.model.train()
        metric_ams = {}
        bar = tqdm.tqdm(self.train_dl, leave=False) if is_main_process() and len(self.train_dl) > 1 else self.train_dl
        begin = time.time()
        for batchid, batch in enumerate(bar):
            self.optimizer.zero_grad()
            batch = to_cuda(batch)
            batch['global_step'] = batchid
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            loss.backward()
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleScheduler):
                self.scheduler.step()
            reduced_loss = reduce_loss(loss)
            metrics = {}
            if 'metrics' in output:
                for k, v in output['metrics'].items():
                    reduced_s = reduce_loss(v)
                    metrics[k] = reduced_s
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']
                self.tb_writer.add_scalar('train/loss', reduced_loss.item(), self.global_steps)
                for k, v in loss_dict.items():
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                self.tb_writer.add_scalar('train/lr', lr, self.global_steps)
                if self.global_steps % 200 == 0:
                    self.image_grid_on_tb_writer(output['rendered_masks'], self.tb_writer,
                                                 'train/rendered_masks', self.global_steps)
                    self.image_grid_on_tb_writer(output['ref_masks'], self.tb_writer,
                                                 'train/ref_masks', self.global_steps)
                    self.image_grid_on_tb_writer(output['error_maps'], self.tb_writer,
                                                 "train/error_maps", self.global_steps)
                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}
                for k, v in metrics.items():
                    if k not in metric_ams.keys():
                        metric_ams[k] = AverageMeter()
                    metric_ams[k].update(v.item())
                    self.tb_writer.add_scalar(f'train/{k}', v.item(), self.global_steps)
                    bar_vals[k] = metric_ams[k].avg
                if isinstance(bar, tqdm.tqdm):
                    bar.set_postfix(bar_vals)
            self.global_steps += 1
            if self.global_steps % self.save_freq == 0:
                self.try_to_save(epoch, 'iteration')
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process() and epoch % self.cfg.solver.log_interval == 0:
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()
        return metric_ams

    def image_grid_on_tb_writer(self, images, tb_writer, tag, global_step):
        plt_utils.image_grid(images, show=False)
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        tb_writer.add_image(tag, np.array(Image.open(img_buf))[:, :, :3].transpose(2, 0, 1), global_step)
        plt.close("all")

    def do_fit(self, explore_it):
        os.makedirs(self.output_dir, exist_ok=True)
        num_epochs = self.num_epochs
        begin = time.time()

        for epoch in tqdm.trange(num_epochs):
            metric_ams = self.train(epoch)
            synchronize()
            if not self.save_every and epoch % self.cfg.solver.val_freq == 0:
                self.val_loss = self.val(epoch)
                synchronize()
            self.try_to_save(explore_it * num_epochs + epoch, "epoch")

            synchronize()
        if is_main_process():
            logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))
        return metric_ams

    def fit(self, **kwargs):
        """train the model to get the camera_2_robot extriniscMatrix. 
        Params:
            batch_imgs_paths: list of str, the paths of the images.
            qposes: list of np.ndarray, the qposes of the robot.
            camera_intrinsics: list of np.ndarray, the camera intrinsics.
            grounded_sam_func: function, the function to get the grounded sam mask.
            gt_local_to_world_matrices: list of np.ndarray, the local to world matrices.
            H: int, the height of the image.
            W: int, the width of the image.
            urdf_path: str, the path of the urdf file.
            mesh_paths: list of str, the paths of the meshes.
            local_save_path: str, the path to save the outputs.
        Note:
            batch_imgs_paths and qposes are compulsory as raw-captured images and robot-joints-positions are required to train the EasyHeC model.
            However, the rest of the parameters actually can be optional since they are not involved directly into the training proceudure.
            We include them here for debugging purposes(render the gt_mask and pred_mask).
        Returns:
            np.ndarray: the fitted Tc_c2b 4x4 Matrix.
        """
        Tc_c2b = None
        batch_imgs_paths = kwargs["batch_imgs_paths"]
        qposes = kwargs["qposes"]
        camera_intrinsics = kwargs["camera_intrinsics"]
        grounded_sam_func = kwargs["grounded_sam_func"]
        gt_local_to_world_matrices = kwargs["local_to_world_matrices"]
        H = kwargs["H"]
        W = kwargs["W"]
        urdf_path = kwargs["urdf_path"]
        mesh_paths = kwargs['mesh_paths']
        local_save_path = kwargs["local_save_path"]
        sampling_num = kwargs["sampling_num"] if "sampling_num" in kwargs else -1

        assert (
            len(batch_imgs_paths) == len(qposes) and len(batch_imgs_paths) > 0
        ), "The number of images and qposes should be the same and greater than 0, get {} and {}".format(
            len(batch_imgs_paths), len(qposes)
        )
        if self.cfg.model.rbsolver_iter.use_realarm.use_grounded_sam.enable:
            grounded_sam_func = kwargs["grounded_sam_func"]
        self.clean_data_dir()

        if sampling_num == -1:
            sampled_inds = np.arange(len(batch_imgs_paths))
        else:
            sampled_inds = np.random.choice(len(batch_imgs_paths), sampling_num, replace=False)

        for explore_it in sampled_inds:
            self.capture_data(
                batch_imgs_paths[explore_it],
                camera_intrinsics[explore_it],
                grounded_sam_func,
            )
            # Initialize camera intrinsics to avoid calling self.rebuild() accidentally initialize K.add()
            self.cfg.defrost()
            self.cfg.model.rbsolver_iter.init_K = camera_intrinsics[explore_it]
            self.cfg.freeze()
            # Initialize Tc_c2b, K if self.cfg.model.rbsolver_iter.init_Tc_c2b/K is empty.
            # Make train/test data loader, optimizer, scheduler.
            self.rebuild()
            metric_ams, Tc_c2b = self.do_fit(explore_it)
            self.cfg.defrost()
            self.cfg.model.rbsolver_iter.init_Tc_c2b = (
                Tc_c2b.detach().cpu().numpy().tolist()
            )
            self.cfg.freeze()
            for k, am in metric_ams.items():
                self.tb_writer.add_scalar("val/" + k, am.avg, explore_it)
            to_zero = explore_it == self.cfg.solver.explore_iters - 1
            self.explore_next_state(explore_it, to_zero, qposes[explore_it])
            logger.info(f"explore_it: {explore_it}, Tc_c2b: {Tc_c2b}, Camera_K: {camera_intrinsics[explore_it]}")

            # render gt mask and pred mask and overlay.
            gt_mask = render_mask(urdf_path, mesh_paths, gt_local_to_world_matrices[explore_it], np.array(camera_intrinsics[explore_it]), H, W, qposes[explore_it])
            pred_mask = render_mask(urdf_path, mesh_paths, Tc_c2b.detach().cpu().numpy(), np.array(camera_intrinsics[explore_it]), H, W, qposes[explore_it])
            overlay_img_path = os.path.join(local_save_path, "outputs", "overlay_img.png")
            overlay_img = overlay_mask_on_img(
                cv2.imread(batch_imgs_paths[explore_it]),
                gt_mask,
                pred_mask,
                rgb1=(255, 0, 255),
                rgb2=(0, 255, 255),
                alpha=0.5,
                show=True,
                save_to_disk=True,
                img_save_path=overlay_img_path,
            )

        self.reset_to_zero_qpos()

        return Tc_c2b.detach().cpu().numpy()

    def capture_data(
        self, fake_img_path=None, fake_camera_intrinsics=None, grounded_sam_func=None
    ):
        outdir = self.cfg.model.rbsolver_iter.data_dir
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(osp.join(outdir, "color"), exist_ok=True)
        os.makedirs(osp.join(outdir, "depth"), exist_ok=True)
        os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
        os.makedirs(osp.join(outdir, "qpos"), exist_ok=True)

        np.savetxt(osp.join(outdir, "Tc_c2b.txt"), np.eye(4))  # fake ground-truth Tc_c2b

        vis3d = Vis3D(
            xyz_pattern=("x", "-y", "-z"),
            out_folder="dbg",
            sequence="rbsolver_iter_realsense_capture_data",
            auto_increase=True,
            enable=True,
        )
        index = len(self.qposes) - 1
        qpose = self.qposes[-1]
        vis3d.add_xarm(qpose.tolist() + [0, 0])

        if (
            self.cfg.model.rbsolver_iter.use_realarm.enable is True
            and self.cfg.use_xarm is True
        ):
            arm = self.arm
            speed = self.cfg.model.rbsolver_iter.use_realarm.speed
            if len(self.qposes) == 1:
                arm.set_servo_angle(angle=qpose, is_radian=True, speed=speed, wait=True)
            else:
                plan_qposes = self.plan_result["position"]
                if not self.cfg.model.rbsolver_iter.use_realarm.speed_control:
                    for plan_qpose in tqdm.tqdm(plan_qposes):
                        arm.set_servo_angle(
                            angle=plan_qpose, is_radian=True, speed=speed, wait=True
                        )
                else:
                    safety_factor = (
                        self.cfg.model.rbsolver_iter.use_realarm.safety_factor
                    )
                    timestep = self.cfg.model.rbsolver_iter.use_realarm.timestep
                    arm.set_mode(4)
                    arm.set_state(state=0)
                    time.sleep(1)
                    if 'PYCHARM_HOSTED' not in os.environ:
                        input("Please visualize the next joint pose in Wis3D and press Enter to drive the robot...")
                    for ti, target_qpos in enumerate(tqdm.tqdm(plan_qposes)):
                        code, joint_state = arm.get_joint_states(is_radian=True)
                        joint_pos = joint_state[0][:7]
                        diff = target_qpos[:7] - joint_pos
                        qvel = diff / timestep / safety_factor
                        qvel_cliped = np.clip(qvel, -0.3, 0.3)
                        arm.vc_set_joint_velocity(qvel_cliped, is_radian=True, is_sync=True, duration=timestep)
                        time.sleep(timestep)
                    arm.set_mode(0)
                    time.sleep(1)
            time.sleep(self.cfg.model.rbsolver_iter.use_realarm.wait_time)
        elif (
            self.cfg.use_xarm is False
            and self.cfg.model.rbsolver_iter.use_realarm.enable is True
        ):
            if len(self.qposes) == 1:
                arm.set_servo_angle(angle=qpose)
            else:
                plan_qposes = self.plan_result['position']
                self.arm.set_servo_angle(angle = plan_qposes)

        # capture data
        qpos = (
            self.plan_result["position"]
            if hasattr(self, "plan_result")
            else self.cfg.model.rbsolver_iter.start_qpos
        )

        fake_img = cv2.imread(fake_img_path)
        fake_img_dir = osp.dirname(fake_img_path)
        fake_img_name, fake_img_ext = osp.splitext(osp.basename(fake_img_path))
        mask_dir = osp.join(fake_img_dir, "%s.d" % fake_img_name)
        os.makedirs(mask_dir, exist_ok=True)
        mask_save_path = osp.join(mask_dir, fake_img_name + "_mask" + fake_img_ext)
        cv2.imshow("fake_img", fake_img)
        cv2.waitKey(20)

        rgb, K = RealSenseAPI.capture_phoneic_data(
            cfg=self.cfg,
            qpos=qpos,
            fake_img_data=fake_img,
            fake_camera_intrinsics=fake_camera_intrinsics,
        )
        np.savetxt(osp.join(outdir, "K.txt"), K)
        self.K = K
        self.cfg.defrost()
        self.cfg.model.rbsolver_iter.H = self.cfg.model.space_explorer.height = rgb.shape[0]
        self.cfg.model.rbsolver_iter.W = self.cfg.model.space_explorer.width = rgb.shape[1]
        self.cfg.freeze()

        imageio.imwrite(osp.join(outdir, f"color/{index:06d}.png"), rgb)
        vis3d.add_image(rgb, name='img')

        if self.cfg.model.rbsolver_iter.use_realarm.enable is True:
            retcode, curr_radian = arm.get_servo_angle(is_radian=True)
            assert retcode == 0
        else:
            curr_radian = qpos[:7]
        np.savetxt(osp.join(outdir, f"qpos/{index:06d}.txt"), curr_radian)

        POINTREND_DIR = osp.join(
            osp.abspath("."), "third_party/detectron2/projects/PointRend"
        )
        pointrend_cfg_file = self.cfg.model.rbsolver_iter.pointrend_cfg_file
        pointrend_model_weight = self.cfg.model.rbsolver_iter.pointrend_model_weight
        config_file = osp.join(POINTREND_DIR, pointrend_cfg_file)
        model_weight = osp.join(POINTREND_DIR, pointrend_model_weight)
        image_path = osp.join(outdir, f"color/{index:06d}.png")

        if self.cfg.model.rbsolver_iter.use_realarm.use_grounded_sam.enable is True:
            grounded_sam_func(
                frame_save_path=fake_img_path, mask_save_path=mask_save_path
            )
            pred_binary_mask = cv2.imread(mask_save_path)

        elif self.cfg.model.rbsolver_iter.use_realarm.use_sam.enable is True:
            point_drawer = PointDrawer(
                screen_scale=1.75,
                sam_checkpoint=self.cfg.model.rbsolver_iter.use_realarm.use_sam.sam_checkpoint,
            )
            _, _, pred_binary_mask = point_drawer.run(rgb)
            pred_binary_mask = pred_binary_mask.astype(np.uint8)
        else:
            from easyhec.utils.pointrend_api import pointrend_api

            pred_binary_mask = pointrend_api(config_file, model_weight, image_path)

        pred_binary_mask = pred_binary_mask.astype(np.uint8)
        # pred_binary_mask = cv2.resize(pred_binary_mask, (1920, 1080))
        cv2.imshow("pred_binary_mask", pred_binary_mask)
        cv2.waitKey(100)

        outpath = osp.join(outdir, "mask", osp.basename(image_path))
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        imageio.imsave(outpath, pred_binary_mask)
        # tmp = plt_utils.vis_mask(rgb, pred_binary_mask.astype(np.uint8), [255, 0, 0])
        vis3d.add_image(pred_binary_mask, name="hover_pred_mask")

    def explore_next_state(self, explore_it, to_zero=False, fake_qpos=None):
        print("the fake qpos is %s" % str(fake_qpos))
        if fake_qpos is not None:
            outputs = {"qpos": fake_qpos, "plan_results": {"position": fake_qpos}}
        else:
            space_explorer = SpaceExplorer(self.cfg)
            dps = next(iter(self.train_dl))
            dps["to_zero"] = to_zero
            outputs, _ = space_explorer(dps)
            self.tb_writer.add_scalar(
                "explore/var_max", outputs["var_max"].item(), explore_it
            )
            self.tb_writer.add_scalar(
                "explore/var_min", outputs["var_min"].item(), explore_it
            )
            self.tb_writer.add_scalar(
                "explore/var_mean", outputs["var_mean"].item(), explore_it
            )
            self.tb_writer.add_scalar(
                "explore/variance", outputs["variance"].item(), explore_it
            )
        next_qpos = np.array(outputs["qpos"])
        self.qposes = np.concatenate([self.qposes, next_qpos[:7][None]], axis=0)
        plan_result = outputs["plan_results"]
        self.plan_result = plan_result

    def rebuild(self):
        """
        if self.cfg.model.rbsolve_iter.init_Tc_c2b/K is empty, call initializing routine.
        Additionally, make train/test data loader, optimizer, scheduler.
        """
        if self.cfg.model.rbsolver_iter.init_Tc_c2b == []:
            self.initialize_Tc_c2b()
        if self.cfg.model.rbsolver_iter.init_K == []:
            self.initialize_K()
        self.model: nn.Module = build_model(self.cfg).to(
            torch.device(self.cfg.model.device)
        )
        self.train_dl = make_data_loader(self.cfg, is_train=True)
        self.valid_dl = make_data_loader(self.cfg, is_train=False)
        self.optimizer = make_optimizer(self.cfg, self.model)
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer,
                                           self.cfg.solver.num_epochs * len(self.train_dl))

    def clean_data_dir(self):
        data_dir = self.cfg.model.rbsolver_iter.data_dir
        archive_runs(data_dir)
        os.makedirs(data_dir, exist_ok=True)

    @torch.no_grad()
    def get_preds(self):
        return torch.empty([])

    def reset_to_zero_qpos(self):
        if (
            self.cfg.model.rbsolver_iter.use_realarm.enable is True
            and self.cfg.use_xarm is True
        ):
            arm = self.arm
            speed = self.cfg.model.rbsolver_iter.use_realarm.speed
            plan_qposes = self.plan_result['position']
            if not self.cfg.model.rbsolver_iter.use_realarm.speed_control:
                for plan_qpose in tqdm.tqdm(plan_qposes):
                    arm.set_servo_angle(angle=plan_qpose, is_radian=True, speed=speed, wait=True)
            else:
                safety_factor = self.cfg.model.rbsolver_iter.use_realarm.safety_factor
                timestep = self.cfg.model.rbsolver_iter.use_realarm.timestep
                arm.set_mode(4)
                arm.set_state(state=0)
                time.sleep(1)
                for target_qpos in tqdm.tqdm(plan_qposes):
                    code, joint_state = arm.get_joint_states(is_radian=True)
                    joint_pos = joint_state[0][:7]
                    diff = target_qpos[:7] - joint_pos
                    qvel = diff / timestep / safety_factor
                    qvel_cliped = np.clip(qvel, -0.3, 0.3)
                    arm.vc_set_joint_velocity(qvel_cliped, is_radian=True, is_sync=True, duration=timestep)
                    time.sleep(timestep)
                arm.set_mode(0)
                time.sleep(1)
                print()
        elif self.cfg.use_xarm is False:
            self.arm.go_to_rest_pose()

    def initialize_K(self):
        output = self.cfg.model.rbsolver_iter.init_K
        init_K = np.array(output).reshape(3, 3)
        self.cfg.defrost()
        self.cfg.model.rbsolver_iter.init_K = init_K.tolist()
        self.cfg.freeze()

    def initialize_Tc_c2b(self):
        if self.cfg.use_xarm is True:
            cmd = f"cd {osp.abspath('.')}/third_party/pvnet && " \
                f"{sys.executable} run_demo_xarm7.py -c configs/xarm7/10k.yaml " \
                "demo_dir ../../data/xarm7/example" \
                " demo_pattern 'color/*png'" \
                " dbg True" \
                f" custom.K '{self.K.tolist()}'"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip().split('\n')
            nums = "".join(output[-4:]).replace("[", " ").replace("]", " ").replace(",", " ").strip().split()
            init_Tc_c2b = np.array(list((map(float, nums)))).reshape(4, 4)
        else:
            output = self.cfg.model.rbsolver_iter.init_Tc_c2b
            init_Tc_c2b = np.array(output).reshape(4, 4)
        self.cfg.defrost()
        self.cfg.model.rbsolver_iter.init_Tc_c2b = init_Tc_c2b.tolist()
        self.cfg.freeze()
