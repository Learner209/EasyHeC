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
from easycalib.utils.point_drawer import PointDrawer
from easyhec.utils.realsense_api import RealSenseAPI
from easyhec.utils.utils_3d import se3_log_map, se3_exp_map

from easycalib.utils.utilities import overlay_mask_on_img, render_mask, run_grounded_sam
from easycalib.utils.setup_logger import setup_logger

logger = setup_logger(__name__)


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

    def train(self, epoch):
        loss_meter = AverageMeter()
        self.model.train()
        metric_ams = {}
        bar = tqdm.tqdm(self.train_dl, leave=False) if is_main_process() and len(self.train_dl) > 1 else self.train_dl
        begin = time.time()
        for batchid, batch in enumerate(self.train_dl):
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
        Tc_c2b = se3_exp_map(self.model.dof[None]).permute(0, 2, 1)[0]
        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if epoch % self.cfg.solver.log_interval == 0:
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            for metric, v in metric_ams.items():
                metric_msgs.append('%s %.4f' % (metric, v.avg))
            s = ', '.join(metric_msgs)
            logger.info(s)
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()
        return metric_ams, Tc_c2b

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
        Tc_c2b = None

        for epoch in tqdm.trange(num_epochs):
            metric_ams, Tc_c2b = self.train(epoch)
            synchronize()
            if not self.save_every and epoch % self.cfg.solver.val_freq == 0:
                self.val_loss = self.val(epoch)
                synchronize()
            self.try_to_save(explore_it * num_epochs + epoch, "epoch")

            synchronize()
        if is_main_process():
            logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))
        return metric_ams, Tc_c2b

    def fit(self, **kwargs):
        """train the model to get the camera_2_robot extriniscMatrix. 
        Params:
                batch_imgs_paths: list of str, the paths of the images.
                qposes: list of np.ndarray, the qposes of the robot.
                camera_intrinsics: list of np.ndarray, the camera intrinsics.
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
        self.clean_data_dir()

        if sampling_num == -1:
            sampled_inds = np.arange(len(batch_imgs_paths))
        else:
            sampled_inds = np.random.choice(len(batch_imgs_paths), sampling_num, replace=False)

        for explore_it in sampled_inds:
            self.capture_data(
                batch_imgs_paths[explore_it],
                camera_intrinsics[explore_it],
                local_save_path
            )
            # Initialize Tc_c2b, K if self.cfg.model.rbsolver_iter.init_Tc_c2b/K is empty.
            # Make train/test data loader, optimizer, scheduler.
            self.rebuild()
            metric_ams, Tc_c2b = self.do_fit(explore_it)

            for k, am in metric_ams.items():
                self.tb_writer.add_scalar("val/" + k, am.avg, explore_it)
            to_zero = explore_it == self.cfg.solver.explore_iters - 1
            self.explore_next_state(explore_it, to_zero, qposes[explore_it])

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
                save_to_disk=False,
                img_save_path=overlay_img_path,
            )

        self.reset_to_zero_qpos()

        return Tc_c2b.detach().cpu().numpy().tolist()

    def capture_data(
            self, fake_img_path=None, fake_camera_intrinsics=None, local_save_path=None):
        outdir = self.cfg.model.rbsolver_iter.data_dir
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(osp.join(outdir, "color"), exist_ok=True)
        os.makedirs(osp.join(outdir, "depth"), exist_ok=True)
        os.makedirs(osp.join(outdir, "gt_mask"), exist_ok=True)
        os.makedirs(osp.join(outdir, "qpos"), exist_ok=True)

        np.savetxt(osp.join(outdir, "Tc_c2b.txt"), np.eye(4))  # fake ground-truth Tc_c2b

        index = len(self.qposes) - 1
        qpose = self.qposes[-1]
        # capture data
        qpos = (
            self.plan_result["position"]
            if hasattr(self, "plan_result")
            else self.cfg.model.rbsolver_iter.start_qpos
        )

        mask_save_dir = osp.join(local_save_path, "mask")
        os.makedirs(mask_save_dir, exist_ok=True)
        mask_save_path = osp.join(mask_save_dir, "%04s_mask.png" % osp.splitext(osp.basename(fake_img_path))[0])

        rgb = cv2.imread(fake_img_path)
        K = fake_camera_intrinsics

        np.savetxt(osp.join(outdir, "K.txt"), K)
        self.K = K
        self.cfg.defrost()
        self.cfg.model.rbsolver_iter.H = self.cfg.model.space_explorer.height = rgb.shape[0]
        self.cfg.model.rbsolver_iter.W = self.cfg.model.space_explorer.width = rgb.shape[1]
        self.cfg.freeze()

        imageio.imwrite(osp.join(outdir, f"color/{index:06d}.png"), rgb)

        curr_radian = qpos[:7]
        np.savetxt(osp.join(outdir, f"qpos/{index:06d}.txt"), curr_radian)
        image_path = osp.join(outdir, f"color/{index:06d}.png")

        if self.cfg.model.rbsolver_iter.use_grounded_sam.enable is True:
            pred_binary_mask = run_grounded_sam(
                frame_save_path=fake_img_path,
                mask_save_path=mask_save_path,
                text_prompt=self.cfg.model.rbsolver_iter.use_grounded_sam.text_prompt,
                grounded_sam_script=self.cfg.model.rbsolver_iter.use_grounded_sam.grounded_sam_script,
                grounded_sam_config=self.cfg.model.rbsolver_iter.use_grounded_sam.grounded_sam_config,
                grounded_sam_checkpoint_path=self.cfg.model.rbsolver_iter.use_grounded_sam.grounded_sam_checkpoint_path,
                grounded_sam_repo_path=self.cfg.model.rbsolver_iter.use_grounded_sam.grounded_sam_repo_path,
                sam_checkpoint_path=self.cfg.model.rbsolver_iter.use_grounded_sam.sam_checkpoint_path,
            )
        elif self.cfg.model.rbsolver_iter.use_realarm.use_sam.enable:
            pointdrawer = PointDrawer(
                sam_checkpoint=self.cfg.model.rbsolver_iter.use_realarm.use_sam.sam_checkpoint,
                sam_model_type=self.cfg.model.rbsolver_iter.use_realarm.use_sam.sam_type,
                window_name="Easyhec mask segm",
            )
            _, _, pred_binary_mask = pointdrawer.run(rgb)
            pred_binary_mask = (pred_binary_mask * 255).astype(np.uint8)
        else:
            raise Exception("Not a vaild method to generate mask.")

        cv2.imshow("pred_binary_mask", pred_binary_mask)
        cv2.waitKey(10)
        # cv2.destroyAllWindows()

        outpath = osp.join(outdir, "mask", osp.basename(image_path))
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        imageio.imsave(outpath, pred_binary_mask)

    def explore_next_state(self, explore_it, to_zero=False, fake_qpos=None):
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
        return

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
