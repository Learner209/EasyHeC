import os
from yacs.config import CfgNode as CN

_C = CN()

_C.dbg = True
_C.evaltime = False
_C.deterministic = False
_C.backup_src = True
_C.use_xarm = False

_C.sim_mask_data = CN()
_C.sim_mask_data.width = 1920  # width of the rendered image
_C.sim_mask_data.height = 1080  # height of the rendered image
_C.sim_mask_data.K = [
    [1.35220691e03, 0.00000000e00, 9.63346497e02],
    [0.00000000e00, 1.35242883e03, 5.29396179e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]
_C.sim_mask_data.outdir = ""  # output directory of the rendered data
_C.sim_mask_data.random_qpos_number = 100000  # number of random qpos to sample
_C.sim_mask_data.envmaps = []  # list of envmap names
_C.sim_mask_data.n_point_light = 10
_C.sim_mask_data.sample_dof = 7  # number of degree of freedom to sample, should be less than the dof of the robot.
_C.sim_mask_data.urdf_path = "assets/franka/urdf/franka.urdf"  # path to the urdf file, used for rendering and collision checking
_C.sim_mask_data.srdf_path = "assets/franka/urdf/panda.srdf"  # path to the srdf file, used for collision checking
_C.sim_mask_data.move_group = ""  # move group name, used for collision checking
_C.sim_mask_data.add_desk_cube = CN()
_C.sim_mask_data.add_desk_cube.enable = True
_C.sim_mask_data.add_desk_cube.half_size = [0.5, 0.5, 0.5]
_C.sim_mask_data.add_desk_cube.pose = [-0.4, -0.1, -0.5]
_C.sim_mask_data.add_desk_cube.color = [0, 0, 0]

_C.sim_pvnet_data = CN()
_C.sim_pvnet_data.K = [
    [1.35220691e03, 0.00000000e00, 9.63346497e02],
    [0.00000000e00, 1.35242883e03, 5.29396179e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]
_C.sim_pvnet_data.width = 1920  # width of the rendered image
_C.sim_pvnet_data.height = 1080  # height of the rendered image
_C.sim_pvnet_data.min_dist = 0.5
_C.sim_pvnet_data.max_dist = 2.0
_C.sim_pvnet_data.n_dist = 10
_C.sim_pvnet_data.min_elev = 0
_C.sim_pvnet_data.max_elev = 80
_C.sim_pvnet_data.n_elev = 10
_C.sim_pvnet_data.nazim = 30
_C.sim_pvnet_data.trans_noise = [0.03] * 3
_C.sim_pvnet_data.n_point_light = 10
_C.sim_pvnet_data.envmaps = ["SaintPetersSquare2"]
_C.sim_pvnet_data.outdir = ""
_C.sim_pvnet_data.urdf_path = "assets/franka/urdf/franka_another.urdf"
_C.sim_pvnet_data.add_desk_cube = CN()
_C.sim_pvnet_data.add_desk_cube.enable = False
_C.sim_pvnet_data.add_desk_cube.half_size = [0.5, 0.5, 0.5]
_C.sim_pvnet_data.add_desk_cube.color = [0, 0, 0]
_C.sim_pvnet_data.add_desk_cube.pose = [-0.3, 0, -0.5]

_C.output_dir = ""

_C.dataset = CN()
_C.dataset.xarm_real = CN()
_C.dataset.xarm_real.urdf_path = "assets/franka/urdf/franka_another.urdf"
_C.dataset.xarm_real.use_links = [1, 2, 3, 4, 5, 6, 7]

_C.model = CN()
_C.model.meta_architecture = ""
_C.model.device = "cuda"

_C.model.space_explorer = CN()
_C.model.space_explorer.qpos_choices_pad_left = 0
_C.model.space_explorer.qpos_choices_pad_right = 0

_C.model.space_explorer.start = 40
_C.model.space_explorer.sample = 10  # num camera sampled
_C.model.space_explorer.n_sample_qposes = 100  # for random
_C.model.space_explorer.sample_dof = 6
_C.model.space_explorer.width = 1920
_C.model.space_explorer.height = 1080
_C.model.space_explorer.K = [
    [1.35220691e03, 0.00000000e00, 9.63346497e02],
    [0.00000000e00, 1.35242883e03, 5.29396179e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]
_C.model.space_explorer.urdf_path = ""
_C.model.space_explorer.srdf_path = ""
_C.model.space_explorer.move_group = ""

_C.model.space_explorer.self_collision_check = CN()
_C.model.space_explorer.self_collision_check.enable = True

_C.model.space_explorer.max_dist_constraint = CN()
_C.model.space_explorer.max_dist_constraint.enable = False
_C.model.space_explorer.max_dist_constraint.max_dist = 0.5
_C.model.space_explorer.max_dist_constraint.max_dist_center_compute_n = 20000

_C.model.space_explorer.collision_check = CN()
_C.model.space_explorer.collision_check.enable = True
_C.model.space_explorer.collision_check.timestep = 0.01
_C.model.space_explorer.collision_check.planning_time = 1.0

_C.model.rbsolver_iter = CN()
_C.model.rbsolver_iter.data_dir = ""
_C.model.rbsolver_iter.lrs = []
_C.model.rbsolver_iter.start_qpos = [
    5.928617003472516e-05,
    -0.7848036409260933,
    -0.000308854746172659,
    -2.357726806912310,
    -0.00011798564528483742,
    1.570464383098814,
    0.7852387161304554,
]
_C.model.rbsolver_iter.start_index = 0  # for data pool
_C.model.rbsolver_iter.mesh_paths = [
    "assets/franka/franka_description/meshes/visual/link1.dae",
    "assets/franka/franka_description/meshes/visual/link2.dae",
    "assets/franka/franka_description/meshes/visual/link3.dae",
    "assets/franka/franka_description/meshes/visual/link4.dae",
    "assets/franka/franka_description/meshes/visual/link5.dae",
    "assets/franka/franka_description/meshes/visual/link6.dae",
    "assets/franka/franka_description/meshes/visual/link7.dae",
]
_C.model.rbsolver_iter.urdf_path = "assets/franka/urdf/franka_another.urdf"

_C.model.rbsolver_iter.init_K = []
_C.model.rbsolver_iter.init_Tc_c2b = []

_C.model.rbsolver_iter.use_realarm = CN()
_C.model.rbsolver_iter.use_realarm.enable = False
_C.model.rbsolver_iter.use_realarm.ip = "192.168.1.209"
_C.model.rbsolver_iter.use_realarm.speed = 0.1
_C.model.rbsolver_iter.use_realarm.wait_time = 1
_C.model.rbsolver_iter.use_realarm.speed_control = False
_C.model.rbsolver_iter.use_realarm.timestep = 0.1
_C.model.rbsolver_iter.use_realarm.safety_factor = 3
_C.model.rbsolver_iter.use_realarm.use_sam = CN()
_C.model.rbsolver_iter.use_realarm.use_sam.enable = True  # use SAM to predict mask
_C.model.rbsolver_iter.use_realarm.use_sam.sam_type = "vit_h"  # use SAM to predict mask
_C.model.rbsolver_iter.use_realarm.use_sam.sam_checkpoint = (
    "pretrained_checkpoints/sam_vit_h_4b8939.pth"
)
_C.model.rbsolver_iter.use_grounded_sam = CN()
_C.model.rbsolver_iter.use_grounded_sam.enable = (
    False  # use SAM to predict mask
)
_C.model.rbsolver_iter.use_grounded_sam.text_prompt = None
_C.model.rbsolver_iter.use_grounded_sam.grounded_sam_script = None
_C.model.rbsolver_iter.use_grounded_sam.grounded_sam_config = None
_C.model.rbsolver_iter.use_grounded_sam.grounded_sam_checkpoint_path = None
_C.model.rbsolver_iter.use_grounded_sam.grounded_sam_repo_path = None
_C.model.rbsolver_iter.use_grounded_sam.sam_checkpoint_path = None

_C.model.rbsolver_iter.pointrend_cfg_file = "configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_xarm.yaml"
_C.model.rbsolver_iter.pointrend_model_weight = "output/model_0099999.pth"

_C.solver = CN()
_C.solver.explore_iters = 10  # explore iterations for rendering-based solver
_C.solver.num_epochs = 500
_C.solver.max_lr = 0.001
_C.solver.end_lr = 0.0001
_C.solver.bias_lr_factor = 1
_C.solver.momentum = 0.9
_C.solver.weight_decay = 0.0005
_C.solver.weight_decay_bias = 0.0
_C.solver.gamma = 0.1
_C.solver.lrate_decay = 250
_C.solver.steps = (30000,)
_C.solver.warmup_factor = 1.0 / 3
_C.solver.warmup_iters = 500
_C.solver.warmup_method = "linear"
_C.solver.num_iters = 1000
_C.solver.min_factor = 0.1
_C.solver.log_interval = 1

_C.solver.optimizer = 'Adam'
_C.solver.scheduler = 'ConstantScheduler'
_C.solver.scheduler_decay_thresh = 0.00005
_C.solver.do_grad_clip = False
_C.solver.grad_clip_type = 'norm'  # norm or value
_C.solver.grad_clip = 1.0
_C.solver.ds_len = -1
_C.solver.batch_size = 1
_C.solver.loss_function = ''
#### save ckpt configs#####
_C.solver.save_min_loss = 20.0
_C.solver.save_every = False
_C.solver.save_freq = 1
_C.solver.save_mode = 'epoch'  # epoch or iteration
_C.solver.val_freq = 1
_C.solver.save_last_only = False
_C.solver.empty_cache = True
_C.solver.trainer = "base"
_C.solver.load_model = ""
_C.solver.load = ""
_C.solver.broadcast_buffers = False
_C.solver.find_unused_parameters = False
_C.solver.resume = False
_C.solver.dist = CN()
_C.solver.save_optimizer = True
_C.solver.save_scheduler = True

_C.dataloader = CN()
_C.dataloader.num_workers = 0
_C.dataloader.collator = 'DefaultBatchCollator'
_C.dataloader.pin_memory = False

_C.datasets = CN()
_C.datasets.train = ()
_C.datasets.test = ""

_C.input = CN()
_C.input.transforms = []
_C.input.shuffle = True

_C.paths_catalog = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

_C.test = CN()
_C.test.batch_size = 1
_C.test.evaluators = []
_C.test.visualizer = ''
_C.test.force_recompute = True
_C.test.do_evaluation = False
_C.test.do_visualization = False
_C.test.save_predictions = False
