import numpy as np
import torch


def K_to_projection(K, H, W, n=0.001, f=10.0):
    # fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # K:[fu, 0, cu]
    #   [0, fv, cv,]
    #   [0, 0,   1]

    K2proj1 = torch.tensor(
        [[2/W, 0, -2/W],
        [0, 2/H, 2/H],
        [0, 0, 0]]
    ).cuda().float()
    K2proj2 = torch.tensor(
        [[0     ,0      ,1],
         [0     ,0      ,-1],
         [0     ,0      ,0]]
    ).cuda().float()
    proj = torch.zeros(4,4)
    proj[:3,:3] = K * K2proj1 + K2proj2
    proj[2,] = torch.as_tensor([0,          0,                     -(f + n) / (f - n),    -2 * f * n / (f - n)])
    proj[3,2] = -1

    # proj = torch.tensor([[2 * fu / W, 0,                     -2 * cu / W + 1,            0],
    #                      [0,          2 * fv / H,             2 * cv / H - 1,            0],
    #                      [0,          0,                     -(f + n) / (f - n),    -2 * f * n / (f - n)],
    #                      [0,          0,                     -1,                      0]]).cuda().float()
    return proj.cuda().float()


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]
