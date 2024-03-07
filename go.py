import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import os
from stl import mesh
import itertools
import math
from typing import List, Tuple, Literal, Dict
import random
import copy


class Metasurface:
    def __init__(self, pos, size, send_dir, recv_dir, blackhole, max_phase):
        self.pos = pos
        self.size = size
        self.send_dir = send_dir
        self.recv_dir = recv_dir
        self.blackhole = blackhole
        self.max_phase = max_phase / 180 * math.pi


class Model(nn.Module):
    def __init__(
        self,
        name: str,
        channels: Dict[str, torch.Tensor] | list[Dict[str, torch.Tensor]],
        ms_list: list[Metasurface],
        max_ms_reflex: int = 2,
        code_num: int = 60,
        w: float = 1,
        global_random: bool = False,
    ):
        """
        channels (Dict[str, torch.Tensor] | list[Dict[str, torch.Tensor]]): A dictionary of channel matrix, or a list of it. If input is a list, then a random one is chosen when forwarding.
        ms_num (int, optional): Number of metasurfaces. Defaults to 0.
        max_ms_reflex (int, optional): Maximum times for a channel to be reflected by any metasurfaces. Defaults to 2.
        code_num (int, optional): Number of codes. Defaults to 60.
        w (float, optional): Weight of the 1st loss. Defaults to 1.
        """
        super(Model, self).__init__()

        if isinstance(channels, dict):
            channels = [channels]

        self.name = name
        self.ms_list = ms_list
        self.max_ms_reflex = max_ms_reflex
        self.channels = channels
        self.loss_list = []
        self.w = w
        self.global_random = global_random

        # Init antenna config: Amplitude and phase of each antenna
        antenna_num = channels[0]["src_to_dst"].shape[0]
        self.antenna_A = torch.ones(
            code_num, antenna_num, dtype=torch.float32, device="cuda"
        ) / math.sqrt(antenna_num / 36)
        # Get rid of 4 cornors if 6x6 AP
        if antenna_num == 36:
            # Set to zero amplitude since 4 cornor don't work.
            self.antenna_A[:, [0, 5, 30, 35]] = 0

        self.antenna_theta = nn.Parameter(
            # Randomize the init phase of each antenna
            torch.rand(code_num, antenna_num, dtype=torch.float32, device="cuda")
            * 2
            * torch.pi
            # rad (0~2pi)
        )

        # Init metasurface config: Amplitude and phase of each element/unit
        self.ms_A = []
        self.ms_theta = nn.ParameterList()
        for i in range(len(ms_list)):
            unit_num = channels[0]["src_to_ms{%d}" % i].shape[1]
            self.ms_A.append(torch.ones(unit_num, dtype=torch.float32, device="cuda"))
            self.ms_theta.append(
                # Set initial phase of each surface element.
                torch.zeros(unit_num, dtype=torch.float32, device="cuda")
                # rad (0~2pi)
            )
        self.forward()

    def postprocess(self):
        # Discretize the phase of each antenna (unit pi/2)
        self.antenna_theta = nn.Parameter(
            torch.round(self.antenna_theta / (torch.pi / 2)) * (torch.pi / 2)
        )
        # Limit the phase of each metasurface element to [0, max_phase]
        for i in range(len(self.ms_list)):
            self.ms_theta[i] = self.ms_theta[i] % self.ms_list[i].max_phase
        self.forward()

    def pick_channel(self, choose_new=False):
        if self.global_random:
            if choose_new:
                self.picked_channel = random.choice(self.channels)
            return self.picked_channel
        else:
            return random.choice(self.channels)

    def get_channel(self):
        channel = self.pick_channel(True)["src_to_dst"]

        # iterate over all possible times of reflection though metasurfaces
        # e.g., if max_ms_reflex = 2, then we will iterate over 1, 2 times of reflection
        for ms_path_len in range(1, self.max_ms_reflex + 1):
            # iterate over all possible paths
            # e.g., if ms_path_len=2 and ms_num = 2, then we will iterate over (0, 1), (1, 0)
            for ms_path in itertools.product(
                range(len(self.ms_list)), repeat=ms_path_len
            ):
                # skip if there are duplicated neighbouring metasurfaces in the path
                skip = False
                for i in range(len(ms_path) - 1):
                    if ms_path[i] == ms_path[i + 1]:
                        skip = True
                        break
                if skip:
                    continue

                # calculate the channel matrix of the path by multiplying all the channel matrices

                # 1. src to the first metasurface
                ms_channel = self.pick_channel()["src_to_ms{%d}" % ms_path[0]]
                # 2. metasurface to metasurface then to dst
                for i, ms_id in enumerate(ms_path):
                    # real phases are limited by achievable phase shift range of metasurface
                    real_phase = self.ms_theta[ms_id] % self.ms_list[ms_id].max_phase
                    ms = self.ms_A[ms_id] * torch.exp(1j * real_phase)
                    ms_channel = ms_channel * ms
                    # if the last one, then multiply the channel matrix from the last metasurface to dst
                    if i == len(ms_path) - 1:
                        ms_channel = torch.matmul(
                            ms_channel,
                            self.pick_channel()["ms{%d}_to_dst" % ms_id],
                        )
                    # if not the last one, then multiply the channel matrix from the current metasurface to the next metasurface
                    else:
                        ms_channel = torch.matmul(
                            ms_channel,
                            self.pick_channel()[
                                "ms{%d}_to_ms{%d}" % (ms_id, ms_path[i + 1])
                            ]
                            if ms_id < ms_path[i + 1]
                            else self.pick_channel()[
                                "ms{%d}_to_ms{%d}" % (ms_path[i + 1], ms_id)
                            ].T,
                        )
                # Add the channel matrix of the current path to the total channel matrix
                channel = channel + ms_channel
        return channel

    def forward(self):
        # get the channel matrix
        channel = self.get_channel()
        # calculate the received signal by multiplying the AP codebook
        antenna = self.antenna_A * torch.exp(-1j * self.antenna_theta)
        recv_sig = torch.matmul(antenna, channel)

        # Get the amplitude of signal at each Rx point (size :[Tx codes, Rx points])
        recv_amp = recv_sig.abs()

        # record the result of signal amplitudes.
        self.result = recv_amp.detach().clone()

        # get the max value among all Tx weights/codes
        max_recv_amp, max_code_ids = recv_amp.max(0)
        self.max_code_ids = max_code_ids.detach().clone()

        rx_rss = recv_amp  # load Rx signal amplitude
        offset = 0  # adjust the power to match the power of real hardware device.
        # threshold = -100
        # Convert to dB scale, truncate values below the threshold
        # rx_rss = 20 * torch.log10( rx_rss*(10**(offset/20)) + 10**(threshold/20))
        rx_rss = 20 * torch.log10(rx_rss + torch.finfo(torch.float32).eps) + offset

        noise_floor = -90  # dBm
        noise_power = 10 ** (noise_floor / 10)  # mW
        rss_power = 10 ** (rx_rss / 10)  # mW

        # channel capacity base on noise floor and signal power.
        cap = torch.log2(1 + torch.div(rss_power, noise_power))

        self.cap = cap.detach().clone()

        # ========= change loss function =======
        loss_mode = "capacity"
        # ======================================
        if loss_mode == "sum":
            # set loss as the sum of all RSS?
            loss = -max_recv_amp.sum()
        elif loss_mode.find("dB") != -1:
            # sum/mean of dB power
            if loss_mode.find("mean") != -1:
                loss = -rx_rss.mean()
            elif loss_mode.find("tail") != -1:
                # Optimize sum of rx_rss values less than 70th tail value
                rx_rss.clamp_(max=rx_rss.quantile(0.3))
                loss = -rx_rss.sum()
            elif loss_mode.find("coverage") != -1:
                # Optimize sum of rx_rss values less than coverage_threshold
                coverage_threshold = -50
                rx_rss.clamp_(max=coverage_threshold)
                loss = -rx_rss.sum()

        elif loss_mode == "capacity":
            # #no need for capacity to go above this upper_limit?
            # upper_limit = torch.tensor(-50).to(device)
            # cap_threshold = torch.log2(1 + 10**(upper_limit/10) / noise_power )
            # cap.clamp_(max=cap_threshold)

            # randomly set some rows of cap to 0 and store as rand_cap
            rand_cap = cap.clone()
            rand_cap[torch.rand(cap.shape[0]) < 0.1] = 0
            # Optimize sum of capacity!
            max_cap, _ = rand_cap.max(0)
            mask = max_cap != rand_cap
            masked_cap = rand_cap * mask

            loss = -self.w * max_cap.sum() + masked_cap.norm("fro")

        ## FAILED: Add a term of loss to enforce the use of different codes.
        # 1. the sum of max/top10percents of all Tx weights/codes
        # beam_amp_sum = cap[cap.T.ge(cap.quantile(0.9, dim=1)).T].sum()
        beam_amp_sum = cap.sum() / cap.shape[0]
        beam_amp_loss = -beam_amp_sum
        # This loss makes all code similarly high and cover all regions?
        # 2. the sum of difference among different Tx weights/codes results.
        # This loss forces the some heatmaps to be very low/empty.
        # mean_cap = cap.mean(dim=0)
        # tmp_max_cap, _ = cap.max(dim=0)
        # diff_sum = (cap - tmp_max_cap.detach()).abs().sum()
        diff_sum = 0
        for i in range(cap.shape[0]):
            diff_sum += (cap - cap[i, :]).abs().sum()
        diff_loss = -diff_sum / cap.shape[0]

        self.loss = loss + beam_amp_loss * 0 + diff_loss * 0
        return self.loss


def gen_direction_from_angle(
    theta: float,
    phi: float,
    up: list[float] | torch.Tensor,
    polar_axis: list[float] | torch.Tensor,
    unit: Literal["deg", "rad"] = "deg",
) -> torch.Tensor:
    """
    Generate a direction vector from the given angle.

    Args:
        theta (float): The polar angle.
        phi (float): The azimuth angle.
        up (list | torch.Tensor): The up vector.
        polar_axis (list | torch.Tensor): The polar axis.
        unit (Literal["deg", "rad"], optional): The unit of the given angle. Defaults to "deg".

    Returns:
        torch.Tensor: The direction vector.
    """
    if unit == "deg":
        theta = theta / 180 * math.pi
        phi = phi / 180 * math.pi

    if isinstance(up, list):
        up = torch.tensor(up, dtype=torch.float32, device="cuda")
    if isinstance(polar_axis, list):
        polar_axis = torch.tensor(polar_axis, dtype=torch.float32, device="cuda")

    i_direction = F.normalize(polar_axis, dim=-1)
    j_direction = F.normalize(torch.cross(up, i_direction), dim=-1)
    k_direction = torch.cross(i_direction, j_direction)

    ijk = torch.tensor(
        [
            math.cos(theta) * math.cos(phi),
            math.cos(theta) * math.sin(phi),
            math.sin(theta),
        ],
        dtype=torch.float32,
        device="cuda",
    )

    transform = torch.stack([i_direction, j_direction, k_direction], dim=0)
    xyz = ijk @ transform
    return xyz


def optimize_model(
    model: Model,
    lr: float = 0.3,
    step: int = 200,
):
    """
    Do optimization.

    Args:
        model (Model): The model to be optimized.
        lr (float, optional): Learning rate. Defaults to 0.3.
        step (int, optional): Number of steps. Defaults to 200.

    Returns:
        Tuple[Model, List[float]]: The optimized model and the loss list.
    """

    # Build the model with channels and parameters

    paras = model.parameters()
    optimizer = optim.Adam(paras, lr=lr)

    bar = tqdm(range(step))

    best_loss = float("inf")
    # Optimize with adam optimizer.
    for _ in bar:
        loss = model()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_antenna_theta = copy.deepcopy(model.antenna_theta)
            best_ms_theta = copy.deepcopy(model.ms_theta)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        bar.set_description_str("loss: {}".format(loss.item()))
        model.loss_list.append(loss.item())

    model.antenna_theta = best_antenna_theta
    model.ms_theta = best_ms_theta
    model.forward()


def gen_pos_matrix(
    center: list[float] | torch.Tensor,
    k_direction: list[float] | torch.Tensor,
    i_direction: list[float] | torch.Tensor,
    size: Tuple[int, int],
    spacing: float,
) -> torch.Tensor:
    """
    Generate positions of each array element.

    Args:
        center: [x, y, z], the center of array
        k_direction: [x, y, z], the direction of front position of array
        i_direction: [x, y, z], the direction of down position of array
        size: [i, j], the size of array
        spacing: the spacing between each array element

    Returns:
        pos: [num, 3], the positions of each array element
    """
    size_i, size_j = size
    if type(center) is not torch.Tensor:
        center = torch.tensor(center, dtype=torch.float32, device="cuda")

    if type(i_direction) is not torch.Tensor:
        i_direction = torch.tensor(i_direction, dtype=torch.float32, device="cuda")
    if type(k_direction) is not torch.Tensor:
        k_direction = torch.tensor(k_direction, dtype=torch.float32, device="cuda")
    j_direction = torch.cross(k_direction, i_direction)

    i_direction = F.normalize(i_direction, dim=-1)
    j_direction = F.normalize(j_direction, dim=-1)
    k_direction = F.normalize(k_direction, dim=-1)

    pos_i = torch.linspace(-(size_i - 1) / 2, (size_i - 1) / 2, size_i, device="cuda")
    pos_j = torch.linspace(-(size_j - 1) / 2, (size_j - 1) / 2, size_j, device="cuda")

    pos_i, pos_j = torch.meshgrid([pos_i, pos_j], indexing="ij")
    pos_i = pos_i.flatten()
    pos_j = pos_j.flatten()
    pos_k = torch.zeros(size_i * size_j, dtype=torch.float32, device="cuda")

    pos_ijk = torch.stack([pos_i, pos_j, pos_k], dim=-1)
    transform = torch.stack([i_direction, j_direction, k_direction], dim=0)

    pos = ((pos_ijk * spacing) @ transform) + center
    return pos.reshape(size + (3,))


def gen_ms(
    center: list[float] | torch.Tensor,
    k_direction: list[float] | torch.Tensor,
    i_direction: list[float] | torch.Tensor,
    size: Tuple[int, int],
    spacing: float,
    thickness: float = 0.001,
) -> Tuple[torch.Tensor, list[list[list[float]]], torch.Tensor]:
    """
    Generate positions of each metasurface element and other properties.

    Args:
        center: [x, y, z], the center of metasurface
        k_direction: [x, y, z], the direction of front position of metasurface
        i_direction: [x, y, z], the direction of down position of metasurface
        size: [i, j], the size of metasurface
        spacing: the spacing between each metasurface element
        thickness (float, optional): thickness of metasurface. Defaults to 0.001.

    Returns:
        pos: [num, 3], the positions of each metasurface element
        blackhole: list[list[list[float]]], the blackhole list
        normal: [num, 3], the normal of each metasurface element
    """
    size_i, size_j = size
    if type(center) is not torch.Tensor:
        center = torch.tensor(center, dtype=torch.float32, device="cuda")

    if type(i_direction) is not torch.Tensor:
        i_direction = torch.tensor(i_direction, dtype=torch.float32, device="cuda")
    if type(k_direction) is not torch.Tensor:
        k_direction = torch.tensor(k_direction, dtype=torch.float32, device="cuda")
    j_direction = torch.cross(k_direction, i_direction)

    i_direction = F.normalize(i_direction, dim=-1)
    j_direction = F.normalize(j_direction, dim=-1)
    k_direction = F.normalize(k_direction, dim=-1)

    center += k_direction * thickness

    pos = gen_pos_matrix(center, k_direction, i_direction, size, spacing)

    i_offset = i_direction * size_i / 2 * spacing
    j_offset = j_direction * size_j / 2 * spacing

    blackhole_upleft = (center - i_offset - j_offset).tolist()
    blackhole_downright = (center + i_offset + j_offset).tolist()
    blackhole_upright = (center + i_offset - j_offset).tolist()
    blackhole_downleft = (center - i_offset + j_offset).tolist()

    blackhole_buffer = [
        [blackhole_upleft, blackhole_upright, blackhole_downleft],
        [blackhole_downleft, blackhole_downright, blackhole_upright],
    ]

    return pos, blackhole_buffer, k_direction


def gen_sth_from_wall(
    wall: list[list[float]],
    ij: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate center, k_direction, i_direction from wall.

    Args:
        wall (list): [v_upleft, v_downleft, v_downright], the wall.
        ij (Tuple[float, float]): [i, j], Percentage of center from the top and left of the wall.

    Returns:
        center: [x, y, z], the center of metasurface
        k_direction: [x, y, z], the direction of front position of wall
        i_direction: [x, y, z], the direction of down position of wall
    """
    up, left = ij
    v_upleft_aslist, v_downleft_aslist, v_downright_aslist = wall
    v_upleft = torch.tensor(v_upleft_aslist, dtype=torch.float32, device="cuda")
    v_downleft = torch.tensor(v_downleft_aslist, dtype=torch.float32, device="cuda")
    v_downright = torch.tensor(v_downright_aslist, dtype=torch.float32, device="cuda")

    v_upright = v_upleft + v_downright - v_downleft
    v_up = (1 - left) * v_upleft + left * v_upright
    v_down = (1 - left) * v_downleft + left * v_downright
    v = up * v_down + (1 - up) * v_up

    i_direction = F.normalize(v_downleft - v_upleft, dim=-1)
    j_direction = F.normalize(v_downright - v_downleft, dim=-1)
    k_direction = torch.cross(i_direction, j_direction)

    return v, k_direction, i_direction


def gen_pos_from_endpoint(x, y, z) -> torch.Tensor:
    """
    Generate element positions based on endpoint position.

    Args:
        x: [x_start, x_end, x_num], the start position, the end position, and the number of elements in x axis.
        y: [y_start, y_end, y_num], the start position, the end position, and the number of elements in y axis.
        z: [z_start, z_end, z_num], the start position, the end position, and the number of elements in z axis.

    Returns:
        torch.Tensor: [num, 3], the positions of each element.
    """
    pos_list = []
    num_list = []
    for arg in [x, y, z]:
        if type(arg) != list:
            arg = [arg, arg, 1]
        start, end, num = arg
        num_list.append(num)
        pos = torch.linspace(
            start,
            end,
            num,
            dtype=torch.float32,
            device="cuda",
        )
        pos_list.append(pos)

    num_list.remove(1)
    num_list += [-1]
    res = torch.meshgrid(pos_list, indexing="ij")
    return torch.stack(res, dim=-1).reshape(num_list)


def init(
    op_path: str = "./build/lib/liboptixPathTracer.so",
):
    """
    Initialize the environment.

    Args:
        op_path (str, optional): Path to the liboptixPathTracer.so. Defaults to "./build/lib/liboptixPathTracer.so".

    Raises:
        Exception: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")

    torch.ops.load_library(op_path)
    torch.manual_seed(5)


def get_ap(
    scene_folder: str,
):
    """
    Get phased array center and direction from scene folder.

    Args:
        scene_folder (str): The scene folder.

    Returns:
        center: [x, y, z], the center of phased array.
        direction: [x, y, z], the direction of phased array.
    """
    ap_file = os.path.join(scene_folder, "ap.txt")
    with open(ap_file, "r") as f:
        lines = f.readlines()
        center = [float(x) for x in lines[0].split()]
        direction = [float(x) for x in lines[1].split()]
    return center, direction


def get_doors(
    scene_folder: str,
    angle: float,
):
    """
    Get walls from scene folder.

    Args:
        scene_folder (str): The scene folder.
        angle (float): The angle of door, in rad.

    Returns:
        doors_buffer: list[list[list[float]]], the walls.
    """
    doors = []
    door_file = os.path.join(scene_folder, "door.txt")
    with open(door_file, "r") as f:
        lines = f.readlines()
        door_num = len(lines) // 4
        for i in range(door_num):
            door = []
            triangle = []
            for j in range(3):
                line = lines[4 * i + j]
                triangle.append([float(x) for x in line.split()])
            theta0 = math.atan2(
                triangle[2][1] - triangle[1][1], triangle[2][0] - triangle[1][0]
            )
            theta = theta0 + angle * (1 - 2 * int(lines[4 * i + 3]))
            triangle[2][0] = triangle[1][0] + math.cos(theta)
            triangle[2][1] = triangle[1][1] + math.sin(theta)
            door.append(triangle)

            triangle2 = copy.deepcopy(triangle)
            triangle2[1][2] = triangle2[0][2]
            triangle2[1][0] = triangle2[2][0]
            triangle2[1][1] = triangle2[2][1]
            door.append(triangle2)

            doors.append(door)
    return doors


def get_walls(
    scene_folder: str,
):
    """
    Get walls from scene folder.

    Args:
        scene_folder (str): The scene folder.

    Returns:
        walls: list[list[list[float]]], the walls.
    """
    walls = []
    wall_file = os.path.join(scene_folder, "wall.txt")
    with open(wall_file, "r") as f:
        lines = f.readlines()
        wall_num = len(lines) // 4
        for i in range(wall_num):
            wall = []
            for j in range(3):
                line = lines[4 * i + 1 + j]
                wall.append([float(x) for x in line.split()])
            walls.append(wall)
    return walls


def build_channel(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_dir: torch.Tensor,
    dst_dir: torch.Tensor,
    src_bptype: int,
    dst_bptype: int,
    wave_len: float,
    scene_folder: str,
    max_reflection: int = 4,
    dst_radius: float = 0.06,
    src_radius: float = 0.5e-5,
    samples_per_launch: int = 1,
    reserved_size: int = 100,
    blackhole_buffer: list[list[list[list[float]]]] = [],
    reflector_buffer: list[list[list[list[float]]]] = [],
    door_buffer: list[list[list[list[float]]]] = [],
    dst_gridsize: int = 5,
) -> torch.Tensor:
    """
    Build a channel.

    Args:
        src (torch.Tensor): [number of source, 3], the source position.
        dst (torch.Tensor): [number of destination, 3], the destination position.
        src_dir (torch.Tensor): [number of source, 3], the source direction.
        dst_dir (torch.Tensor): [number of destination, 3], the destination direction.
        src_bptype (int): Beampattern type of source. See get_beampattern_factor in optixPathTracer.cu for more details.
        dst_bptype (int): Beampattern type of destination. See get_beampattern_factor in optixPathTracer.cu for more details.
        wave_len (float): The wavelength of the channel.
        scene_folder (str): The folder of the scene.
        max_reflection (int, optional): The maximum number of reflections. Defaults to 4.
        dst_radius (float, optional): The radius of destination antenna. Defaults to 0.06.
        src_radius (float, optional): The radius of source antenna. Defaults to 0.5e-5.
        samples_per_launch (int, optional): The number of samples per optix launch. Defaults to 1.
        reserved_size (int, optional): The reserved size of trace item buffer for each kernel thread. Defaults to 100.
        blackhole_buffer (list, optional): The blackhole buffer. Obtained from gen_ms Defaults to [].
        reflector_buffer (list, optional): The reflector buffer. Defaults to [].
        door_buffer (list, optional): The door buffer. Defaults to [].
        dst_gridsize (int, optional): The grid size of destination. Defaults to 5.

    Returns:
        torch.Tensor: [number of source, number of destination], the channel matrix, dtype=complex64.

    """

    src_buffer = torch.cat([src - src_radius, src + src_radius], dim=-1)
    dst_buffer = torch.cat([dst - dst_radius, dst + dst_radius], dim=-1)
    result = torch.zeros(
        src.shape[0] * src.shape[1],
        dst.shape[0] * dst.shape[1],
        2,
        dtype=torch.float32,
        device="cuda",
    )

    # prepare scene stl file
    scene_files = []
    for file in os.listdir(scene_folder):
        if file.endswith(".stl"):
            scene_file = os.path.join(scene_folder, file)
            scene_files.append(scene_file)

    mat_list = []
    mat_prop_aslist = []
    with open(scene_folder + "/mat.txt", "r") as f:
        for line in f:
            line_vals = line.split()
            mat_prop_item = []
            mat_list.append(line_vals[0])
            for i in range(1, len(line_vals)):
                mat_prop_item.append(float(line_vals[i]))
            mat_prop_aslist.append(mat_prop_item)

    mat_aslist = []
    scene_buffer_list = []
    for scene_file in scene_files:
        your_mesh = mesh.Mesh.from_file(scene_file)
        mat_name = scene_file.split("_")[-1].split(".stl")[0]
        mat_id = mat_list.index(mat_name)
        for triangle in your_mesh.vectors:
            for v in triangle:
                scene_buffer_list.append([v[0], v[1], v[2]])
            mat_aslist.append(mat_id)

    if len(blackhole_buffer) > 0:
        mat_prop_aslist.append([0 for _ in range(19)])  # add mat for blackhole
        for blackhole_buffer in blackhole_buffer:
            for triangle in blackhole_buffer:
                for v in triangle:
                    scene_buffer_list.append([v[0], v[1], v[2]])
                mat_aslist.append(len(mat_list))
        mat_list.append("blackhole")

    if len(reflector_buffer) > 0:
        mat_prop_aslist.append([1 for _ in range(19)])  # add mat for reflector
        for reflector_buffer in reflector_buffer:
            for triangle in reflector_buffer:
                for v in triangle:
                    scene_buffer_list.append([v[0], v[1], v[2]])
                mat_aslist.append(len(mat_list))
        mat_list.append("reflector")

    if len(door_buffer) > 0:
        mat_prop_aslist.append([-0.7 for _ in range(19)])  # add mat for reflector
        for door_buffer in door_buffer:
            for triangle in door_buffer:
                for v in triangle:
                    scene_buffer_list.append([v[0], v[1], v[2]])
                mat_aslist.append(len(mat_list))
        mat_list.append("door")

    scene_buffer = torch.tensor(scene_buffer_list, dtype=torch.float32, device="cuda")
    mat = torch.tensor(mat_aslist, dtype=torch.int32, device="cuda")
    mat_prop = torch.tensor(mat_prop_aslist, dtype=torch.float32, device="cuda")

    launch_x = 100
    launch_y = 100
    # Call Optix-based C++ code
    torch.ops.pathtrace.torch_launch_pathtrace(
        result,
        src,
        dst,
        src_buffer,
        dst_buffer,
        scene_buffer,
        mat,
        samples_per_launch,
        launch_x,
        launch_y,
        wave_len,
        max_reflection,
        reserved_size,
        mat_prop,
        src_dir.cpu(),
        dst_dir.cpu(),
        1,
        src_bptype,
        dst_bptype,
        dst_gridsize,
        src_radius,
        dst_radius,
    )
    result = torch.complex(result[..., 0], result[..., 1])
    return result


def mark(
    ax,
    dst,
    shape,
    blackhole_buffer=[],
    reflector_buffer=[],
    door_buffer=[],
):
    if len(shape) == 2:
        shape = (1, 1) + shape
    o = dst[0, 0]
    i_len = (dst[1, 0] - dst[0, 0]).norm(p=2).cpu().numpy()
    j_len = (dst[0, 1] - dst[0, 0]).norm(p=2).cpu().numpy()
    i_dir = F.normalize(dst[1, 0] - dst[0, 0], p=2, dim=-1)
    j_dir = F.normalize(dst[0, 1] - dst[0, 0], p=2, dim=-1)
    for buffer, color in zip(
        [door_buffer, reflector_buffer, blackhole_buffer],
        ["black", "pink", "red"],
    ):
        for rectangle in buffer:
            for triangle in rectangle:
                triangle = torch.tensor(triangle, dtype=torch.float32, device="cuda")
                p0i = (triangle[0] - o).dot(i_dir).cpu().numpy() / i_len
                p0j = (triangle[0] - o).dot(j_dir).cpu().numpy() / j_len
                p1i = (triangle[1] - o).dot(i_dir).cpu().numpy() / i_len
                p1j = (triangle[1] - o).dot(j_dir).cpu().numpy() / j_len
                p2i = (triangle[2] - o).dot(i_dir).cpu().numpy() / i_len
                p2j = (triangle[2] - o).dot(j_dir).cpu().numpy() / j_len

                for ii in range(shape[0]):
                    for jj in range(shape[1]):
                        if ii * jj >= dst.shape[0] / shape[2] / shape[3]:
                            break
                        ioffset = ii * shape[2]
                        joffset = jj * shape[3]
                        ax.plot(
                            [p0j + joffset, p1j + joffset],
                            [p0i + ioffset, p1i + ioffset],
                            color=color,
                            linewidth=0.5,
                        )
                        ax.plot(
                            [p1j + joffset, p2j + joffset],
                            [p1i + ioffset, p2i + ioffset],
                            color=color,
                            linewidth=0.5,
                        )
                        ax.plot(
                            [p2j + joffset, p0j + joffset],
                            [p2i + ioffset, p0i + ioffset],
                            color=color,
                            linewidth=0.5,
                        )


def draw_channel(
    channel: torch.Tensor,
    shape: Tuple[int, int, int, int] | Tuple[int, int],
    filename: str = "result.png",
    sum_axis: int = 0,
    img_norm=None,
    cmap="jet",
    threshold: float = -150,
    offset: float = 0,
    dst=None,
    blackhole_buffer=[],
    reflector_buffer=[],
    door_buffer=[],
):
    """
    draw channel

    Args:
        channel (torch.Tensor): Channel matrx.
        shape (Tuple[int, int, int, int] | Tuple[int, int]): Shape of receivers array.
        filename (str, optional): Filename to save. Defaults to "result.png".
        sum_axis (int, optional): Dim index of antenna. Defaults to 0.
        threshold (float, optional): Min value of amplitude in dB. Defaults to -150.
    """
    channel = channel.sum(sum_axis).abs()

    image = amp_data_to_image(channel, shape, threshold=threshold, offset=offset)

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(image.cpu().numpy(), norm=img_norm, cmap=cmap)
    plt.colorbar()

    ## draw lines
    if len(blackhole_buffer) > 0 or len(reflector_buffer) > 0 or len(door_buffer) > 0:
        mark(ax, dst, shape, blackhole_buffer, reflector_buffer, door_buffer)

    plt.savefig(filename)
    plt.close()


def draw_per_code_results(
    size: Tuple[int, int, int, int] | Tuple[int, int],
    model: Model,
    output_image_folder: str = "output",
):
    """
    Draw per code results -- received signal power heatmap for each AP codeword.

    Args:
        shape (Tuple[int, int, int, int] | Tuple[int, int]): Shape of receivers array.
        model (Model): Model.
        output_image_folder (str, optional): folder to save images. Defaults to "output".
    """
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)

    print("Drawing max code id distribution ...")
    plt.figure()
    image = model.max_code_ids
    if len(size) == 4:
        image = image.reshape(-1, size[2], size[3])
        image = F.pad(
            image, (0, 0, 0, 0, 0, size[0] * size[1] - image.size(0)), "constant", 0
        )
        image = image.reshape(size).transpose(1, 2)
        image = image.reshape(size[0] * size[2], size[1] * size[3])
    else:
        image = image.reshape(size)
    plt.imshow(
        image.cpu().numpy(),
        cmap="viridis",
    )
    plt.colorbar()
    plt.savefig(os.path.join(output_image_folder, "max_code_id.png"))
    plt.close()

    print("Processing per code results...")
    rss_norm = matplotlib.colors.Normalize(vmin=-70, vmax=-20)
    for i in range(len(model.result)):
        draw_channel(
            model.result[i : i + 1, :],
            size,
            os.path.join(output_image_folder, "code#%d.png" % i),
            img_norm=rss_norm,
            cmap="viridis",
        )


def draw_metasurface_pattern(
    model: Model,
    output_image_folder: str = "output",
):
    """
    Draw metasurface pattern.

    Args:
        model (Model): Model.
        output_image_folder (str, optional): folder to save images. Defaults to 'output'.

    """
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    for i, theta in enumerate(model.ms_theta):
        plt.figure()
        plt.title("metasurface pattern")
        plt.imshow(
            (theta % (torch.pi * 2))
            .reshape(model.ms_list[i].size)
            .detach()
            .cpu()
            .numpy(),
            cmap="twilight",
        )
        plt.colorbar()
        plt.savefig(os.path.join(output_image_folder, "pattern_{%d}" % i + ".png"))
        plt.close()


def amp_data_to_image(
    data: torch.Tensor,
    shape: Tuple[int, int, int, int] | Tuple[int, int],
    threshold: float = -70,
    offset: float = 0,
):
    if len(shape) == 4:
        data = data.reshape(-1, shape[2], shape[3])
        data = F.pad(
            data, (0, 0, 0, 0, 0, shape[0] * shape[1] - data.size(0)), "constant", 0
        )
        data = data.reshape(shape).transpose(1, 2)
        data = data.reshape(shape[0] * shape[2], shape[1] * shape[3])
    else:
        data = data.reshape(shape)

    # convert to dB scale. Truncate and offset.
    data = 20 * torch.log10(data)
    data = torch.where(data.isnan(), threshold, data)
    data = (data + offset).clamp_(min=threshold)
    return data


def draw_optimization_result(
    shape: Tuple[int, int, int, int],
    models: list[Model],
    output_image_folder: str = "output",
    max_pos=None,
    threshold=-70,
    offset=0,
    dst=None,
    blackhole_buffer4=[[] for _ in range(4)],
    reflector_buffer4=[[] for _ in range(4)],
    door_buffer4=[[] for _ in range(4)],
):
    """
    Draw optimization result.

    Args:
        shape (Tuple[int, int, int, int]): Shape of receivers array.
        models (list[list[Model]]): 2x2 Model list.
        output_image_folder (str, optional): folder to save images. Defaults to 'output'.
    """

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Plot1: Loss curve with and w\o surfaces.
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.title(models[i].name)
        loss_x = torch.arange(len(models[i].loss_list))
        plt.plot(loss_x, models[i].loss_list)

    plt.savefig(os.path.join(output_image_folder, "loss.png"))
    plt.close()

    # Plot2:
    plt.figure()
    results = [model.result for model in models]
    names = [model.name for model in models]

    fig, axs = plt.subplots(3, 3)

    pos_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
    image_list = {}

    # process different results.
    for id in range(4):
        name = names[id]
        result = results[id]
        if result.ndim == 2:
            if max_pos is None:
                result, _ = result.max(0)
            else:
                _, maxid = result[:, max_pos].max(0)
                result = result[maxid]

        image = amp_data_to_image(result, shape, threshold, offset)
        # assign value to image output.
        image_list[pos_list[id]] = image

    image_list[(2, 0)] = image_list[(1, 0)] - image_list[(0, 0)]
    image_list[(2, 1)] = image_list[(1, 1)] - image_list[(0, 1)]

    image_list[(0, 2)] = image_list[(0, 1)] - image_list[(0, 0)]
    image_list[(1, 2)] = image_list[(1, 1)] - image_list[(1, 0)]

    image_list[(2, 2)] = image_list[(1, 1)] - image_list[(0, 0)]

    img_max = max(
        [image_list[(i, j)].max() for (i, j) in image_list if i != 2 and j != 2]
    )
    img_min = max(
        [image_list[(i, j)].min() for (i, j) in image_list if i != 2 and j != 2]
    )
    img_max = min(img_max, -20)  # No greater than -20
    img_min = max(img_min, -90)  # No less than -90
    diff_max = max(
        [image_list[(i, j)].max() for (i, j) in image_list if j == 2 or i == 2]
    )
    diff_min = min(
        [image_list[(i, j)].min() for (i, j) in image_list if j == 2 or i == 2]
    )
    diff_max = min(diff_max, 30)  # No greater than 30
    diff_min = max(diff_min, 0)  # greater than 0

    diff_norm = matplotlib.colors.Normalize(vmin=diff_min, vmax=diff_max)
    norm = matplotlib.colors.Normalize(vmin=img_min, vmax=img_max)

    for pos in image_list:
        name = names[pos_list.index(pos)] if pos in pos_list else ""
        image = image_list[pos]
        image = image.detach().cpu().numpy()

        axs[pos].clear()
        axs[pos].set_title(name, fontsize=8)
        if pos[0] == 2 or pos[1] == 2:
            axs[pos].imshow(image, norm=diff_norm, cmap=plt.cm.inferno)
        else:
            axs[pos].imshow(image, norm=norm, cmap=plt.cm.viridis)
            ## draw lines
            if (
                len(blackhole_buffer4[pos_list.index(pos)]) > 0
                or len(reflector_buffer4[pos_list.index(pos)]) > 0
                or len(door_buffer4[pos_list.index(pos)]) > 0
            ):
                mark(
                    axs[pos],
                    dst,
                    shape,
                    blackhole_buffer4[pos_list.index(pos)],
                    reflector_buffer4[pos_list.index(pos)],
                    door_buffer4[pos_list.index(pos)],
                )

    for i in range(3):
        for j in range(3):
            if i == 2 and j == 0:
                axs[i, j].set_title("difference", fontsize=8)
            if i == 2 and j == 1:
                axs[i, j].set_title("difference", fontsize=8)
            if j == 2:
                axs[i, j].set_title("difference", fontsize=8)
            axs[i, j].axis("off")

    # remove last printed colorbar
    for ax in fig.axes:
        if ax.name == "colorbar":
            ax.remove()

    caxpos1 = matplotlib.transforms.Bbox.from_extents(
        axs[0, 2].get_position().x1 + 0.04,
        axs[2, 2].get_position().y0,
        axs[0, 2].get_position().x1 + 0.06,
        axs[0, 2].get_position().y1,
    )
    cax1 = fig.add_axes(caxpos1)
    cax1.name = "colorbar"
    plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=diff_norm, cmap=plt.cm.inferno),
        cax=cax1,
        location="right",
    )

    caxpos2 = matplotlib.transforms.Bbox.from_extents(
        axs[0, 0].get_position().x0 - 0.06,
        axs[2, 2].get_position().y0,
        axs[0, 0].get_position().x0 - 0.04,
        axs[0, 2].get_position().y1,
    )
    cax2 = fig.add_axes(caxpos2)
    cax2.name = "colorbar"
    plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
        cax=cax2,
        location="left",
    )

    plt.savefig(os.path.join(output_image_folder, "optimization_result.png"), dpi=150)
    plt.close()
    torch.save(
        {
            "image_list": image_list,
            "names": names,
        },
        os.path.join(output_image_folder, "optimization_result.pt"),
    )


def build_channels(
    src: torch.Tensor,
    dst: torch.Tensor,
    ms_list: list[Metasurface],
    ap_dir: list[float] | torch.Tensor,
    scene_folder: str,
    wave_len: float,
    max_reflection: int,
    reflector_buffer: list[list[list[list[float]]]] = [],
    door_buffer: list[list[list[list[float]]]] = [],
) -> Dict[str, torch.Tensor]:
    """
    Build channels.

    Args:
        src (torch.Tensor): antenna positions.
        dst (torch.Tensor): receiver positions.
        ms_list (list[Metasurface]): metasurface list.
        ap_dir (list[float] | torch.Tensor): antenna direction.
        scene_folder (str): folder to load scene.
        wave_len (float): wavelength.
        max_reflection (int): max reflection number.
        reflector_buffer (list[list[list[list[float]]]], optional): reflector buffer. Defaults to [].
        door_buffer (list[list[list[list[float]]]], optional): door buffer. Defaults to [].

    Returns:
        Dict[str, torch.Tensor]: A dictionary of channel matrix.

    """
    dir_no = torch.tensor([0, 0, 0], dtype=torch.float32, device="cpu")
    if isinstance(ap_dir, list):
        ap_dir = torch.tensor(ap_dir, dtype=torch.float32, device="cpu")

    ms_num = len(ms_list)
    # store all channels
    channels = {}

    # src to dst without metasurfacess
    print("building channel: src to dst")
    channels["src_to_dst"] = build_channel(
        src,
        dst,
        ap_dir,
        dir_no,
        2,
        0,
        wave_len,
        scene_folder,
        max_reflection,
        blackhole_buffer=[ms_list[i].blackhole for i in range(ms_num)],
        reflector_buffer=reflector_buffer,
        door_buffer=door_buffer,
    )

    # metasurface related channels
    for ms_id in range(ms_num):
        print("building channel: src to ms{%d}" % ms_id)
        channels["src_to_ms{%d}" % ms_id] = build_channel(
            src,
            ms_list[ms_id].pos,
            ap_dir,
            ms_list[ms_id].recv_dir,
            2,
            1,
            wave_len,
            scene_folder,
            max_reflection,
            blackhole_buffer=[
                ms_list[i].blackhole for i in range(ms_num) if i != ms_id
            ],
            reflector_buffer=reflector_buffer,
            door_buffer=door_buffer,
        )
        print("building channel: ms{%d} to dst" % ms_id)
        channels["ms{%d}_to_dst" % ms_id] = build_channel(
            ms_list[ms_id].pos,
            dst,
            ms_list[ms_id].send_dir,
            dir_no,
            1,
            0,
            wave_len,
            scene_folder,
            max_reflection,
            blackhole_buffer=[
                ms_list[i].blackhole for i in range(ms_num) if i != ms_id
            ],
            reflector_buffer=reflector_buffer,
            door_buffer=door_buffer,
        )
        # channels between metasurfaces
        for ms_next_id in range(ms_id + 1, ms_num):
            print("building channel: ms{%d} to ms{%d}" % (ms_id, ms_next_id))
            channels["ms{%d}_to_ms{%d}" % (ms_id, ms_next_id)] = build_channel(
                ms_list[ms_id].pos,
                ms_list[ms_next_id].pos,
                ms_list[ms_id].send_dir,
                ms_list[ms_next_id].recv_dir,
                1,
                1,
                wave_len,
                scene_folder,
                max_reflection,
                blackhole_buffer=[
                    ms_list[i].blackhole
                    for i in range(ms_num)
                    if (i != ms_id) and (i != ms_next_id)
                ],
                reflector_buffer=reflector_buffer,
                door_buffer=door_buffer,
            )

    return channels


def draw_channels(
    channels: Dict[str, torch.Tensor],
    ms_list: list[Metasurface],
    dst_size: Tuple[int, int, int, int],
    output_image_folder: str = "output",
    dst=None,
    blackhole_buffer=[],
    reflector_buffer=[],
    door_buffer=[],
):
    """
    Draw channels.

    Args:
        channels (Dict[str, torch.Tensor]): A dictionary of channel matrix.
        ms_list (list[Metasurface]): metasurface list.
        dst_size (Tuple[int, int, int, int]): shape of receiver array.
        output_image_folder (str, optional): folder to save images. Defaults to 'output'.
    """
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    draw_channel(
        channels["src_to_dst"],
        dst_size,
        os.path.join(output_image_folder, "src_to_dst.png"),
        dst=dst,
        blackhole_buffer=blackhole_buffer,
        reflector_buffer=reflector_buffer,
        door_buffer=door_buffer,
    )

    # metasurface related channels
    ms_num = len(ms_list)
    for ms_id in range(ms_num):
        draw_channel(
            channels["src_to_ms{%d}" % ms_id],
            ms_list[ms_id].size,
            os.path.join(output_image_folder, "src_to_ms{%d}.png" % ms_id),
        )
        draw_channel(
            channels["ms{%d}_to_dst" % ms_id],
            dst_size,
            os.path.join(output_image_folder, "ms{%d}_to_dst.png" % ms_id),
            dst=dst,
            blackhole_buffer=blackhole_buffer,
            reflector_buffer=reflector_buffer,
            door_buffer=door_buffer,
        )
        # channels between metasurfaces
        for ms_next_id in range(ms_id + 1, ms_num):
            draw_channel(
                channels["ms{%d}_to_ms{%d}" % (ms_id, ms_next_id)],
                ms_list[ms_next_id].size,
                os.path.join(
                    output_image_folder, "ms{%d}_to_ms{%d}" % (ms_id, ms_next_id)
                ),
            )


def create_vertex_buffer(
    center,
    xlen: float = 0.2,
    ylen: float = 0.2,
    zlen: float = 1.5,
):
    triangle_buffer_offset = (
        torch.tensor(
            [
                [[-1, -1, 1], [1, -1, 1], [1, 1, 1]],  # top
                [[-1, -1, 1], [-1, 1, 1], [1, 1, 1]],
                [[-1, -1, -1], [1, -1, -1], [1, 1, -1]],  # bottom
                [[-1, -1, -1], [-1, 1, -1], [1, 1, -1]],
                [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1]],  # left
                [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1]],
                [[1, -1, -1], [1, -1, 1], [1, 1, 1]],  # right
                [[1, -1, -1], [1, 1, -1], [1, 1, 1]],
                [[-1, 1, -1], [1, 1, -1], [1, 1, 1]],  # front
                [[-1, 1, -1], [-1, 1, 1], [1, 1, 1]],
                [[-1, -1, -1], [1, -1, -1], [1, -1, 1]],  # back
                [[-1, -1, -1], [-1, -1, 1], [1, -1, 1]],
            ],
            dtype=torch.float32,
        )
        / 2
        * torch.tensor([xlen, ylen, zlen], dtype=torch.float32)
    )
    buffer = triangle_buffer_offset + torch.tensor(center, dtype=torch.float32)
    buffer = buffer.numpy().tolist()
    return buffer


if __name__ == "__main__":
    # arguments
    wave_len = 3e8 / 60e9
    max_reflection = 4
    scene_folder = "./stl/4321"
    op_path = "./build/lib/liboptixPathTracer.so"

    init(op_path)

    # Size of AP(Tx) antenna array
    src_size = (6, 6)
    # Size of receiver locations.
    dst_size_i, dst_size_j = 84, 75
    dst_size = (dst_size_i, dst_size_j)

    # generate the locations of the AP and receiver.
    src = gen_pos_matrix(
        center=[-3, 3.4, -0.51],
        k_direction=[-1, 0, 0],
        i_direction=[0, 0, -1],
        size=src_size,
        spacing=wave_len / 2,
    )
    dst = gen_pos_from_endpoint([-6, 2.4, dst_size_i], [-1, 6.5, dst_size_j], -0.51)

    channel = build_channel(
        src=src,
        dst=dst,
        src_dir=torch.tensor([-1, 0, 0], dtype=torch.float32),
        dst_dir=torch.tensor([0, 0, 0], dtype=torch.float32),
        src_bptype=2,
        dst_bptype=0,
        wave_len=wave_len,
        scene_folder=scene_folder,
        max_reflection=max_reflection,
    )

    draw_channel(channel, dst_size, "channel.png", dst=dst)
