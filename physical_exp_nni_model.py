import nni
from go import *

SINGLE_RUN = True
## manual setting param for a single test run.
# For 4321
params = {
    "ms_j": 0.11939808933765397,
    "wall_id": 1,
}

if SINGLE_RUN == False:
    params.update(nni.get_next_parameter())

trial_id = nni.get_trial_id()
seq_id = nni.get_sequence_id()
output_path = "./output/{}_{}".format(seq_id, trial_id)


# arguments
wave_len = 3e8 / 60e9
max_reflection = 4
scene_folder = "./stl/4321"

init("./build/lib/liboptixPathTracer.so")

# Size of AP(Tx) antenna array
src_size = (6, 6)
# Size of receiver locations.
# dst_size_i, dst_size_j = round(84/2), round(75/2)
dst_size_i, dst_size_j = 84, 75
dst_size = (2, 2, dst_size_i, dst_size_j)
# size of metasurface elements
ms_size = (160, 200)

# generate the orientation of the directional AP.
ap_dir = gen_direction_from_angle(0, 23.4352 - 90, [0, 0, 1], [0, 1, 0])

# Explitly define the locations of walls which can host the metasurfaces.
walls = get_walls(scene_folder)

center_a, k_dir_a, i_dir_a = gen_sth_from_wall(walls[0], (0.5, 0.5))
center_a[2] = -1.4161 + 2.5

# generate the locations of the AP and receiver.
src = gen_pos_matrix(
    center_a + k_dir_a * 0.3,
    ap_dir,
    [0, 0, -1],
    src_size,
    wave_len * 0.58,
)

dst1 = gen_pos_from_endpoint([-6, 2.4, dst_size_i], [-1, 6.5, dst_size_j], -1.4161 + 0.9)
dst2 = gen_pos_from_endpoint([-6, 2.4, dst_size_i], [-1, 6.5, dst_size_j], -1.4161 + 1.05)
dst3 = gen_pos_from_endpoint([-6, 2.4, dst_size_i], [-1, 6.5, dst_size_j], -1.4161 + 1.2)
dst = torch.cat([dst1, dst2, dst3], dim=0)

center1, k_dir1, i_dir1 = gen_sth_from_wall(
    walls[params["wall_id"]], (0.5, params["ms_j"])
)
center1[2] = -1.4161 + 2.0
ms1, bh1, n1 = gen_ms(
    center1,
    k_dir1,
    i_dir1,
    ms_size,
    wave_len / 2,
)
ms_1 = Metasurface(ms1, ms_size, n1, n1, bh1, 300)
ap_code_num = 30

channels_noms = build_channels(
    src=src,
    dst=dst,
    ms_list=[],
    ap_dir=ap_dir,
    scene_folder=scene_folder,
    wave_len=wave_len,
    max_reflection=max_reflection,
)
channels_ms = build_channels(
    src=src,
    dst=dst,
    ms_list=[ms_1],
    ap_dir=ap_dir,
    scene_folder=scene_folder,
    wave_len=wave_len,
    max_reflection=max_reflection,
)

draw_channels(channels_noms, [], dst_size, output_image_folder=output_path + "/noms")
draw_channels(channels_ms, [ms_1], dst_size, output_image_folder=output_path + "/ms")

# Optimize the metasurface pattern and AP code book.
models = [
    Model(
        "AP no code book",
        channels=channels_noms,
        ms_list=[],
        max_ms_reflex=0,
        code_num=ap_code_num,
        w=0.5,
    ),
    Model(
        "AP code book",
        channels=channels_noms,
        ms_list=[],
        max_ms_reflex=0,
        code_num=ap_code_num,
        w=0.5,
    ),
    Model(
        "AP code book + MS not optimized",
        channels=channels_ms,
        ms_list=[ms_1],
        max_ms_reflex=1,
        code_num=ap_code_num,
        w=0.5,
    ),
    Model(
        "AP code book + MS optimized",
        channels=channels_ms,
        ms_list=[ms_1],
        max_ms_reflex=1,
        code_num=ap_code_num,
        w=0.5,
    ),
]

models[2].ms_theta[0].requires_grad = False
optimize_model(models[1])
optimize_model(models[2])
optimize_model(models[3])

draw_optimization_result(
    dst_size,
    models,
    output_image_folder=output_path,
)
draw_per_code_results(dst_size, models[1], output_image_folder=output_path + "/1")
draw_per_code_results(dst_size, models[2], output_image_folder=output_path + "/2")
draw_per_code_results(dst_size, models[3], output_image_folder=output_path + "/3")

draw_metasurface_pattern(models[2], output_path + "/2")
draw_metasurface_pattern(models[3], output_path + "/3")


nni.report_final_result(models[3].loss_list[-1])
print("ok")

if SINGLE_RUN:
    torch.save(
        {
            "ms_theta0": models[3].ms_theta[0],
            # "ms_theta1": models[3].ms_theta[1],
            "ap_noms": models[1].antenna_theta,
            "ap_ms": models[3].antenna_theta,
        },
        output_path+"/ms_and_codebook.pt",
    )
