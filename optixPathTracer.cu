#include <optix.h>
#include <sutil/vec_math.h>

#include "optixPathTracer.h"
#include "random.h"

extern "C" {
__constant__ Params params;
}
class PRD {
public:
  float3 origin;
  float3 direction;
  uint64_t src;
  int success = -1;
  uint ttl;
  uint64_t is_reflect;
};

class ForwardPRD : public PRD {
public:
  sutil::Matrix3x4 image_transform = sutil::Matrix3x4::affineIdentity();
  ForwardPRD *prd_group;
};
class BackwardPRD : public PRD {
public:
  uint64_t dst;
  uint64_t id;
  int supposed_hit_cnt;
  float amplitude;
  float phase;
  float far_dist;
  float d;
  float d_reserve;
};

static __forceinline__ __device__ void *unpackPointer(uint i0, uint i1) {
  const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
  void *ptr = reinterpret_cast<void *>(uptr);
  return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, uint &i0,
                                                   uint &i1) {
  const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ PRD *getPRD() {
  const uint u0 = optixGetPayload_0();
  const uint u1 = optixGetPayload_1();
  return reinterpret_cast<PRD *>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ float3 rand_dir(uint &seed, uint id_x,
                                                  uint id_y) {
  float rnd1 = (rnd(seed) + id_x) / params.launch_x;
  float rnd2 = (rnd(seed) + id_y) / params.launch_y;
  float rand_theta = 2 * M_PIf * rnd1;
  float rand_r = sqrtf(rnd2);
  float rand_u = rand_r * cosf(rand_theta);
  float rand_v = rand_r * sinf(rand_theta);
  float x = 2 * rand_u * sqrtf(1 - rand_r * rand_r);
  float y = 2 * rand_v * sqrtf(1 - rand_r * rand_r);
  float z = 1 - 2 * rand_r * rand_r;
  return make_float3(x, y, z);
}

static __forceinline__ __device__ void
updateImageTransform(sutil::Matrix3x4 &mat, float3 N, float3 A) {
  auto m1 = sutil::Matrix3x2{0, N.x, 0, N.y, 0, N.z};
  auto m2 = m1.transpose();
  mat = (sutil::Matrix3x3::identity() - 2 * m1 * m2) * mat;
  mat.setCol(3, mat.getCol(3) + 2 * dot(A, N) * N);
}

static __forceinline__ __device__ float3 getImage(sutil::Matrix3x4 mat,
                                                  float3 p) {
  return mat * make_float4(p.x, p.y, p.z, 1);
}

static __forceinline__ __device__ int dst_to_grid(int dst) {
  int dst_0 = dst / params.n1_dst;
  int dst_1 = dst % params.n1_dst;
  int grid_0 = dst_0 / params.dst_gridsize;
  int grid_1 = dst_1 / params.dst_gridsize;
  int grid_idx = grid_0 * params.n1_dst_grid + grid_1;
  return grid_idx;
}

static __forceinline__ __device__ void rayTrace(OptixTraversableHandle handle,
                                                float3 ray_origin,
                                                float3 ray_direction,
                                                bool forward, PRD *prd) {
  uint u0, u1;
  packPointer(prd, u0, u1);
  optixTrace(handle, ray_origin, ray_direction, 1e-5f, 1e16f,
             0.0f, // rayTime
             OptixVisibilityMask(forward ? 5 : 3), OPTIX_RAY_FLAG_NONE,
             forward ? RAY_TYPE_FORWARD : RAY_TYPE_BACKWARD, // SBT offset
             RAY_TYPE_COUNT,                                 // SBT stride
             forward ? RAY_TYPE_FORWARD : RAY_TYPE_BACKWARD, // missSBTIndex
             u0, u1);
}

static __forceinline__ __device__ float
get_beampattern_factor(float3 dir1, float3 dir2, int type) {
  float cos_theta = dot(dir1, dir2) / (length(dir1) * length(dir2));
  switch (type) {
  case 0:
    return 1.0f;
  case 1:
    if (cos_theta < 0.f)
      return 0.f;
    else
      return cos_theta * 2.4495f;
  case 2: {
    float theta = acosf(cos_theta);
    if (theta > M_PIf * 2 / 3)
      return 0.1f;
    else if (theta > M_PIf * 5 / 12)
      return powf(10.f, (M_PIf * 2 / 3 - theta) / M_PIf * 4 *
                                (log10(cosf(M_PIf * 5 / 12)) + 1) -
                            1) *
             2.4495f;
    else
      return cos_theta * 2.4495;
  }
  case 3:
    if (cos_theta < 0.f)
      return 0.f;
    else
      return 1.4142f;
  }
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dims = optixGetLaunchDimensions();
  int stage = params.stage;
  if (stage == 1) {
    uint seed = tea<4>(idx.x * params.launch_y + idx.y, 0);
    int launch_cnt = params.samples_per_launch;
    int res_cnt = 0;
    int offset = idx.x * params.launch_y + idx.y;
    ForwardRecord *tmp_result =
        params.tmp_results + offset * params.reserve_size;
    int *min_hc = params.min_hc + offset * params.n_dst_grid;
    while (launch_cnt--) {
      const uint src = rnd(seed) * params.n_src;
      ForwardPRD prd[1 << MAX_HIT_N];
      for (int i = 0; i < 1 << MAX_HIT_N; i++)
        prd[i].ttl = 0;
      prd[0].prd_group = prd;
      prd[0].is_reflect = 0;
      prd[0].ttl = params.max_reflection + 1;
      prd[0].src = src;
      prd[0].origin = params.src[src];
      prd[0].direction = rand_dir(seed, idx.x, idx.y);
      bool goflag;
      do {
        goflag = false;
        for (int i = 0; i < (1 << MAX_HIT_N); i++) {
          while (prd[i].ttl > 0) {
            goflag = true;
            rayTrace(params.handle, prd[i].origin, prd[i].direction, true,
                     &prd[i]);
            if (prd[i].success == -1)
              continue;
            int hit_dst = prd[i].success;
            prd[i].success = -1;
            if (res_cnt < params.reserve_size) {
              int grid_idx = dst_to_grid(hit_dst);
              int hit_cnt = params.max_reflection + 1 - prd[i].ttl;
              if (min_hc[grid_idx] <= hit_cnt) {
                min_hc[grid_idx] = hit_cnt + 1;
                tmp_result[res_cnt].path.hit_cnt = hit_cnt;
                tmp_result[res_cnt].dst_grid_idx = grid_idx;
                tmp_result[res_cnt].path.image_transform =
                    prd[i].image_transform;
                tmp_result[res_cnt].path.is_reflect = prd[i].is_reflect;
                res_cnt++;
              }
            }
          }
        }
      } while (goflag);
    }
    params.tmp_result_counts[idx.x * params.launch_y + idx.y] = res_cnt;
  } else {
    const int src = idx.x;
    const int dstPerGrid = params.dst_gridsize * params.dst_gridsize;
    const int nDetailTask = params.traceTaskLen * dstPerGrid;
    const int startTask = (long long)nDetailTask * idx.y / dims.y;
    const int endTask =
        min(nDetailTask, (int)((long long)nDetailTask * (idx.y + 1) / dims.y));
    for (int taskId = startTask; taskId < endTask; taskId++) {
      uint64_t globalTaskId = nDetailTask * src + taskId;
#ifdef EXPORT_TASKINFO
      params.taskInfo[globalTaskId].successful = false;
#endif
      int gridTaskID = taskId / dstPerGrid;
      if (gridTaskID >= params.traceTaskLen)
#ifdef EXPORT_TASKINFO
        continue;
#else
        break;
#endif
      int grid = params.traceTasks[gridTaskID].dst_grid_idx;
      int grid_i = grid / params.n1_dst_grid;
      int grid_j = grid % params.n1_dst_grid;
      int inner_dst = taskId % dstPerGrid;
      int inner_dst_i = inner_dst / params.dst_gridsize;
      int inner_dst_j = inner_dst % params.dst_gridsize;
      int dst_i = inner_dst_i + grid_i * params.dst_gridsize;
      int dst_j = inner_dst_j + grid_j * params.dst_gridsize;
      int dst = dst_i * params.n1_dst + dst_j;
      if (dst >= params.n_dst)
        continue;
      int pathId = params.traceTasks[gridTaskID].path_idx;

      BackwardPRD prd;
      prd.id = globalTaskId;
      prd.ttl = params.pathList[pathId].hit_cnt + 1;
      prd.supposed_hit_cnt = params.pathList[pathId].hit_cnt;
      prd.is_reflect = params.pathList[pathId].is_reflect;
      prd.src = src;
      prd.dst = dst;
      prd.origin = params.dst[dst];
      prd.direction = normalize(
          getImage(params.pathList[pathId].image_transform, params.src[src]) -
          prd.origin);
      float beam_pattern_factor = get_beampattern_factor(
          prd.direction, params.dst_dir, params.dst_bptype);

      prd.amplitude = beam_pattern_factor;
      prd.phase = 0.0f;
      prd.far_dist = 1e6;
      prd.d = 0.f;
      prd.d_reserve = 0.f;

#ifdef EXPORT_TASKINFO
      params.taskInfo[globalTaskId].src = src;
      params.taskInfo[globalTaskId].dst = dst;
      params.taskInfo[globalTaskId].receive_angle =
          dot(prd.direction, params.dst_dir) /
          (length(prd.direction) * length(params.dst_dir));
      params.taskInfo[globalTaskId].loss = 1;
      params.taskInfo[globalTaskId].hit_cnt = 0;
#endif

      while (prd.ttl > 0)
        rayTrace(params.handle, prd.origin, prd.direction, false, &prd);

      if (prd.success >= 0) {
        atomicAdd(&(params.stats[0]), 1);

        uint64_t result_offset = ((uint64_t)src * params.n_dst + dst) * 2;

        float *real_addr = &params.result[result_offset];
        float *image_addr = &params.result[result_offset + 1];
        atomicAdd(real_addr, prd.amplitude * cosf(prd.phase));
        atomicAdd(image_addr, -prd.amplitude * sinf(prd.phase));
        params.taskSuccess[gridTaskID] = true;
#ifdef EXPORT_TASKINFO
        params.taskInfo[globalTaskId].successful = true;
#endif
      } else {
        atomicAdd(&(params.stats[1]), 1);
      }
    }
  }
}

extern "C" __global__ void __miss__forward() {
  auto prd = static_cast<PRD *>(getPRD());
  prd->ttl = 0;
}
extern "C" __global__ void __miss__backward() {
  auto prd = static_cast<PRD *>(getPRD());
  prd->ttl = 0;
}

extern "C" __global__ void __closesthit__forward() {
  const int prim_idx = optixGetPrimitiveIndex();
  const float3 ray_dir = optixGetWorldRayDirection();
  const int instance_id = optixGetInstanceId();

  auto prd = static_cast<ForwardPRD *>(getPRD());
  float delta_t = optixGetRayTmax();
  prd->origin = optixGetWorldRayOrigin() + delta_t * ray_dir;

  if (instance_id == 0 && prd->ttl > 1) // hit scene
  {
    float3 *vertices = (float3 *)params.scene_triangle_buffer;
    const float3 v0 = vertices[prim_idx * 3 + 0];
    const float3 v1 = vertices[prim_idx * 3 + 1];
    const float3 v2 = vertices[prim_idx * 3 + 2];
    const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));
    const float3 N = faceforward(N_0, -ray_dir, N_0);

    int mat = params.mat[prim_idx];
    float cos_theta = dot(ray_dir, N) / (length(N) * length(ray_dir));
    float theta = acosf(fabsf(cos_theta));
    int theta_id = (theta / (M_PIf / 2) * 36 + 1) / 2;
    bool penetration = params.mat_prop[theta_id + mat * 19] < 0;

    prd->ttl--;

    int hit_cnt = params.max_reflection + 1 - prd->ttl;

    int new_prdid = prd->is_reflect | (1 << (hit_cnt - 1));
    ForwardPRD *new_prd = &(prd->prd_group[new_prdid]);
    (*new_prd).prd_group = prd->prd_group;
    (*new_prd).is_reflect = new_prdid;
    (*new_prd).ttl = prd->ttl;
    (*new_prd).src = prd->src;
    (*new_prd).origin = prd->origin;
    (*new_prd).direction = ray_dir - 2.0f * dot(ray_dir, N) * N;
    (*new_prd).image_transform = prd->image_transform;

    float3 V = v0 - getImage((*new_prd).image_transform, params.src[prd->src]);
    float3 N_image = faceforward(N_0, V, N_0);
    updateImageTransform((*new_prd).image_transform, N_image, v0);

    if (!penetration)
      prd->ttl = 0;
  } else if (instance_id == 2) // hit destination
    prd->success = prim_idx;
}
extern "C" __global__ void __closesthit__backward() {
  const int prim_idx = optixGetPrimitiveIndex();
  const float3 ray_dir = optixGetWorldRayDirection();
  const int instance_id = optixGetInstanceId();

  auto prd = static_cast<BackwardPRD *>(getPRD());

  float delta_t = optixGetRayTmax();

  prd->origin = optixGetWorldRayOrigin() + delta_t * ray_dir;

  prd->d += delta_t;
  prd->phase += delta_t / params.wave_len * 2 * M_PIf;

  if (instance_id == 0) // hit scene
  {
    if (prd->ttl == 1)
      prd->ttl = 0;
    else {
      float3 *vertices = (float3 *)params.scene_triangle_buffer;
      const float3 v0 = vertices[prim_idx * 3 + 0];
      const float3 v1 = vertices[prim_idx * 3 + 1];
      const float3 v2 = vertices[prim_idx * 3 + 2];
      const float3 N_0 = normalize(cross(v1 - v0, v2 - v0));
      const float3 N = faceforward(N_0, -ray_dir, N_0);

      int mat = params.mat[prim_idx];
      float cos_theta = dot(ray_dir, N) / (length(N) * length(ray_dir));
      float theta = acosf(fabsf(cos_theta));
      int theta_id = (theta / (M_PIf / 2) * 36 + 1) / 2;

      float hit_loss = fabsf(params.mat_prop[theta_id + mat * 19]);
      prd->ttl--;

      int hit_cnt = prd->supposed_hit_cnt + 1 - prd->ttl;
      bool is_reflect = ((prd->is_reflect) >> (hit_cnt - 1)) & 1;

      float max_d =
          fmaxf(fmaxf(length(v0 - v1), length(v1 - v2)), length(v2 - v0));
      float fspl = 1;
      if (prd->d > prd->far_dist) {
        fspl = params.wave_len / (4 * M_PIf * prd->d_reserve);
        prd->d_reserve = 0.f;
      }
      prd->amplitude *= hit_loss * fspl;
      prd->d_reserve += prd->d;
#ifdef EXPORT_TASKINFO
      int info_hit_cnt = params.taskInfo[prd->id].hit_cnt;
      params.taskInfo[prd->id].hit_loss_idx[info_hit_cnt] = theta_id + mat * 19;
      params.taskInfo[prd->id].loss *= fspl;
      params.taskInfo[prd->id].hit_cnt++;
#endif

      float far_dist = 2.0 * max_d * max_d / params.wave_len;
      if (prd->d > far_dist)
        prd->far_dist = far_dist;
      else
        prd->far_dist = 1e6;

      prd->d = 0.f;

      if (is_reflect)
        prd->direction = ray_dir - 2.0f * dot(ray_dir, N) * N;
    }
  } else if (instance_id == 1 && prim_idx == prd->src) // hit source
  {
    if (prd->ttl == 1) {
      float beam_pattern_factor =
          get_beampattern_factor(-ray_dir, params.src_dir, params.src_bptype);

      float amplitude_factor =
          params.wave_len / (4 * M_PIf * (prd->d_reserve + prd->d));
      float loss = beam_pattern_factor * amplitude_factor;
      prd->amplitude *= loss;
      prd->success = 0;
#ifdef EXPORT_TASKINFO
      params.taskInfo[prd->id].loss *= amplitude_factor;
      params.taskInfo[prd->id].launch_angle =
          dot(-ray_dir, params.src_dir) /
          (length(-ray_dir) * length(params.src_dir));
      params.taskInfo[prd->id].phase = prd->phase;
#endif
    }
    prd->ttl = 0;
  }
}

extern "C" __global__ void __intersection__sphere() {
  const float3 ray_orig = optixGetWorldRayOrigin();
  const float3 ray_dir = optixGetWorldRayDirection();

  const int prim_idx = optixGetPrimitiveIndex();
  const int instance_id = optixGetInstanceId();

  float3 center;
  float radius;
  if (instance_id == 1) {
    center = params.src[prim_idx];
    radius = params.src_r;
  } else if (instance_id == 2) {
    center = params.dst[prim_idx];
    radius = params.dst_r;
  }

  const float3 O = center - ray_orig;
  const float3 D = ray_dir / length(ray_dir);

  float b = dot(O, D);
  float c = dot(O, O);

  float accepted_dist = fmaxf(5e-5, radius * radius);
  if ((c - b * b < accepted_dist) && b > 0.0f && length(O) > radius)
    optixReportIntersection(b, 0);
}