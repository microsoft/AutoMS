#include <stdint.h>
#include <sutil/Matrix.h>

// #define EXPORT_TASKINFO
#define MAX_HIT_N 5

enum RayType { RAY_TYPE_FORWARD = 0, RAY_TYPE_BACKWARD = 1, RAY_TYPE_COUNT };

struct Path {
  sutil::Matrix3x4 image_transform;
  int hit_cnt;
  int is_reflect;
};

void printPath(Path path) {
  printf("image_transform:\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n",
         path.image_transform.getData()[0], path.image_transform.getData()[1],
         path.image_transform.getData()[2], path.image_transform.getData()[3],
         path.image_transform.getData()[4], path.image_transform.getData()[5],
         path.image_transform.getData()[6], path.image_transform.getData()[7],
         path.image_transform.getData()[8], path.image_transform.getData()[9],
         path.image_transform.getData()[10],
         path.image_transform.getData()[11]);
  printf("hit_cnt: %d\n", path.hit_cnt);
  printf("is_reflect: %d\n", path.is_reflect);
}

bool operator<(const Path &a, const Path &b) {
  if (a.hit_cnt < b.hit_cnt)
    return true;
  if (a.hit_cnt > b.hit_cnt)
    return false;
  if (a.is_reflect < b.is_reflect)
    return true;
  if (a.is_reflect > b.is_reflect)
    return false;
  return a.image_transform < b.image_transform;
}

#ifdef EXPORT_TASKINFO
struct ExportedTaskInfo {
  bool successful;
  float launch_angle;  // cos value
  float receive_angle; // cos value
  float loss;
  float phase;
  int src;
  int dst;
  int hit_cnt;
  int hit_loss_idx[MAX_HIT_N];
};
#endif

struct TraceTask {
  int path_idx;
  int dst_grid_idx;
};

struct ForwardRecord {
  Path path;
  int dst_grid_idx;
};

struct Params {
  OptixTraversableHandle handle;

  int samples_per_launch;
  int launch_x;
  int launch_y;

  // my
  float3 *src;
  float3 *dst;
  float *dst_aabb_buffer;
  float *src_aabb_buffer;
  float *scene_triangle_buffer;
  int *mat;
  float *mat_prop;
  int n_scene_triangle;
  int n_src;
  int n_dst;
  int n0_dst;
  int n1_dst;
  int n0_dst_grid;
  int n1_dst_grid;
  int n_dst_grid;
  int dst_gridsize;

  int max_reflection;
  unsigned long long *stats;
  float wave_len;

  int *tmp_result_counts;
  int *min_hc;
  Path *pathList;
  TraceTask *traceTasks;
  int traceTaskLen;
  bool *taskSuccess;
#ifdef EXPORT_TASKINFO
  ExportedTaskInfo *taskInfo;
#endif
  ForwardRecord *tmp_results;

  float *result;
  float3 src_dir;
  float3 dst_dir;
  float src_r;
  float dst_r;

  int src_bptype;
  int dst_bptype;

  int stage;
  int reserve_size;
};

struct RayGenData {};

struct MissData {};

struct HitGroupData {};
