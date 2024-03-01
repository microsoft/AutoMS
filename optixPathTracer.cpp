#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <stdio.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <torch/extension.h>

#include "cudaMemManager.h"
#include "optixPathTracer.h"

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

template <typename T> struct Record {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitGroupRecord;

struct PathTracerState {
  OptixDeviceContext context = 0;

  OptixTraversableHandle pas_handle[3] = {0};
  OptixTraversableHandle ias_handle = 0;
  CUdeviceptr d_pas_output_buffer[3] = {0};
  CUdeviceptr d_ias_output_buffer = 0;
  CUdeviceptr d_mat = 0;

  OptixModule ptx_module = 0;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixPipeline pipeline = 0;

  OptixProgramGroup raygen_prog_group = 0;
  OptixProgramGroup forward_miss_group = 0;
  OptixProgramGroup forward_hit_group = 0;
  OptixProgramGroup backward_miss_group = 0;
  OptixProgramGroup backward_hit_group = 0;

  CUstream stream = 0;
  Params params;
  Params *d_params;

  int optix_log_level = 0;

  OptixShaderBindingTable sbt = {};

#ifdef EXPORT_TASKINFO
  FILE *exportedTaskInfo;
  int exportedRecordCount = 0;
#endif
};

//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------

void initLaunchParams(PathTracerState &state) {
  state.params.handle = state.ias_handle;
  CMM_CHECK(cmmMalloc((void **)&state.params.tmp_results,
                      state.params.reserve_size * state.params.launch_x *
                          state.params.launch_y * sizeof(ForwardRecord)));
  CMM_CHECK(cmmMalloc((void **)&state.params.min_hc,
                      state.params.n_dst_grid * state.params.launch_x *
                          state.params.launch_y * sizeof(int)));
  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(state.params.min_hc), 0,
                        state.params.n_dst_grid * state.params.launch_x *
                            state.params.launch_y * sizeof(int)));
  CMM_CHECK(
      cmmMalloc(reinterpret_cast<void **>(&state.params.tmp_result_counts),
                sizeof(int) * (state.params.launch_x * state.params.launch_y)));
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&state.params.stats),
                      sizeof(uint64_t) * 2));
  CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(state.params.stats), 0,
                        sizeof(uint64_t) * 2));
  CUDA_CHECK(cudaStreamCreate(&state.stream));
  CMM_CHECK(
      cmmMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
}

void launchSubframe(PathTracerState &state) {
  // Launch
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(state.d_params),
                             &state.params, sizeof(Params),
                             cudaMemcpyHostToDevice, state.stream));
  int stage = 1;
  CMM_CHECK(cmmMemcpy(&(state.d_params->stage), &stage, sizeof(int),
                      cudaMemcpyHostToDevice));

  if (state.optix_log_level > 0)
    printf("\033[38;5;208mstep 1\033[0m\nraytricing... ");

  cudaEvent_t start, stop;
  float totalTime = 0.0f;
  if (state.optix_log_level > 0) {
    cudaEventCreate(&start);   // 起始时间
    cudaEventCreate(&stop);    // 结束时间
    cudaEventRecord(start, 0); // 记录起始时间
  }
  OPTIX_CHECK(optixLaunch(state.pipeline, state.stream,
                          reinterpret_cast<CUdeviceptr>(state.d_params),
                          sizeof(Params), &state.sbt, state.params.launch_x,
                          state.params.launch_y, 1));
  if (state.optix_log_level > 0) {
    cudaEventRecord(stop, 0); // 执行完代码，记录结束时间
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf(" %.3fs\n", elapsedTime / 1000);
    totalTime += elapsedTime;
  }
  CUDA_SYNC_CHECK();

  // count result
  int debug[2];
  CMM_CHECK(cmmMemcpy(debug, state.params.stats, sizeof(uint64_t) * 2,
                      cudaMemcpyDeviceToHost));
  int tmp_result_counts_cpu[state.params.launch_x * state.params.launch_y];
  CMM_CHECK(
      cmmMemcpy(tmp_result_counts_cpu, state.params.tmp_result_counts,
                sizeof(int) * state.params.launch_x * state.params.launch_y,
                cudaMemcpyDeviceToHost));
  int trace_count_original = 0;
  int trace_count_max = 0;
  for (int i = 0; i < state.params.launch_x * state.params.launch_y; i++) {
    trace_count_original += tmp_result_counts_cpu[i];
    trace_count_max = max(trace_count_max, tmp_result_counts_cpu[i]);
  }
  if (state.optix_log_level > 0) {
    std::cout << "trace_count_max: " << trace_count_max << std::endl;
    std::cout << "tmp_result_max_size: " << state.params.reserve_size
              << std::endl;
    std::cout << "trace_count_original: " << trace_count_original << std::endl;
  }

  if (trace_count_original == 0) {
    std::cout << "No path found!" << std::endl;
    return;
  }
  // combine tmp_results
  ForwardRecord *tmp_result_gpu;
  CMM_CHECK(cmmMalloc((void **)&tmp_result_gpu,
                      sizeof(ForwardRecord) * trace_count_original));
  ForwardRecord *tmp_result_gpu_tail = tmp_result_gpu;
  for (int i = 0; i < state.params.launch_x * state.params.launch_y; i++) {
    if (tmp_result_counts_cpu[i] == 0)
      continue;
    CMM_CHECK(
        cmmMemcpy(tmp_result_gpu_tail,
                  state.params.tmp_results + i * state.params.reserve_size,
                  sizeof(ForwardRecord) * tmp_result_counts_cpu[i],
                  cudaMemcpyDeviceToDevice));
    tmp_result_gpu_tail += tmp_result_counts_cpu[i];
  }

  // to tensor
  int floats_of_ForwardRecord = sizeof(ForwardRecord) / sizeof(float);
  auto optionsf =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto optionsi =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor tensor_trace_all = torch::from_blob(
      tmp_result_gpu, {trace_count_original, floats_of_ForwardRecord},
      optionsi);

  // unique
  auto trace_unique_raw = at::unique_dim(tensor_trace_all, 0);
  auto trace_int = std::get<0>(trace_unique_raw);
  int trace_count = trace_int.size(0);
  auto trace_float =
      torch::from_blob(trace_int.data_ptr<int>(),
                       {trace_count, floats_of_ForwardRecord}, optionsf);
  float *trace_float_ptr = trace_float.data_ptr<float>();
  if (state.optix_log_level > 0)
    std::cout << "trace_count: " << trace_count << std::endl;
  // copy trace to CPU
  ForwardRecord *trace_cpu_ptr;
  trace_cpu_ptr = (ForwardRecord *)malloc(sizeof(ForwardRecord) * trace_count);
  CMM_CHECK(cmmMemcpy(trace_cpu_ptr, trace_float_ptr,
                      sizeof(ForwardRecord) * trace_count,
                      cudaMemcpyDeviceToHost));
  // generate inverted list of path
  int grid_num = state.params.n_dst_grid;
  std::map<Path, int> pathListInv;
  std::set<int> tracedPath[grid_num];
  std::vector<int> gridTraceTasks[grid_num];
  int traceTaskLen;

  for (int i = 0; i < trace_count; i++)
    pathListInv[trace_cpu_ptr[i].path] = 0;

  // generate path list
  Path pathList[pathListInv.size()];
  int idx = 0;
  for (auto it = pathListInv.begin(); it != pathListInv.end(); it++) {
    pathList[idx] = (*it).first;
    it->second = idx;
    idx++;
  }
  if (state.optix_log_level > 0)
    std::cout << "pathList size: " << pathListInv.size() << std::endl;
  // copy data to GPU

  CMM_CHECK(cmmMalloc((void **)&(state.params.pathList),
                      sizeof(Path) * pathListInv.size()));
  CMM_CHECK(cmmMemcpy(&(state.d_params->pathList), &(state.params.pathList),
                      sizeof(Path *), cudaMemcpyHostToDevice));
  CMM_CHECK(cmmMemcpy(state.params.pathList, pathList,
                      sizeof(Path) * pathListInv.size(),
                      cudaMemcpyHostToDevice));

  stage = 2;
  CMM_CHECK(cmmMemcpy(&(state.d_params->stage), &stage, sizeof(int),
                      cudaMemcpyHostToDevice));
  if (state.optix_log_level > 0)
    printf("\033[38;5;208mstep 2\033[0m\n");
  for (int round = 1;; round++) {
    // generate raytracinig task for each grid
    if (round == 1) {
      // initialize with forward result
      for (int i = 0; i < trace_count; i++) {
        auto trace = trace_cpu_ptr[i];
        int task = pathListInv[trace.path];
        gridTraceTasks[trace.dst_grid_idx].push_back(task);
        tracedPath[trace.dst_grid_idx].insert(task);
      }
      free(trace_cpu_ptr);
    } else {
      bool goflag = false;
      // add successful path of neighbors' in the last round
      bool taskSuccess[traceTaskLen];
#ifdef EXPORT_TASKINFO
      int taskInfoLen = traceTaskLen * state.params.n_src *
                        state.params.dst_gridsize * state.params.dst_gridsize;
      ExportedTaskInfo *buffer =
          (ExportedTaskInfo *)malloc(sizeof(ExportedTaskInfo) * taskInfoLen);
      CMM_CHECK(cmmMemcpy(buffer, state.params.taskInfo,
                          sizeof(ExportedTaskInfo) * taskInfoLen,
                          cudaMemcpyDeviceToHost));
      fwrite(buffer, sizeof(ExportedTaskInfo), taskInfoLen,
             state.exportedTaskInfo);
      free(buffer);
      state.exportedRecordCount += taskInfoLen;
#endif
      TraceTask lastTraceTasks[traceTaskLen];
      CMM_CHECK(cmmMemcpy(taskSuccess, state.params.taskSuccess,
                          sizeof(bool) * traceTaskLen, cudaMemcpyDeviceToHost));
      CMM_CHECK(cmmMemcpy(lastTraceTasks, state.params.traceTasks,
                          sizeof(TraceTask) * traceTaskLen,
                          cudaMemcpyDeviceToHost));

      for (int i = 0; i < grid_num; i++)
        gridTraceTasks[i].clear();
      for (int i = 0; i < traceTaskLen; i++) {
        int path_idx = lastTraceTasks[i].path_idx;
        int dst_grid_idx = lastTraceTasks[i].dst_grid_idx;
        if (taskSuccess[i]) {
          int neighbors[] = {dst_grid_idx - 1, dst_grid_idx + 1,
                             dst_grid_idx - state.params.n1_dst_grid,
                             dst_grid_idx + state.params.n1_dst_grid};
          for (int neighbor : neighbors) {
            if (neighbor < 0 || neighbor >= grid_num)
              continue;
            if (tracedPath[neighbor].find(path_idx) ==
                tracedPath[neighbor].end()) {
              goflag = true;
              gridTraceTasks[neighbor].push_back(path_idx);
              tracedPath[neighbor].insert(path_idx);
            }
          }
        }
      }
      CMM_CHECK(cmmFree(state.params.traceTasks));
      CMM_CHECK(cmmFree(state.params.taskSuccess));
#ifdef EXPORT_TASKINFO
      CMM_CHECK(cmmFree(state.params.taskInfo));
#endif
      if (!goflag)
        break;
    }

    std::vector<TraceTask> traceTasks;
    for (int grid = 0; grid < grid_num; grid++) {
      for (int path : gridTraceTasks[grid]) {
        TraceTask traceTask = {path, grid};
        traceTasks.push_back(traceTask);
      }
    }
    traceTaskLen = traceTasks.size();

    CMM_CHECK(cmmMalloc((void **)&(state.params.traceTasks),
                        sizeof(TraceTask) * traceTaskLen));
    CMM_CHECK(cmmMalloc((void **)&(state.params.taskSuccess),
                        sizeof(bool) * traceTaskLen));
#ifdef EXPORT_TASKINFO
    CMM_CHECK(cmmMalloc((void **)&(state.params.taskInfo),
                        sizeof(ExportedTaskInfo) * traceTaskLen *
                            state.params.n_src * state.params.dst_gridsize *
                            state.params.dst_gridsize));

    CMM_CHECK(cmmMemcpy(&(state.d_params->taskInfo), &(state.params.taskInfo),
                        sizeof(ExportedTaskInfo *), cudaMemcpyHostToDevice));
#endif
    CUDA_CHECK(cudaMemset(state.params.taskSuccess, false,
                          sizeof(bool) * traceTaskLen));
    CMM_CHECK(cmmMemcpy(&(state.d_params->traceTasks),
                        &(state.params.traceTasks), sizeof(int *),
                        cudaMemcpyHostToDevice));
    CMM_CHECK(cmmMemcpy(&(state.d_params->taskSuccess),
                        &(state.params.taskSuccess), sizeof(bool *),
                        cudaMemcpyHostToDevice));
    CMM_CHECK(cmmMemcpy(state.params.traceTasks, traceTasks.data(),
                        sizeof(TraceTask) * traceTaskLen,
                        cudaMemcpyHostToDevice));
    CMM_CHECK(cmmMemcpy(&(state.d_params->traceTaskLen), &traceTaskLen,
                        sizeof(int), cudaMemcpyHostToDevice));
    // launch ray tracing
    if (state.optix_log_level > 0)
      printf("(round %d, task %d) raytracing...", round, traceTaskLen);
    int idealThreadNum =
        traceTaskLen * state.params.dst_gridsize * state.params.dst_gridsize;
    int threadNum = min(100000000 / state.params.n_src, idealThreadNum);

    if (state.optix_log_level > 0) {
      cudaEventCreate(&start);   // 起始时间
      cudaEventCreate(&stop);    // 结束时间
      cudaEventRecord(start, 0); // 记录起始时间
    }
    OPTIX_CHECK(optixLaunch(state.pipeline, state.stream,
                            reinterpret_cast<CUdeviceptr>(state.d_params),
                            sizeof(Params), &state.sbt, state.params.n_src,
                            threadNum, 1));
    CUDA_SYNC_CHECK();
    if (state.optix_log_level > 0) {
      cudaEventRecord(stop, 0); // 执行完代码，记录结束时间
      cudaEventSynchronize(stop);
      float elapsedTime;
      cudaEventElapsedTime(&elapsedTime, start, stop);
      printf(" %.3fs\n", elapsedTime / 1000);
      totalTime += elapsedTime;
    }
  }
  if (state.optix_log_level > 0)
    printf("total raytrace time: %.3fs\n", totalTime / 1000);

  CMM_CHECK(cmmFree(state.params.pathList));
  CMM_CHECK(cmmFree(tmp_result_gpu));
}

static void context_log_cb(uint level, const char *tag, const char *message,
                           void * /*cbdata */) {
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
            << "]: " << message << "\n";
}

void createContext(PathTracerState &state) {
  // Initialize CUDA
  CMM_CHECK(cmmFree(0));

  OptixDeviceContext context;
  CUcontext cu_ctx = 0; // zero means take the current context
  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &context_log_cb;
  options.logCallbackLevel = state.optix_log_level;
  OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

  state.context = context;
}
void buildInstanceAccel(PathTracerState &state) {
  int instance_len = 3;
  CUdeviceptr d_instances;
  size_t instance_size_in_bytes = sizeof(OptixInstance) * instance_len;
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_instances),
                      instance_size_in_bytes));

  OptixBuildInput instance_input = {};

  instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
  instance_input.instanceArray.instances = d_instances;
  instance_input.instanceArray.numInstances = instance_len;

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes ias_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options,
                                           &instance_input,
                                           1, // num build inputs
                                           &ias_buffer_sizes));

  CUdeviceptr d_temp_buffer;
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_temp_buffer),
                      ias_buffer_sizes.tempSizeInBytes));
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&state.d_ias_output_buffer),
                      ias_buffer_sizes.outputSizeInBytes));

  // Use the identity matrix for the instance transform
  float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

  OptixInstance optix_instances[instance_len];
  memset(optix_instances, 0, instance_size_in_bytes);

  for (int i = 0; i < instance_len; i++) {
    optix_instances[i].traversableHandle = state.pas_handle[i];
    optix_instances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
    optix_instances[i].instanceId = i;
    optix_instances[i].sbtOffset = 0;
    optix_instances[i].visibilityMask = 1 << i;
    memcpy(optix_instances[i].transform, transform, sizeof(float) * 12);
  }
  CMM_CHECK(cmmMemcpy(reinterpret_cast<void *>(d_instances), &optix_instances,
                      instance_size_in_bytes, cudaMemcpyHostToDevice));

  OPTIX_CHECK(optixAccelBuild(state.context,
                              0, // CUDA stream
                              &accel_options, &instance_input,
                              1, // num build inputs
                              d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
                              state.d_ias_output_buffer,
                              ias_buffer_sizes.outputSizeInBytes,
                              &state.ias_handle,
                              nullptr, // emitted property list
                              0        // num emitted properties
                              ));

  CMM_CHECK(cmmFree((void *)d_temp_buffer));
  CMM_CHECK(cmmFree((void *)d_instances));
}
void buildPrimitiveAccel(OptixDeviceContext context, bool triangle,
                         int n_primitive, float *buffer,
                         OptixTraversableHandle *handle,
                         CUdeviceptr *output_buffer) {

  CUdeviceptr d_vertices = 0;
  OptixBuildInput optix_build_input = {};
  uint32_t input_flags[1]; // One per SBT record for this build input
  if (triangle) {
    CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_vertices),
                        n_primitive * 3 * 4 * sizeof(float)));
    for (int i = 0; i < n_primitive * 3; i++) {
      CMM_CHECK(cmmMemcpy(
          reinterpret_cast<void *>(d_vertices + i * 4 * sizeof(float)),
          buffer + i * 3, 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    input_flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    optix_build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    optix_build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    optix_build_input.triangleArray.vertexStrideInBytes = sizeof(float) * 4;
    optix_build_input.triangleArray.numVertices = n_primitive * 3;
    optix_build_input.triangleArray.vertexBuffers = &d_vertices;
    optix_build_input.triangleArray.flags = input_flags;
    optix_build_input.triangleArray.numSbtRecords = 1;
  } else {
    input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;
    optix_build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    optix_build_input.customPrimitiveArray.aabbBuffers = (CUdeviceptr *)&buffer;
    optix_build_input.customPrimitiveArray.numPrimitives = n_primitive;
    optix_build_input.customPrimitiveArray.flags = input_flags;
    optix_build_input.customPrimitiveArray.numSbtRecords = 1;
  }

  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
                                           &optix_build_input,
                                           1, // num_build_inputs
                                           &gas_buffer_sizes));

  CUdeviceptr d_temp_buffer;
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_temp_buffer),
                      gas_buffer_sizes.tempSizeInBytes));

  // non-compacted output
  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset =
      roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CMM_CHECK(cmmMalloc(
      reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
      compactedSizeOffset + 8));

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.result =
      (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size +
                    compactedSizeOffset);

  OPTIX_CHECK(optixAccelBuild(context,
                              0, // CUDA stream
                              &accel_options, &optix_build_input,
                              1, // num build inputs
                              d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
                              d_buffer_temp_output_gas_and_compacted_size,
                              gas_buffer_sizes.outputSizeInBytes, handle,
                              &emitProperty, // emitted property list
                              1              // num emitted properties
                              ));

  CMM_CHECK(cmmFree((void *)d_temp_buffer));
  CMM_CHECK(cmmFree((void *)d_vertices));

  size_t compacted_gas_size;
  CMM_CHECK(cmmMemcpy(&compacted_gas_size, (void *)emitProperty.result,
                      sizeof(size_t), cudaMemcpyDeviceToHost));

  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(output_buffer),
                        compacted_gas_size));

    // use handle as input and output
    OPTIX_CHECK(optixAccelCompact(context, 0, *handle, *output_buffer,
                                  compacted_gas_size, handle));

    CMM_CHECK(cmmFree((void *)d_buffer_temp_output_gas_and_compacted_size));
  } else {
    *output_buffer = d_buffer_temp_output_gas_and_compacted_size;
  }
}
void buildAccel(PathTracerState &state) {
  buildPrimitiveAccel(state.context, true, state.params.n_scene_triangle,
                      state.params.scene_triangle_buffer, &state.pas_handle[0],
                      &state.d_pas_output_buffer[0]);
  buildPrimitiveAccel(state.context, false, state.params.n_src,
                      state.params.src_aabb_buffer, &state.pas_handle[1],
                      &state.d_pas_output_buffer[1]);
  buildPrimitiveAccel(state.context, false, state.params.n_dst,
                      state.params.dst_aabb_buffer, &state.pas_handle[2],
                      &state.d_pas_output_buffer[2]);
  buildInstanceAccel(state);
}

void createModule(PathTracerState &state) {
  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
  state.pipeline_compile_options.numPayloadValues = 2;
  state.pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur
             // significant performance cost and should only be done during
             // development.
  state.pipeline_compile_options.exceptionFlags =
      OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
      OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  size_t inputSize = 0;
  const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR,
                                          "optixPathTracer.cu", inputSize);

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
      state.context, &module_compile_options, &state.pipeline_compile_options,
      input, inputSize, log, &sizeof_log, &state.ptx_module));
}

void createProgramGroups(PathTracerState &state) {
  OptixProgramGroupOptions program_group_options = {};

  char log[2048];
  size_t sizeof_log = sizeof(log);

  {
    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.ptx_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options, log, &sizeof_log, &state.raygen_prog_group));
  }

  {
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.ptx_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__forward";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &miss_prog_group_desc,
        1, // num program groups
        &program_group_options, log, &sizeof_log, &state.forward_miss_group));
    memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module =
        state.ptx_module; // NULL miss program for backward rays
    miss_prog_group_desc.miss.entryFunctionName = "__miss__backward";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &miss_prog_group_desc,
        1, // num program groups
        &program_group_options, log, &sizeof_log, &state.backward_miss_group));
  }

  {
    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__forward";
    hit_prog_group_desc.hitgroup.moduleIS = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context, &hit_prog_group_desc,
        1, // num program groups
        &program_group_options, log, &sizeof_log, &state.forward_hit_group));
    memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__backward";
    hit_prog_group_desc.hitgroup.moduleIS = state.ptx_module;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
    sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        state.context, &hit_prog_group_desc,
        1, // num program groups
        &program_group_options, log, &sizeof_log, &state.backward_hit_group));
  }
}

void createPipeline(PathTracerState &state) {
  OptixProgramGroup program_groups[] = {
      state.raygen_prog_group,  state.forward_miss_group,
      state.forward_hit_group,  state.backward_miss_group,
      state.backward_hit_group,
  };

  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 2;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

  char log[2048];
  size_t sizeof_log = sizeof(log);
  OPTIX_CHECK_LOG(optixPipelineCreate(
      state.context, &state.pipeline_compile_options, &pipeline_link_options,
      program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log,
      &sizeof_log, &state.pipeline));

  // We need to specify the max traversal
  // depth.  Calculate the stack sizes, so
  // we can specify all parameters to
  // optixPipelineSetStackSize.
  OptixStackSizes stack_sizes = {};
  OPTIX_CHECK(
      optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes));
  OPTIX_CHECK(
      optixUtilAccumulateStackSizes(state.forward_miss_group, &stack_sizes));
  OPTIX_CHECK(
      optixUtilAccumulateStackSizes(state.forward_hit_group, &stack_sizes));
  OPTIX_CHECK(
      optixUtilAccumulateStackSizes(state.backward_miss_group, &stack_sizes));
  OPTIX_CHECK(
      optixUtilAccumulateStackSizes(state.backward_hit_group, &stack_sizes));

  uint32_t max_trace_depth = 2;
  uint32_t max_cc_depth = 0;
  uint32_t max_dc_depth = 0;
  uint32_t direct_callable_stack_size_from_traversal;
  uint32_t direct_callable_stack_size_from_state;
  uint32_t continuation_stack_size;
  OPTIX_CHECK(optixUtilComputeStackSizes(
      &stack_sizes, max_trace_depth, max_cc_depth, max_dc_depth,
      &direct_callable_stack_size_from_traversal,
      &direct_callable_stack_size_from_state, &continuation_stack_size));

  const uint32_t max_traversal_depth = 1;
  OPTIX_CHECK(optixPipelineSetStackSize(
      state.pipeline, direct_callable_stack_size_from_traversal,
      direct_callable_stack_size_from_state, continuation_stack_size,
      max_traversal_depth));
}

void createSBT(PathTracerState &state) {
  CUdeviceptr d_raygen_record;
  const size_t raygen_record_size = sizeof(RayGenRecord);
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_raygen_record),
                      raygen_record_size));

  RayGenRecord rg_sbt = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

  CMM_CHECK(cmmMemcpy(reinterpret_cast<void *>(d_raygen_record), &rg_sbt,
                      raygen_record_size, cudaMemcpyHostToDevice));

  CUdeviceptr d_miss_records;
  const size_t miss_record_size = sizeof(MissRecord);
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_miss_records),
                      miss_record_size * RAY_TYPE_COUNT));

  MissRecord ms_sbt[2];
  OPTIX_CHECK(optixSbtRecordPackHeader(state.forward_miss_group, &ms_sbt[0]));
  OPTIX_CHECK(optixSbtRecordPackHeader(state.backward_miss_group, &ms_sbt[1]));

  CMM_CHECK(cmmMemcpy(reinterpret_cast<void *>(d_miss_records), ms_sbt,
                      miss_record_size * RAY_TYPE_COUNT,
                      cudaMemcpyHostToDevice));

  CUdeviceptr d_hitgroup_records;
  const size_t hitgroup_record_size = sizeof(HitGroupRecord);
  CMM_CHECK(cmmMalloc(reinterpret_cast<void **>(&d_hitgroup_records),
                      hitgroup_record_size * RAY_TYPE_COUNT));

  HitGroupRecord hitgroup_records[RAY_TYPE_COUNT];

  OPTIX_CHECK(
      optixSbtRecordPackHeader(state.forward_hit_group, &hitgroup_records[0]));
  OPTIX_CHECK(
      optixSbtRecordPackHeader(state.backward_hit_group, &hitgroup_records[1]));

  CMM_CHECK(cmmMemcpy(reinterpret_cast<void *>(d_hitgroup_records),
                      hitgroup_records, hitgroup_record_size * RAY_TYPE_COUNT,
                      cudaMemcpyHostToDevice));

  state.sbt.raygenRecord = d_raygen_record;
  state.sbt.missRecordBase = d_miss_records;
  state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
  state.sbt.missRecordCount = RAY_TYPE_COUNT;
  state.sbt.hitgroupRecordBase = d_hitgroup_records;
  state.sbt.hitgroupRecordStrideInBytes =
      static_cast<uint32_t>(hitgroup_record_size);
  state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT;
}

void cleanupState(PathTracerState &state) {
  OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
  OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.forward_miss_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.forward_hit_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.backward_miss_group));
  OPTIX_CHECK(optixProgramGroupDestroy(state.backward_hit_group));
  OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
  OPTIX_CHECK(optixDeviceContextDestroy(state.context));

  CMM_CHECK(cmmFree((void *)state.sbt.raygenRecord));
  CMM_CHECK(cmmFree((void *)state.sbt.missRecordBase));
  CMM_CHECK(cmmFree((void *)state.sbt.hitgroupRecordBase));

  CMM_CHECK(cmmFree((void *)state.d_ias_output_buffer));
  for (int i = 0; i < 3; i++)
    CMM_CHECK(cmmFree((void *)state.d_pas_output_buffer[i]));
  CMM_CHECK(cmmFree(state.d_params));
  CMM_CHECK(cmmFree(state.params.stats));
  CMM_CHECK(cmmFree(state.params.tmp_results));
  CMM_CHECK(cmmFree(state.params.min_hc));
  CMM_CHECK(cmmFree(state.params.tmp_result_counts));
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void torch_launch_pathtrace(
    torch::Tensor &result, const torch::Tensor &src, const torch::Tensor &dst,
    const torch::Tensor &src_aabb_buffer, const torch::Tensor &dst_aabb_buffer,
    const torch::Tensor &scene_triangle_buffer, const torch::Tensor &mat,
    int64_t samples_per_launch, int64_t launch_x, int64_t launch_y,
    double wave_len, int64_t max_reflection, int64_t reserve_size,
    const torch::Tensor &mat_prop, const torch::Tensor &src_dir,
    const torch::Tensor &dst_dir, int64_t optix_log_level, int64_t src_bptype,
    int64_t dst_bptype, int64_t dst_girdsize, double src_r, double dst_r

) {
  try {
    PathTracerState state;
    state.optix_log_level = optix_log_level;
    state.params.max_reflection = max_reflection;
    state.params.samples_per_launch = samples_per_launch;
    state.params.launch_x = launch_x;
    state.params.launch_y = launch_y;
    state.params.src_bptype = src_bptype;
    state.params.dst_bptype = dst_bptype;
    state.params.dst_gridsize = dst_girdsize;
    state.params.result =
        reinterpret_cast<float *const>(result.data_ptr<float>());
    state.params.mat = reinterpret_cast<int *const>(mat.data_ptr<int>());
    state.params.mat_prop =
        reinterpret_cast<float *const>(mat_prop.data_ptr<float>());
    state.params.wave_len = wave_len;
    state.params.src = reinterpret_cast<float3 *const>(src.data_ptr<float>());
    state.params.dst = reinterpret_cast<float3 *const>(dst.data_ptr<float>());
    state.params.dst_aabb_buffer = dst_aabb_buffer.data_ptr<float>();
    state.params.src_aabb_buffer = src_aabb_buffer.data_ptr<float>();
    state.params.scene_triangle_buffer =
        scene_triangle_buffer.data_ptr<float>();
    state.params.n_src = src.size(0) * src.size(1);
    state.params.n_dst = dst.size(0) * dst.size(1);
    state.params.n0_dst = dst.size(0);
    state.params.n1_dst = dst.size(1);
    state.params.n0_dst_grid = (dst.size(0) - 1) / dst_girdsize + 1;
    state.params.n1_dst_grid = (dst.size(1) - 1) / dst_girdsize + 1;
    state.params.n_dst_grid =
        state.params.n0_dst_grid * state.params.n1_dst_grid;
    state.params.n_scene_triangle = scene_triangle_buffer.size(0) / 3;
    state.params.reserve_size = reserve_size;
    state.params.src_r = src_r;
    state.params.dst_r = dst_r;

    state.params.src_dir = *((float3 *)src_dir.data_ptr<float>());
    state.params.dst_dir = *((float3 *)dst_dir.data_ptr<float>());

#ifdef EXPORT_TASKINFO
    state.exportedTaskInfo = fopen("path.dat", "w");
#endif

    createContext(state);
    buildAccel(state);
    createModule(state);
    createProgramGroups(state);
    createPipeline(state);
    createSBT(state);
    initLaunchParams(state);
    launchSubframe(state);

    uint64_t stats_cpu[2];
    CMM_CHECK(cmmMemcpy(stats_cpu, state.params.stats, 2 * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
    if (state.optix_log_level > 0)
      printf("total rays: %lu (%.2f%% valid)\n", stats_cpu[0] + stats_cpu[1],
             100.0 * stats_cpu[0] / (stats_cpu[0] + stats_cpu[1]));

#ifdef EXPORT_TASKINFO
    fclose(state.exportedTaskInfo);
    printf("Exported task info to path.dat. Record num: %d\n",
           state.exportedRecordCount);
#endif

    cleanupState(state);
    cmmCheckMem();
  } catch (std::exception &e) {
    std::cerr << "Caught exception: " << e.what() << "\n";
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_launch_pathtrace", &torch_launch_pathtrace,
        "pathtrace kernel warpper");
}

TORCH_LIBRARY(pathtrace, m) {
  m.def("torch_launch_pathtrace", torch_launch_pathtrace);
}