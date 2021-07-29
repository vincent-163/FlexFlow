/* Copyright 2020 Stanford, NVIDIA, Facebook
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/linear.h"
#include "cuda_helper.h"

using namespace Legion;

Tensor FFModel::dense(const Tensor input,
                      int outDim,
                      ActiMode activation,
                      bool use_bias,
                      const Op* shared_op,
                      Initializer* kernel_initializer,
                      Initializer* bias_initializer,
                      const char *name)
{
  Linear* li = new Linear(*this, input, outDim, activation, use_bias, false, name);
  layers.push_back(li);
  return li->outputs[0];
}

size_t Linear::get_params_hash() const {
  return this->get_params().get_hash(this->inputs[0]);
}

/*
  regions[0](O): output
  regions[1](I): kernel
  regions[2](I): bias
*/
OpMeta* Linear::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime* runtime)
{
  Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (out_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return init_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  return NULL;
}

bool Linear::use_cudnn_activation(ActiMode mode)
{
  switch (mode) {
    case AC_MODE_RELU:
    case AC_MODE_SIGMOID:
    case AC_MODE_TANH:
      return true;
  }
  return false;
}

template<int NDIM>
OpMeta* Linear::init_task_with_dim(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
  assert(regions.size() == task->regions.size());
  assert(regions.size() == 2 || regions.size() == 3);
  const Linear* linear = (Linear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  //TensorAccessorR<float, 2> acc_input(
  //    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorW<float, 3> acc_kernel(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  // TensorAccessorR<float, 1> acc_bias(
  //     regions[3], task->regions[3], FID_DATA, ctx, runtime);
  //int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  printf("init linear (input): in_dim(%d) out_dim(%d) batch_size(%d)\n",
      in_dim, out_dim, batch_size);
  LinearMeta* m = new LinearMeta(handle, batch_size);
  m->activation = linear->activation;
  m->use_bias = linear->use_bias;
  m->profiling = linear->profiling;
  m->trainableInputs[0] = linear->trainableInputs[0];
  std::strcpy(m->op_name, linear->name);

  if (use_cudnn_activation(m->activation)) {
    cudnnActivationMode_t mode;
    switch (linear->activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          batch_size, out_dim, 1, 1));
  }
  return m;
}

void Linear::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  //assert(check_output_input_weight_same_machine_view());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LINEAR_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  //launcher.add_region_requirement(
  //    RegionRequirement(input_lps[0], 0/*projection id*/,
  //                      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  //launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(1, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    //launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    //launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*static*/
void Linear::forward_kernel(const LinearMeta* m,
                            const float* input_ptr,
                            float* output_ptr,
                            const float* kernel_ptr,
                            const float* bias_ptr,
                            int in_dim, int out_dim, int batch_size,
                            cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        out_dim, batch_size, in_dim,
                        &alpha, kernel_ptr, in_dim,
                        input_ptr, in_dim, &beta,
                        output_ptr, out_dim));
  // use_bias = True 
  if (bias_ptr != NULL) { 
    checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                          out_dim, batch_size, 1,
                          &alpha, bias_ptr, 1,
                          m->one_ptr, 1, &alpha,
                          output_ptr, out_dim));
  }
  if (use_cudnn_activation(m->activation)) {
    checkCUDNN(cudnnActivationForward(m->handle.dnn, m->actiDesc,
        &alpha, m->outputTensor, output_ptr,
        &beta, m->outputTensor, output_ptr));
  } else if (m->activation == AC_MODE_GELU) {
    size_t elements = (size_t)out_dim * (size_t) batch_size;
    constexpr float B = 0.7978845608028654f;   // sqrt(2.0/M_PI)
    constexpr float C = 0.035677408136300125f; // 0.044715 * sqrt(2.0/M_PI)
    gelu_forward_kernel<<<GET_BLOCKS(elements), CUDA_NUM_THREADS>>>(
        elements, B, C, output_ptr);
  } else if (m->activation == AC_MODE_NONE) {
    // Do nothing
  } else {
    assert(false && "Unsupported activation for Linear");
  }
}

__host__
void Linear::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return forward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
  regions[3](I): bias
*/
template<int NDIM>
void Linear::forward_task_with_dim(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime)
{
  //Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  assert(regions.size() == (3 + int(m->use_bias)));
  assert(task->regions.size() == (3 + int(m->use_bias)));
  
  TensorAccessorR<float, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, NDIM> acc_output(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 3> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_input.rect.volume() == in_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  const float* acc_bias_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorR<float, 3> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(acc_bias.rect.volume() == out_dim);
    acc_bias_ptr = acc_bias.ptr;
  }

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Linear::forward_kernel(m, acc_input.ptr, acc_output.ptr,
      acc_kernel.ptr, acc_bias_ptr, in_dim, out_dim, batch_size, stream);

  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("%s [Linear] forward time = %.2lfms\n", m->op_name, elapsed);
    //print_tensor<NDIM, float>(acc_input.ptr, acc_input.rect, "[Linear:forward:input]");
    //print_tensor<2, float>(acc_kernel.ptr, acc_kernel.rect, "[Linear:forward:kernel]");
    //print_tensor<1, float>(acc_bias.ptr, acc_bias.rect, "[Linear:forward:bias]");
    //print_tensor<NDIM, float>(acc_output.ptr, acc_output.rect, "[Linear:forward:output]");
  }
}

void Linear::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(LINEAR_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, weights[0]->region));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(
        RegionRequirement(weights[1]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

__global__
void sigmoid_backward(float *grad_ptr, const float *output, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = grad_ptr[i] * output[i] * (1 - output[i]);
  }
}

/*static*/
void Linear::backward_kernel(const LinearMeta* m,
                             const float* input_ptr,
                             float* input_grad_ptr,
                             const float* output_ptr,
                             float* output_grad_ptr,
                             const float* kernel_ptr,
                             float* kernel_grad_ptr,
                             float* bias_grad_ptr,
                             int in_dim, int out_dim, int batch_size,
                             cudaStream_t stream)
{
  checkCUDA(cublasSetStream(m->handle.blas, stream));
  checkCUDNN(cudnnSetStream(m->handle.dnn, stream));

  float alpha = 1.0f;
  int output_size = out_dim * batch_size;
  if (m->activation == AC_MODE_RELU) {
    reluBackward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, output_size);
  } else if (m->activation == AC_MODE_SIGMOID) {
    sigmoid_backward<<<GET_BLOCKS(output_size), CUDA_NUM_THREADS, 0, stream>>>(
        output_grad_ptr, output_ptr, output_size);
  } else {
    // TODO: only support relu and sigmoid for now
    assert(m->activation == AC_MODE_NONE);
  }
  // Compute weight gradiant
  // NOTE: we use alpha=1 for kernel_grad to accumulate gradients
  checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        in_dim, out_dim, batch_size,
                        &alpha, input_ptr, in_dim,
                        output_grad_ptr, out_dim,
                        &alpha, kernel_grad_ptr, in_dim));
  // Compute bias gradiant
  // NOTE: we use alpha=1 for bias_grad to accumulate gradients
  // use_bias = True
  if (bias_grad_ptr != NULL) {
    checkCUDA(cublasSgemv(m->handle.blas, CUBLAS_OP_N,
                          out_dim, batch_size,
                          &alpha, output_grad_ptr, out_dim,
                          m->one_ptr, 1,
                          &alpha, bias_grad_ptr, 1));
  }
  // Compute data gradiant
  // NOTE: we use alpha=1 for input_grad to accumulate gradients
  if (input_grad_ptr != NULL) {
    checkCUDA(cublasSgemm(m->handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                          in_dim, batch_size, out_dim,
                          &alpha, kernel_ptr, in_dim,
                          output_grad_ptr, out_dim,
                          &alpha, input_grad_ptr, in_dim));
  }
}

void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM) \
    case DIM: \
      return backward_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I): input
  regions[1](I/O): replica_grad or input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
template<int NDIM>
__host__
void Linear::backward_task_with_dim(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime)
{
  //Linear* linear = (Linear*) task->args;
  const LinearMeta* m = *((LinearMeta**) task->local_args);
  assert(regions.size() == (5 + int(m->trainableInputs[0]) + int(m->use_bias)));
  assert(task->regions.size() == (5 + int(m->trainableInputs[0]) + int(m->use_bias)));
  float* input_grad = NULL;
  size_t rid = 0;
  TensorAccessorR<float, NDIM> acc_input(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  if (m->trainableInputs[0]) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[rid].region.get_index_space());
    if (domain.get_dim() == NDIM+1) {
      assert(domain.get_volume() == acc_input.rect.volume());
      input_grad = helperGetTensorPointerWO<float>(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
    } else {
      TensorAccessorW<float, NDIM> acc_replica_grad(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
          true/*readOutput*/);
      assert(acc_replica_grad.rect.volume() == acc_input.rect.volume());
      input_grad = acc_replica_grad.ptr;
    }
    rid++;
  }
  TensorAccessorR<float, NDIM> acc_output(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, NDIM> acc_output_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid++;
  TensorAccessorR<float, 3> acc_kernel(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<float, 3> acc_kernel_grad(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
      true/*readOutput*/);
  rid++;
  // make sure the sizes match
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == out_dim * batch_size);
  assert(acc_output_grad.rect.volume() == out_dim * batch_size);
  assert(acc_kernel.rect.volume() == in_dim * out_dim);
  assert(acc_kernel_grad.rect.volume() == in_dim * out_dim);
  float* acc_bias_grad_ptr = NULL;
  if (m->use_bias) {
    TensorAccessorW<float, 3> acc_bias_grad(
        regions[rid], task->regions[rid], FID_DATA, ctx, runtime,
        true/*readOutput*/);
    rid++;
    assert(acc_bias_grad.rect.volume() == out_dim);
    acc_bias_grad_ptr = static_cast<float*>(acc_bias_grad.ptr);
  }
  assert(rid == regions.size());

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  cudaEvent_t t_start, t_end;
  if (m->profiling) {
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_end);
    cudaEventRecord(t_start, stream);
  }
  Linear::backward_kernel(m, acc_input.ptr, input_grad,
      acc_output.ptr, acc_output_grad.ptr,
      acc_kernel.ptr, acc_kernel_grad.ptr,
      acc_bias_grad_ptr, in_dim, out_dim, batch_size, stream);
  if (m->profiling) {
    cudaEventRecord(t_end, stream);
    checkCUDA(cudaEventSynchronize(t_end));
    float elapsed = 0;
    checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_end);
    printf("Linear backward time = %.2lfms\n", elapsed);
    //print_tensor<NDIM, float>(acc_output_grad.ptr, acc_output_grad.rect, "[Linear:backward:output_grad]");
    //print_tensor<2, float>(acc_kernel_grad.ptr, acc_kernel_grad.rect, "[Linear:backward:kernel_grad]");
    //print_tensor<1, float>(acc_bias_grad.ptr, acc_bias_grad.rect, "[Linear:backward:bias_grad]");
    //print_tensor<2, float>(input_grad, acc_input.rect, "[Linear:backward:input_grad]");
  }
}

void Linear::backward(const FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  {
    ArgumentMap argmap;
    set_argumentmap_for_backward(ff, argmap);
    IndexLauncher launcher(LINEAR_BWD_TASK_ID, parallel_is,
                           TaskArgument(NULL, 0), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           outputs[0]->machine_view.hash());
    int rid = 0;
    // regions[0](I): input
    launcher.add_region_requirement(
        RegionRequirement(inputs[0]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, inputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[1](I/O): replica_grad
    assert(replica == NULL);
    launcher.add_region_requirement(
        RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
    // regions[2](I): output
    launcher.add_region_requirement(
        RegionRequirement(outputs[0]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, outputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[3](I/O): output_grad
    launcher.add_region_requirement(
        RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, outputs[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
    // regions[4](I): filter
    launcher.add_region_requirement(
        RegionRequirement(weights[0]->part, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, weights[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[5](I/O): filter_grad
    launcher.add_region_requirement(
        RegionRequirement(weights[0]->part_grad, 0/*projection id*/,
                          READ_WRITE, EXCLUSIVE, weights[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
    if (use_bias) {
      // regions[6](I/O): bias_grad
      launcher.add_region_requirement(
          RegionRequirement(weights[1]->part_grad, 0/*projection id*/,
                            READ_WRITE, EXCLUSIVE, weights[1]->region_grad));
      launcher.add_field(rid++, FID_DATA);
    }
    runtime->execute_index_space(ctx, launcher);
  }
  assert(replica == NULL);
}

/*
__host__
Parameter* Linear::get_parameter(int index)
{
  if (index == 0) {
    return &weights[0];
  } else if (index == 1){
    return &weights[1];
  } else {
    assert(0);
    return NULL;
  }
}
*/

__host__
void Linear::print_layer(const FFModel& ff)
{
  printf("linear layer\n");
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  RegionRequirement kernel_req(weights[0]->region, READ_WRITE, EXCLUSIVE, weights[0]->region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

  RegionRequirement bias_req(weights[1]->region, READ_WRITE, EXCLUSIVE, weights[1]->region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();

  TensorAccessorW<float, 2> acc_kernel(kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
  TensorAccessorW<float, 1> acc_bias(bias_region, bias_req, FID_DATA, ctx, runtime, true);

  const float *kernel_ptr = acc_kernel.ptr;
  const float *bias_ptr = acc_bias.ptr;

  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  size_t bias_size = acc_bias.rect.volume();
  printf("kernel, %p, %zu, [%d, %d]\n", kernel_ptr, kernel_size, kernel_dim1, kernel_dim2);
  printf("bias, %p, %zu\n", bias_ptr, bias_size);

  for (int i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");

  for (int i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");

  runtime->unmap_region(ctx, kernel_region);
  runtime->unmap_region(ctx, bias_region);

}

LinearMeta::LinearMeta(FFHandler handler, int batch_size)
: OpMeta(handler)
{
  // Allocate an all-one's vector
  float* dram_one_ptr = (float *) malloc(sizeof(float) * batch_size);
  for (int i = 0; i < batch_size; i++)
    dram_one_ptr[i] = 1.0f;
  float* fb_one_ptr;
  checkCUDA(cudaMalloc(&fb_one_ptr, sizeof(float) * batch_size));
  checkCUDA(cudaMemcpy(fb_one_ptr, dram_one_ptr,
                       sizeof(float) * batch_size, cudaMemcpyHostToDevice));
  one_ptr = (const float*) fb_one_ptr;
  // Allocate descriptors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
}

bool Linear::measure_operator_cost(Simulator* sim,
                                   const ParallelConfig& pc,
                                   CostMetrics& cost_metrics) const
{
  TensorBase sub_output, sub_input;
  if (!outputs[0]->get_output_sub_tensor(pc, sub_output, OP_LINEAR))
    return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_LINEAR))
    return false;
  int input_c = sub_input.dims[0].size;
  int input_n = sub_input.get_volume() / input_c;
  int output_c = sub_output.dims[0].size;
  int output_n = sub_output.get_volume() / output_c;
  LinearMeta* m = sim->linear_meta;
  m->activation = activation;
  if (use_cudnn_activation(m->activation)) {
    cudnnActivationMode_t mode;
    switch (activation) {
      case AC_MODE_RELU:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case AC_MODE_SIGMOID:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      default:
        // Unsupported activation mode
        assert(false);
    }
    checkCUDNN(cudnnSetActivationDescriptor(m->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnSetTensor4dDescriptor(m->outputTensor,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_n, output_c, 1, 1));
  }
  // allocate tensors in simulator
  sim->free_all();
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  float *output_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  float* kernel_ptr = (float*)sim->allocate((size_t)output_c * input_c, DT_FLOAT);
  float* bias_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
  assert(bias_ptr != NULL);

  cudaStream_t stream;
  checkCUDA(get_legion_stream(&stream));

  bool out_of_memory = (input_ptr == NULL) || (output_ptr == NULL)
                       || (kernel_ptr == NULL) || (bias_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel(m, input_ptr, output_ptr, kernel_ptr, bias_ptr,
        input_c, output_c, input_n, stream);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float* input_grad_ptr = NULL;
    if (trainableInputs[0]) {
      input_grad_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    }
    float *output_grad_ptr = (float*)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    float* kernel_grad_ptr = (float*)sim->allocate((size_t)output_c * input_c, DT_FLOAT);
    float* bias_grad_ptr = (float*)sim->allocate(output_c, DT_FLOAT);
    out_of_memory = (input_grad_ptr == NULL) || (output_grad_ptr == NULL)
                    || (kernel_grad_ptr == NULL) || (bias_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel(m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr,
          kernel_ptr, kernel_grad_ptr, bias_grad_ptr, input_c, output_c, input_n, stream);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf) backward_time(%.4lf)\n",
           name, input_n, input_c, output_n, output_c,
           cost_metrics.forward_time, cost_metrics.backward_time);
  } else {
    log_measure.debug("[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf)\n",
           name, input_n, input_c, output_n, output_c,
           cost_metrics.forward_time);
  }
  return true;
}

bool Linear::estimate_sync_cost(Simulator* sim,
                                const MachineView& view,
                                CostMetrics& cost_metrics) const
{
  // Estimate the cost of sync weights
  TensorShape tensor_shape;
  tensor_shape.num_dims = 3;
  tensor_shape.data_type = DT_FLOAT;
  tensor_shape.dims[0] = inputs[0]->dims[0];
  tensor_shape.dims[1] = inputs[0]->dims[inputs[0]->num_dims-1];
  tensor_shape.dims[2] = inputs[0]->dims[inputs[0]->num_dims-2];
  tensor_shape.dims[1].size = out_channels;
  tensor_shape.dims[1].degree = 1;
  tensor_shape.dims[2].degree = inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  tensor_shape.dims[2].size = inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  cost_metrics.sync_time = sim->default_estimate_sync_cost(tensor_shape, view, 1);
  //printf("[Estimate Linear] name(%s) sync_time(%.4lf)\n", name, cost_metrics.sync_time);
  return true;
}

ParallelConfig Linear::get_random_parallel_config(const FFModel& ff) const
{
  if (!ff.config.enable_parameter_parallel)
    return Op::get_random_parallel_config(ff);
  std::vector<int> batch_candidates;
  std::vector<int> channel_candidates;
  int batch = outputs[0]->dims[outputs[0]->num_dims-1].size;
  int channel = outputs[0]->dims[0].size;
  int total_devices = ff.config.workersPerNode * ff.config.numNodes;
  for (int i = 1; i <= ff.config.workersPerNode; i++)
    if (channel % i == 0)
      for (int j = 1; i * j <= total_devices; j++)
        if (batch % j == 0) {
          batch_candidates.push_back(j);
          channel_candidates.push_back(i);
        }
  assert(batch_candidates.size() > 0);
  int idx = std::rand() % batch_candidates.size();
  int num_par_c = channel_candidates[idx];
  int num_par_b = batch_candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  pc.dim[0] = num_par_c;
  pc.dim[pc.nDims-1] = num_par_b;
  for (int i = 1; i < pc.nDims - 1; i++)
    pc.dim[i] = 1;
  int start_idx = std::rand() % (total_devices - num_par_c * num_par_b + 1);
  start_idx = start_idx - start_idx % num_par_c;
  for (int i = 0; i < num_par_c * num_par_b; i++)
    pc.device_ids[i] = start_idx + i;
  return pc;
}

bool Linear::get_int_parameter(PMParameter para, int* value) const
{
  switch(para) {
    case PM_ACTI:
      *value = (int) activation;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Linear::is_valid_parallel_config(const FFModel& ff, const ParallelConfig& pc) const
{
  if (!ff.config.enable_parameter_parallel)
    return Op::is_valid_parallel_config(ff, pc);
  // Support data and parameter parallel
  if (pc.nDims != outputs[0]->num_dims)
    return false;
  for (int i = 1; i < pc.nDims-1; i++)
    if (pc.dim[i] != 1)
      return false;
  return true;
}
