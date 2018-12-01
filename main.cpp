#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <utility>
#include <omp.h>
#include <PZSDKHelper.h>
#include <pzcl/pzcl_ocl_wrapper.h>
#include "types.hpp"
#include "solver.hpp"

constexpr int MAX_BIN_SIZE = 1000000;

const char *getErrorString(cl_int error)
{
  switch(error){
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
  }
}

using pfnPezyExtSetPerThreadStackSize = CL_API_ENTRY cl_int (CL_API_CALL *) (pzcl_kernel kernel, size_t size);

std::string to_string(const Board& bd) {
  std::string res;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      int index = i*8+j;
      if ((bd.me >> index) & 1) {
        res += 'x';
      } else if ((bd.op >> index) & 1) {
        res += 'o';
      } else {
        res += ' ';
      }
    }
    res += '\n';
  }
  return res;
}

class NQueenExpand {
 public:
  NQueenExpand(int N, int M) : N(N), M(M) {}
  int N, M;
  std::vector<Problem> probs;
  void expand(int depth = 0, uint32_t left = 0, uint32_t mid = 0, uint32_t right = 0) {
    if (depth == M) {
      probs.push_back((Problem){N, depth, left, mid, right});
      return;
    }
    for (uint32_t pos = (((uint32_t)1 << N) - 1) & ~(left | mid | right);
      pos; pos &= pos-1) {
      uint32_t bit = pos & -pos;
      expand(depth+1, (left | bit) << 1, mid | bit, (right | bit) >> 1);
    }
  }
};

uint64_t solve(int N, int depth, uint32_t left, uint32_t mid, uint32_t right) {
  if (depth == N) return 1;
  uint64_t sum = 0;
  for (uint32_t pos = (((uint32_t)1 << N) - 1) & ~(left | mid | right);
      pos; pos &= pos-1) {
    uint32_t bit = pos & -pos;
    sum += solve(N, depth+1, (left | bit) << 1, mid | bit, (right | bit) >> 1);
  }
  return sum;
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  int M = atoi(argv[2]);
  std::cerr << N << ' ' << M << std::endl;
  NQueenExpand nqe(N, M);
  nqe.expand();
  const size_t length = nqe.probs.size();
  std::cerr << length << std::endl;

  cl_platform_id platform_id = nullptr;
  cl_uint num_platforms = 0;
  cl_int result = 0;
  result = clGetPlatformIDs(1, &platform_id, &num_platforms);
  std::cerr << "Number of platforms: " << num_platforms << std::endl;

  cl_device_id device_id = nullptr;
  cl_uint num_devices = 0;
  result = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);

  cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &result);

  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &result);

  std::cerr << "load program" << std::endl;
  unsigned char *binary = (unsigned char *)malloc(MAX_BIN_SIZE * sizeof(char));
  FILE *fp = fopen("kernel.sc1-64/solver.pz", "rb");
  std::size_t size = fread(binary, sizeof(char), MAX_BIN_SIZE, fp);
  fclose(fp);

  std::cerr << "create program" << std::endl;
  cl_int binary_status = 0;
  cl_program program = clCreateProgramWithBinary(context, 1, &device_id, &size, (const unsigned char **)&binary, &binary_status, &result);
  
  std::cerr << "create kernel" << std::endl;
  cl_kernel kernel = clCreateKernel(program, "Solve", &result);

  std::cerr << "create buffer" << std::endl;
  const size_t global_work_size = 8192; // max size
  cl_mem memProb = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Problem)*length, nullptr, &result);
  cl_mem memRes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint64_t)*global_work_size, nullptr, &result);

  clEnqueueWriteBuffer(command_queue, memProb, CL_TRUE, 0, sizeof(Problem)*length, nqe.probs.data(), 0, nullptr, nullptr);

  result = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memProb);
  if (result != CL_SUCCESS) {
    std::cerr << "kernel arg set error: " << getErrorString(result) << std::endl;
  } else {
    std::cerr << "kernel arg set success" << std::endl;
  }
  result = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memRes);
  if (result != CL_SUCCESS) {
    std::cerr << "kernel arg set error: " << getErrorString(result) << std::endl;
  } else {
    std::cerr << "kernel arg set success" << std::endl;
  }
  result = clSetKernelArg(kernel, 2, sizeof(size_t), (void *)&length);
  if (result != CL_SUCCESS) {
    std::cerr << "kernel arg set error: " << getErrorString(result) << std::endl;
  } else {
    std::cerr << "kernel arg set success" << std::endl;
  }
  
  std::cerr << "start" << std::endl;
  auto start = std::chrono::system_clock::now();
  result = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, nullptr);
  if (result != CL_SUCCESS) {
    std::cerr << "kernel launch error: " << getErrorString(result) << std::endl;
  } else {
    std::cerr << "kernel launch success" << std::endl;
  }

  std::vector<uint64_t> results(global_work_size);
  result = clEnqueueReadBuffer(command_queue, memRes, CL_TRUE, 0, sizeof(uint64_t)*global_work_size, results.data(), 0, nullptr, nullptr);
  if (result != CL_SUCCESS) {
    std::cerr << "readbuffer launch error: " << getErrorString(result) << std::endl;
  }
  auto end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  std::cerr << "elapsed: " << elapsed << std::endl;

  uint64_t sum = 0;
  for (size_t i = 0; i < global_work_size; ++i) sum += results[i];
  std::cerr << sum << std::endl;

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(memProb);
  clReleaseMemObject(memRes);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  free(binary);

  return 0;
}
