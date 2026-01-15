// filter.cu  (NO OpenCV dependency)
// Uses stb_image + stb_image_write for I/O (PNG/JPG).
// CUDA applies 3x3 W filter per channel with zero-border.
//
// Build:
//   nvcc -O3 -std=c++17 filter.cu -o filter
//
// Run:
//   ./filter input.png output.png blockX blockY [reps]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__device__ __forceinline__ unsigned char clamp_u8(float x) {
  x = fminf(255.0f, fmaxf(0.0f, x));
  return (unsigned char)(x + 0.5f);
}

// Input/Output are interleaved RGB (3 channels)
__global__ void wfilter_rgb_u8(const unsigned char* __restrict__ in,
                               unsigned char* __restrict__ out,
                               int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int w00 = 1, w01 = 2, w02 = 1;
  const int w10 = 3, w11 = 4, w12 = 3;
  const int w20 = 1, w21 = 2, w22 = 1;

  auto at = [&](int xx, int yy, int c) -> float {
    if (xx < 0 || yy < 0 || xx >= width || yy >= height) return 0.0f;
    int idx = (yy * width + xx) * 3 + c;
    return (float)in[idx];
  };

  int out_idx = (y * width + x) * 3;

  for (int c = 0; c < 3; ++c) {
    float acc = 0.0f;
    acc += w00 * at(x - 1, y - 1, c);
    acc += w01 * at(x + 0, y - 1, c);
    acc += w02 * at(x + 1, y - 1, c);

    acc += w10 * at(x - 1, y + 0, c);
    acc += w11 * at(x + 0, y + 0, c);
    acc += w12 * at(x + 1, y + 0, c);

    acc += w20 * at(x - 1, y + 1, c);
    acc += w21 * at(x + 0, y + 1, c);
    acc += w22 * at(x + 1, y + 1, c);

    acc *= (1.0f / 16.0f);
    out[out_idx + c] = clamp_u8(acc);
  }
}

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage: %s <input_png/jpg> <output_png> <blockX> <blockY> [reps]\n",
          prog);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    usage(argv[0]);
    return 1;
  }

  std::string input_path  = argv[1];
  std::string output_path = argv[2];
  int blockX = std::atoi(argv[3]);
  int blockY = std::atoi(argv[4]);
  int reps   = (argc >= 6) ? std::max(1, std::atoi(argv[5])) : 1;

  if (blockX <= 0 || blockY <= 0) {
    fprintf(stderr, "Error: blockX/blockY must be positive.\n");
    return 1;
  }

  int w = 0, h = 0, comp = 0;
  unsigned char* img = stbi_load(input_path.c_str(), &w, &h, &comp, 3);
  if (!img) {
    fprintf(stderr, "Error: cannot read image: %s\n", input_path.c_str());
    return 1;
  }

  size_t bytes = (size_t)w * (size_t)h * 3;

  unsigned char* d_in = nullptr;
  unsigned char* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaEvent_t e0, e1, e2, e3, ek0, ek1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  CUDA_CHECK(cudaEventCreate(&e3));
  CUDA_CHECK(cudaEventCreate(&ek0));
  CUDA_CHECK(cudaEventCreate(&ek1));

  dim3 block(blockX, blockY);
  dim3 grid((w + block.x - 1) / block.x,
            (h + block.y - 1) / block.y);

  CUDA_CHECK(cudaEventRecord(e0, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_in, img, bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(e1, stream));

  // warmup
  wfilter_rgb_u8<<<grid, block, 0, stream>>>(d_in, d_out, w, h);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(ek0, stream));
  for (int r = 0; r < reps; ++r) {
    wfilter_rgb_u8<<<grid, block, 0, stream>>>(d_in, d_out, w, h);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(ek1, stream));

  unsigned char* out = (unsigned char*)std::malloc(bytes);
  if (!out) {
    fprintf(stderr, "Error: malloc failed.\n");
    return 1;
  }

  CUDA_CHECK(cudaEventRecord(e2, stream));
  CUDA_CHECK(cudaMemcpyAsync(out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(e3, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  float h2d_ms=0, kernel_ms_total=0, d2h_ms=0, total_ms=0;
  CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms_total, ek0, ek1));
  CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, e2, e3));
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e3));
  float kernel_ms_avg = kernel_ms_total / (float)reps;

  // write PNG (stride = w*3)
  int ok = stbi_write_png(output_path.c_str(), w, h, 3, out, w * 3);
  if (!ok) {
    fprintf(stderr, "Error: cannot write output PNG: %s\n", output_path.c_str());
    return 1;
  }

  printf("RESULT,%s,%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
         input_path.c_str(), output_path.c_str(),
         w, h, blockX, blockY, reps,
         h2d_ms, kernel_ms_avg, d2h_ms, total_ms);

  stbi_image_free(img);
  std::free(out);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
  cudaEventDestroy(ek0); cudaEventDestroy(ek1);
  cudaStreamDestroy(stream);

  return 0;
}


