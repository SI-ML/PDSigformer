
// Derived from PyTorch v2.0.0/v2.8.0 ActivationGeluKernel.cu, approximate=none.
// float32 opmath semantics, exact erf expression; no fast-math flag.
extern "C" __global__ void gelu_exact(const float* x, float* y, long long n) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    constexpr float kAlpha = 0.707106781186547524400844362104849039;
    const float v = x[i];
    y[i] = v * float(0.5) * (float(1) + erf(v * kAlpha));
  }
}
