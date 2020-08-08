#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define USE_MKL

#ifdef USE_MKL
#include <mkl.h>
#endif
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#define D (80 * 80)
#define H (100)
#define A (1)

void print_w(float* w, size_t n) {
  for (size_t ii = 0; ii < n; ii++) printf("%4zu: %.7f\n", ii, w[ii]);
}

// c = a * b
void BLAS_mmul( float* __restrict c, float* __restrict a, float* __restrict b,
                int aT, int bT, size_t c_rows, size_t c_cols, size_t a_rows, size_t a_cols ) {

#ifdef USE_MKL
  mkl_set_num_threads(1);
#endif
#ifdef USE_OPENBLAS
  openblas_set_num_threads(1);
#endif

  CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

  size_t M = c_rows; size_t N = c_cols; size_t K = aT ? a_rows : a_cols;
  float alpha = 1.0f; float beta = 0.0f;
  size_t lda = aT ? K : M; size_t ldb = bT ? N : K; size_t ldc = M;

  cblas_sgemm( CblasColMajor, transA, transB, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc );

}

float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

void forward(float *x, float* h, float* logp, float* p, float* w1, float* w2) {

  //printf("%p %p %p w1[0] %f w1[1] %f w1[2] %f\n", x, w1, w2, w1[0], w1[1], w1[2]);
  // h = dot(w1, x)
  BLAS_mmul( h, w1, x, 1, 1, H, 1, D, H);
  // relu h = h[h>0]
  for (size_t hh = 0; hh < H; hh++) {
    h[hh] = h[hh] < 0.0f ? 0.0f : h[hh];
  }
  // logp = dot(w2,h)
  BLAS_mmul( logp, w2, h, 1, 1, 1, 1, H, 1);
  *p = sigmoid(*logp);
}

void copy1(float* dst, unsigned int idx, float val) {
  dst[idx] = val;
}

void copyn(float* dst, unsigned int idx, float* src, unsigned int n) {
  memcpy(&dst[idx], src, n * sizeof(float));
}

void adapt(float *w, float *m, float *g, unsigned int n, float lr, float decay) {
  for (unsigned int i=0; i < n; i++) {
    float _g = g[i];
    m[i] = decay * m[i] + (1.0f-decay)*_g*_g;
    float _m = _g / ( sqrtf(m[i]) + 0.0001f);
    w[i] = w[i] + lr * _m;
  }
}

void modulate(float *logp, float* r, float gamma, unsigned int len) {

  float running_add = 0.0f; float sum = 0.0f;

  for (unsigned int j=0; j<len; j++) {
    if (fabs(r[len-j-1]) > 0.0f) running_add = 0.0f;
    running_add = running_add * gamma + r[len-j-1];
    r[len-j-1] = running_add;
    sum += r[len-j-1];
  }

  float mean = sum / (float)len;

  for (unsigned int i = 0; i < len; i++) {
    logp[i] *= (r[i] - mean);
  }
}

void backward(float *eph, float* epdlogp, float *epx, float* dW1, float* dW2, float* dh, float *W2, int ep_length) {

  BLAS_mmul( dW2, eph, epdlogp, 0, 1, H, 1, H, ep_length);
  BLAS_mmul( dh, W2, epdlogp, 0, 1, H, ep_length, H, 1);

  for (size_t hh = 0; hh < H*ep_length; hh++) {
    dh[hh] = eph[hh] <= 0.0f ? 0.0f : dh[hh];
  }

  BLAS_mmul( dW1, epx, dh, 0, 1, D, H, H, ep_length);
}
