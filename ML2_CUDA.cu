#include <cuda_runtime.h>
#include <limits>

// NOTE(8/10/2026 2:18:05): some function declarations use restrict,
//                          C++ compilers wont compile that so this is a hack
#define restrict __restrict
#include "ML2.h"
#undef restrict

#ifdef _MSC_VER
    #define ML2_CUDA_UNREACHABLE(msg, ...)                       \
        do {                                                     \
            printf("CUDA UNREACHABLE: \"" msg "\"\n"             \
                   "    file:     |%s|\n"                        \
                   "    function: |%s|\n"                        \
                   "    line:     |%d|\n",                       \
                   ##__VA_ARGS__, __FILE__, __func__, __LINE__); \
            __trap();                                            \
        } while (0)
#else
    #define ML2_CUDA_UNREACHABLE(msg, ...)                                  \
        do {                                                                \
            printf("CUDA UNREACHABLE: \"" msg "\"\n"                        \
                   "    file:     |%s|\n"                                   \
                   "    function: |%s|\n"                                   \
                   "    line:     |%d|\n",                                  \
                   __VA_OPT__(__VA_ARGS__, ) __FILE__, __func__, __LINE__); \
            __trap();                                                       \
        } while (0)
#endif

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Exp(ML2_Scalar x) {
    return expf(x);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Exp2(ML2_Scalar x) {
    return exp2f(x);
}

// __device__ static __forceinline__ ML2_Scalar ML2_CUDA_NLog(ML2_Scalar x) {
//     return logf(x);
// }

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Log2(ML2_Scalar x) {
    return log2f(x);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Sqrt(ML2_Scalar x) {
    return sqrtf(x);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Sine(ML2_Scalar x) {
    return sinf(x);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Cosine(ML2_Scalar x) {
    return cosf(x);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_HTangent(ML2_Scalar x) {
    return tanhf(x);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Fmax(ML2_Scalar a, ML2_Scalar b) {
    return fmaxf(a, b);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_Fmin [[maybe_unused]](ML2_Scalar a, ML2_Scalar b) {
    return fminf(a, b);
}

__device__ static constexpr ML2_Scalar ML2_CUDA_Epsilon = ML2_SCALAR(1e-7);
__device__ static constexpr ML2_Scalar ML2_CUDA_Pi [[maybe_unused]] = ML2_SCALAR(3.141592653589793238);
// TODO(6/10/2026 5:15:50): i fucking hate msvc
#ifdef _MSC_VER
__device__ static constexpr ML2_Scalar ML2_CUDA_NegInf = -std::numeric_limits<ML2_Scalar>::infinity();
#else
__device__ static constexpr ML2_Scalar ML2_CUDA_NegInf = ML2_SCALAR(-INFINITY);
#endif
__device__ static constexpr ML2_Scalar ML2_CUDA_Zero = ML2_SCALAR(0.0);
__device__ static constexpr ML2_Scalar ML2_CUDA_One = ML2_SCALAR(1.0);
// NOTE(7/11/2026 16:21:04): ML2_CUDA_Log2e = Log2(e);
__device__ static constexpr ML2_Scalar ML2_CUDA_Log2e = ML2_SCALAR(1.442695040888963387);

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarReLUForward(ML2_Scalar input) {
    return input > ML2_CUDA_Zero ? input : ML2_CUDA_Zero;
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarReLUBackwardInput(ML2_Scalar input) {
    return input > ML2_CUDA_Zero ? ML2_CUDA_One : ML2_CUDA_Zero;
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarSigmoidForward(ML2_Scalar input) {
    return ML2_CUDA_One / (ML2_CUDA_One + ML2_CUDA_Exp(-input));
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarSigmoidBackwardOutput(ML2_Scalar output) {
    return output * (ML2_CUDA_One - output);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarSinForward(ML2_Scalar input) {
    return ML2_CUDA_Sine(input);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarSinBackwardInput(ML2_Scalar input) {
    return ML2_CUDA_Cosine(input);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarCosForward(ML2_Scalar input) {
    return ML2_CUDA_Cosine(input);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarCosBackwardInput(ML2_Scalar input) {
    return -ML2_CUDA_Sine(input);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarTanhForward(ML2_Scalar input) {
    return ML2_CUDA_HTangent(input);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarTanhBackwardOutput(ML2_Scalar output) {
    return ML2_CUDA_One - output * output;
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarLeakyReLUForward(ML2_Scalar input) {
    return input > ML2_CUDA_Zero ? input : input * ML2_SCALAR(0.01);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarLeakyReLUBackwardInput(ML2_Scalar input) {
    return input > ML2_CUDA_Zero ? ML2_CUDA_One : ML2_SCALAR(0.01);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarSiLUForward(ML2_Scalar input) {
    return input * ML2_CUDA_ScalarSigmoidForward(input);
}

__device__ static __forceinline__ ML2_Scalar ML2_CUDA_ScalarSiLUBackwardInputOutput(ML2_Scalar input, ML2_Scalar output) {
    return ML2_CUDA_ScalarSigmoidForward(input) * (ML2_CUDA_One + input - output);
}

extern "C" {

__global__ void ML2_CUDA_FunctionSum(
    ML2_Int scalars,
    ML2_Scalar *__restrict__ dst, const ML2_Scalar *__restrict__ src
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;
    dst[c] += src[c];
}

__global__ void ML2_CUDA_FunctionSumMatrixVector(
    ML2_Int rows, ML2_Int cols,
    ML2_Scalar *__restrict__ dst,    // [rows][cols]
    const ML2_Scalar *__restrict__ V // [cols]
) {
    ML2_Int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= rows * cols) return;
    ML2_Int c = idx % cols;
    dst[idx] += V[c];
    // TODO(21/10/2026 18:52:36): in the future i should do this in case the number of threads hit a cuda hardware limit
    // ML2_Int scalars = rows * cols;
    // ML2_Int threads = gridDim.x * blockDim.x;
    // for (
    //     ML2_Int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    //     idx < scalars;
    //     idx += threads
    // ) {
    //     ML2_Int c = idx % cols;
    //     dst[idx] += V[c];
    // }
}

__global__ void ML2_CUDA_FunctionSumVectorMatrix(
    ML2_Int rows, ML2_Int cols,
    ML2_Scalar *__restrict__ dst,    // [cols]
    const ML2_Scalar *__restrict__ M // [rows][cols]
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= cols) return;

    ML2_Scalar sum = {};
    for (ML2_Int r = 0; r < rows; r++) {
        sum += M[r * cols + c];
    }
    dst[c] += sum;
}

__global__ void ML2_CUDA_FunctionActivationForward(
    ML2_Int scalars,
    ML2_ActivationType activation,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ output
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;
    switch (activation) {
        case ML2_ActivationTypeReLU: {
            output[c] = ML2_CUDA_ScalarReLUForward(input[c]);
        } break;
        case ML2_ActivationTypeSigmoid: {
            output[c] = ML2_CUDA_ScalarSigmoidForward(input[c]);
        } break;
        case ML2_ActivationTypeSin: {
            output[c] = ML2_CUDA_ScalarSinForward(input[c]);
        } break;
        case ML2_ActivationTypeCos: {
            output[c] = ML2_CUDA_ScalarCosForward(input[c]);
        } break;
        case ML2_ActivationTypeTanh: {
            output[c] = ML2_CUDA_ScalarTanhForward(input[c]);
        } break;
        case ML2_ActivationTypeLeakyReLU: {
            output[c] = ML2_CUDA_ScalarLeakyReLUForward(input[c]);
        } break;
        case ML2_ActivationTypeSiLU: {
            output[c] = ML2_CUDA_ScalarSiLUForward(input[c]);
        } break;
        default: ML2_CUDA_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

__global__ void ML2_CUDA_FunctionActivationCacheBackwardInput(
    ML2_Int scalars,
    ML2_ActivationType activation,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;
    switch (activation) {
        case ML2_ActivationTypeReLU: {
            inputGradient[c] += ML2_CUDA_ScalarReLUBackwardInput(input[c]) * outputGradient[c];
        } break;
        case ML2_ActivationTypeSigmoid: ML2_CUDA_UNREACHABLE("Sigmoid backward takes the output");
        case ML2_ActivationTypeSin: {
            inputGradient[c] += ML2_CUDA_ScalarSinBackwardInput(input[c]) * outputGradient[c];
        } break;
        case ML2_ActivationTypeCos: {
            inputGradient[c] += ML2_CUDA_ScalarCosBackwardInput(input[c]) * outputGradient[c];
        } break;
        case ML2_ActivationTypeTanh: ML2_CUDA_UNREACHABLE("Tanh backward takes the output");
        case ML2_ActivationTypeLeakyReLU: {
            inputGradient[c] += ML2_CUDA_ScalarLeakyReLUBackwardInput(input[c]) * outputGradient[c];
        } break;
        case ML2_ActivationTypeSiLU: ML2_CUDA_UNREACHABLE("SiLU backward takes both the input and the output");
        default: ML2_CUDA_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

__global__ void ML2_CUDA_FunctionActivationCacheBackwardOutput(
    ML2_Int scalars,
    ML2_ActivationType activation,
    ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ output, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;
    switch (activation) {
        case ML2_ActivationTypeReLU: ML2_CUDA_UNREACHABLE("ReLU backward takes the input");
        case ML2_ActivationTypeSigmoid: {
            inputGradient[c] += ML2_CUDA_ScalarSigmoidBackwardOutput(output[c]) * outputGradient[c];
        } break;
        case ML2_ActivationTypeSin: ML2_CUDA_UNREACHABLE("Sin backward takes the input");
        case ML2_ActivationTypeCos: ML2_CUDA_UNREACHABLE("Cos backward takes the input");
        case ML2_ActivationTypeTanh: {
            inputGradient[c] += ML2_CUDA_ScalarTanhBackwardOutput(output[c]) * outputGradient[c];
        } break;
        case ML2_ActivationTypeLeakyReLU: ML2_CUDA_UNREACHABLE("LeakyReLU backward takes the input");
        case ML2_ActivationTypeSiLU: ML2_CUDA_UNREACHABLE("SiLU backward takes both the input and the output");
        default: ML2_CUDA_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

__global__ void ML2_CUDA_FunctionActivationCacheBackwardInputOutput(
    ML2_Int scalars,
    ML2_ActivationType activation,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ output, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;
    switch (activation) {
        case ML2_ActivationTypeReLU: ML2_CUDA_UNREACHABLE("ReLU backward takes the input");
        case ML2_ActivationTypeSigmoid: ML2_CUDA_UNREACHABLE("Sigmoid backward takes the output");
        case ML2_ActivationTypeSin: ML2_CUDA_UNREACHABLE("Sin backward takes the input");
        case ML2_ActivationTypeCos: ML2_CUDA_UNREACHABLE("Cos backward takes the input");
        case ML2_ActivationTypeTanh: ML2_CUDA_UNREACHABLE("Tanh backward takes the output");
        case ML2_ActivationTypeLeakyReLU: ML2_CUDA_UNREACHABLE("LeakyReLU backward takes the input");
        case ML2_ActivationTypeSiLU: {
            inputGradient[c] += ML2_CUDA_ScalarSiLUBackwardInputOutput(input[c], output[c]) * outputGradient[c];
        } break;
        default: ML2_CUDA_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

__global__ void ML2_CUDA_FunctionAttentionForwardMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ output
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Int keyVectorsEnd = qv + 1;
    ML2_Scalar scaleScalar = ML2_CUDA_Log2e / ML2_CUDA_Sqrt(headScalars);
    ML2_Int headOffset = h * headScalars;

    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Scalar scoreMax = {};
    /* kv = 0 */ {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreMax += inputQueryPtr[hc] * inputKeyPtr[hc];
            outputPtr[hc] = inputValuePtr[hc];
        }
        scoreMax *= scaleScalar;
    }
    ML2_Scalar softmaxSum = ML2_CUDA_One;
    for (ML2_Int kv = 1; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleScalar;
        ML2_Scalar scoreNewMax = ML2_CUDA_Fmax(scoreMax, scoreScalar);
        ML2_Scalar correction = ML2_CUDA_Exp2(scoreMax - scoreNewMax);
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreNewMax);
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            outputPtr[hc] = outputPtr[hc] * correction + inputValuePtr[hc] * softmaxScalar;
        }
        softmaxSum = softmaxSum * correction + softmaxScalar;
        scoreMax = scoreNewMax;
    }
    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        outputPtr[hc] /= softmaxSum;
    }
}

__global__ void ML2_CUDA_FunctionAttentionForwardStateMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    ML2_Scalar *__restrict__ state, const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ output
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Int keyVectorsEnd = qv + 1;
    ML2_Scalar scaleScalar = ML2_CUDA_Log2e / ML2_CUDA_Sqrt(headScalars);
    ML2_Int headOffset = h * headScalars;

    ML2_Scalar *__restrict__ statePtr = &state[(s * queryVectors + qv) * heads];
    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Scalar scoreMax = {};
    /* kv = 0 */ {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreMax += inputQueryPtr[hc] * inputKeyPtr[hc];
            outputPtr[hc] = inputValuePtr[hc];
        }
        scoreMax *= scaleScalar;
    }
    ML2_Scalar softmaxSum = ML2_CUDA_One;
    for (ML2_Int kv = 1; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleScalar;
        ML2_Scalar scoreNewMax = ML2_CUDA_Fmax(scoreMax, scoreScalar);
        ML2_Scalar correction = ML2_CUDA_Exp2(scoreMax - scoreNewMax);
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreNewMax);
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            outputPtr[hc] = outputPtr[hc] * correction + inputValuePtr[hc] * softmaxScalar;
        }
        softmaxSum = softmaxSum * correction + softmaxScalar;
        scoreMax = scoreNewMax;
    }
    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        outputPtr[hc] /= softmaxSum;
    }
    statePtr[h] = scoreMax + ML2_CUDA_Log2(softmaxSum);
}

__global__ void ML2_CUDA_FunctionAttentionCacheBackwardMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ output, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Scalar scaleScalar = ML2_CUDA_One / ML2_CUDA_Sqrt(headScalars);
    ML2_Scalar scaleLog2eScalar = ML2_CUDA_Log2e * scaleScalar;
    ML2_Int headOffset = h * headScalars;

    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ inputGradientQueryPtr = &inputGradient[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputGradientPtr = &outputGradient[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Int keyVectorsEnd = qv + 1;

    ML2_Scalar scoreMax = {};
    /* kv = 0 */ {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreMax += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreMax *= scaleScalar;
    }
    ML2_Scalar softmaxSum = ML2_CUDA_One;
    for (ML2_Int kv = 1; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleScalar;
        ML2_Scalar scoreNewMax = ML2_CUDA_Fmax(scoreMax, scoreScalar);
        ML2_Scalar correction = ML2_CUDA_Exp2(scoreMax - scoreNewMax);
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreNewMax);
        softmaxSum = softmaxSum * correction + softmaxScalar;
        scoreMax = scoreNewMax;
    }

    ML2_Scalar softmaxGradientSum = {};
    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        softmaxGradientSum += outputPtr[hc] * outputGradientPtr[hc];
    }

    for (ML2_Int kv = 0; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientKeyPtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientValuePtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleLog2eScalar;
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreMax) / softmaxSum;

        ML2_Scalar softmaxGradientScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            softmaxGradientScalar += inputValuePtr[hc] * outputGradientPtr[hc];
            atomicAdd(&inputGradientValuePtr[hc], softmaxScalar * outputGradientPtr[hc]);
        }

        ML2_Scalar scoreGradientScalar = softmaxScalar * (softmaxGradientScalar - softmaxGradientSum) * scaleScalar;

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            inputGradientQueryPtr[hc] += inputKeyPtr[hc] * scoreGradientScalar;
            atomicAdd(&inputGradientKeyPtr[hc], inputQueryPtr[hc] * scoreGradientScalar);
        }
    }
}

__global__ void ML2_CUDA_FunctionAttentionCacheBackwardStateMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    const ML2_Scalar *__restrict__ state, const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ output, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Scalar scaleScalar = ML2_CUDA_One / ML2_CUDA_Sqrt(headScalars);
    ML2_Scalar scaleLog2eScalar = ML2_CUDA_Log2e * scaleScalar;
    ML2_Int headOffset = h * headScalars;

    const ML2_Scalar *__restrict__ statePtr = &state[(s * queryVectors + qv) * heads];
    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ inputGradientQueryPtr = &inputGradient[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputGradientPtr = &outputGradient[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Int keyVectorsEnd = qv + 1;

    ML2_Scalar maxLogSum = statePtr[h];
    ML2_Scalar softmaxGradientSum = {};

    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        softmaxGradientSum += outputPtr[hc] * outputGradientPtr[hc];
    }

    for (ML2_Int kv = 0; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientKeyPtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientValuePtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar * scaleLog2eScalar - maxLogSum);

        ML2_Scalar softmaxGradientScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            softmaxGradientScalar += inputValuePtr[hc] * outputGradientPtr[hc];
            atomicAdd(&inputGradientValuePtr[hc], softmaxScalar * outputGradientPtr[hc]);
        }

        ML2_Scalar scoreGradientScalar = softmaxScalar * (softmaxGradientScalar - softmaxGradientSum) * scaleScalar;

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            inputGradientQueryPtr[hc] += inputKeyPtr[hc] * scoreGradientScalar;
            atomicAdd(&inputGradientKeyPtr[hc], inputQueryPtr[hc] * scoreGradientScalar);
        }
    }
}

__global__ void ML2_CUDA_FunctionAttentionFullForwardMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ output
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Int keyVectorsEnd = queryVectors;
    ML2_Scalar scaleScalar = ML2_CUDA_Log2e / ML2_CUDA_Sqrt(headScalars);
    ML2_Int headOffset = h * headScalars;

    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Scalar scoreMax = {};
    /* kv = 0 */ {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreMax += inputQueryPtr[hc] * inputKeyPtr[hc];
            outputPtr[hc] = inputValuePtr[hc];
        }
        scoreMax *= scaleScalar;
    }
    ML2_Scalar softmaxSum = ML2_CUDA_One;
    for (ML2_Int kv = 1; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleScalar;
        ML2_Scalar scoreNewMax = ML2_CUDA_Fmax(scoreMax, scoreScalar);
        ML2_Scalar correction = ML2_CUDA_Exp2(scoreMax - scoreNewMax);
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreNewMax);
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            outputPtr[hc] = outputPtr[hc] * correction + inputValuePtr[hc] * softmaxScalar;
        }
        softmaxSum = softmaxSum * correction + softmaxScalar;
        scoreMax = scoreNewMax;
    }
    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        outputPtr[hc] /= softmaxSum;
    }
}

__global__ void ML2_CUDA_FunctionAttentionFullForwardStateMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    ML2_Scalar *__restrict__ state, const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ output
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Int keyVectorsEnd = queryVectors;
    ML2_Scalar scaleScalar = ML2_CUDA_Log2e / ML2_CUDA_Sqrt(headScalars);
    ML2_Int headOffset = h * headScalars;

    ML2_Scalar *__restrict__ statePtr = &state[(s * queryVectors + qv) * heads];
    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Scalar scoreMax = {};
    /* kv = 0 */ {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreMax += inputQueryPtr[hc] * inputKeyPtr[hc];
            outputPtr[hc] = inputValuePtr[hc];
        }
        scoreMax *= scaleScalar;
    }
    ML2_Scalar softmaxSum = ML2_CUDA_One;
    for (ML2_Int kv = 1; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleScalar;
        ML2_Scalar scoreNewMax = ML2_CUDA_Fmax(scoreMax, scoreScalar);
        ML2_Scalar correction = ML2_CUDA_Exp2(scoreMax - scoreNewMax);
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreNewMax);
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            outputPtr[hc] = outputPtr[hc] * correction + inputValuePtr[hc] * softmaxScalar;
        }
        softmaxSum = softmaxSum * correction + softmaxScalar;
        scoreMax = scoreNewMax;
    }
    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        outputPtr[hc] /= softmaxSum;
    }
    statePtr[h] = scoreMax + ML2_CUDA_Log2(softmaxSum);
}

__global__ void ML2_CUDA_FunctionAttentionFullCacheBackwardMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ output, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Scalar scaleScalar = ML2_CUDA_One / ML2_CUDA_Sqrt(headScalars);
    ML2_Scalar scaleLog2eScalar = ML2_CUDA_Log2e * scaleScalar;
    ML2_Int headOffset = h * headScalars;

    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ inputGradientQueryPtr = &inputGradient[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputGradientPtr = &outputGradient[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Int keyVectorsEnd = queryVectors;
    
    ML2_Scalar scoreMax = {};
    /* kv = 0 */ {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[(s * queryVectors * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreMax += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreMax *= scaleScalar;
    }
    ML2_Scalar softmaxSum = ML2_CUDA_One;
    for (ML2_Int kv = 1; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleScalar;
        ML2_Scalar scoreNewMax = ML2_CUDA_Fmax(scoreMax, scoreScalar);
        ML2_Scalar correction = ML2_CUDA_Exp2(scoreMax - scoreNewMax);
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreNewMax);
        softmaxSum = softmaxSum * correction + softmaxScalar;
        scoreMax = scoreNewMax;
    }

    ML2_Scalar softmaxGradientSum = {};
    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        softmaxGradientSum += outputPtr[hc] * outputGradientPtr[hc];
    }

    for (ML2_Int kv = 0; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientKeyPtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientValuePtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        scoreScalar *= scaleLog2eScalar;
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar - scoreMax) / softmaxSum;

        ML2_Scalar softmaxGradientScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            softmaxGradientScalar += inputValuePtr[hc] * outputGradientPtr[hc];
            atomicAdd(&inputGradientValuePtr[hc], softmaxScalar * outputGradientPtr[hc]);
        }

        ML2_Scalar scoreGradientScalar = softmaxScalar * (softmaxGradientScalar - softmaxGradientSum) * scaleScalar;

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            inputGradientQueryPtr[hc] += inputKeyPtr[hc] * scoreGradientScalar;
            atomicAdd(&inputGradientKeyPtr[hc], inputQueryPtr[hc] * scoreGradientScalar);
        }
    }
}

__global__ void ML2_CUDA_FunctionAttentionFullCacheBackwardStateMatrices(
    ML2_Int samples, ML2_Int matrices, ML2_Int scalars, ML2_Int heads,
    const ML2_Scalar *__restrict__ state, const ML2_Scalar *__restrict__ input, ML2_Scalar *__restrict__ inputGradient, const ML2_Scalar *__restrict__ output, const ML2_Scalar *__restrict__ outputGradient
) {
    ML2_Int queryVectors = matrices;

    ML2_Int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid >= samples * queryVectors * heads) return;
    ML2_Int s = tid / heads / queryVectors;
    ML2_Int qv = tid / heads % queryVectors;
    ML2_Int h = tid % heads;

    ML2_Int vectors = ML2_AttentionVectorsCount;
    ML2_Int headScalars = scalars / heads;
    ML2_Scalar scaleScalar = ML2_CUDA_One / ML2_CUDA_Sqrt(headScalars);
    ML2_Scalar scaleLog2eScalar = ML2_CUDA_Log2e * scaleScalar;
    ML2_Int headOffset = h * headScalars;

    const ML2_Scalar *__restrict__ statePtr = &state[(s * queryVectors + qv) * heads];
    const ML2_Scalar *__restrict__ inputQueryPtr = &input[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    ML2_Scalar *__restrict__ inputGradientQueryPtr = &inputGradient[((s * queryVectors + qv) * vectors + ML2_AttentionInputIndexQuery) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputPtr = &output[(s * queryVectors + qv) * scalars + headOffset];
    const ML2_Scalar *__restrict__ outputGradientPtr = &outputGradient[(s * queryVectors + qv) * scalars + headOffset];

    ML2_Int keyVectorsEnd = queryVectors;

    ML2_Scalar maxLogSum = statePtr[h];
    ML2_Scalar softmaxGradientSum = {};

    for (ML2_Int hc = 0; hc < headScalars; hc++) {
        softmaxGradientSum += outputPtr[hc] * outputGradientPtr[hc];
    }

    for (ML2_Int kv = 0; kv < keyVectorsEnd; kv++) {
        const ML2_Scalar *__restrict__ inputKeyPtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientKeyPtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexKey) * scalars + headOffset];
        const ML2_Scalar *__restrict__ inputValuePtr = &input[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];
        ML2_Scalar *__restrict__ inputGradientValuePtr = &inputGradient[((s * queryVectors + kv) * vectors + ML2_AttentionInputIndexValue) * scalars + headOffset];

        ML2_Scalar scoreScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            scoreScalar += inputQueryPtr[hc] * inputKeyPtr[hc];
        }
        ML2_Scalar softmaxScalar = ML2_CUDA_Exp2(scoreScalar * scaleLog2eScalar - maxLogSum);

        ML2_Scalar softmaxGradientScalar = {};
        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            softmaxGradientScalar += inputValuePtr[hc] * outputGradientPtr[hc];
            atomicAdd(&inputGradientValuePtr[hc], softmaxScalar * outputGradientPtr[hc]);
        }

        ML2_Scalar scoreGradientScalar = softmaxScalar * (softmaxGradientScalar - softmaxGradientSum) * scaleScalar;

        for (ML2_Int hc = 0; hc < headScalars; hc++) {
            inputGradientQueryPtr[hc] += inputKeyPtr[hc] * scoreGradientScalar;
            atomicAdd(&inputGradientKeyPtr[hc], inputQueryPtr[hc] * scoreGradientScalar);
        }
    }
}

__global__ void ML2_CUDA_FunctionLossBackwardSoftmaxCrossEntropy(
    ML2_Int vectors, ML2_Int scalars,
    const ML2_Scalar *__restrict__ predicted, const ML2_Scalar *__restrict__ expected, ML2_Scalar *__restrict__ gradient
) {
    ML2_Int v = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (v >= vectors) return;

    const ML2_Scalar *__restrict__ predictedPtr = &predicted[v * scalars];
    const ML2_Scalar *__restrict__ expectedPtr = &expected[v * scalars];
    ML2_Scalar *__restrict__ gradientPtr = &gradient[v * scalars];
    ML2_Scalar max = ML2_CUDA_NegInf;
    for (ML2_Int c = 0; c < scalars; c++) {
        max = ML2_CUDA_Fmax(max, predictedPtr[c]);
    }
    ML2_Scalar sum = {};
    for (ML2_Int c = 0; c < scalars; c++) {
        sum += ML2_CUDA_Exp(predictedPtr[c] - max);
    }
    for (ML2_Int c = 0; c < scalars; c++) {
        ML2_Scalar predictedScalar = ML2_CUDA_Exp(predictedPtr[c] - max) / sum;
        ML2_Scalar expectedScalar = expectedPtr[c];
        gradientPtr[c] += (predictedScalar - expectedScalar) / vectors;
    }
}

__global__ void ML2_CUDA_FunctionAdamOptimize(
    ML2_Int scalars,
    ML2_Scalar *__restrict__ parameters, const ML2_Scalar *__restrict__ parametersGradient, ML2_Scalar *__restrict__ decayingSum, ML2_Scalar *__restrict__ decayingSquareSum,
    ML2_Scalar learningRate,
    ML2_Scalar decayRate1, ML2_Scalar decayRate2,
    ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;

    ML2_Scalar parameter = parameters[c];
    ML2_Scalar parameterGradient = parametersGradient[c];
    ML2_Scalar decayingSumScalar = decayingSum[c];
    ML2_Scalar decayingSquareSumScalar = decayingSquareSum[c];

    decayingSumScalar = decayRate1 * decayingSumScalar + (ML2_CUDA_One - decayRate1) * parameterGradient;
    decayingSquareSumScalar = decayRate2 * decayingSquareSumScalar + (ML2_CUDA_One - decayRate2) * parameterGradient * parameterGradient;
    ML2_Scalar decayingSumWeight = decayingSumScalar / (ML2_CUDA_One - decayingWeight1);
    ML2_Scalar decayingSquareSumWeight = decayingSquareSumScalar / (ML2_CUDA_One - decayingWeight2);
    parameter -= learningRate * decayingSumWeight / (ML2_CUDA_Sqrt(decayingSquareSumWeight) + ML2_CUDA_Epsilon);

    parameters[c] = parameter;
    decayingSum[c] = decayingSumScalar;
    decayingSquareSum[c] = decayingSquareSumScalar;
}

__global__ void ML2_CUDA_FunctionAdamWOptimize(
    ML2_Int scalars,
    ML2_Scalar *__restrict__ parameters, const ML2_Scalar *__restrict__ parametersGradient, ML2_Scalar *__restrict__ decayingSum, ML2_Scalar *__restrict__ decayingSquareSum,
    ML2_Scalar learningRate,
    ML2_Scalar decayRate1, ML2_Scalar decayRate2,
    ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2, ML2_Scalar parameterDecay
) {
    ML2_Int c = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (c >= scalars) return;

    ML2_Scalar parameter = parameters[c];
    ML2_Scalar parameterGradient = parametersGradient[c];
    ML2_Scalar decayingSumScalar = decayingSum[c];
    ML2_Scalar decayingSquareSumScalar = decayingSquareSum[c];

    decayingSumScalar = decayRate1 * decayingSumScalar + (ML2_CUDA_One - decayRate1) * parameterGradient;
    decayingSquareSumScalar = decayRate2 * decayingSquareSumScalar + (ML2_CUDA_One - decayRate2) * parameterGradient * parameterGradient;
    ML2_Scalar decayingSumWeight = decayingSumScalar / (ML2_CUDA_One - decayingWeight1);
    ML2_Scalar decayingSquareSumWeight = decayingSquareSumScalar / (ML2_CUDA_One - decayingWeight2);
    parameter -= learningRate * (decayingSumWeight / (ML2_CUDA_Sqrt(decayingSquareSumWeight) + ML2_CUDA_Epsilon) + parameterDecay * parameter);

    parameters[c] = parameter;
    decayingSum[c] = decayingSumScalar;
    decayingSquareSum[c] = decayingSquareSumScalar;
}

} // extern "C"