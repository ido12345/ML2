#ifndef ML2_H_INCLUDE_GUARD
#define ML2_H_INCLUDE_GUARD

/*
    this library aims for minimal memory allocations

    *New calls allocate memory, so use *Destroy to free the memory
*/

#ifndef ML2_FN
    #define ML2_FN static inline
#endif // ML2_FN

#ifndef ML2_VAR
    #define ML2_VAR static
#endif // ML2_VAR

#define ML2_BOOL_TO_STR(b) ((b) ? ("true") : ("false"))

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef ML2_CUDA_ENABLE
    #include <cuda.h>

    #define ML2_CUDA_CHECK_ERROR(result, msg, ...)                                                          \
        do {                                                                                                \
            CUresult _result = (result);                                                                    \
            if (_result != CUDA_SUCCESS) {                                                                  \
                const char *_errorName = "Unknown";                                                         \
                const char *_errorDetails = "Unknown";                                                      \
                cuGetErrorName(_result, &_errorName);                                                       \
                cuGetErrorString(_result, &_errorDetails);                                                  \
                fprintf(stderr, "CUDA DRIVER ERROR: \"" msg "\"\n"                                          \
                                "    Error:    |%s|\n"                                                      \
                                "    Details:  |%s|\n"                                                      \
                                "    File:     |%s|\n"                                                      \
                                "    Function: |%s|\n"                                                      \
                                "    Line:     |%d|\n",                                                     \
                        __VA_OPT__(__VA_ARGS__, ) _errorName, _errorDetails, __FILE__, __func__, __LINE__); \
                abort();                                                                                    \
            }                                                                                               \
        } while (0)
#endif // ML2_CUDA_ENABLE

#define ML2_UNREACHABLE(msg, ...)                                        \
    do {                                                                 \
        fprintf(stderr, "UNREACHABLE: \"" msg "\"\n"                     \
                        "    file:     |%s|\n"                           \
                        "    function: |%s|\n"                           \
                        "    line:     |%d|\n",                          \
                __VA_OPT__(__VA_ARGS__, ) __FILE__, __func__, __LINE__); \
        abort();                                                         \
    } while (0)

#define ML2_TODO(msg, ...)                                               \
    do {                                                                 \
        fprintf(stderr, "TODO: \"" msg "\"\n"                            \
                        "    file:     |%s|\n"                           \
                        "    function: |%s|\n"                           \
                        "    line:     |%d|\n",                          \
                __VA_OPT__(__VA_ARGS__, ) __FILE__, __func__, __LINE__); \
        abort();                                                         \
    } while (0)

#define ML2_CRASH(msg, ...)                                              \
    do {                                                                 \
        fprintf(stderr, "CRASH: \"" msg "\"\n"                           \
                        "    file:     |%s|\n"                           \
                        "    function: |%s|\n"                           \
                        "    line:     |%d|\n",                          \
                __VA_OPT__(__VA_ARGS__, ) __FILE__, __func__, __LINE__); \
        abort();                                                         \
    } while (0)

#define ML2_TO_STR_HELPER(x) #x
#define ML2_TO_STR(x) ML2_TO_STR_HELPER(x)

// NOTE(28/9/2026 20:25:13): fputs is used because it does not have internal allocations
#define ML2_ABORT(msg)                                      \
    do {                                                    \
        fputs("ABORT: \"" msg "\"\n"                        \
              "    file:     |" __FILE__ "|\n"              \
              "    function: |",                            \
              stderr);                                      \
        fputs(__func__, stderr);                            \
        fputs("|\n"                                         \
              "    line:     |" ML2_TO_STR(__LINE__) "|\n", \
              stderr);                                      \
        abort();                                            \
    } while (0)

#ifndef ML2_NO_DEBUG
    #define ML2_SOFT_ASSERT(cond)                             \
        do {                                                  \
            if (!(cond)) {                                    \
                fprintf(stderr, "SOFT ASSERTION FAILED:\n"    \
                                "    condition: |%s|\n"       \
                                "    file:      |%s|\n"       \
                                "    function:  |%s|\n"       \
                                "    line:      |%d|\n",      \
                        #cond, __FILE__, __func__, __LINE__); \
                abort();                                      \
            }                                                 \
        } while (0)
#else
    #define ML2_SOFT_ASSERT(cond)
#endif // ML2_NO_DEBUG

#define ML2_HARD_ASSERT(cond)                             \
    do {                                                  \
        if (!(cond)) {                                    \
            fprintf(stderr, "HARD ASSERTION FAILED:\n"    \
                            "    condition: |%s|\n"       \
                            "    file:      |%s|\n"       \
                            "    function:  |%s|\n"       \
                            "    line:      |%d|\n",      \
                    #cond, __FILE__, __func__, __LINE__); \
            abort();                                      \
        }                                                 \
    } while (0)

// ML2_Int ⬇️

#ifndef ML2_INT_TYPE
    // TODO(21/4/2026 3:49:27pm): i can make this an int_fast32_t, ill wait until i can measure performance
    // TODO(21/4/2026 5:13:36pm): i should figure out if 32 are actually needed, would be useful to know if it can be 16 or something
    #define ML2_INT_TYPE int32_t
#endif // ML2_INT_TYPE
typedef ML2_INT_TYPE ML2_Int;

// ML2_Int ⬆️

// ML2_Size ⬇️

#ifndef ML2_SIZE_TYPE
    #define ML2_SIZE_TYPE size_t
#endif // ML2_SIZE_TYPE
typedef ML2_SIZE_TYPE ML2_Size;

// ML2_Size ⬆️

#ifndef ML2_CALLOC
    #include <stdlib.h>
static inline void *ML2_Calloc(ML2_Size size) {
    #ifdef ML2_PRINT_ALLOC
    {
        // NOTE: global byte counter
        static ML2_Size total = 0;
        total += size;

        constexpr double measure = 1000.0;
        double bytes = (double)size;

        const char *units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
        ML2_Int unitCount = sizeof(units) / sizeof(*units);
        ML2_Int unit = {};
        while (bytes >= measure && unit < unitCount - 1) {
            unit++;
            bytes /= measure;
        }
        printf("%s: %.3lf %s\n", __func__, bytes, units[unit]);

        double totalBytes = (double)total;
        unit = 0;
        while (totalBytes >= measure && unit < unitCount - 1) {
            unit++;
            totalBytes /= measure;
        }
        printf("%s: Total: %.3lf %s\n", __func__, totalBytes, units[unit]);
    }
    #endif // ML2_PRINT_ALLOC

    void *ptr = calloc(size, 1);
    if (!ptr) {
        ML2_ABORT("Failed to allocate memory");
    }
    return ptr;
}
    #define ML2_CALLOC ML2_Calloc
#endif // ML2_CALLOC

#ifndef ML2_FREE
    #include <stdlib.h>
    #define ML2_FREE free
#endif // ML2_FREE

// ML2_Scalar ⬇️

// NOTE: Customizable, but make sure to define all of these
#ifndef ML2_SCALAR_TYPE
    #define ML2_SCALAR_TYPE float

    #define ML2_SCALAR_LITERAL(x) x##f
    #define ML2_EXP expf
    #define ML2_LN logf
    #define ML2_SQRT sqrtf
    #define ML2_SIN sinf
    #define ML2_COS cosf
    #define ML2_TANH tanhf
    #define ML2_FMAX fmaxf
    #define ML2_FMIN fminf
#endif // ML2_SCALAR_TYPE
typedef ML2_SCALAR_TYPE ML2_Scalar;

ML2_VAR constexpr ML2_Scalar ML2_Epsilon = ML2_SCALAR_LITERAL(1e-7);
ML2_VAR constexpr ML2_Scalar ML2_Pi = ML2_SCALAR_LITERAL(3.141592653589793);
ML2_VAR constexpr ML2_Scalar ML2_NegInf = (ML2_Scalar)(-INFINITY);

// TODO: figure out how to align the floats nicely
#define ML2_SCALAR_FMT "%.3f"

// ML2_Scalar ⬆️

#ifndef ML2_INDENTATION
    #define ML2_INDENTATION 4
#endif // ML2_INDENTATION
ML2_VAR constexpr ML2_Int ML2_Indentation = ML2_INDENTATION;

#define ML2_INDENT(fstr, i) "%*s" fstr, (i), ""

// ML2_Device ⬇️

typedef enum : ML2_Int {
    ML2_DeviceCPU = 1,
#ifdef ML2_CUDA_ENABLE
    ML2_DeviceCUDA,
#endif // ML2_CUDA_ENABLE
} ML2_Device;

// ML2_CPU ⬇️

typedef struct {
    // TODO(29/9/2026 4:30:05): in the future this should support different data types
    ML2_Scalar *ptr;
} ML2_CPU_DeviceData;

// ML2_CPU ⬆️

// ML2_CUDA ⬇️

#ifdef ML2_CUDA_ENABLE
typedef struct {
    CUdeviceptr ptr;
} ML2_CUDA_DeviceData;
#endif // ML2_CUDA_ENABLE

// ML2_CUDA ⬆️

// TODO(1/10/2026 1:21:53): there should be a way to signify that a ML2_DeviceData is a view, meaning its not meant to be freed
typedef struct {
    ML2_Device device;
    union {
        ML2_CPU_DeviceData cpu;
#ifdef ML2_CUDA_ENABLE
        ML2_CUDA_DeviceData cuda;
#endif // ML2_CUDA_ENABLE
    } as;
} ML2_DeviceData;

// ML2_Device ⬆️

// ML2_Layer ⬇️

typedef enum : ML2_Int {
    ML2_LayerTypeWeights = 1,
    ML2_LayerTypeBiases,
    ML2_LayerTypeActivation,
    ML2_LayerTypeSoftmax,
    ML2_LayerTypeLinear,
    ML2_LayerTypeFilters,
    ML2_LayerTypeConv,
    ML2_LayerTypeFlatten,
    ML2_LayerTypePool,
    ML2_LayerTypeFusedWeights,
    ML2_LayerTypeAttention,
} ML2_LayerType;

// ML2_LayerWeights ⬇️

typedef struct {
    ML2_Int outputs;
    ML2_Int inputs;
} ML2_LayerInfoWeights;

typedef struct {
    ML2_LayerInfoWeights info;
    ML2_DeviceData data;
} ML2_LayerWeights;

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

typedef struct {
    ML2_Int inputs;
} ML2_LayerInfoBiases;

typedef struct {
    ML2_LayerInfoBiases info;
    ML2_DeviceData data;
} ML2_LayerBiases;

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

typedef enum : ML2_Int {
    ML2_ActivationTypeReLU = 1,
    ML2_ActivationTypeSigmoid,
    ML2_ActivationTypeSin,
    ML2_ActivationTypeCos,
    ML2_ActivationTypeTanh,
    ML2_ActivationTypeLeakyReLU,
    ML2_ActivationTypeSiLU,
} ML2_ActivationType;

typedef struct {
    ML2_ActivationType type;
} ML2_LayerInfoActivation;

typedef ML2_Scalar(ML2_ScalarActivationForward)(ML2_Scalar input);
typedef ML2_Scalar(ML2_ScalarActivationBackwardInput)(ML2_Scalar input);
typedef ML2_Scalar(ML2_ScalarActivationBackwardOutput)(ML2_Scalar output);
typedef ML2_Scalar(ML2_ScalarActivationBackwardInputOutput)(ML2_Scalar input, ML2_Scalar output);

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

typedef struct {
    ML2_Int outputs;
    ML2_Int inputs;
} ML2_LayerInfoLinear;

typedef struct {
    ML2_LayerInfoLinear info;
    ML2_DeviceData data;
} ML2_LayerLinear;

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

typedef struct {
    ML2_Int height;
    ML2_Int width;
    ML2_Int inputs;
    ML2_Int outputs;
    ML2_Int paddingHeight;
    ML2_Int paddingWidth;
    ML2_Int strideHeight;
    ML2_Int strideWidth;
} ML2_LayerInfoFilters;

typedef struct {
    ML2_LayerInfoFilters info;
    ML2_DeviceData data;
} ML2_LayerFilters;

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

typedef struct {
    ML2_Int height;
    ML2_Int width;
    ML2_Int inputs;
    ML2_Int outputs;
    ML2_Int paddingHeight;
    ML2_Int paddingWidth;
    ML2_Int strideHeight;
    ML2_Int strideWidth;
} ML2_LayerInfoConv;

typedef struct {
    ML2_LayerInfoConv info;
    ML2_DeviceData data;
} ML2_LayerConv;

// ML2_LayerConv ⬆️

// ML2_LayerPool ⬇️

typedef enum : ML2_Int {
    ML2_PoolTypeMax = 1,
    ML2_PoolTypeAverage,
    ML2_PoolTypeSoftmax,
} ML2_PoolType;

// ML2_LayerPoolMax ⬇️

typedef struct {
    ML2_Int height;
    ML2_Int width;
    ML2_Int paddingHeight;
    ML2_Int paddingWidth;
    ML2_Int strideHeight;
    ML2_Int strideWidth;
} ML2_LayerInfoPoolMax;

// ML2_LayerPoolMax ⬆️

// ML2_LayerPoolAverage ⬇️

typedef struct {
    ML2_Int height;
    ML2_Int width;
    ML2_Int paddingHeight;
    ML2_Int paddingWidth;
    ML2_Int strideHeight;
    ML2_Int strideWidth;
} ML2_LayerInfoPoolAverage;

// ML2_LayerPoolAverage ⬆️

// ML2_LayerPoolSoftmax ⬇️

typedef struct {
    ML2_Int height;
    ML2_Int width;
    ML2_Int paddingHeight;
    ML2_Int paddingWidth;
    ML2_Int strideHeight;
    ML2_Int strideWidth;
} ML2_LayerInfoPoolSoftmax;

// ML2_LayerPoolSoftmax ⬆️

typedef struct {
    ML2_PoolType type;
    union {
        ML2_LayerInfoPoolMax max;
        ML2_LayerInfoPoolAverage average;
        ML2_LayerInfoPoolSoftmax softmax;
    } as;
} ML2_LayerInfoPool;

// ML2_LayerPool ⬆️

// ML2_LayerFusedWeights ⬇️

typedef struct {
    ML2_Int weights;
    ML2_Int outputs;
    ML2_Int inputs;
} ML2_LayerInfoFusedWeights;

typedef struct {
    ML2_LayerInfoFusedWeights info;
    ML2_DeviceData data;
} ML2_LayerFusedWeights;

// ML2_LayerFusedWeights ⬆️

// ML2_LayerAttention ⬇️

// NOTE(3/9/2026 20:06:17):
//      the input to attention is (SxMxVxC)
//      in the case of tokens its (SxTx3xC)
//      the 3 means that for each token i have the vectors of Query Key and Value sequentially
//      its best this way due to the access patterns in Attention
//      because it will access those dimensions in that exact order
//      this global variable exists so that i dont have a hardcoded 3 everywhere
ML2_VAR constexpr ML2_Int ML2_AttentionVectorsCount = 3;
ML2_VAR constexpr ML2_Int ML2_AttentionInputIndexQuery = 0;
ML2_VAR constexpr ML2_Int ML2_AttentionInputIndexKey = 1;
ML2_VAR constexpr ML2_Int ML2_AttentionInputIndexValue = 2;

// TODO(3/9/2026 19:45:41): for now Attention has no data, and i dont know if it will
typedef struct {
    ML2_Int heads;
    bool masked;
} ML2_LayerInfoAttention;

// ML2_LayerAttention ⬆️

typedef struct {
    ML2_LayerType type;
    union {
        ML2_LayerInfoWeights weights;
        ML2_LayerInfoBiases biases;
        ML2_LayerInfoActivation activation;
        ML2_LayerInfoLinear linear;
        ML2_LayerInfoFilters filters;
        ML2_LayerInfoConv conv;
        ML2_LayerInfoPool pool;
        ML2_LayerInfoFusedWeights fusedWeights;
        ML2_LayerInfoAttention attention;
    } as;
} ML2_LayerInfo;

typedef struct {
    ML2_LayerType type;
    union {
        ML2_LayerWeights weights;
        ML2_LayerBiases biases;
        ML2_LayerInfoActivation activation;
        ML2_LayerLinear linear;
        ML2_LayerFilters filters;
        ML2_LayerConv conv;
        ML2_LayerInfoPool pool;
        ML2_LayerFusedWeights fusedWeights;
        ML2_LayerInfoAttention attention;
    } as;
} ML2_Layer;

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

typedef enum : ML2_Int {
    ML2_LayerCacheTypeScalars = 1,
    ML2_LayerCacheTypeVectors,
    ML2_LayerCacheTypeMatrices,
} ML2_LayerCacheType;

// ML2_LayerCacheScalars ⬇️

typedef struct {
    ML2_Int samples;
    ML2_Int scalars;
} ML2_LayerCacheInfoScalars;

typedef struct {
    ML2_LayerCacheInfoScalars info;
    ML2_DeviceData data;
} ML2_LayerCacheScalars;

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheVectors ⬇️

typedef struct {
    ML2_Int samples;
    ML2_Int vectors;
    ML2_Int scalars;
} ML2_LayerCacheInfoVectors;

typedef struct {
    ML2_LayerCacheInfoVectors info;
    ML2_DeviceData data;
} ML2_LayerCacheVectors;

// ML2_LayerCacheVectors ⬆️

// ML2_LayerCacheMatrices ⬇️

typedef struct {
    ML2_Int samples;
    ML2_Int matrices;
    ML2_Int vectors;
    ML2_Int scalars;
} ML2_LayerCacheInfoMatrices;

typedef struct {
    ML2_LayerCacheInfoMatrices info;
    ML2_DeviceData data;
} ML2_LayerCacheMatrices;

// ML2_LayerCacheMatrices ⬆️

typedef struct {
    ML2_LayerCacheType type;
    union {
        ML2_LayerCacheInfoScalars scalars;
        ML2_LayerCacheInfoVectors vectors;
        ML2_LayerCacheInfoMatrices matrices;
    } as;
} ML2_LayerCacheInfo;

typedef struct {
    ML2_LayerCacheType type;
    union {
        ML2_LayerCacheScalars scalars;
        ML2_LayerCacheVectors vectors;
        ML2_LayerCacheMatrices matrices;
    } as;
} ML2_LayerCache;

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

typedef struct {
    ML2_Int layerCount;
    ML2_LayerInfo *infos;
} ML2_Arch;

// ML2_Arch ⬆️

// ML2_Model ⬇️

typedef struct {
    ML2_Int layerCount;
    ML2_Layer *layers;
} ML2_Model;

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

typedef struct {
    ML2_Int layerCount;
    ML2_LayerCache *layers;
} ML2_ModelCache;

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

typedef struct {
    ML2_LayerCacheInfo input;
    ML2_LayerCacheInfo output;
} ML2_BatchInfo;

typedef struct {
    ML2_LayerCache input;
    ML2_LayerCache output;
} ML2_Batch;

// ML2_Batch ⬆️

// ML2_Optimizer ⬇️

typedef enum : ML2_Int {
    ML2_OptimizerTypeGradientDescent = 1,
    ML2_OptimizerTypeMomentum,
    ML2_OptimizerTypeAdagrad,
    ML2_OptimizerTypeRMSProp,
    ML2_OptimizerTypeAdam,
} ML2_OptimizerType;

// ML2_OptimizerGradientDescent ⬇️

typedef struct {
    ML2_Scalar learningRate;
} ML2_OptimizerParametersGradientDescent;

typedef struct {
    ML2_OptimizerParametersGradientDescent parameters;
} ML2_OptimizerGradientDescent;

// ML2_OptimizerGradientDescent ⬆️

// ML2_OptimizerMomentum ⬇️

typedef struct {
    ML2_Scalar learningRate;
    ML2_Scalar decayRate;
} ML2_OptimizerParametersMomentum;

typedef struct {
    ML2_OptimizerParametersMomentum parameters;
    struct {
        ML2_Int layerCount;
        ML2_Layer *averageLayers;
    } state;
} ML2_OptimizerMomentum;

// ML2_OptimizerMomentum ⬆️

// ML2_OptimizerAdagrad ⬇️

typedef struct {
    ML2_Scalar learningRate;
} ML2_OptimizerParametersAdagrad;

typedef struct {
    ML2_OptimizerParametersAdagrad parameters;
    struct {
        ML2_Int layerCount;
        ML2_Layer *squareSumLayers;
    } state;
} ML2_OptimizerAdagrad;

// ML2_OptimizerAdagrad ⬆️

// ML2_OptimizerRMSProp ⬇️

typedef struct {
    ML2_Scalar learningRate;
    ML2_Scalar decayRate;
} ML2_OptimizerParametersRMSProp;

typedef struct {
    ML2_OptimizerParametersRMSProp parameters;
    struct {
        ML2_Int layerCount;
        ML2_Layer *squareAverageLayers;
    } state;
} ML2_OptimizerRMSProp;

// ML2_OptimizerRMSProp ⬆️

// ML2_OptimizerAdam ⬇️

typedef struct {
    ML2_Scalar learningRate;
    ML2_Scalar decayRate1;
    ML2_Scalar decayRate2;
} ML2_OptimizerParametersAdam;

typedef struct {
    ML2_OptimizerParametersAdam parameters;
    struct {
        ML2_Int layerCount;
        ML2_Layer *averageLayers;
        ML2_Layer *squareAverageLayers;
        ML2_Scalar decayingWeight1;
        ML2_Scalar decayingWeight2;
    } state;
} ML2_OptimizerAdam;

// ML2_OptimizerAdam ⬆️

typedef struct {
    ML2_OptimizerType type;
    union {
        ML2_OptimizerParametersGradientDescent gradientDescent;
        ML2_OptimizerParametersMomentum momentum;
        ML2_OptimizerParametersAdagrad adagrad;
        ML2_OptimizerParametersRMSProp RMSProp;
        ML2_OptimizerParametersAdam adam;
    } as;
} ML2_OptimizerParameters;

typedef struct {
    ML2_OptimizerType type;
    union {
        ML2_OptimizerGradientDescent gradientDescent;
        ML2_OptimizerMomentum momentum;
        ML2_OptimizerAdagrad adagrad;
        ML2_OptimizerRMSProp RMSProp;
        ML2_OptimizerAdam adam;
    } as;
} ML2_Optimizer;

// ML2_Optimizer ⬆️

// ML2_Loss ⬇️

typedef ML2_Scalar (*ML2_LossForward)(ML2_LayerCache predicted, ML2_LayerCache expected);
typedef void (*ML2_LossBackward)(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

// ML2_Loss ⬆️

// ML2_Int ⬇️

ML2_FN ML2_Int ML2_IntMax(ML2_Int a, ML2_Int b);
ML2_FN ML2_Int ML2_IntMin(ML2_Int a, ML2_Int b);

// ML2_Int ⬆️

// ML2_Scalar ⬇️

ML2_FN ML2_Scalar ML2_ScalarRand(ML2_Scalar low, ML2_Scalar high);
ML2_FN ML2_Scalar ML2_ScalarRandNormal(ML2_Scalar mean, ML2_Scalar std);
ML2_FN ML2_ScalarActivationForward ML2_ScalarReLUForward;
ML2_FN ML2_ScalarActivationBackwardInput ML2_ScalarReLUBackwardInput;
ML2_FN ML2_ScalarActivationForward ML2_ScalarSigmoidForward;
ML2_FN ML2_ScalarActivationBackwardOutput ML2_ScalarSigmoidBackwardOutput;
ML2_FN ML2_ScalarActivationForward ML2_ScalarSinForward;
ML2_FN ML2_ScalarActivationBackwardInput ML2_ScalarSinBackwardInput;
ML2_FN ML2_ScalarActivationForward ML2_ScalarCosForward;
ML2_FN ML2_ScalarActivationBackwardInput ML2_ScalarCosBackwardInput;
ML2_FN ML2_ScalarActivationForward ML2_ScalarTanhForward;
ML2_FN ML2_ScalarActivationBackwardOutput ML2_ScalarTanhBackwardOutput;
ML2_FN ML2_ScalarActivationForward ML2_ScalarLeakyReLUForward;
ML2_FN ML2_ScalarActivationBackwardInput ML2_ScalarLeakyReLUBackwardInput;
ML2_FN ML2_ScalarActivationForward ML2_ScalarSiLUForward;
ML2_FN ML2_ScalarActivationBackwardInputOutput ML2_ScalarSiLUBackwardInputOutput;

// ML2_Scalar ⬆️

// ML2_Device ⬇️

ML2_FN void ML2_DeviceInit(ML2_Device device);
ML2_FN ML2_DeviceData ML2_DeviceDataNew(ML2_Device device, ML2_Size size);
ML2_FN void ML2_DeviceDataDestroy(ML2_DeviceData *data);
ML2_FN ML2_CPU_DeviceData ML2_DeviceDataAsCPU(ML2_DeviceData data);
ML2_FN ML2_CPU_DeviceData *ML2_DeviceDataPtrAsCPU(ML2_DeviceData *data);
#ifdef ML2_CUDA_ENABLE
ML2_FN ML2_CUDA_DeviceData ML2_DeviceDataAsCUDA(ML2_DeviceData data);
ML2_FN ML2_CUDA_DeviceData *ML2_DeviceDataPtrAsCUDA(ML2_DeviceData *data);
#endif // ML2_CUDA_ENABLE

// ML2_CPU ⬇️

ML2_FN ML2_DeviceData ML2_CPU_DeviceDataNew(ML2_Size size);
ML2_FN void ML2_CPU_DeviceDataDestroy(ML2_DeviceData *data);
ML2_FN ML2_Scalar *ML2_CPU_DeviceDataAt(ML2_DeviceData data, ML2_Int index);

ML2_FN ML2_Scalar *ML2_CPU_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i);
ML2_FN void ML2_CPU_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent);
ML2_FN void ML2_CPU_WeightsClear(ML2_LayerWeights weights);
ML2_FN void ML2_CPU_WeightsCopy(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_CPU_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_CPU_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_WeightsXavierInit(ML2_LayerWeights weights);
ML2_FN void ML2_CPU_WeightsHeInit(ML2_LayerWeights weights);

ML2_FN void ML2_CPU_WeightsForwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CPU_WeightsForwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CPU_WeightsBackwardScalars(ML2_LayerWeights weightsGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_WeightsBackwardVectors(ML2_LayerWeights weightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CPU_WeightsCacheBackwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_WeightsCacheBackwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient);

ML2_FN void ML2_CPU_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CPU_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average);
ML2_FN void ML2_CPU_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum);
ML2_FN void ML2_CPU_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage);
ML2_FN void ML2_CPU_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Scalar *ML2_CPU_BiasesAt(ML2_LayerBiases biases, ML2_Int i);
ML2_FN void ML2_CPU_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent);
ML2_FN void ML2_CPU_BiasesClear(ML2_LayerBiases biases);
ML2_FN void ML2_CPU_BiasesCopy(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_CPU_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_CPU_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_BiasesXavierInit(ML2_LayerBiases biases);
ML2_FN void ML2_CPU_BiasesHeInit(ML2_LayerBiases biases);

ML2_FN void ML2_CPU_BiasesForwardScalars(ML2_LayerBiases biases, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CPU_BiasesForwardMatrices(ML2_LayerBiases biases, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_BiasesBackwardScalars(ML2_LayerBiases biasesGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_BiasesBackwardMatrices(ML2_LayerBiases biasesGradient, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CPU_BiasesCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_BiasesCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CPU_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average);
ML2_FN void ML2_CPU_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum);
ML2_FN void ML2_CPU_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage);
ML2_FN void ML2_CPU_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN void ML2_CPU_ActivationForwardScalars(ML2_LayerInfoActivation activation, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CPU_ActivationForwardVectors(ML2_LayerInfoActivation activation, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CPU_ActivationForwardMatrices(ML2_LayerInfoActivation activation, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_ActivationCacheBackwardInputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardInputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardInputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardInputOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardInputOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CPU_ActivationCacheBackwardInputOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_SoftmaxForwardScalars(ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CPU_SoftmaxForwardVectors(ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CPU_SoftmaxCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_SoftmaxCacheBackwardVectors(ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);

ML2_FN ML2_Scalar *ML2_CPU_LinearWeightsAt(ML2_LayerLinear linear, ML2_Int o, ML2_Int i);
ML2_FN ML2_Scalar *ML2_CPU_LinearBiasesAt(ML2_LayerLinear linear, ML2_Int o);
ML2_FN ML2_LayerWeights ML2_CPU_LinearWeights(ML2_LayerLinear linear);
ML2_FN ML2_LayerBiases ML2_CPU_LinearBiases(ML2_LayerLinear linear);
ML2_FN void ML2_CPU_LinearPrint(ML2_LayerLinear linear, ML2_Int indent);
ML2_FN void ML2_CPU_LinearClear(ML2_LayerLinear linear);
ML2_FN void ML2_CPU_LinearCopy(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_CPU_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_CPU_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_LinearXavierInit(ML2_LayerLinear linear);
ML2_FN void ML2_CPU_LinearHeInit(ML2_LayerLinear linear);

ML2_FN void ML2_CPU_LinearForwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CPU_LinearBackwardScalars(ML2_LayerLinear linearGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CPU_LinearCacheBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);

ML2_FN void ML2_CPU_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CPU_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average);
ML2_FN void ML2_CPU_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum);
ML2_FN void ML2_CPU_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage);
ML2_FN void ML2_CPU_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Scalar *ML2_CPU_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN void ML2_CPU_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent);
ML2_FN void ML2_CPU_FiltersClear(ML2_LayerFilters filters);
ML2_FN void ML2_CPU_FiltersCopy(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_CPU_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_CPU_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_FiltersXavierInit(ML2_LayerFilters filters);
ML2_FN void ML2_CPU_FiltersHeInit(ML2_LayerFilters filters);

ML2_FN void ML2_CPU_FiltersForwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_FiltersBackwardMatrices(ML2_LayerFilters filtersGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CPU_FiltersCacheBackwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CPU_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average);
ML2_FN void ML2_CPU_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum);
ML2_FN void ML2_CPU_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage);
ML2_FN void ML2_CPU_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Scalar *ML2_CPU_ConvFiltersAt(ML2_LayerConv conv, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN ML2_Scalar *ML2_CPU_ConvBiasesAt(ML2_LayerConv conv, ML2_Int o);
ML2_FN ML2_LayerFilters ML2_CPU_ConvFilters(ML2_LayerConv conv);
ML2_FN ML2_LayerBiases ML2_CPU_ConvBiases(ML2_LayerConv conv);
ML2_FN void ML2_CPU_ConvPrint(ML2_LayerConv conv, ML2_Int indent);
ML2_FN void ML2_CPU_ConvClear(ML2_LayerConv conv);
ML2_FN void ML2_CPU_ConvCopy(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_CPU_ConvSum(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_CPU_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_ConvXavierInit(ML2_LayerConv conv);
ML2_FN void ML2_CPU_ConvHeInit(ML2_LayerConv conv);

ML2_FN void ML2_CPU_ConvForwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_ConvBackwardMatrices(ML2_LayerConv convGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CPU_ConvCacheBackwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CPU_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average);
ML2_FN void ML2_CPU_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum);
ML2_FN void ML2_CPU_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage);
ML2_FN void ML2_CPU_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN void ML2_CPU_FlattenForwardMatrices(ML2_LayerCacheMatrices input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CPU_FlattenCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheScalars outputGradient);

ML2_FN void ML2_CPU_PoolMaxForwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_PoolMaxCacheBackwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_PoolAverageForwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_PoolAverageCacheBackwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_PoolSoftmaxForwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_PoolSoftmaxCacheBackwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

ML2_FN ML2_Scalar *ML2_CPU_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i);
ML2_FN ML2_LayerWeights ML2_CPU_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w);
ML2_FN void ML2_CPU_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent);
ML2_FN void ML2_CPU_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_CPU_FusedWeightsCopy(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_CPU_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_CPU_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_CPU_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights);

ML2_FN void ML2_CPU_FusedWeightsForwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CPU_FusedWeightsBackwardVectors(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CPU_FusedWeightsCacheBackwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CPU_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CPU_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average);
ML2_FN void ML2_CPU_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum);
ML2_FN void ML2_CPU_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage);
ML2_FN void ML2_CPU_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN void ML2_CPU_AttentionForwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CPU_AttentionCacheBackwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheVectors outputGradient);

ML2_FN ML2_Scalar *ML2_CPU_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int s, ML2_Int c);
ML2_FN void ML2_CPU_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent);
ML2_FN void ML2_CPU_ScalarsClear(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_CPU_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_CPU_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_CPU_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_ScalarsXavierInit(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_CPU_ScalarsHeInit(ML2_LayerCacheScalars scalars);

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CPU_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CPU_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CPU_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CPU_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar *ML2_CPU_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int s, ML2_Int v, ML2_Int c);
ML2_FN void ML2_CPU_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent);
ML2_FN void ML2_CPU_VectorsClear(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_CPU_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_CPU_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_CPU_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_VectorsXavierInit(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_CPU_VectorsHeInit(ML2_LayerCacheVectors vectors);

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CPU_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CPU_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CPU_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CPU_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar *ML2_CPU_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c);
ML2_FN void ML2_CPU_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent);
ML2_FN void ML2_CPU_MatricesClear(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_CPU_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_CPU_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_CPU_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CPU_MatricesXavierInit(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_CPU_MatricesHeInit(ML2_LayerCacheMatrices matrices);

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CPU_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CPU_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CPU_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CPU_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

// ML2_CPU ⬆️

// ML2_CUDA ⬇️

#ifdef ML2_CUDA_ENABLE
ML2_FN void ML2_CUDA_DeviceInit();
ML2_FN ML2_DeviceData ML2_CUDA_DeviceDataNew(ML2_Size size);
ML2_FN void ML2_CUDA_DeviceDataDestroy(ML2_DeviceData *data);

ML2_FN ML2_Scalar *ML2_CUDA_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i);
ML2_FN void ML2_CUDA_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent);
ML2_FN void ML2_CUDA_WeightsClear(ML2_LayerWeights weights);
ML2_FN void ML2_CUDA_WeightsCopy(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_CUDA_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_CUDA_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_WeightsXavierInit(ML2_LayerWeights weights);
ML2_FN void ML2_CUDA_WeightsHeInit(ML2_LayerWeights weights);

ML2_FN void ML2_CUDA_WeightsCopyHostToDevice(ML2_LayerWeights destDevice, ML2_LayerWeights srcHost);
ML2_FN void ML2_CUDA_WeightsCopyDeviceToHost(ML2_LayerWeights destHost, ML2_LayerWeights srcDevice);

ML2_FN void ML2_CUDA_WeightsForwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_WeightsForwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CUDA_WeightsBackwardScalars(ML2_LayerWeights weightsGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_WeightsBackwardVectors(ML2_LayerWeights weightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CUDA_WeightsCacheBackwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_WeightsCacheBackwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient);

ML2_FN void ML2_CUDA_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CUDA_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average);
ML2_FN void ML2_CUDA_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum);
ML2_FN void ML2_CUDA_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage);
ML2_FN void ML2_CUDA_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Scalar *ML2_CUDA_BiasesAt(ML2_LayerBiases biases, ML2_Int i);
ML2_FN void ML2_CUDA_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent);
ML2_FN void ML2_CUDA_BiasesClear(ML2_LayerBiases biases);
ML2_FN void ML2_CUDA_BiasesCopy(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_CUDA_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_CUDA_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_BiasesXavierInit(ML2_LayerBiases biases);
ML2_FN void ML2_CUDA_BiasesHeInit(ML2_LayerBiases biases);

ML2_FN void ML2_CUDA_BiasesCopyHostToDevice(ML2_LayerBiases destDevice, ML2_LayerBiases srcHost);
ML2_FN void ML2_CUDA_BiasesCopyDeviceToHost(ML2_LayerBiases destHost, ML2_LayerBiases srcDevice);

ML2_FN void ML2_CUDA_BiasesForwardScalars(ML2_LayerBiases biases, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_BiasesForwardMatrices(ML2_LayerBiases biases, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_BiasesBackwardScalars(ML2_LayerBiases biasesGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_BiasesBackwardMatrices(ML2_LayerBiases biasesGradient, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CUDA_BiasesCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_BiasesCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CUDA_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average);
ML2_FN void ML2_CUDA_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum);
ML2_FN void ML2_CUDA_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage);
ML2_FN void ML2_CUDA_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN void ML2_CUDA_ActivationForwardScalars(ML2_LayerInfoActivation activation, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_ActivationForwardVectors(ML2_LayerInfoActivation activation, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CUDA_ActivationForwardMatrices(ML2_LayerInfoActivation activation, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_ActivationCacheBackwardInputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardInputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardInputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardInputOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardInputOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);
ML2_FN void ML2_CUDA_ActivationCacheBackwardInputOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_SoftmaxForwardScalars(ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_SoftmaxForwardVectors(ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CUDA_SoftmaxCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN void ML2_CUDA_SoftmaxCacheBackwardVectors(ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);

ML2_FN ML2_Scalar *ML2_CUDA_LinearWeightsAt(ML2_LayerLinear linear, ML2_Int o, ML2_Int i);
ML2_FN ML2_Scalar *ML2_CUDA_LinearBiasesAt(ML2_LayerLinear linear, ML2_Int o);
ML2_FN ML2_LayerWeights ML2_CUDA_LinearWeights(ML2_LayerLinear linear);
ML2_FN ML2_LayerBiases ML2_CUDA_LinearBiases(ML2_LayerLinear linear);
ML2_FN void ML2_CUDA_LinearPrint(ML2_LayerLinear linear, ML2_Int indent);
ML2_FN void ML2_CUDA_LinearClear(ML2_LayerLinear linear);
ML2_FN void ML2_CUDA_LinearCopy(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_CUDA_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_CUDA_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_LinearXavierInit(ML2_LayerLinear linear);
ML2_FN void ML2_CUDA_LinearHeInit(ML2_LayerLinear linear);

ML2_FN void ML2_CUDA_LinearCopyHostToDevice(ML2_LayerLinear destDevice, ML2_LayerLinear srcHost);
ML2_FN void ML2_CUDA_LinearCopyDeviceToHost(ML2_LayerLinear destHost, ML2_LayerLinear srcDevice);

ML2_FN void ML2_CUDA_LinearForwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_LinearBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_LinearCacheBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);

ML2_FN void ML2_CUDA_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CUDA_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average);
ML2_FN void ML2_CUDA_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum);
ML2_FN void ML2_CUDA_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage);
ML2_FN void ML2_CUDA_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Scalar *ML2_CUDA_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN void ML2_CUDA_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent);
ML2_FN void ML2_CUDA_FiltersClear(ML2_LayerFilters filters);
ML2_FN void ML2_CUDA_FiltersCopy(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_CUDA_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_CUDA_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_FiltersXavierInit(ML2_LayerFilters filters);
ML2_FN void ML2_CUDA_FiltersHeInit(ML2_LayerFilters filters);

ML2_FN void ML2_CUDA_FiltersCopyHostToDevice(ML2_LayerFilters destDevice, ML2_LayerFilters srcHost);
ML2_FN void ML2_CUDA_FiltersCopyDeviceToHost(ML2_LayerFilters destHost, ML2_LayerFilters srcDevice);

ML2_FN void ML2_CUDA_FiltersForwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_FiltersBackwardMatrices(ML2_LayerFilters filtersGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CUDA_FiltersCacheBackwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CUDA_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average);
ML2_FN void ML2_CUDA_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum);
ML2_FN void ML2_CUDA_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage);
ML2_FN void ML2_CUDA_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Scalar *ML2_CUDA_ConvFiltersAt(ML2_LayerConv conv, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN ML2_Scalar *ML2_CUDA_ConvBiasesAt(ML2_LayerConv conv, ML2_Int o);
ML2_FN ML2_LayerFilters ML2_CUDA_ConvFilters(ML2_LayerConv conv);
ML2_FN ML2_LayerBiases ML2_CUDA_ConvBiases(ML2_LayerConv conv);
ML2_FN void ML2_CUDA_ConvPrint(ML2_LayerConv conv, ML2_Int indent);
ML2_FN void ML2_CUDA_ConvClear(ML2_LayerConv conv);
ML2_FN void ML2_CUDA_ConvCopy(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_CUDA_ConvSum(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_CUDA_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_ConvXavierInit(ML2_LayerConv conv);
ML2_FN void ML2_CUDA_ConvHeInit(ML2_LayerConv conv);

ML2_FN void ML2_CUDA_ConvCopyHostToDevice(ML2_LayerConv destDevice, ML2_LayerConv srcHost);
ML2_FN void ML2_CUDA_ConvCopyDeviceToHost(ML2_LayerConv destHost, ML2_LayerConv srcDevice);

ML2_FN void ML2_CUDA_ConvForwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_ConvBackwardMatrices(ML2_LayerConv convGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CUDA_ConvCacheBackwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CUDA_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average);
ML2_FN void ML2_CUDA_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum);
ML2_FN void ML2_CUDA_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage);
ML2_FN void ML2_CUDA_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN void ML2_CUDA_FlattenForwardMatrices(ML2_LayerCacheMatrices input, ML2_LayerCacheScalars output);
ML2_FN void ML2_CUDA_FlattenCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheScalars outputGradient);

ML2_FN void ML2_CUDA_PoolMaxForwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_PoolMaxCacheBackwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_PoolAverageForwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_PoolAverageCacheBackwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_PoolSoftmaxForwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_PoolSoftmaxCacheBackwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

ML2_FN ML2_Scalar *ML2_CUDA_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i);
ML2_FN ML2_LayerWeights ML2_CUDA_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w);
ML2_FN void ML2_CUDA_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent);
ML2_FN void ML2_CUDA_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_CUDA_FusedWeightsCopy(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_CUDA_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_CUDA_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_CUDA_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights);

ML2_FN void ML2_CUDA_FusedWeightsCopyHostToDevice(ML2_LayerFusedWeights destDevice, ML2_LayerFusedWeights srcHost);
ML2_FN void ML2_CUDA_FusedWeightsCopyDeviceToHost(ML2_LayerFusedWeights destHost, ML2_LayerFusedWeights srcDevice);

ML2_FN void ML2_CUDA_FusedWeightsForwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices output);
ML2_FN void ML2_CUDA_FusedWeightsBackwardVectors(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices outputGradient);
ML2_FN void ML2_CUDA_FusedWeightsCacheBackwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_CUDA_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_CUDA_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average);
ML2_FN void ML2_CUDA_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum);
ML2_FN void ML2_CUDA_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage);
ML2_FN void ML2_CUDA_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN void ML2_CUDA_AttentionForwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheVectors output);
ML2_FN void ML2_CUDA_AttentionCacheBackwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheVectors outputGradient);

ML2_FN ML2_Scalar *ML2_CUDA_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int s, ML2_Int c);
ML2_FN void ML2_CUDA_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent);
ML2_FN void ML2_CUDA_ScalarsClear(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_CUDA_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_CUDA_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_CUDA_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_ScalarsXavierInit(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_CUDA_ScalarsHeInit(ML2_LayerCacheScalars scalars);

ML2_FN void ML2_CUDA_ScalarsCopyHostToDevice(ML2_LayerCacheScalars destDevice, ML2_LayerCacheScalars srcHost);
ML2_FN void ML2_CUDA_ScalarsCopyDeviceToHost(ML2_LayerCacheScalars destHost, ML2_LayerCacheScalars srcDevice);

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CUDA_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CUDA_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CUDA_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_CUDA_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar *ML2_CUDA_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int s, ML2_Int v, ML2_Int c);
ML2_FN void ML2_CUDA_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent);
ML2_FN void ML2_CUDA_VectorsClear(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_CUDA_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_CUDA_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_CUDA_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_VectorsXavierInit(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_CUDA_VectorsHeInit(ML2_LayerCacheVectors vectors);

ML2_FN void ML2_CUDA_VectorsCopyHostToDevice(ML2_LayerCacheVectors destDevice, ML2_LayerCacheVectors srcHost);
ML2_FN void ML2_CUDA_VectorsCopyDeviceToHost(ML2_LayerCacheVectors destHost, ML2_LayerCacheVectors srcDevice);

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CUDA_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CUDA_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CUDA_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_CUDA_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar *ML2_CUDA_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c);
ML2_FN void ML2_CUDA_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent);
ML2_FN void ML2_CUDA_MatricesClear(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_CUDA_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_CUDA_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_CUDA_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_CUDA_MatricesXavierInit(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_CUDA_MatricesHeInit(ML2_LayerCacheMatrices matrices);

ML2_FN void ML2_CUDA_MatricesCopyHostToDevice(ML2_LayerCacheMatrices destDevice, ML2_LayerCacheMatrices srcHost);
ML2_FN void ML2_CUDA_MatricesCopyDeviceToHost(ML2_LayerCacheMatrices destHost, ML2_LayerCacheMatrices srcDevice);

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CUDA_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CUDA_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CUDA_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_CUDA_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);
#endif // ML2_CUDA_ENABLE

// ML2_CUDA ⬆️

// ML2_Device ⬆️

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_FN ML2_LayerInfo ML2_Weights(ML2_Int inputs, ML2_Int outputs);
ML2_FN ML2_LayerWeights ML2_WeightsNew(ML2_Device device, ML2_LayerInfoWeights info);
ML2_FN void ML2_WeightsDestroy(ML2_LayerWeights *weights);
ML2_FN ML2_LayerInfoWeights ML2_LayerInfoAsWeights(ML2_LayerInfo info);
ML2_FN ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer);
ML2_FN void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, ML2_Int indent);
ML2_FN bool ML2_WeightsInfoForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_WeightsInfoForward(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_FN bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b);

ML2_FN ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i);
ML2_FN void ML2_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent);
ML2_FN void ML2_WeightsClear(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsCopy(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_WeightsXavierInit(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsHeInit(ML2_LayerWeights weights);

ML2_FN void ML2_WeightsCopyHostToDevice(ML2_LayerWeights destDevice, ML2_LayerWeights srcHost);
ML2_FN void ML2_WeightsCopyDeviceToHost(ML2_LayerWeights destHost, ML2_LayerWeights srcDevice);

ML2_FN bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_WeightsForwardCompatibleScalars(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output);
ML2_FN void ML2_WeightsForwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN bool ML2_WeightsForwardCompatibleVectors(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors output);
ML2_FN void ML2_WeightsForwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);

ML2_FN bool ML2_WeightsBackwardCompatible(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_WeightsBackward(ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_WeightsBackwardCompatibleScalars(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_WeightsBackwardScalars(ML2_LayerWeights weightsGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_WeightsBackwardCompatibleVectors(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_WeightsBackwardVectors(ML2_LayerWeights weightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheVectors outputGradient);

ML2_FN bool ML2_WeightsCacheBackwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_WeightsCacheBackward(ML2_LayerWeights weights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_WeightsCacheBackwardCompatibleScalars(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_WeightsCacheBackwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_WeightsCacheBackwardCompatibleVectors(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_WeightsCacheBackwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient);

ML2_FN void ML2_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average);
ML2_FN void ML2_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum);
ML2_FN void ML2_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage);
ML2_FN void ML2_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_WeightsSizeof(ML2_LayerWeights weights);

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_FN ML2_LayerInfo ML2_Biases(ML2_Int inputs);
ML2_FN ML2_LayerBiases ML2_BiasesNew(ML2_Device device, ML2_LayerInfoBiases info);
ML2_FN void ML2_BiasesDestroy(ML2_LayerBiases *biases);
ML2_FN ML2_LayerInfoBiases ML2_LayerInfoAsBiases(ML2_LayerInfo info);
ML2_FN ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer);
ML2_FN void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, ML2_Int indent);
ML2_FN bool ML2_BiasesInfoForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_BiasesInfoForward(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_FN bool ML2_BiasesInfoSame(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b);

ML2_FN ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, ML2_Int i);
ML2_FN void ML2_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent);
ML2_FN void ML2_BiasesClear(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesCopy(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_BiasesXavierInit(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesHeInit(ML2_LayerBiases biases);

ML2_FN void ML2_BiasesCopyHostToDevice(ML2_LayerBiases destDevice, ML2_LayerBiases srcHost);
ML2_FN void ML2_BiasesCopyDeviceToHost(ML2_LayerBiases destHost, ML2_LayerBiases srcDevice);

ML2_FN bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_BiasesForwardCompatibleScalars(ML2_LayerInfoBiases biases, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output);
ML2_FN void ML2_BiasesForwardScalars(ML2_LayerBiases biases, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN bool ML2_BiasesForwardCompatibleMatrices(ML2_LayerInfoBiases biases, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_BiasesForwardMatrices(ML2_LayerBiases biases, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_BiasesBackwardCompatible(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_BiasesBackward(ML2_LayerBiases biasesGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_BiasesBackwardCompatibleScalars(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_BiasesBackwardScalars(ML2_LayerBiases biasesGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_BiasesBackwardCompatibleMatrices(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_BiasesBackwardMatrices(ML2_LayerBiases biasesGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN bool ML2_BiasesCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_BiasesCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_BiasesCacheBackwardCompatibleScalars(ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_BiasesCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_BiasesCacheBackwardCompatibleMatrices(ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_BiasesCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate);
ML2_FN void ML2_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average);
ML2_FN void ML2_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum);
ML2_FN void ML2_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage);
ML2_FN void ML2_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_BiasesSizeof(ML2_LayerBiases biases);

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

// TODO(6/4/2026 2:32:30am): maybe delete this function its more pleasent to use the ones below it
ML2_FN ML2_LayerInfo ML2_Activation(ML2_ActivationType type);
ML2_FN ML2_LayerInfo ML2_ReLU();
ML2_FN ML2_LayerInfo ML2_Sigmoid();
ML2_FN ML2_LayerInfo ML2_Sin();
ML2_FN ML2_LayerInfo ML2_Cos();
ML2_FN ML2_LayerInfo ML2_Tanh();
ML2_FN ML2_LayerInfo ML2_LeakyReLU();
ML2_FN ML2_LayerInfo ML2_SiLU();
ML2_FN ML2_LayerInfoActivation ML2_LayerInfoAsActivation(ML2_LayerInfo info);
ML2_FN ML2_LayerInfoActivation ML2_LayerAsActivation(ML2_Layer layer);
ML2_FN const char *ML2_ActivationNameOf(ML2_ActivationType type);
ML2_FN void ML2_ActivationInfoPrint(ML2_LayerInfoActivation info, ML2_Int indent);
ML2_FN bool ML2_ActivationInfoSame(ML2_LayerInfoActivation a, ML2_LayerInfoActivation b);
ML2_FN ML2_ScalarActivationForward *ML2_ActivationForwardOf(ML2_ActivationType type);
ML2_FN ML2_ScalarActivationBackwardInput *ML2_ActivationBackwardInputOf(ML2_ActivationType type);
ML2_FN ML2_ScalarActivationBackwardOutput *ML2_ActivationBackwardOutputOf(ML2_ActivationType type);
ML2_FN ML2_ScalarActivationBackwardInputOutput *ML2_ActivationBackwardInputOutputOf(ML2_ActivationType type);

ML2_FN bool ML2_ActivationForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_ActivationForward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ActivationForwardCompatibleScalars(ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output);
ML2_FN void ML2_ActivationForwardScalars(ML2_LayerInfoActivation activation, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN bool ML2_ActivationForwardCompatibleVectors(ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors output);
ML2_FN void ML2_ActivationForwardVectors(ML2_LayerInfoActivation activation, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);
ML2_FN bool ML2_ActivationForwardCompatibleMatrices(ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_ActivationForwardMatrices(ML2_LayerInfoActivation activation, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_ActivationCacheBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationCacheBackward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);

ML2_FN bool ML2_ActivationIsBackwardInput(ML2_ActivationType type);
ML2_FN bool ML2_ActivationCacheBackwardInputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardInputCompatibleScalars(ML2_ActivationType type, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardInputCompatibleVectors(ML2_ActivationType type, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardInputCompatibleMatrices(ML2_ActivationType type, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN bool ML2_ActivationIsBackwardOutput(ML2_ActivationType type);
ML2_FN bool ML2_ActivationCacheBackwardOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationCacheBackwardOutput(ML2_ActivationType type, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardOutputCompatibleScalars(ML2_ActivationType type, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars output, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_ActivationCacheBackwardOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardOutputCompatibleVectors(ML2_ActivationType type, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors output, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_ActivationCacheBackwardOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardOutputCompatibleMatrices(ML2_ActivationType type, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices output, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_ActivationCacheBackwardOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

ML2_FN bool ML2_ActivationIsBackwardInputOutput(ML2_ActivationType type);
ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputOutput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatibleScalars(ML2_ActivationType type, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars output, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatibleVectors(ML2_ActivationType type, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors output, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);
ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatibleMatrices(ML2_ActivationType type, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices output, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_ActivationCacheBackwardInputOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

// ML2_LayerActivation ⬆️

// ML2_LayerSoftmax ⬇️

ML2_FN ML2_LayerInfo ML2_Softmax();
ML2_FN void ML2_SoftmaxInfoPrint(ML2_Int indent);

ML2_FN bool ML2_SoftmaxForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_SoftmaxForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_SoftmaxForwardCompatibleScalars(ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output);
ML2_FN void ML2_SoftmaxForwardScalars(ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);
ML2_FN bool ML2_SoftmaxForwardCompatibleVectors(ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors output);
ML2_FN void ML2_SoftmaxForwardVectors(ML2_LayerCacheVectors input, ML2_LayerCacheVectors output);

ML2_FN bool ML2_SoftmaxCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_SoftmaxCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_SoftmaxCacheBackwardCompatibleScalars(ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars output, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_SoftmaxCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient);
ML2_FN bool ML2_SoftmaxCacheBackwardCompatibleVectors(ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors output, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_SoftmaxCacheBackwardVectors(ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient);

// ML2_LayerSoftmax ⬆️

// ML2_LayerLinear ⬇️

ML2_FN ML2_LayerInfo ML2_Linear(ML2_Int inputs, ML2_Int outputs);
ML2_FN ML2_LayerLinear ML2_LinearNew(ML2_Device device, ML2_LayerInfoLinear info);
ML2_FN void ML2_LinearDestroy(ML2_LayerLinear *linear);
ML2_FN ML2_LayerInfoLinear ML2_LayerInfoAsLinear(ML2_LayerInfo info);
ML2_FN ML2_LayerLinear ML2_LayerAsLinear(ML2_Layer layer);
ML2_FN ML2_LayerInfoWeights ML2_LinearInfoWeights(ML2_LayerInfoLinear linear);
ML2_FN ML2_LayerInfoBiases ML2_LinearInfoBiases(ML2_LayerInfoLinear linear);
ML2_FN void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, ML2_Int indent);
ML2_FN bool ML2_LinearInfoForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_LinearInfoForward(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_FN bool ML2_LinearInfoSame(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b);

ML2_FN ML2_Scalar *ML2_LinearWeightsAt(ML2_LayerLinear linear, ML2_Int o, ML2_Int i);
ML2_FN ML2_Scalar *ML2_LinearBiasesAt(ML2_LayerLinear linear, ML2_Int o);
ML2_FN ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear);
ML2_FN ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear);
ML2_FN void ML2_LinearPrint(ML2_LayerLinear linear, ML2_Int indent);
ML2_FN void ML2_LinearClear(ML2_LayerLinear linear);
ML2_FN void ML2_LinearCopy(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LinearXavierInit(ML2_LayerLinear linear);
ML2_FN void ML2_LinearHeInit(ML2_LayerLinear linear);

ML2_FN void ML2_LinearCopyHostToDevice(ML2_LayerLinear destDevice, ML2_LayerLinear srcHost);
ML2_FN void ML2_LinearCopyDeviceToHost(ML2_LayerLinear destHost, ML2_LayerLinear srcDevice);

ML2_FN bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_LinearForwardCompatibleScalars(ML2_LayerInfoLinear linear, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output);
ML2_FN void ML2_LinearForwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output);

ML2_FN bool ML2_LinearBackwardCompatible(ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LinearBackward(ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_LinearBackwardCompatibleScalars(ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_LinearBackwardScalars(ML2_LayerLinear linearGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient);

ML2_FN bool ML2_LinearCacheBackwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LinearCacheBackward(ML2_LayerLinear linear, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_LinearCacheBackwardCompatibleScalars(ML2_LayerInfoLinear linear, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_LinearCacheBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient);

ML2_FN void ML2_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate);
ML2_FN void ML2_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average);
ML2_FN void ML2_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum);
ML2_FN void ML2_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage);
ML2_FN void ML2_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_LinearSizeof(ML2_LayerLinear linear);

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_FN ML2_LayerInfo ML2_Filters(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerFilters ML2_FiltersNew(ML2_Device device, ML2_LayerInfoFilters info);
ML2_FN void ML2_FiltersDestroy(ML2_LayerFilters *filters);
ML2_FN ML2_LayerInfoFilters ML2_LayerInfoAsFilters(ML2_LayerInfo info);
ML2_FN ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer);
ML2_FN void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, ML2_Int indent);
ML2_FN ML2_Int ML2_FiltersInfoOutputHeight(ML2_LayerInfoFilters filters, ML2_Int inputHeight);
ML2_FN ML2_Int ML2_FiltersInfoOutputWidth(ML2_LayerInfoFilters filters, ML2_Int inputWidth);
ML2_FN bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_FN bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b);

ML2_FN ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN void ML2_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent);
ML2_FN void ML2_FiltersClear(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersCopy(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_FiltersXavierInit(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersHeInit(ML2_LayerFilters filters);

ML2_FN void ML2_FiltersCopyHostToDevice(ML2_LayerFilters destDevice, ML2_LayerFilters srcHost);
ML2_FN void ML2_FiltersCopyDeviceToHost(ML2_LayerFilters destHost, ML2_LayerFilters srcDevice);

ML2_FN bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FiltersForwardCompatibleMatrices(ML2_LayerInfoFilters filters, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_FiltersForwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_FiltersBackwardCompatible(ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FiltersBackward(ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FiltersBackwardCompatibleMatrices(ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_FiltersBackwardMatrices(ML2_LayerFilters filtersGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient);

ML2_FN bool ML2_FiltersCacheBackwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FiltersCacheBackward(ML2_LayerFilters filters, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FiltersCacheBackwardCompatibleMatrices(ML2_LayerInfoFilters filters, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_FiltersCacheBackwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate);
ML2_FN void ML2_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average);
ML2_FN void ML2_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum);
ML2_FN void ML2_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage);
ML2_FN void ML2_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_FiltersSizeof(ML2_LayerFilters filters);

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_FN ML2_LayerInfo ML2_Conv(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerConv ML2_ConvNew(ML2_Device device, ML2_LayerInfoConv info);
ML2_FN void ML2_ConvDestroy(ML2_LayerConv *conv);
ML2_FN ML2_LayerInfoConv ML2_LayerInfoAsConv(ML2_LayerInfo info);
ML2_FN ML2_LayerConv ML2_LayerAsConv(ML2_Layer layer);
ML2_FN ML2_LayerInfoFilters ML2_ConvInfoFilters(ML2_LayerInfoConv conv);
ML2_FN ML2_LayerInfoBiases ML2_ConvInfoBiases(ML2_LayerInfoConv conv);
ML2_FN void ML2_ConvInfoPrint(ML2_LayerInfoConv info, ML2_Int indent);
ML2_FN bool ML2_ConvInfoForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_ConvInfoForward(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_FN bool ML2_ConvInfoSame(ML2_LayerInfoConv a, ML2_LayerInfoConv b);

ML2_FN ML2_Scalar *ML2_ConvFiltersAt(ML2_LayerConv conv, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN ML2_Scalar *ML2_ConvBiasesAt(ML2_LayerConv conv, ML2_Int o);
ML2_FN ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv);
ML2_FN ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv);
ML2_FN void ML2_ConvPrint(ML2_LayerConv conv, ML2_Int indent);
ML2_FN void ML2_ConvClear(ML2_LayerConv conv);
ML2_FN void ML2_ConvCopy(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_ConvSum(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ConvXavierInit(ML2_LayerConv conv);
ML2_FN void ML2_ConvHeInit(ML2_LayerConv conv);

ML2_FN void ML2_ConvCopyHostToDevice(ML2_LayerConv destDevice, ML2_LayerConv srcHost);
ML2_FN void ML2_ConvCopyDeviceToHost(ML2_LayerConv destHost, ML2_LayerConv srcDevice);

ML2_FN bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ConvForwardCompatibleMatrices(ML2_LayerInfoConv conv, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_ConvForwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_ConvBackwardCompatible(ML2_LayerInfoConv convGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ConvBackward(ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ConvBackwardCompatibleMatrices(ML2_LayerInfoConv convGradient, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_ConvBackwardMatrices(ML2_LayerConv convGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient);

ML2_FN bool ML2_ConvCacheBackwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ConvCacheBackward(ML2_LayerConv conv, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ConvCacheBackwardCompatibleMatrices(ML2_LayerInfoConv conv, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_ConvCacheBackwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate);
ML2_FN void ML2_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average);
ML2_FN void ML2_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum);
ML2_FN void ML2_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage);
ML2_FN void ML2_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_ConvSizeof(ML2_LayerConv conv);

// ML2_LayerConv ⬆️

// ML2_LayerFlatten ⬇️

ML2_FN ML2_LayerInfo ML2_Flatten();
ML2_FN void ML2_FlattenInfoPrint(ML2_Int indent);
ML2_FN bool ML2_FlattenInfoForwardCompatible(ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerCacheInfo input);

ML2_FN bool ML2_FlattenForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FlattenForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FlattenForwardCompatibleMatrices(ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoScalars output);
ML2_FN void ML2_FlattenForwardMatrices(ML2_LayerCacheMatrices input, ML2_LayerCacheScalars output);

ML2_FN bool ML2_FlattenCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FlattenCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FlattenCacheBackwardCompatibleMatrices(ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoScalars outputGradient);
ML2_FN void ML2_FlattenCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheScalars outputGradient);

// ML2_LayerFlatten ⬆️

// ML2_LayerPool ⬇️

ML2_FN const char *ML2_PoolNameOf(ML2_PoolType type);
ML2_FN void ML2_PoolInfoPrint(ML2_LayerInfoPool pool, ML2_Int indent);
ML2_FN ML2_LayerInfoPool ML2_LayerInfoAsPool(ML2_LayerInfo layer);
ML2_FN ML2_LayerInfoPool ML2_LayerAsPool(ML2_Layer layer);
ML2_FN bool ML2_PoolInfoSame(ML2_LayerInfoPool a, ML2_LayerInfoPool b);
ML2_FN bool ML2_PoolInfoForwardCompatible(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_PoolInfoForward(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input);

ML2_FN bool ML2_PoolForwardCompatible(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_PoolForward(ML2_LayerInfoPool pool, ML2_LayerCache input, ML2_LayerCache output);

ML2_FN bool ML2_PoolCacheBackwardCompatible(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolCacheBackward(ML2_LayerInfoPool pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);

// ML2_LayerPoolMax ⬇️

ML2_FN ML2_LayerInfo ML2_PoolMax(ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerInfoPoolMax ML2_PoolAsMax(ML2_LayerInfoPool pool);
ML2_FN bool ML2_PoolMaxInfoSame(ML2_LayerInfoPoolMax a, ML2_LayerInfoPoolMax b);
ML2_FN ML2_Int ML2_PoolMaxInfoOutputHeight(ML2_LayerInfoPoolMax pool, ML2_Int inputHeight);
ML2_FN ML2_Int ML2_PoolMaxInfoOutputWidth(ML2_LayerInfoPoolMax pool, ML2_Int inputWidth);
ML2_FN bool ML2_PoolMaxInfoForwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_PoolMaxInfoForward(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input);

ML2_FN bool ML2_PoolMaxForwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_PoolMaxForward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_PoolMaxForwardCompatibleMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_PoolMaxForwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_PoolMaxCacheBackwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolMaxCacheBackward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_PoolMaxCacheBackwardCompatibleMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_PoolMaxCacheBackwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

// ML2_LayerPoolMax ⬆️

// ML2_LayerPoolAverage ⬇️

ML2_FN ML2_LayerInfo ML2_PoolAverage(ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerInfoPoolAverage ML2_PoolAsAverage(ML2_LayerInfoPool pool);
ML2_FN bool ML2_PoolAverageInfoSame(ML2_LayerInfoPoolAverage a, ML2_LayerInfoPoolAverage b);
ML2_FN ML2_Int ML2_PoolAverageInfoOutputHeight(ML2_LayerInfoPoolAverage pool, ML2_Int inputHeight);
ML2_FN ML2_Int ML2_PoolAverageInfoOutputWidth(ML2_LayerInfoPoolAverage pool, ML2_Int inputWidth);
ML2_FN bool ML2_PoolAverageInfoForwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_PoolAverageInfoForward(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input);

ML2_FN bool ML2_PoolAverageForwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_PoolAverageForward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_PoolAverageForwardCompatibleMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_PoolAverageForwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_PoolAverageCacheBackwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolAverageCacheBackward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_PoolAverageCacheBackwardCompatibleMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_PoolAverageCacheBackwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient);

// ML2_LayerPoolAverage ⬆️

// ML2_LayerPoolSoftmax ⬇️

ML2_FN ML2_LayerInfo ML2_PoolSoftmax(ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerInfoPoolSoftmax ML2_PoolAsSoftmax(ML2_LayerInfoPool pool);
ML2_FN bool ML2_PoolSoftmaxInfoSame(ML2_LayerInfoPoolSoftmax a, ML2_LayerInfoPoolSoftmax b);
ML2_FN ML2_Int ML2_PoolSoftmaxInfoOutputHeight(ML2_LayerInfoPoolSoftmax pool, ML2_Int inputHeight);
ML2_FN ML2_Int ML2_PoolSoftmaxInfoOutputWidth(ML2_LayerInfoPoolSoftmax pool, ML2_Int inputWidth);
ML2_FN bool ML2_PoolSoftmaxInfoForwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_PoolSoftmaxInfoForward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input);

ML2_FN bool ML2_PoolSoftmaxForwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_PoolSoftmaxForward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_PoolSoftmaxForwardCompatibleMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_PoolSoftmaxForwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_PoolSoftmaxCacheBackwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolSoftmaxCacheBackward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_PoolSoftmaxCacheBackwardCompatibleMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices output, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_PoolSoftmaxCacheBackwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient);

// ML2_LayerPoolSoftmax ⬆️

// ML2_LayerPool ⬆️

// ML2_LayerFusedWeights ⬇️

ML2_FN ML2_LayerInfo ML2_FusedWeights(ML2_Int weights, ML2_Int inputs, ML2_Int outputs);
ML2_FN ML2_LayerFusedWeights ML2_FusedWeightsNew(ML2_Device device, ML2_LayerInfoFusedWeights info);
ML2_FN void ML2_FusedWeightsDestroy(ML2_LayerFusedWeights *fusedWeights);
ML2_FN ML2_LayerInfoFusedWeights ML2_LayerInfoAsFusedWeights(ML2_LayerInfo info);
ML2_FN ML2_LayerFusedWeights ML2_LayerAsFusedWeights(ML2_Layer layer);
ML2_FN ML2_LayerInfoWeights ML2_FusedWeightsInfoWeights(ML2_LayerInfoFusedWeights fusedWeights);
ML2_FN void ML2_FusedWeightsInfoPrint(ML2_LayerInfoFusedWeights info, ML2_Int indent);
ML2_FN bool ML2_FusedWeightsInfoForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FusedWeightsInfoForward(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input);
ML2_FN bool ML2_FusedWeightsInfoSame(ML2_LayerInfoFusedWeights a, ML2_LayerInfoFusedWeights b);

ML2_FN ML2_Scalar *ML2_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i);
ML2_FN ML2_LayerWeights ML2_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w);
ML2_FN void ML2_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent);
ML2_FN void ML2_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_FusedWeightsCopy(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights);

ML2_FN void ML2_FusedWeightsCopyHostToDevice(ML2_LayerFusedWeights destDevice, ML2_LayerFusedWeights srcHost);
ML2_FN void ML2_FusedWeightsCopyDeviceToHost(ML2_LayerFusedWeights destHost, ML2_LayerFusedWeights srcDevice);

ML2_FN bool ML2_FusedWeightsForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FusedWeightsForward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FusedWeightsForwardCompatibleVectors(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoMatrices output);
ML2_FN void ML2_FusedWeightsForwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices output);

ML2_FN bool ML2_FusedWeightsBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FusedWeightsBackward(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FusedWeightsBackwardCompatibleVectors(ML2_LayerInfoFusedWeights fusedWeightsGradient, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_FusedWeightsBackwardVectors(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices outputGradient);

ML2_FN bool ML2_FusedWeightsCacheBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FusedWeightsCacheBackward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FusedWeightsCacheBackwardCompatibleVectors(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoMatrices outputGradient);
ML2_FN void ML2_FusedWeightsCacheBackwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheMatrices outputGradient);

ML2_FN void ML2_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average);
ML2_FN void ML2_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum);
ML2_FN void ML2_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage);
ML2_FN void ML2_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_FusedWeightsSizeof(ML2_LayerFusedWeights fusedWeights);

// ML2_LayerFusedWeights ⬆️

// ML2_LayerAttention ⬇️

ML2_FN ML2_LayerInfo ML2_Attention(ML2_Int heads, bool masked);
ML2_FN ML2_LayerInfoAttention ML2_LayerInfoAsAttention(ML2_LayerInfo info);
ML2_FN ML2_LayerInfoAttention ML2_LayerAsAttention(ML2_Layer layer);
ML2_FN void ML2_AttentionInfoPrint(ML2_LayerInfoAttention info, ML2_Int indent);
ML2_FN bool ML2_AttentionInfoForwardCompatible(ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_AttentionInfoForward(ML2_LayerCacheInfo input);
ML2_FN bool ML2_AttentionInfoSame(ML2_LayerInfoAttention a, ML2_LayerInfoAttention b);

ML2_FN bool ML2_AttentionForwardCompatible(ML2_LayerInfoAttention attention, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_AttentionForward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_AttentionForwardCompatibleMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoVectors output);
ML2_FN void ML2_AttentionForwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheVectors output);

ML2_FN bool ML2_AttentionCacheBackwardCompatible(ML2_LayerInfoAttention attention, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_AttentionCacheBackward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_AttentionCacheBackwardCompatibleMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoVectors outputGradient);
ML2_FN void ML2_AttentionCacheBackwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheVectors outputGradient);

// ML2_LayerAttention ⬆️

ML2_FN ML2_Layer ML2_LayerNew(ML2_Device device, ML2_LayerInfo info);
ML2_FN void ML2_LayerDestroy(ML2_Layer *layer);
ML2_FN ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer);
ML2_FN void ML2_LayerInfoPrint(ML2_LayerInfo layer, ML2_Int indent);
ML2_FN bool ML2_LayerInfoForwardCompatible(ML2_LayerInfo info, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input);
ML2_FN bool ML2_LayerInfoSame(ML2_LayerInfo a, ML2_LayerInfo b);

ML2_FN void ML2_LayerPrint(ML2_Layer layer, ML2_Int indent);
ML2_FN void ML2_LayerClear(ML2_Layer layer);
ML2_FN void ML2_LayerCopy(ML2_Layer dest, ML2_Layer src);
ML2_FN void ML2_LayerSum(ML2_Layer dest, ML2_Layer src);
ML2_FN void ML2_LayerRand(ML2_Layer layer, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LayerXavierInit(ML2_Layer layer);
ML2_FN void ML2_LayerHeInit(ML2_Layer layer);

ML2_FN void ML2_LayerCopyHostToDevice(ML2_Layer destDevice, ML2_Layer srcHost);
ML2_FN void ML2_LayerCopyDeviceToHost(ML2_Layer destHost, ML2_Layer srcDevice);

ML2_FN bool ML2_LayerForwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output);

ML2_FN bool ML2_LayerBackwardCompatible(ML2_LayerInfo gradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LayerBackward(ML2_Layer gradient, ML2_LayerCache input, ML2_LayerCache outputGradient);

ML2_FN bool ML2_LayerCacheBackwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LayerCacheBackward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);

ML2_FN void ML2_LayerGradientDescentOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate);
ML2_FN void ML2_LayerMomentumOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer average);
ML2_FN void ML2_LayerAdagradOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Layer squareSum);
ML2_FN void ML2_LayerRMSPropOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer squareAverage);
ML2_FN void ML2_LayerAdamOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_Layer average, ML2_Layer squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

ML2_FN ML2_Size ML2_LayerSizeof(ML2_Layer layer);

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

// ML2_LayerCacheScalars ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Scalars(ML2_Int samples, ML2_Int scalars);
ML2_FN ML2_LayerCacheScalars ML2_ScalarsNew(ML2_Device device, ML2_LayerCacheInfoScalars info);
ML2_FN void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars);
ML2_FN ML2_LayerCacheInfoScalars ML2_LayerCacheInfoAsScalars(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache);
ML2_FN void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, ML2_Int indent);
ML2_FN bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b);

ML2_FN ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int s, ML2_Int c);
ML2_FN void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent);
ML2_FN void ML2_ScalarsClear(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ScalarsXavierInit(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_ScalarsHeInit(ML2_LayerCacheScalars scalars);

ML2_FN void ML2_ScalarsCopyHostToDevice(ML2_LayerCacheScalars destDevice, ML2_LayerCacheScalars srcHost);
ML2_FN void ML2_ScalarsCopyDeviceToHost(ML2_LayerCacheScalars destHost, ML2_LayerCacheScalars srcDevice);

ML2_FN ML2_Size ML2_ScalarsSizeof(ML2_LayerCacheScalars scalars);

ML2_FN ML2_Scalar ML2_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

ML2_FN ML2_Scalar ML2_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected);
ML2_FN void ML2_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient);

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheVectors ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Vectors(ML2_Int samples, ML2_Int vectors, ML2_Int scalars);
ML2_FN ML2_LayerCacheVectors ML2_VectorsNew(ML2_Device device, ML2_LayerCacheInfoVectors info);
ML2_FN void ML2_VectorsDestroy(ML2_LayerCacheVectors *vectors);
ML2_FN ML2_LayerCacheInfoVectors ML2_LayerCacheInfoAsVectors(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheVectors ML2_LayerCacheAsVectors(ML2_LayerCache cache);
ML2_FN void ML2_VectorsInfoPrint(ML2_LayerCacheInfoVectors info, ML2_Int indent);
ML2_FN bool ML2_VectorsInfoSame(ML2_LayerCacheInfoVectors a, ML2_LayerCacheInfoVectors b);

ML2_FN ML2_Scalar *ML2_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int s, ML2_Int v, ML2_Int c);
ML2_FN void ML2_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent);
ML2_FN void ML2_VectorsClear(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_VectorsXavierInit(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_VectorsHeInit(ML2_LayerCacheVectors vectors);

ML2_FN void ML2_VectorsCopyHostToDevice(ML2_LayerCacheVectors destDevice, ML2_LayerCacheVectors srcHost);
ML2_FN void ML2_VectorsCopyDeviceToHost(ML2_LayerCacheVectors destHost, ML2_LayerCacheVectors srcDevice);

ML2_FN ML2_Size ML2_VectorsSizeof(ML2_LayerCacheVectors vectors);

ML2_FN ML2_Scalar ML2_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

ML2_FN ML2_Scalar ML2_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected);
ML2_FN void ML2_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient);

// ML2_LayerCacheVectors ⬆️

// ML2_LayerCacheMatrices ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Matrices(ML2_Int samples, ML2_Int height, ML2_Int width, ML2_Int channels);
ML2_FN ML2_LayerCacheMatrices ML2_MatricesNew(ML2_Device device, ML2_LayerCacheInfoMatrices info);
ML2_FN void ML2_MatricesDestroy(ML2_LayerCacheMatrices *matrices);
ML2_FN ML2_LayerCacheInfoMatrices ML2_LayerCacheInfoAsMatrices(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheMatrices ML2_LayerCacheAsMatrices(ML2_LayerCache cache);
ML2_FN void ML2_MatricesInfoPrint(ML2_LayerCacheInfoMatrices info, ML2_Int indent);
ML2_FN bool ML2_MatricesInfoSame(ML2_LayerCacheInfoMatrices a, ML2_LayerCacheInfoMatrices b);

ML2_FN ML2_Scalar *ML2_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c);
ML2_FN void ML2_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent);
ML2_FN void ML2_MatricesClear(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_MatricesXavierInit(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_MatricesHeInit(ML2_LayerCacheMatrices matrices);

ML2_FN void ML2_MatricesCopyHostToDevice(ML2_LayerCacheMatrices destDevice, ML2_LayerCacheMatrices srcHost);
ML2_FN void ML2_MatricesCopyDeviceToHost(ML2_LayerCacheMatrices destHost, ML2_LayerCacheMatrices srcDevice);

ML2_FN ML2_Size ML2_MatricesSizeof(ML2_LayerCacheMatrices matrices);

ML2_FN ML2_Scalar ML2_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

ML2_FN ML2_Scalar ML2_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected);
ML2_FN void ML2_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient);

// ML2_LayerCacheMatrices ⬆️

ML2_FN ML2_LayerCache ML2_LayerCacheNew(ML2_Device device, ML2_LayerCacheInfo info);
ML2_FN void ML2_LayerCacheDestroy(ML2_LayerCache *cache);
ML2_FN ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, ML2_Int indent);
ML2_FN bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b);

ML2_FN void ML2_LayerCachePrint(ML2_LayerCache cache, ML2_Int indent);
ML2_FN void ML2_LayerCacheClear(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src);
ML2_FN void ML2_LayerCacheSum(ML2_LayerCache dest, ML2_LayerCache src);
ML2_FN void ML2_LayerCacheRand(ML2_LayerCache cache, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LayerCacheXavierInit(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheHeInit(ML2_LayerCache cache);

ML2_FN void ML2_LayerCacheCopyHostToDevice(ML2_LayerCache destDevice, ML2_LayerCache srcHost);
ML2_FN void ML2_LayerCacheCopyDeviceToHost(ML2_LayerCache destHost, ML2_LayerCache srcDevice);

ML2_FN ML2_Size ML2_LayerCacheSizeof(ML2_LayerCache cache);

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

#define ML2_ArchMake(...) ((ML2_Arch){sizeof((ML2_LayerInfo[]){__VA_ARGS__}) / sizeof(ML2_LayerInfo), (ML2_LayerInfo[]){__VA_ARGS__}})
ML2_FN ML2_Arch ML2_ArchNew(ML2_Int layers, ML2_LayerInfo infos[static layers]);
ML2_FN void ML2_ArchDestroy(ML2_Arch *arch);

// ML2_Arch ⬆️

// ML2_Model ⬇️

ML2_FN ML2_Model ML2_ModelNew(ML2_Device device, ML2_Arch arch);
ML2_FN void ML2_ModelDestroy(ML2_Model *model);
ML2_FN void ML2_ModelInfoPrint(ML2_Model model, ML2_Int indent);
ML2_FN bool ML2_ModelInfoSame(ML2_Model a, ML2_Model b);

ML2_FN void ML2_ModelPrint(ML2_Model model, ML2_Int indent);
ML2_FN void ML2_ModelClear(ML2_Model model);
ML2_FN void ML2_ModelCopy(ML2_Model dest, ML2_Model src);
ML2_FN void ML2_ModelSum(ML2_Model dest, ML2_Model src);
ML2_FN void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ModelXavierInit(ML2_Model model);
ML2_FN void ML2_ModelHeInit(ML2_Model model);

ML2_FN void ML2_ModelCopyHostToDevice(ML2_Model destDevice, ML2_Model srcHost);
ML2_FN void ML2_ModelCopyDeviceToHost(ML2_Model destHost, ML2_Model srcDevice);

ML2_FN bool ML2_ModelForwardCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_LayerCache input);
ML2_FN void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache, ML2_LayerCache input);

ML2_FN bool ML2_ModelBackwardCompatible(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient, ML2_Batch batch);
ML2_FN void ML2_ModelBackward(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient, ML2_Batch batch, ML2_LossBackward lossBackward);

ML2_FN bool ML2_ModelGradientDescentOptimizeCompatible(ML2_Model model, ML2_Model modelGradient);
ML2_FN void ML2_ModelGradientDescentOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerGradientDescent gradientDescent);
ML2_FN bool ML2_ModelMomentumOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerMomentum momentum);
ML2_FN void ML2_ModelMomentumOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerMomentum momentum);
ML2_FN bool ML2_ModelAdagradOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdagrad adagrad);
ML2_FN void ML2_ModelAdagradOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdagrad adagrad);
ML2_FN bool ML2_ModelRMSPropOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerRMSProp RMSProp);
ML2_FN void ML2_ModelRMSPropOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerRMSProp RMSProp);
ML2_FN bool ML2_ModelAdamOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdam adam);
ML2_FN void ML2_ModelAdamOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdam *adam);
ML2_FN void ML2_ModelOptimize(ML2_Model model, ML2_Model modelGradient, ML2_Optimizer *optimizer);

ML2_FN ML2_Size ML2_ModelSizeof(ML2_Model model);

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_FN ML2_ModelCache ML2_ModelCacheNew(ML2_Device device, ML2_Arch arch, ML2_BatchInfo info);
ML2_FN void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache);
ML2_FN void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, ML2_Int indent);
ML2_FN bool ML2_ModelCacheInfoSame(ML2_ModelCache a, ML2_ModelCache b);

ML2_FN void ML2_ModelCachePrint(ML2_ModelCache modelCache, ML2_Int indent);
ML2_FN void ML2_ModelCacheClear(ML2_ModelCache modelCache);

ML2_FN void ML2_ModelCacheCopyHostToDevice(ML2_ModelCache destDevice, ML2_ModelCache srcHost);
ML2_FN void ML2_ModelCacheCopyDeviceToHost(ML2_ModelCache destHost, ML2_ModelCache srcDevice);

ML2_FN ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache);
ML2_FN ML2_Scalar ML2_ModelCacheLoss(ML2_ModelCache modelCache, ML2_LayerCache expected, ML2_LossForward lossForward);
ML2_FN ML2_Size ML2_ModelCacheSizeof(ML2_ModelCache modelCache);

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_FN ML2_BatchInfo ML2_BatchInfoMake(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN ML2_Batch ML2_BatchNew(ML2_Device device, ML2_BatchInfo info);
ML2_FN void ML2_BatchDestroy(ML2_Batch *batch);
ML2_FN void ML2_BatchInfoPrint(ML2_Batch batch, ML2_Int indent);
ML2_FN void ML2_BatchPrint(ML2_Batch batch, ML2_Int indent);
ML2_FN ML2_Size ML2_BatchSizeof(ML2_Batch batch);

// ML2_Batch ⬆️

// ML2_Optimizer ⬇️

// ML2_OptimizerGradientDescent ⬇️

ML2_FN ML2_OptimizerParameters ML2_GradientDescent(ML2_Scalar learningRate);
ML2_FN ML2_OptimizerParametersGradientDescent ML2_OptimizerParametersAsGradientDescent(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerGradientDescent ML2_OptimizerAsGradientDescent(ML2_Optimizer optimizer);
ML2_FN void ML2_GradientDescentInfoPrint(ML2_OptimizerParametersGradientDescent parameters, ML2_Int indent);

// ML2_OptimizerGradientDescent ⬆️

// ML2_OptimizerMomentum ⬇️

ML2_FN ML2_OptimizerParameters ML2_Momentum(ML2_Scalar learningRate, ML2_Scalar decayRate);
ML2_FN ML2_OptimizerMomentum ML2_MomentumNew(ML2_Device device, ML2_OptimizerParametersMomentum parameters, ML2_Arch arch);
ML2_FN void ML2_MomentumDestroy(ML2_OptimizerMomentum *momentum);
ML2_FN void ML2_MomentumReset(ML2_OptimizerMomentum momentum);
ML2_FN ML2_OptimizerParametersMomentum ML2_OptimizerParametersAsMomentum(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerMomentum ML2_OptimizerAsMomentum(ML2_Optimizer optimizer);
ML2_FN void ML2_MomentumInfoPrint(ML2_OptimizerParametersMomentum parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_MomentumSizeof(ML2_OptimizerMomentum momentum);

// ML2_OptimizerMomentum ⬆️

// ML2_OptimizerAdagrad ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adagrad(ML2_Scalar learningRate);
ML2_FN ML2_OptimizerAdagrad ML2_AdagradNew(ML2_Device device, ML2_OptimizerParametersAdagrad parameters, ML2_Arch arch);
ML2_FN void ML2_AdagradDestroy(ML2_OptimizerAdagrad *adagrad);
ML2_FN void ML2_AdagradReset(ML2_OptimizerAdagrad adagrad);
ML2_FN ML2_OptimizerParametersAdagrad ML2_OptimizerParametersAsAdagrad(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerAdagrad ML2_OptimizerAsAdagrad(ML2_Optimizer optimizer);
ML2_FN void ML2_AdagradInfoPrint(ML2_OptimizerParametersAdagrad parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_AdagradSizeof(ML2_OptimizerAdagrad adagrad);

// ML2_OptimizerAdagrad ⬆️

// ML2_OptimizerRMSProp ⬇️

ML2_FN ML2_OptimizerParameters ML2_RMSProp(ML2_Scalar learningRate, ML2_Scalar decayRate);
ML2_FN ML2_OptimizerRMSProp ML2_RMSPropNew(ML2_Device device, ML2_OptimizerParametersRMSProp parameters, ML2_Arch arch);
ML2_FN void ML2_RMSPropDestroy(ML2_OptimizerRMSProp *RMSProp);
ML2_FN void ML2_RMSPropReset(ML2_OptimizerRMSProp RMSProp);
ML2_FN ML2_OptimizerParametersRMSProp ML2_OptimizerParametersAsRMSProp(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerRMSProp ML2_OptimizerAsRMSProp(ML2_Optimizer optimizer);
ML2_FN void ML2_RMSPropInfoPrint(ML2_OptimizerParametersRMSProp parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_RMSPropSizeof(ML2_OptimizerRMSProp RMSProp);

// ML2_OptimizerRMSProp ⬆️

// ML2_OptimizerAdam ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adam(ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2);
ML2_FN ML2_OptimizerAdam ML2_AdamNew(ML2_Device device, ML2_OptimizerParametersAdam parameters, ML2_Arch arch);
ML2_FN void ML2_AdamDestroy(ML2_OptimizerAdam *adam);
ML2_FN void ML2_AdamReset(ML2_OptimizerAdam *adam);
ML2_FN ML2_OptimizerParametersAdam ML2_OptimizerParametersAsAdam(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerAdam ML2_OptimizerAsAdam(ML2_Optimizer optimizer);
ML2_FN void ML2_AdamInfoPrint(ML2_OptimizerParametersAdam parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_AdamSizeof(ML2_OptimizerAdam adam);

// ML2_OptimizerAdam ⬆️

ML2_FN ML2_Optimizer ML2_OptimizerNew(ML2_Device device, ML2_OptimizerParameters parameters, ML2_Arch arch);
ML2_FN void ML2_OptimizerDestroy(ML2_Optimizer *optimizer);
ML2_FN void ML2_OptimizerReset(ML2_Optimizer *optimizer);
ML2_FN ML2_OptimizerParameters ML2_OptimizerAsParameters(ML2_Optimizer optimizer);
ML2_FN void ML2_OptimizerInfoPrint(ML2_OptimizerParameters parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_OptimizerSizeof(ML2_Optimizer optimizer);

// ML2_Optimizer ⬆️

// ML2_Loss ⬇️

ML2_FN ML2_Scalar ML2_LossForwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_FN void ML2_LossBackwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

ML2_FN ML2_Scalar ML2_LossForwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_FN void ML2_LossBackwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

ML2_FN ML2_Scalar ML2_LossForwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_FN void ML2_LossBackwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

ML2_FN ML2_Scalar ML2_LossForwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_FN void ML2_LossBackwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

// ML2_Loss ⬆️

#endif // ML2_H_INCLUDE_GUARD

#ifdef ML2_IMPLEMENTATION

// ML2_Int ⬇️

ML2_FN ML2_Int ML2_IntMax(ML2_Int a, ML2_Int b) {
    return a > b ? a : b;
}

ML2_FN ML2_Int ML2_IntMin(ML2_Int a, ML2_Int b) {
    return a < b ? a : b;
}

// ML2_Int ⬆️

// ML2_Scalar ⬇️

ML2_FN ML2_Scalar ML2_ScalarRand(ML2_Scalar low, ML2_Scalar high) {
    return ((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX) * (high - low) + low;
}

// TODO: i just yoinked this from claude, i dont know if this is what i want it to look like
ML2_FN ML2_Scalar ML2_ScalarRandNormal(ML2_Scalar mean, ML2_Scalar std) {
    ML2_Scalar u1 = ML2_FMAX((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX, ML2_Epsilon);
    ML2_Scalar u2 = (ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX;
    ML2_Scalar magnitude = ML2_SQRT(-ML2_SCALAR_LITERAL(2.0) * ML2_LN(u1)) * ML2_COS(ML2_SCALAR_LITERAL(2.0) * ML2_Pi * u2);
    return mean + (magnitude * std);
}

ML2_FN ML2_Scalar ML2_ScalarReLUForward(ML2_Scalar input) {
    return input > ML2_SCALAR_LITERAL(0.0) ? input : ML2_SCALAR_LITERAL(0.0);
}

ML2_FN ML2_Scalar ML2_ScalarReLUBackwardInput(ML2_Scalar input) {
    return input > ML2_SCALAR_LITERAL(0.0) ? ML2_SCALAR_LITERAL(1.0) : ML2_SCALAR_LITERAL(0.0);
}

ML2_FN ML2_Scalar ML2_ScalarSigmoidForward(ML2_Scalar input) {
    return ML2_SCALAR_LITERAL(1.0) / (ML2_SCALAR_LITERAL(1.0) + ML2_EXP(-input));
}

ML2_FN ML2_Scalar ML2_ScalarSigmoidBackwardOutput(ML2_Scalar output) {
    return output * (ML2_SCALAR_LITERAL(1.0) - output);
}

ML2_FN ML2_Scalar ML2_ScalarSinForward(ML2_Scalar input) {
    return ML2_SIN(input);
}

ML2_FN ML2_Scalar ML2_ScalarSinBackwardInput(ML2_Scalar input) {
    return ML2_COS(input);
}

ML2_FN ML2_Scalar ML2_ScalarCosForward(ML2_Scalar input) {
    return ML2_COS(input);
}

ML2_FN ML2_Scalar ML2_ScalarCosBackwardInput(ML2_Scalar input) {
    return -ML2_SIN(input);
}

ML2_FN ML2_Scalar ML2_ScalarTanhForward(ML2_Scalar input) {
    return ML2_TANH(input);
}

ML2_FN ML2_Scalar ML2_ScalarTanhBackwardOutput(ML2_Scalar output) {
    return ML2_SCALAR_LITERAL(1.0) - output * output;
}

ML2_FN ML2_Scalar ML2_ScalarLeakyReLUForward(ML2_Scalar input) {
    return input > ML2_SCALAR_LITERAL(0.0) ? input : input * ML2_SCALAR_LITERAL(0.01);
}

ML2_FN ML2_Scalar ML2_ScalarLeakyReLUBackwardInput(ML2_Scalar input) {
    return input > ML2_SCALAR_LITERAL(0.0) ? ML2_SCALAR_LITERAL(1.0) : ML2_SCALAR_LITERAL(0.01);
}

ML2_FN ML2_Scalar ML2_ScalarSiLUForward(ML2_Scalar input) {
    return input * ML2_ScalarSigmoidForward(input);
}

ML2_FN ML2_Scalar ML2_ScalarSiLUBackwardInputOutput(ML2_Scalar input, ML2_Scalar output) {
    return ML2_ScalarSigmoidForward(input) * (ML2_SCALAR_LITERAL(1.0) + input - output);
}

// ML2_Scalar ⬆️

// ML2_Device ⬇️

ML2_FN void ML2_DeviceInit(ML2_Device device) {
    switch (device) {
        case ML2_DeviceCPU: /* No action needed */ break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_DeviceInit(); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_DeviceData ML2_DeviceDataNew(ML2_Device device, ML2_Size size) {
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_DeviceDataNew(size);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_DeviceDataNew(size);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_DeviceDataDestroy(ML2_DeviceData *data) {
    switch (data->device) {
        case ML2_DeviceCPU: ML2_CPU_DeviceDataDestroy(data); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_DeviceDataDestroy(data); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_CPU_DeviceData ML2_DeviceDataAsCPU(ML2_DeviceData data) {
    ML2_HARD_ASSERT(data.device == ML2_DeviceCPU);
    return data.as.cpu;
}

ML2_FN ML2_CPU_DeviceData *ML2_DeviceDataPtrAsCPU(ML2_DeviceData *data) {
    ML2_HARD_ASSERT(data->device == ML2_DeviceCPU);
    return &data->as.cpu;
}

#ifdef ML2_CUDA_ENABLE
ML2_FN ML2_CUDA_DeviceData ML2_DeviceDataAsCUDA(ML2_DeviceData data) {
    ML2_HARD_ASSERT(data.device == ML2_DeviceCUDA);
    return data.as.cuda;
}

ML2_FN ML2_CUDA_DeviceData *ML2_DeviceDataPtrAsCUDA(ML2_DeviceData *data) {
    ML2_HARD_ASSERT(data->device == ML2_DeviceCUDA);
    return &data->as.cuda;
}
#endif // ML2_CUDA_ENABLE

// ML2_CPU ⬇️

ML2_FN ML2_DeviceData ML2_CPU_DeviceDataNew(ML2_Size size) {
    void *ptr = ML2_CALLOC(size);
    ML2_DeviceData data = {
        ML2_DeviceCPU,
        .as.cpu = {
            .ptr = ptr,
        },
    };
    return data;
}

ML2_FN void ML2_CPU_DeviceDataDestroy(ML2_DeviceData *data) {
    ML2_HARD_ASSERT(data->device == ML2_DeviceCPU);
    ML2_FREE(ML2_DeviceDataPtrAsCPU(data)->ptr);
    *data = (ML2_DeviceData){};
}

ML2_FN ML2_Scalar *ML2_CPU_DeviceDataAt(ML2_DeviceData data, ML2_Int index) {
    ML2_SOFT_ASSERT(data.device == ML2_DeviceCPU);
    return &data.as.cpu.ptr[index];
}

ML2_FN ML2_Scalar *ML2_CPU_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i) {
    ML2_SOFT_ASSERT(
        0 <= o && o < weights.info.outputs &&
        0 <= i && i < weights.info.inputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(weights.data, o * weights.info.inputs + i);
}

ML2_FN void ML2_CPU_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent) {
    ML2_HARD_ASSERT(weights.data.device == ML2_DeviceCPU);
    ML2_WeightsInfoPrint(weights.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int o = 0; o < weights.info.outputs; o++) {
            printf(ML2_INDENT("", indent));
            for (ML2_Int i = 0; i < weights.info.inputs; i++) {
                printf(ML2_SCALAR_FMT " ", *ML2_CPU_WeightsAt(weights, o, i));
            }
            printf("\n");
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_WeightsClear(ML2_LayerWeights weights) {
    ML2_HARD_ASSERT(weights.data.device == ML2_DeviceCPU);
#if 0
        // TODO(1/10/2026 7:30:00): 2 candidates for how the rest of the library will look, i need to pick 1
        {
            for (ML2_Int o = 0; o < weights.info.outputs; o++) {
                ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
                for (ML2_Int i = 0; i < weights.info.inputs; i++) {
                    weightsPtr[i] = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
        {
            auto restrict weightsPtr = (ML2_Scalar(*)[weights.info.outputs][weights.info.inputs])ML2_CPU_WeightsAt(weights, 0, 0);
            for (ML2_Int o = 0; o < weights.info.outputs; o++) {
                for (ML2_Int i = 0; i < weights.info.inputs; i++) {
                    (*weightsPtr)[o][i] = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
#endif
    for (ML2_Int o = 0; o < weights.info.outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        for (ML2_Int i = 0; i < weights.info.inputs; i++) {
            weightsPtr[i] = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_FN void ML2_CPU_WeightsCopy(ML2_LayerWeights dest, ML2_LayerWeights src) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = dest.info.outputs;
    ML2_Int inputs = dest.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict destPtr = ML2_CPU_WeightsAt(dest, o, 0);
        ML2_Scalar *restrict srcPtr = ML2_CPU_WeightsAt(src, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            destPtr[i] = srcPtr[i];
        }
    }
}

ML2_FN void ML2_CPU_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = dest.info.outputs;
    ML2_Int inputs = dest.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict destPtr = ML2_CPU_WeightsAt(dest, o, 0);
        ML2_Scalar *restrict srcPtr = ML2_CPU_WeightsAt(src, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            destPtr[i] += srcPtr[i];
        }
    }
}

ML2_FN void ML2_CPU_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(weights.data.device == ML2_DeviceCPU);
    for (ML2_Int o = 0; o < weights.info.outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        for (ML2_Int i = 0; i < weights.info.inputs; i++) {
            weightsPtr[i] = ML2_ScalarRand(low, high);
        }
    }
}

ML2_FN void ML2_CPU_WeightsXavierInit(ML2_LayerWeights weights) {
    ML2_HARD_ASSERT(weights.data.device == ML2_DeviceCPU);
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (weights.info.inputs + weights.info.outputs));
    ML2_CPU_WeightsRand(weights, -limit, limit);
}

ML2_FN void ML2_CPU_WeightsHeInit(ML2_LayerWeights weights) {
    ML2_HARD_ASSERT(weights.data.device == ML2_DeviceCPU);
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / weights.info.inputs);
    for (ML2_Int o = 0; o < weights.info.outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        for (ML2_Int i = 0; i < weights.info.inputs; i++) {
            weightsPtr[i] = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
        }
    }
}

ML2_FN void ML2_CPU_WeightsForwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatibleScalars(weights.info, input.info, output.info));
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        for (ML2_Int o = 0; o < outputs; o++) {
            ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
            ML2_Scalar outputScalar = {};
            for (ML2_Int i = 0; i < inputs; i++) {
                outputScalar += inputPtr[i] * weightsPtr[i];
            }
            outputPtr[o] += outputScalar;
        }
    }
}

ML2_FN void ML2_CPU_WeightsForwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatibleVectors(weights.info, input.info, output.info));
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int vectors = input.info.vectors;
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputPtr = ML2_CPU_VectorsAt(input, s, v, 0);
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, v, 0);
            for (ML2_Int o = 0; o < outputs; o++) {
                ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
                ML2_Scalar outputScalar = {};
                for (ML2_Int i = 0; i < inputs; i++) {
                    outputScalar += inputPtr[i] * weightsPtr[i];
                }
                outputPtr[o] += outputScalar;
            }
        }
    }
}

ML2_FN void ML2_CPU_WeightsBackwardScalars(ML2_LayerWeights weightsGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatibleScalars(weightsGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weightsGradient.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int outputs = weightsGradient.info.outputs;
    ML2_Int inputs = weightsGradient.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int o = 0; o < outputs; o++) {
            ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
            for (ML2_Int i = 0; i < inputs; i++) {
                weightsGradientPtr[i] += inputPtr[i] * outputGradientPtr[o];
            }
        }
    }
}

ML2_FN void ML2_CPU_WeightsBackwardVectors(ML2_LayerWeights weightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatibleVectors(weightsGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weightsGradient.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int vectors = input.info.vectors;
    ML2_Int outputs = weightsGradient.info.outputs;
    ML2_Int inputs = weightsGradient.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputPtr = ML2_CPU_VectorsAt(input, s, v, 0);
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, v, 0);
            for (ML2_Int o = 0; o < outputs; o++) {
                ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
                for (ML2_Int i = 0; i < inputs; i++) {
                    weightsGradientPtr[i] += inputPtr[i] * outputGradientPtr[o];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_WeightsCacheBackwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsCacheBackwardCompatibleScalars(weights.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_ScalarsAt(inputGradient, s, 0);
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int o = 0; o < outputs; o++) {
            ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
            for (ML2_Int i = 0; i < inputs; i++) {
                inputGradientPtr[i] += weightsPtr[i] * outputGradientPtr[o];
            }
        }
    }
}

ML2_FN void ML2_CPU_WeightsCacheBackwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsCacheBackwardCompatibleVectors(weights.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputGradientPtr = ML2_CPU_VectorsAt(inputGradient, s, v, 0);
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, v, 0);
            for (ML2_Int o = 0; o < outputs; o++) {
                ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
                for (ML2_Int i = 0; i < inputs; i++) {
                    inputGradientPtr[i] += weightsPtr[i] * outputGradientPtr[o];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        weightsGradient.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            weightsPtr[i] -= weightsGradientPtr[i] * learningRate;
        }
    }
}

ML2_FN void ML2_CPU_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, average.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        weightsGradient.data.device == ML2_DeviceCPU &&
        average.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
        ML2_Scalar *restrict averagePtr = ML2_CPU_WeightsAt(average, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            averagePtr[i] = decayRate * averagePtr[i] - learningRate * weightsGradientPtr[i];
            weightsPtr[i] += averagePtr[i];
        }
    }
}

ML2_FN void ML2_CPU_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        weightsGradient.data.device == ML2_DeviceCPU &&
        squareSum.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
        ML2_Scalar *restrict squareSumPtr = ML2_CPU_WeightsAt(squareSum, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            squareSumPtr[i] += weightsGradientPtr[i] * weightsGradientPtr[i];
            weightsPtr[i] -= weightsGradientPtr[i] * learningRate / ML2_SQRT(squareSumPtr[i] + ML2_Epsilon);
        }
    }
}

ML2_FN void ML2_CPU_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        weightsGradient.data.device == ML2_DeviceCPU &&
        squareAverage.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
        ML2_Scalar *restrict squareAveragePtr = ML2_CPU_WeightsAt(squareAverage, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            squareAveragePtr[i] = decayRate * squareAveragePtr[i] + (ML2_SCALAR_LITERAL(1.0) - decayRate) * weightsGradientPtr[i] * weightsGradientPtr[i];
            weightsPtr[i] -= weightsGradientPtr[i] * learningRate / ML2_SQRT(squareAveragePtr[i] + ML2_Epsilon);
        }
    }
}

ML2_FN void ML2_CPU_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, average.info) &&
        ML2_WeightsInfoSame(weights.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == ML2_DeviceCPU &&
        weightsGradient.data.device == ML2_DeviceCPU &&
        average.data.device == ML2_DeviceCPU &&
        squareAverage.data.device == ML2_DeviceCPU
    );
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        ML2_Scalar *restrict weightsPtr = ML2_CPU_WeightsAt(weights, o, 0);
        ML2_Scalar *restrict weightsGradientPtr = ML2_CPU_WeightsAt(weightsGradient, o, 0);
        ML2_Scalar *restrict averagePtr = ML2_CPU_WeightsAt(average, o, 0);
        ML2_Scalar *restrict squareAveragePtr = ML2_CPU_WeightsAt(squareAverage, o, 0);
        for (ML2_Int i = 0; i < inputs; i++) {
            averagePtr[i] = decayRate1 * averagePtr[i] + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * weightsGradientPtr[i];
            squareAveragePtr[i] = decayRate2 * squareAveragePtr[i] + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * weightsGradientPtr[i] * weightsGradientPtr[i];
            ML2_Scalar averageWeighted = averagePtr[i] / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
            ML2_Scalar squareAverageWeighed = squareAveragePtr[i] / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
            weightsPtr[i] -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_BiasesAt(ML2_LayerBiases biases, ML2_Int i) {
    ML2_SOFT_ASSERT(
        0 <= i && i < biases.info.inputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(biases.data, i);
}

ML2_FN void ML2_CPU_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent) {
    ML2_HARD_ASSERT(biases.data.device == ML2_DeviceCPU);
    ML2_BiasesInfoPrint(biases.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("", indent));
        for (ML2_Int i = 0; i < biases.info.inputs; i++) {
            printf(ML2_SCALAR_FMT " ", *ML2_CPU_BiasesAt(biases, i));
        }
        printf("\n");
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_BiasesClear(ML2_LayerBiases biases) {
    ML2_HARD_ASSERT(biases.data.device == ML2_DeviceCPU);
    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    for (ML2_Int i = 0; i < biases.info.inputs; i++) {
        biasesPtr[i] = ML2_SCALAR_LITERAL(0.0);
    }
}

ML2_FN void ML2_CPU_BiasesCopy(ML2_LayerBiases dest, ML2_LayerBiases src) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );
    ML2_Int inputs = dest.info.inputs;
    ML2_Scalar *restrict destPtr = ML2_CPU_BiasesAt(dest, 0);
    ML2_Scalar *restrict srcPtr = ML2_CPU_BiasesAt(src, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        destPtr[i] = srcPtr[i];
    }
}

ML2_FN void ML2_CPU_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );
    ML2_Int inputs = dest.info.inputs;
    ML2_Scalar *restrict destPtr = ML2_CPU_BiasesAt(dest, 0);
    ML2_Scalar *restrict srcPtr = ML2_CPU_BiasesAt(src, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        destPtr[i] += srcPtr[i];
    }
}

ML2_FN void ML2_CPU_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(biases.data.device == ML2_DeviceCPU);
    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    for (ML2_Int i = 0; i < biases.info.inputs; i++) {
        biasesPtr[i] = ML2_ScalarRand(low, high);
    }
}

ML2_FN void ML2_CPU_BiasesXavierInit(ML2_LayerBiases biases) {
    ML2_CPU_BiasesClear(biases);
}

ML2_FN void ML2_CPU_BiasesHeInit(ML2_LayerBiases biases) {
    ML2_CPU_BiasesClear(biases);
}

ML2_FN void ML2_CPU_BiasesForwardScalars(ML2_LayerBiases biases, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatibleScalars(biases.info, input.info, output.info));
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int scalars = biases.info.inputs;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            outputPtr[c] += inputPtr[c] + biasesPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_BiasesForwardMatrices(ML2_LayerBiases biases, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatibleMatrices(biases.info, input.info, output.info));
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int matrices = input.info.matrices;
    ML2_Int vectors = input.info.vectors;
    ML2_Int scalars = input.info.scalars;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, m, v, 0);
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    outputPtr[c] += inputPtr[c] + biasesPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_BiasesBackwardScalars(ML2_LayerBiases biasesGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatibleScalars(biasesGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        biasesGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = outputGradient.info.samples;
    ML2_Int scalars = outputGradient.info.scalars;

    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            biasesGradientPtr[c] += outputGradientPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_BiasesBackwardMatrices(ML2_LayerBiases biasesGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatibleMatrices(biasesGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        biasesGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = outputGradient.info.samples;
    ML2_Int matrices = outputGradient.info.matrices;
    ML2_Int vectors = outputGradient.info.vectors;
    ML2_Int scalars = outputGradient.info.scalars;

    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    biasesGradientPtr[c] += outputGradientPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_BiasesCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_CPU_ScalarsSum(inputGradient, outputGradient);
}

ML2_FN void ML2_CPU_BiasesCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_CPU_MatricesSum(inputGradient, outputGradient);
}

ML2_FN void ML2_CPU_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        biasesGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int inputs = biases.info.inputs;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        biasesPtr[i] -= biasesGradientPtr[i] * learningRate;
    }
}

ML2_FN void ML2_CPU_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, average.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        biasesGradient.data.device == ML2_DeviceCPU &&
        average.data.device == ML2_DeviceCPU
    );

    ML2_Int inputs = biases.info.inputs;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    ML2_Scalar *restrict averagePtr = ML2_CPU_BiasesAt(average, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        averagePtr[i] = decayRate * averagePtr[i] - learningRate * biasesGradientPtr[i];
        biasesPtr[i] += averagePtr[i];
    }
}

ML2_FN void ML2_CPU_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        biasesGradient.data.device == ML2_DeviceCPU &&
        squareSum.data.device == ML2_DeviceCPU
    );

    ML2_Int inputs = biases.info.inputs;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    ML2_Scalar *restrict squareSumPtr = ML2_CPU_BiasesAt(squareSum, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        squareSumPtr[i] += biasesGradientPtr[i] * biasesGradientPtr[i];
        biasesPtr[i] -= biasesGradientPtr[i] * learningRate / ML2_SQRT(squareSumPtr[i] + ML2_Epsilon);
    }
}

ML2_FN void ML2_CPU_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        biasesGradient.data.device == ML2_DeviceCPU &&
        squareAverage.data.device == ML2_DeviceCPU
    );

    ML2_Int inputs = biases.info.inputs;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    ML2_Scalar *restrict squareAveragePtr = ML2_CPU_BiasesAt(squareAverage, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        squareAveragePtr[i] = decayRate * squareAveragePtr[i] + (ML2_SCALAR_LITERAL(1.0) - decayRate) * biasesGradientPtr[i] * biasesGradientPtr[i];
        biasesPtr[i] -= biasesGradientPtr[i] * learningRate / ML2_SQRT(squareAveragePtr[i] + ML2_Epsilon);
    }
}

ML2_FN void ML2_CPU_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, average.info) &&
        ML2_BiasesInfoSame(biases.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == ML2_DeviceCPU &&
        biasesGradient.data.device == ML2_DeviceCPU &&
        average.data.device == ML2_DeviceCPU &&
        squareAverage.data.device == ML2_DeviceCPU
    );

    ML2_Int inputs = biases.info.inputs;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    ML2_Scalar *restrict biasesGradientPtr = ML2_CPU_BiasesAt(biasesGradient, 0);
    ML2_Scalar *restrict averagePtr = ML2_CPU_BiasesAt(average, 0);
    ML2_Scalar *restrict squareAveragePtr = ML2_CPU_BiasesAt(squareAverage, 0);
    for (ML2_Int i = 0; i < inputs; i++) {
        averagePtr[i] = decayRate1 * averagePtr[i] + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * biasesGradientPtr[i];
        squareAveragePtr[i] = decayRate2 * squareAveragePtr[i] + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * biasesGradientPtr[i] * biasesGradientPtr[i];
        ML2_Scalar averageWeighted = averagePtr[i] / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
        ML2_Scalar squareAverageWeighed = squareAveragePtr[i] / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
        biasesPtr[i] -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
    }
}

ML2_FN void ML2_CPU_ActivationForwardScalars(ML2_LayerInfoActivation activation, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatibleScalars(input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationForward *forward = ML2_ActivationForwardOf(activation.type);
    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            outputPtr[c] += forward(inputPtr[c]);
        }
    }
}

ML2_FN void ML2_CPU_ActivationForwardVectors(ML2_LayerInfoActivation activation, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatibleVectors(input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationForward *forward = ML2_ActivationForwardOf(activation.type);
    ML2_Int samples = input.info.samples;
    ML2_Int vectors = input.info.vectors;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputPtr = ML2_CPU_VectorsAt(input, s, v, 0);
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                outputPtr[c] += forward(inputPtr[c]);
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationForwardMatrices(ML2_LayerInfoActivation activation, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatibleMatrices(input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationForward *forward = ML2_ActivationForwardOf(activation.type);
    ML2_Int samples = input.info.samples;
    ML2_Int matrices = input.info.matrices;
    ML2_Int vectors = input.info.vectors;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, m, v, 0);
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    outputPtr[c] += forward(inputPtr[c]);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardInputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatibleScalars(type, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardInput *backwardInput = ML2_ActivationBackwardInputOf(type);
    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_ScalarsAt(inputGradient, s, 0);
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            inputGradientPtr[c] += backwardInput(inputPtr[c]) * outputGradientPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardInputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatibleVectors(type, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardInput *backwardInput = ML2_ActivationBackwardInputOf(type);
    ML2_Int samples = input.info.samples;
    ML2_Int vectors = input.info.vectors;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputPtr = ML2_CPU_VectorsAt(input, s, v, 0);
            ML2_Scalar *restrict inputGradientPtr = ML2_CPU_VectorsAt(inputGradient, s, v, 0);
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                inputGradientPtr[c] += backwardInput(inputPtr[c]) * outputGradientPtr[c];
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardInputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatibleMatrices(type, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardInput *backwardInput = ML2_ActivationBackwardInputOf(type);
    ML2_Int samples = input.info.samples;
    ML2_Int matrices = input.info.matrices;
    ML2_Int vectors = input.info.vectors;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, m, v, 0);
                ML2_Scalar *restrict inputGradientPtr = ML2_CPU_MatricesAt(inputGradient, s, m, v, 0);
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    inputGradientPtr[c] += backwardInput(inputPtr[c]) * outputGradientPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatibleScalars(type, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardOutput *backwardOutput = ML2_ActivationBackwardOutputOf(type);
    ML2_Int samples = inputGradient.info.samples;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_ScalarsAt(inputGradient, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            inputGradientPtr[c] += backwardOutput(outputPtr[c]) * outputGradientPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatibleVectors(type, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardOutput *backwardOutput = ML2_ActivationBackwardOutputOf(type);
    ML2_Int samples = inputGradient.info.samples;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputGradientPtr = ML2_CPU_VectorsAt(inputGradient, s, v, 0);
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, v, 0);
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                inputGradientPtr[c] += backwardOutput(outputPtr[c]) * outputGradientPtr[c];
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatibleMatrices(type, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardOutput *backwardOutput = ML2_ActivationBackwardOutputOf(type);
    ML2_Int samples = inputGradient.info.samples;
    ML2_Int matrices = inputGradient.info.matrices;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputGradientPtr = ML2_CPU_MatricesAt(inputGradient, s, m, v, 0);
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, m, v, 0);
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    inputGradientPtr[c] += backwardOutput(outputPtr[c]) * outputGradientPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardInputOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatibleScalars(type, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardInputOutput *backwardInputOutput = ML2_ActivationBackwardInputOutputOf(type);
    ML2_Int samples = inputGradient.info.samples;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_ScalarsAt(inputGradient, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            inputGradientPtr[c] += backwardInputOutput(inputPtr[c], outputPtr[c]) * outputGradientPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardInputOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatibleVectors(type, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardInputOutput *backwardInputOutput = ML2_ActivationBackwardInputOutputOf(type);
    ML2_Int samples = inputGradient.info.samples;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputPtr = ML2_CPU_VectorsAt(input, s, v, 0);
            ML2_Scalar *restrict inputGradientPtr = ML2_CPU_VectorsAt(inputGradient, s, v, 0);
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, v, 0);
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                inputGradientPtr[c] += backwardInputOutput(inputPtr[c], outputPtr[c]) * outputGradientPtr[c];
            }
        }
    }
}

ML2_FN void ML2_CPU_ActivationCacheBackwardInputOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatibleMatrices(type, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_ScalarActivationBackwardInputOutput *backwardInputOutput = ML2_ActivationBackwardInputOutputOf(type);
    ML2_Int samples = inputGradient.info.samples;
    ML2_Int matrices = inputGradient.info.matrices;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, m, v, 0);
                ML2_Scalar *restrict inputGradientPtr = ML2_CPU_MatricesAt(inputGradient, s, m, v, 0);
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, m, v, 0);
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    inputGradientPtr[c] += backwardInputOutput(inputPtr[c], outputPtr[c]) * outputGradientPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_SoftmaxForwardScalars(ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_SoftmaxForwardCompatibleScalars(input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = output.info.samples;
    ML2_Int scalars = output.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputPtr = ML2_CPU_ScalarsAt(input, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);

        ML2_Scalar max = ML2_NegInf;
        for (ML2_Int c = 0; c < scalars; c++) {
            max = ML2_FMAX(max, inputPtr[c]);
        }
        ML2_Scalar sum = {};
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar e = ML2_EXP(inputPtr[c] - max);
            sum += e;
        }
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar e = ML2_EXP(inputPtr[c] - max);
            outputPtr[c] += e / sum;
        }
    }
}

ML2_FN void ML2_CPU_SoftmaxForwardVectors(ML2_LayerCacheVectors input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_SoftmaxForwardCompatibleVectors(input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = output.info.samples;
    ML2_Int vectors = output.info.vectors;
    ML2_Int scalars = output.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputPtr = ML2_CPU_VectorsAt(input, s, v, 0);
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, v, 0);

            ML2_Scalar max = ML2_NegInf;
            for (ML2_Int c = 0; c < scalars; c++) {
                max = ML2_FMAX(max, inputPtr[c]);
            }
            ML2_Scalar sum = {};
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar e = ML2_EXP(inputPtr[c] - max);
                sum += e;
            }
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar e = ML2_EXP(inputPtr[c] - max);
                outputPtr[c] += e / sum;
            }
        }
    }
}

ML2_FN void ML2_CPU_SoftmaxCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_SoftmaxCacheBackwardCompatibleScalars(inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_ScalarsAt(inputGradient, s, 0);
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);

        ML2_Scalar sum = {};
        for (ML2_Int c = 0; c < scalars; c++) {
            sum += outputPtr[c] * outputGradientPtr[c];
        }
        for (ML2_Int c = 0; c < scalars; c++) {
            inputGradientPtr[c] += outputPtr[c] * (outputGradientPtr[c] - sum);
        }
    }
}

ML2_FN void ML2_CPU_SoftmaxCacheBackwardVectors(ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_SoftmaxCacheBackwardCompatibleVectors(inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict inputGradientPtr = ML2_CPU_VectorsAt(inputGradient, s, v, 0);
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, v, 0);
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, v, 0);

            ML2_Scalar sum = {};
            for (ML2_Int c = 0; c < scalars; c++) {
                sum += outputPtr[c] * outputGradientPtr[c];
            }
            for (ML2_Int c = 0; c < scalars; c++) {
                inputGradientPtr[c] += outputPtr[c] * (outputGradientPtr[c] - sum);
            }
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_LinearWeightsAt(ML2_LayerLinear linear, ML2_Int o, ML2_Int i) {
    ML2_SOFT_ASSERT(
        0 <= o && o < linear.info.outputs &&
        0 <= i && i < linear.info.inputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(linear.data, o * linear.info.inputs + i);
}

ML2_FN ML2_Scalar *ML2_CPU_LinearBiasesAt(ML2_LayerLinear linear, ML2_Int o) {
    ML2_SOFT_ASSERT(
        0 <= o && o < linear.info.outputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(linear.data, o + linear.info.outputs * linear.info.inputs);
}

ML2_FN ML2_LayerWeights ML2_CPU_LinearWeights(ML2_LayerLinear linear) {
    ML2_HARD_ASSERT(linear.data.device == ML2_DeviceCPU);
    return (ML2_LayerWeights){ML2_LinearInfoWeights(linear.info), {linear.data.device, .as.cpu = {ML2_CPU_LinearWeightsAt(linear, 0, 0)}}};
}

ML2_FN ML2_LayerBiases ML2_CPU_LinearBiases(ML2_LayerLinear linear) {
    ML2_HARD_ASSERT(linear.data.device == ML2_DeviceCPU);
    return (ML2_LayerBiases){ML2_LinearInfoBiases(linear.info), {linear.data.device, .as.cpu = {ML2_CPU_LinearBiasesAt(linear, 0)}}};
}

ML2_FN void ML2_CPU_LinearPrint(ML2_LayerLinear linear, ML2_Int indent) {
    ML2_HARD_ASSERT(linear.data.device == ML2_DeviceCPU);
    ML2_LinearInfoPrint(linear.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        ML2_CPU_WeightsPrint(ML2_CPU_LinearWeights(linear), indent);
        ML2_CPU_BiasesPrint(ML2_CPU_LinearBiases(linear), indent);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_LinearClear(ML2_LayerLinear linear) {
    ML2_CPU_WeightsClear(ML2_CPU_LinearWeights(linear));
    ML2_CPU_BiasesClear(ML2_CPU_LinearBiases(linear));
}

ML2_FN void ML2_CPU_LinearCopy(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_CPU_WeightsCopy(ML2_CPU_LinearWeights(dest), ML2_CPU_LinearWeights(src));
    ML2_CPU_BiasesCopy(ML2_CPU_LinearBiases(dest), ML2_CPU_LinearBiases(src));
}

ML2_FN void ML2_CPU_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_CPU_WeightsSum(ML2_CPU_LinearWeights(dest), ML2_CPU_LinearWeights(src));
    ML2_CPU_BiasesSum(ML2_CPU_LinearBiases(dest), ML2_CPU_LinearBiases(src));
}

ML2_FN void ML2_CPU_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high) {
    ML2_CPU_WeightsRand(ML2_CPU_LinearWeights(linear), low, high);
    ML2_CPU_BiasesRand(ML2_CPU_LinearBiases(linear), low, high);
}

ML2_FN void ML2_CPU_LinearXavierInit(ML2_LayerLinear linear) {
    ML2_CPU_WeightsXavierInit(ML2_CPU_LinearWeights(linear));
    ML2_CPU_BiasesXavierInit(ML2_CPU_LinearBiases(linear));
}

ML2_FN void ML2_CPU_LinearHeInit(ML2_LayerLinear linear) {
    ML2_CPU_WeightsHeInit(ML2_CPU_LinearWeights(linear));
    ML2_CPU_BiasesHeInit(ML2_CPU_LinearBiases(linear));
}

ML2_FN void ML2_CPU_LinearForwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_LinearForwardCompatibleScalars(linear.info, input.info, output.info));
    ML2_HARD_ASSERT(
        linear.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_CPU_WeightsForwardScalars(ML2_CPU_LinearWeights(linear), input, output);
    ML2_LayerBiases biases = ML2_CPU_LinearBiases(linear);
    ML2_Int samples = output.info.samples;
    ML2_Int scalars = output.info.scalars;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            outputPtr[c] += biasesPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_LinearBackwardScalars(ML2_LayerLinear linearGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearBackwardCompatibleScalars(linearGradient.info, input.info, outputGradient.info));
    ML2_CPU_WeightsBackwardScalars(ML2_CPU_LinearWeights(linearGradient), input, outputGradient);
    ML2_CPU_BiasesBackwardScalars(ML2_CPU_LinearBiases(linearGradient), outputGradient);
}

ML2_FN void ML2_CPU_LinearCacheBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearCacheBackwardCompatibleScalars(linear.info, inputGradient.info, outputGradient.info));
    ML2_CPU_WeightsCacheBackwardScalars(ML2_CPU_LinearWeights(linear), inputGradient, outputGradient);
}

ML2_FN void ML2_CPU_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_CPU_WeightsGradientDescentOptimize(ML2_CPU_LinearWeights(linear), ML2_CPU_LinearWeights(linearGradient), learningRate);
    ML2_CPU_BiasesGradientDescentOptimize(ML2_CPU_LinearBiases(linear), ML2_CPU_LinearBiases(linearGradient), learningRate);
}

ML2_FN void ML2_CPU_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, average.info));
    ML2_CPU_WeightsMomentumOptimize(ML2_CPU_LinearWeights(linear), ML2_CPU_LinearWeights(linearGradient), learningRate, decayRate, ML2_CPU_LinearWeights(average));
    ML2_CPU_BiasesMomentumOptimize(ML2_CPU_LinearBiases(linear), ML2_CPU_LinearBiases(linearGradient), learningRate, decayRate, ML2_CPU_LinearBiases(average));
}

ML2_FN void ML2_CPU_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, squareSum.info));
    ML2_CPU_WeightsAdagradOptimize(ML2_CPU_LinearWeights(linear), ML2_CPU_LinearWeights(linearGradient), learningRate, ML2_CPU_LinearWeights(squareSum));
    ML2_CPU_BiasesAdagradOptimize(ML2_CPU_LinearBiases(linear), ML2_CPU_LinearBiases(linearGradient), learningRate, ML2_CPU_LinearBiases(squareSum));
}

ML2_FN void ML2_CPU_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, squareAverage.info));
    ML2_CPU_WeightsRMSPropOptimize(ML2_CPU_LinearWeights(linear), ML2_CPU_LinearWeights(linearGradient), learningRate, decayRate, ML2_CPU_LinearWeights(squareAverage));
    ML2_CPU_BiasesRMSPropOptimize(ML2_CPU_LinearBiases(linear), ML2_CPU_LinearBiases(linearGradient), learningRate, decayRate, ML2_CPU_LinearBiases(squareAverage));
}

ML2_FN void ML2_CPU_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, average.info));
    ML2_CPU_WeightsAdamOptimize(ML2_CPU_LinearWeights(linear), ML2_CPU_LinearWeights(linearGradient), learningRate, decayRate1, decayRate2, ML2_CPU_LinearWeights(average), ML2_CPU_LinearWeights(squareAverage), decayingWeight1, decayingWeight2);
    ML2_CPU_BiasesAdamOptimize(ML2_CPU_LinearBiases(linear), ML2_CPU_LinearBiases(linearGradient), learningRate, decayRate1, decayRate2, ML2_CPU_LinearBiases(average), ML2_CPU_LinearBiases(squareAverage), decayingWeight1, decayingWeight2);
}

ML2_FN ML2_Scalar *ML2_CPU_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o) {
    ML2_SOFT_ASSERT(
        0 <= y && y < filters.info.height &&
        0 <= x && x < filters.info.width &&
        0 <= i && i < filters.info.inputs &&
        0 <= o && o < filters.info.outputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(filters.data, ((y * filters.info.width + x) * filters.info.inputs + i) * filters.info.outputs + o);
}

ML2_FN void ML2_CPU_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent) {
    ML2_HARD_ASSERT(filters.data.device == ML2_DeviceCPU);
    ML2_FiltersInfoPrint(filters.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int y = 0; y < filters.info.height; y++) {
            printf(ML2_INDENT("", indent));
            for (ML2_Int x = 0; x < filters.info.width; x++) {
                for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                    for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                        printf(ML2_INDENT("Weights[%d,%d]:\n", indent), i, o);
                        printf(ML2_INDENT("{\n", indent));
                        {
                            indent += ML2_Indentation;
                            printf(ML2_SCALAR_FMT " ", *ML2_CPU_FiltersAt(filters, y, x, i, o));
                            indent -= ML2_Indentation;
                        }
                        printf(ML2_INDENT("}\n", indent));
                    }
                }
                printf("\n");
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_FiltersClear(ML2_LayerFilters filters) {
    ML2_HARD_ASSERT(filters.data.device == ML2_DeviceCPU);
    for (ML2_Int y = 0; y < filters.info.height; y++) {
        for (ML2_Int x = 0; x < filters.info.width; x++) {
            for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                    filtersPtr[o] = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersCopy(ML2_LayerFilters dest, ML2_LayerFilters src) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );
    ML2_Int height = dest.info.height;
    ML2_Int width = dest.info.width;
    ML2_Int inputs = dest.info.inputs;
    ML2_Int outputs = dest.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict destPtr = ML2_CPU_FiltersAt(dest, y, x, i, 0);
                ML2_Scalar *restrict srcPtr = ML2_CPU_FiltersAt(src, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    destPtr[o] = srcPtr[o];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );
    ML2_Int height = dest.info.height;
    ML2_Int width = dest.info.width;
    ML2_Int inputs = dest.info.inputs;
    ML2_Int outputs = dest.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict destPtr = ML2_CPU_FiltersAt(dest, y, x, i, 0);
                ML2_Scalar *restrict srcPtr = ML2_CPU_FiltersAt(src, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    destPtr[o] += srcPtr[o];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(filters.data.device == ML2_DeviceCPU);
    for (ML2_Int y = 0; y < filters.info.height; y++) {
        for (ML2_Int x = 0; x < filters.info.width; x++) {
            for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                    filtersPtr[o] = ML2_ScalarRand(low, high);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersXavierInit(ML2_LayerFilters filters) {
    ML2_HARD_ASSERT(filters.data.device == ML2_DeviceCPU);
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / ((filters.info.inputs + filters.info.outputs) * filters.info.height * filters.info.width));
    ML2_CPU_FiltersRand(filters, -limit, limit);
}

ML2_FN void ML2_CPU_FiltersHeInit(ML2_LayerFilters filters) {
    ML2_HARD_ASSERT(filters.data.device == ML2_DeviceCPU);
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (filters.info.height * filters.info.width * filters.info.inputs));
    for (ML2_Int y = 0; y < filters.info.height; y++) {
        for (ML2_Int x = 0; x < filters.info.width; x++) {
            for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                    filtersPtr[o] = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersForwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_FiltersForwardCompatibleMatrices(filters.info, input.info, output.info));
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int outputs = filters.info.outputs;
    ML2_Int inputs = filters.info.inputs;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = output.info.matrices;
    ML2_Int outputWidth = output.info.vectors;
    ML2_Int filtersHeight = filters.info.height;
    ML2_Int filtersWidth = filters.info.width;
    ML2_Int paddingHeight = filters.info.paddingHeight;
    ML2_Int paddingWidth = filters.info.paddingWidth;
    ML2_Int strideHeight = filters.info.strideHeight;
    ML2_Int strideWidth = filters.info.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, yo, xo, 0);
                ML2_Int yfMin = ML2_IntMax(paddingHeight - yo * strideHeight, 0);
                ML2_Int yfMax = ML2_IntMin(inputHeight - 1 + paddingHeight - yo * strideHeight, filtersHeight - 1);
                ML2_Int xfMin = ML2_IntMax(paddingWidth - xo * strideWidth, 0);
                ML2_Int xfMax = ML2_IntMin(inputWidth - 1 + paddingWidth - xo * strideWidth, filtersWidth - 1);
                for (ML2_Int yf = yfMin; yf <= yfMax; yf++) {
                    for (ML2_Int xf = xfMin; xf <= xfMax; xf++) {
                        ML2_Int yi = yo * strideHeight + yf - paddingHeight;
                        ML2_Int xi = xo * strideWidth + xf - paddingWidth;
                        ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, yi, xi, 0);
                        for (ML2_Int i = 0; i < inputs; i++) {
                            ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, yf, xf, i, 0);
                            ML2_Scalar inputScalar = inputPtr[i];
                            for (ML2_Int o = 0; o < outputs; o++) {
                                outputPtr[o] += inputScalar * filtersPtr[o];
                            }
                        }
                    }
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersBackwardMatrices(ML2_LayerFilters filtersGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersBackwardCompatibleMatrices(filtersGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        filtersGradient.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int outputs = filtersGradient.info.outputs;
    ML2_Int inputs = filtersGradient.info.inputs;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = outputGradient.info.matrices;
    ML2_Int outputWidth = outputGradient.info.vectors;
    ML2_Int filtersHeight = filtersGradient.info.height;
    ML2_Int filtersWidth = filtersGradient.info.width;
    ML2_Int paddingHeight = filtersGradient.info.paddingHeight;
    ML2_Int paddingWidth = filtersGradient.info.paddingWidth;
    ML2_Int strideHeight = filtersGradient.info.strideHeight;
    ML2_Int strideWidth = filtersGradient.info.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, yo, xo, 0);
                ML2_Int yfMin = ML2_IntMax(paddingHeight - yo * strideHeight, 0);
                ML2_Int yfMax = ML2_IntMin(inputHeight - 1 + paddingHeight - yo * strideHeight, filtersHeight - 1);
                ML2_Int xfMin = ML2_IntMax(paddingWidth - xo * strideWidth, 0);
                ML2_Int xfMax = ML2_IntMin(inputWidth - 1 + paddingWidth - xo * strideWidth, filtersWidth - 1);
                for (ML2_Int yf = yfMin; yf <= yfMax; yf++) {
                    for (ML2_Int xf = xfMin; xf <= xfMax; xf++) {
                        ML2_Int yi = yo * strideHeight + yf - paddingHeight;
                        ML2_Int xi = xo * strideWidth + xf - paddingWidth;
                        ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, yi, xi, 0);
                        for (ML2_Int i = 0; i < inputs; i++) {
                            ML2_Scalar *restrict filtersGradientPtr = ML2_CPU_FiltersAt(filtersGradient, yf, xf, i, 0);
                            ML2_Scalar inputScalar = inputPtr[i];
                            for (ML2_Int o = 0; o < outputs; o++) {
                                filtersGradientPtr[o] += inputScalar * outputGradientPtr[o];
                            }
                        }
                    }
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersCacheBackwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersCacheBackwardCompatibleMatrices(filters.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int outputs = filters.info.outputs;
    ML2_Int inputs = filters.info.inputs;

    ML2_Int inputHeight = inputGradient.info.matrices;
    ML2_Int inputWidth = inputGradient.info.vectors;
    ML2_Int outputHeight = outputGradient.info.matrices;
    ML2_Int outputWidth = outputGradient.info.vectors;
    ML2_Int filtersHeight = filters.info.height;
    ML2_Int filtersWidth = filters.info.width;
    ML2_Int paddingHeight = filters.info.paddingHeight;
    ML2_Int paddingWidth = filters.info.paddingWidth;
    ML2_Int strideHeight = filters.info.strideHeight;
    ML2_Int strideWidth = filters.info.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, yo, xo, 0);
                ML2_Int yfMin = ML2_IntMax(paddingHeight - yo * strideHeight, 0);
                ML2_Int yfMax = ML2_IntMin(inputHeight - 1 + paddingHeight - yo * strideHeight, filtersHeight - 1);
                ML2_Int xfMin = ML2_IntMax(paddingWidth - xo * strideWidth, 0);
                ML2_Int xfMax = ML2_IntMin(inputWidth - 1 + paddingWidth - xo * strideWidth, filtersWidth - 1);
                for (ML2_Int yf = yfMin; yf <= yfMax; yf++) {
                    for (ML2_Int xf = xfMin; xf <= xfMax; xf++) {
                        ML2_Int yi = yo * strideHeight + yf - paddingHeight;
                        ML2_Int xi = xo * strideWidth + xf - paddingWidth;
                        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_MatricesAt(inputGradient, s, yi, xi, 0);
                        for (ML2_Int i = 0; i < inputs; i++) {
                            ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, yf, xf, i, 0);
                            ML2_Scalar inputGradientScalar = {};
                            for (ML2_Int o = 0; o < outputs; o++) {
                                inputGradientScalar += filtersPtr[o] * outputGradientPtr[o];
                            }
                            inputGradientPtr[i] += inputGradientScalar;
                        }
                    }
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        filtersGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;

    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                ML2_Scalar *restrict filtersGradientPtr = ML2_CPU_FiltersAt(filtersGradient, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    filtersPtr[o] -= filtersGradientPtr[o] * learningRate;
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, average.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        filtersGradient.data.device == ML2_DeviceCPU &&
        average.data.device == ML2_DeviceCPU
    );

    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;

    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                ML2_Scalar *restrict filtersGradientPtr = ML2_CPU_FiltersAt(filtersGradient, y, x, i, 0);
                ML2_Scalar *restrict averagePtr = ML2_CPU_FiltersAt(average, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    averagePtr[o] = decayRate * averagePtr[o] - learningRate * filtersGradientPtr[o];
                    filtersPtr[o] += averagePtr[o];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        filtersGradient.data.device == ML2_DeviceCPU &&
        squareSum.data.device == ML2_DeviceCPU
    );

    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;

    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                ML2_Scalar *restrict filtersGradientPtr = ML2_CPU_FiltersAt(filtersGradient, y, x, i, 0);
                ML2_Scalar *restrict squareSumPtr = ML2_CPU_FiltersAt(squareSum, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    squareSumPtr[o] += filtersGradientPtr[o] * filtersGradientPtr[o];
                    filtersPtr[o] -= filtersGradientPtr[o] * learningRate / ML2_SQRT(squareSumPtr[o] + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        filtersGradient.data.device == ML2_DeviceCPU &&
        squareAverage.data.device == ML2_DeviceCPU
    );

    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;

    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                ML2_Scalar *restrict filtersGradientPtr = ML2_CPU_FiltersAt(filtersGradient, y, x, i, 0);
                ML2_Scalar *restrict squareAveragePtr = ML2_CPU_FiltersAt(squareAverage, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    squareAveragePtr[o] = decayRate * squareAveragePtr[o] + (ML2_SCALAR_LITERAL(1.0) - decayRate) * filtersGradientPtr[o] * filtersGradientPtr[o];
                    filtersPtr[o] -= filtersGradientPtr[o] * learningRate / ML2_SQRT(squareAveragePtr[o] + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, average.info) &&
        ML2_FiltersInfoSame(filters.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == ML2_DeviceCPU &&
        filtersGradient.data.device == ML2_DeviceCPU &&
        average.data.device == ML2_DeviceCPU &&
        squareAverage.data.device == ML2_DeviceCPU
    );

    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;

    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                ML2_Scalar *restrict filtersPtr = ML2_CPU_FiltersAt(filters, y, x, i, 0);
                ML2_Scalar *restrict filtersGradientPtr = ML2_CPU_FiltersAt(filtersGradient, y, x, i, 0);
                ML2_Scalar *restrict averagePtr = ML2_CPU_FiltersAt(average, y, x, i, 0);
                ML2_Scalar *restrict squareAveragePtr = ML2_CPU_FiltersAt(squareAverage, y, x, i, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    averagePtr[o] = decayRate1 * averagePtr[o] + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * filtersGradientPtr[o];
                    squareAveragePtr[o] = decayRate2 * squareAveragePtr[o] + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * filtersGradientPtr[o] * filtersGradientPtr[o];
                    ML2_Scalar averageWeighted = averagePtr[o] / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
                    ML2_Scalar squareAverageWeighed = squareAveragePtr[o] / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
                    filtersPtr[o] -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_ConvFiltersAt(ML2_LayerConv conv, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o) {
    ML2_SOFT_ASSERT(
        0 <= y && y < conv.info.height &&
        0 <= x && x < conv.info.width &&
        0 <= i && i < conv.info.inputs &&
        0 <= o && o < conv.info.outputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(conv.data, ((y * conv.info.width + x) * conv.info.inputs + i) * conv.info.outputs + o);
}

ML2_FN ML2_Scalar *ML2_CPU_ConvBiasesAt(ML2_LayerConv conv, ML2_Int o) {
    ML2_SOFT_ASSERT(
        0 <= o && o < conv.info.outputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(conv.data, o + conv.info.height * conv.info.width * conv.info.inputs * conv.info.outputs);
}

ML2_FN ML2_LayerFilters ML2_CPU_ConvFilters(ML2_LayerConv conv) {
    ML2_HARD_ASSERT(conv.data.device == ML2_DeviceCPU);
    return (ML2_LayerFilters){ML2_ConvInfoFilters(conv.info), {conv.data.device, .as.cpu = {ML2_CPU_ConvFiltersAt(conv, 0, 0, 0, 0)}}};
}

ML2_FN ML2_LayerBiases ML2_CPU_ConvBiases(ML2_LayerConv conv) {
    ML2_HARD_ASSERT(conv.data.device == ML2_DeviceCPU);
    return (ML2_LayerBiases){ML2_ConvInfoBiases(conv.info), {conv.data.device, .as.cpu = {ML2_CPU_ConvBiasesAt(conv, 0)}}};
}

ML2_FN void ML2_CPU_ConvPrint(ML2_LayerConv conv, ML2_Int indent) {
    ML2_HARD_ASSERT(conv.data.device == ML2_DeviceCPU);
    ML2_ConvInfoPrint(conv.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        ML2_CPU_FiltersPrint(ML2_CPU_ConvFilters(conv), indent);
        ML2_CPU_BiasesPrint(ML2_CPU_ConvBiases(conv), indent);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_ConvClear(ML2_LayerConv conv) {
    ML2_CPU_FiltersClear(ML2_CPU_ConvFilters(conv));
    ML2_CPU_BiasesClear(ML2_CPU_ConvBiases(conv));
}

ML2_FN void ML2_CPU_ConvCopy(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(dest.info, src.info));
    ML2_CPU_FiltersCopy(ML2_CPU_ConvFilters(dest), ML2_CPU_ConvFilters(src));
    ML2_CPU_BiasesCopy(ML2_CPU_ConvBiases(dest), ML2_CPU_ConvBiases(src));
}

ML2_FN void ML2_CPU_ConvSum(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(dest.info, src.info));
    ML2_CPU_FiltersSum(ML2_CPU_ConvFilters(dest), ML2_CPU_ConvFilters(src));
    ML2_CPU_BiasesSum(ML2_CPU_ConvBiases(dest), ML2_CPU_ConvBiases(src));
}

ML2_FN void ML2_CPU_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high) {
    ML2_CPU_FiltersRand(ML2_CPU_ConvFilters(conv), low, high);
    ML2_CPU_BiasesRand(ML2_CPU_ConvBiases(conv), low, high);
}

ML2_FN void ML2_CPU_ConvXavierInit(ML2_LayerConv conv) {
    ML2_CPU_FiltersXavierInit(ML2_CPU_ConvFilters(conv));
    ML2_CPU_BiasesXavierInit(ML2_CPU_ConvBiases(conv));
}

ML2_FN void ML2_CPU_ConvHeInit(ML2_LayerConv conv) {
    ML2_CPU_FiltersHeInit(ML2_CPU_ConvFilters(conv));
    ML2_CPU_BiasesHeInit(ML2_CPU_ConvBiases(conv));
}

ML2_FN void ML2_CPU_ConvForwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_ConvForwardCompatibleMatrices(conv.info, input.info, output.info));
    ML2_HARD_ASSERT(
        conv.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_CPU_FiltersForwardMatrices(ML2_CPU_ConvFilters(conv), input, output);
    // TODO(3/9/2026 18:51:27): why doesnt this call ML2_BiasesForward(), i remember there was a reason but didnt i fix it? i dont remember, should be investigated
    // (26/9/2026 14:20:25): because it needs to add biases inplace, the solution is inplace versions of functions which can be interesting in the future
    ML2_LayerBiases biases = ML2_CPU_ConvBiases(conv);
    ML2_Int samples = output.info.samples;
    ML2_Int matrices = output.info.matrices;
    ML2_Int vectors = output.info.vectors;
    ML2_Int scalars = output.info.scalars;

    ML2_Scalar *restrict biasesPtr = ML2_CPU_BiasesAt(biases, 0);
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    outputPtr[c] += biasesPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_ConvBackwardMatrices(ML2_LayerConv convGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvBackwardCompatibleMatrices(convGradient.info, input.info, outputGradient.info));
    ML2_CPU_FiltersBackwardMatrices(ML2_CPU_ConvFilters(convGradient), input, outputGradient);
    ML2_CPU_BiasesBackwardMatrices(ML2_CPU_ConvBiases(convGradient), outputGradient);
}

ML2_FN void ML2_CPU_ConvCacheBackwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvCacheBackwardCompatibleMatrices(conv.info, inputGradient.info, outputGradient.info));
    ML2_CPU_FiltersCacheBackwardMatrices(ML2_CPU_ConvFilters(conv), inputGradient, outputGradient);
}

ML2_FN void ML2_CPU_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_CPU_FiltersGradientDescentOptimize(ML2_CPU_ConvFilters(conv), ML2_CPU_ConvFilters(convGradient), learningRate);
    ML2_CPU_BiasesGradientDescentOptimize(ML2_CPU_ConvBiases(conv), ML2_CPU_ConvBiases(convGradient), learningRate);
}

ML2_FN void ML2_CPU_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, average.info));
    ML2_CPU_FiltersMomentumOptimize(ML2_CPU_ConvFilters(conv), ML2_CPU_ConvFilters(convGradient), learningRate, decayRate, ML2_CPU_ConvFilters(average));
    ML2_CPU_BiasesMomentumOptimize(ML2_CPU_ConvBiases(conv), ML2_CPU_ConvBiases(convGradient), learningRate, decayRate, ML2_CPU_ConvBiases(average));
}

ML2_FN void ML2_CPU_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, squareSum.info));
    ML2_CPU_FiltersAdagradOptimize(ML2_CPU_ConvFilters(conv), ML2_CPU_ConvFilters(convGradient), learningRate, ML2_CPU_ConvFilters(squareSum));
    ML2_CPU_BiasesAdagradOptimize(ML2_CPU_ConvBiases(conv), ML2_CPU_ConvBiases(convGradient), learningRate, ML2_CPU_ConvBiases(squareSum));
}

ML2_FN void ML2_CPU_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, squareAverage.info));
    ML2_CPU_FiltersRMSPropOptimize(ML2_CPU_ConvFilters(conv), ML2_CPU_ConvFilters(convGradient), learningRate, decayRate, ML2_CPU_ConvFilters(squareAverage));
    ML2_CPU_BiasesRMSPropOptimize(ML2_CPU_ConvBiases(conv), ML2_CPU_ConvBiases(convGradient), learningRate, decayRate, ML2_CPU_ConvBiases(squareAverage));
}

ML2_FN void ML2_CPU_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, average.info));
    ML2_CPU_FiltersAdamOptimize(ML2_CPU_ConvFilters(conv), ML2_CPU_ConvFilters(convGradient), learningRate, decayRate1, decayRate2, ML2_CPU_ConvFilters(average), ML2_CPU_ConvFilters(squareAverage), decayingWeight1, decayingWeight2);
    ML2_CPU_BiasesAdamOptimize(ML2_CPU_ConvBiases(conv), ML2_CPU_ConvBiases(convGradient), learningRate, decayRate1, decayRate2, ML2_CPU_ConvBiases(average), ML2_CPU_ConvBiases(squareAverage), decayingWeight1, decayingWeight2);
}

ML2_FN void ML2_CPU_FlattenForwardMatrices(ML2_LayerCacheMatrices input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_FlattenForwardCompatibleMatrices(input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int matrices = input.info.matrices;
    ML2_Int vectors = input.info.vectors;
    ML2_Int scalars = input.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict outputPtr = ML2_CPU_ScalarsAt(output, s, 0);
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputPtr = ML2_CPU_MatricesAt(input, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    outputPtr[(m * vectors + v) * scalars + c] += inputPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FlattenCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_FlattenCacheBackwardCompatibleMatrices(inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int matrices = inputGradient.info.matrices;
    ML2_Int vectors = inputGradient.info.vectors;
    ML2_Int scalars = inputGradient.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict outputGradientPtr = ML2_CPU_ScalarsAt(outputGradient, s, 0);
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict inputGradientPtr = ML2_CPU_MatricesAt(inputGradient, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    inputGradientPtr[c] += outputGradientPtr[(m * vectors + v) * scalars + c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_PoolMaxForwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_PoolMaxForwardCompatibleMatrices(pool, input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = output.info.matrices;
    ML2_Int outputWidth = output.info.vectors;
    ML2_Int poolHeight = pool.height;
    ML2_Int poolWidth = pool.width;
    ML2_Int paddingHeight = pool.paddingHeight;
    ML2_Int paddingWidth = pool.paddingWidth;
    ML2_Int strideHeight = pool.strideHeight;
    ML2_Int strideWidth = pool.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, yo, xo, 0);
                ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                // TODO(25/4/2026 1:32:54am): another implementation is to make an array of size channels
                //                            but i dont know how slow VLAs are, needs to be tested
                /*
                ML2_Scalar max[channels] = {ML2_NegInf, ...};
                for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                    for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            max[c] = ML2_FMAX(max[c], *ML2_CPU_MatricesAt(input, s, yi, xi, c));
                        }
                    }
                }
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_CPU_MatricesAt(output, s, yo, xo, c) += max[c];
                }
                */
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar max = ML2_NegInf;
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            max = ML2_FMAX(max, *ML2_CPU_MatricesAt(input, s, yi, xi, c));
                        }
                    }
                    outputPtr[c] += max;
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_PoolMaxCacheBackwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolMaxCacheBackwardCompatibleMatrices(pool, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = outputGradient.info.matrices;
    ML2_Int outputWidth = outputGradient.info.vectors;
    ML2_Int poolHeight = pool.height;
    ML2_Int poolWidth = pool.width;
    ML2_Int paddingHeight = pool.paddingHeight;
    ML2_Int paddingWidth = pool.paddingWidth;
    ML2_Int strideHeight = pool.strideHeight;
    ML2_Int strideWidth = pool.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, yo, xo, 0);
                ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                // TODO(25/4/2026 1:32:54am): another implementation is to make an array of size channels
                //                            but i dont know how slow VLAs are, needs to be tested
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar max = ML2_NegInf;
                    ML2_Int yIdx = -1;
                    ML2_Int xIdx = -1;
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            if (*ML2_CPU_MatricesAt(input, s, yi, xi, c) > max) {
                                max = *ML2_CPU_MatricesAt(input, s, yi, xi, c);
                                yIdx = yi;
                                xIdx = xi;
                            }
                        }
                    }
                    *ML2_CPU_MatricesAt(inputGradient, s, yIdx, xIdx, c) += outputGradientPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_PoolAverageForwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_PoolAverageForwardCompatibleMatrices(pool, input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = output.info.matrices;
    ML2_Int outputWidth = output.info.vectors;
    ML2_Int poolHeight = pool.height;
    ML2_Int poolWidth = pool.width;
    ML2_Int paddingHeight = pool.paddingHeight;
    ML2_Int paddingWidth = pool.paddingWidth;
    ML2_Int strideHeight = pool.strideHeight;
    ML2_Int strideWidth = pool.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, yo, xo, 0);
                ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                ML2_Scalar poolArea = (yiMax - yiMin + 1) * (xiMax - xiMin + 1);
                // TODO(25/4/2026 1:32:54am): another implementation is to just divide by the poolArea in the loop
                //                            but i dont know how slow that would be, needs to be tested
                // (27/9/2026 4:27:42): what the hell does that mean???, oh wait i get it, add in the c loop and divide there and bring the yi xi loops out
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar sum = {};
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            sum += *ML2_CPU_MatricesAt(input, s, yi, xi, c);
                        }
                    }
                    outputPtr[c] += sum / poolArea;
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_PoolAverageCacheBackwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolAverageCacheBackwardCompatibleMatrices(pool, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = inputGradient.info.samples;
    ML2_Int scalars = inputGradient.info.scalars;

    ML2_Int inputHeight = inputGradient.info.matrices;
    ML2_Int inputWidth = inputGradient.info.vectors;
    ML2_Int outputHeight = outputGradient.info.matrices;
    ML2_Int outputWidth = outputGradient.info.vectors;
    ML2_Int poolHeight = pool.height;
    ML2_Int poolWidth = pool.width;
    ML2_Int paddingHeight = pool.paddingHeight;
    ML2_Int paddingWidth = pool.paddingWidth;
    ML2_Int strideHeight = pool.strideHeight;
    ML2_Int strideWidth = pool.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, yo, xo, 0);
                ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                ML2_Scalar poolArea = (yiMax - yiMin + 1) * (xiMax - xiMin + 1);
                for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                    for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                        ML2_Scalar *restrict inputGradientPtr = ML2_CPU_MatricesAt(inputGradient, s, yi, xi, 0);
                        for (ML2_Int c = 0; c < scalars; c++) {
                            inputGradientPtr[c] += outputGradientPtr[c] / poolArea;
                        }
                    }
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_PoolSoftmaxForwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxForwardCompatibleMatrices(pool, input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = output.info.matrices;
    ML2_Int outputWidth = output.info.vectors;
    ML2_Int poolHeight = pool.height;
    ML2_Int poolWidth = pool.width;
    ML2_Int paddingHeight = pool.paddingHeight;
    ML2_Int paddingWidth = pool.paddingWidth;
    ML2_Int strideHeight = pool.strideHeight;
    ML2_Int strideWidth = pool.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, yo, xo, 0);
                ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                // TODO(25/4/2026 1:32:54am): another implementation is to make an array for each thing
                //                            but that feels like a lot of memory, needs to be tested
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar max = ML2_NegInf;
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            max = ML2_FMAX(max, *ML2_CPU_MatricesAt(input, s, yi, xi, c));
                        }
                    }
                    ML2_Scalar sum = {};
                    ML2_Scalar weightedSum = {};
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            ML2_Scalar inputScalar = *ML2_CPU_MatricesAt(input, s, yi, xi, c);
                            ML2_Scalar e = ML2_EXP(inputScalar - max);
                            sum += e;
                            weightedSum += e * inputScalar;
                        }
                    }
                    outputPtr[c] += weightedSum / sum;
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_PoolSoftmaxCacheBackwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxCacheBackwardCompatibleMatrices(pool, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = input.info.samples;
    ML2_Int scalars = input.info.scalars;

    ML2_Int inputHeight = input.info.matrices;
    ML2_Int inputWidth = input.info.vectors;
    ML2_Int outputHeight = output.info.matrices;
    ML2_Int outputWidth = output.info.vectors;
    ML2_Int poolHeight = pool.height;
    ML2_Int poolWidth = pool.width;
    ML2_Int paddingHeight = pool.paddingHeight;
    ML2_Int paddingWidth = pool.paddingWidth;
    ML2_Int strideHeight = pool.strideHeight;
    ML2_Int strideWidth = pool.strideWidth;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int yo = 0; yo < outputHeight; yo++) {
            for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, yo, xo, 0);
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, yo, xo, 0);
                ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                // TODO(25/4/2026 1:32:54am): another implementation is to make an array for each thing
                //                            but that feels like a lot of memory, needs to be tested
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar max = ML2_NegInf;
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            max = ML2_FMAX(max, *ML2_CPU_MatricesAt(input, s, yi, xi, c));
                        }
                    }
                    ML2_Scalar sum = {};
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            ML2_Scalar e = ML2_EXP(*ML2_CPU_MatricesAt(input, s, yi, xi, c) - max);
                            sum += e;
                        }
                    }
                    for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                        for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                            ML2_Scalar inputScalar = *ML2_CPU_MatricesAt(input, s, yi, xi, c);
                            ML2_Scalar outputScalar = outputPtr[c];
                            ML2_Scalar e = ML2_EXP(inputScalar - max);
                            *ML2_CPU_MatricesAt(inputGradient, s, yi, xi, c) += e / sum * (inputScalar + ML2_SCALAR_LITERAL(1.0) - outputScalar) * outputGradientPtr[c];
                        }
                    }
                }
            }
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i) {
    ML2_SOFT_ASSERT(
        0 <= w && w < fusedWeights.info.weights &&
        0 <= o && o < fusedWeights.info.outputs &&
        0 <= i && i < fusedWeights.info.inputs &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(fusedWeights.data, (w * fusedWeights.info.outputs + o) * fusedWeights.info.inputs + i);
}

ML2_FN ML2_LayerWeights ML2_CPU_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w) {
    // NOTE(1/10/2026 00:21:43): ML2_HARD_ASSERT is used here because this function isnt used in performant code, just like ML2_LinearWeights
    ML2_HARD_ASSERT(
        0 <= w && w < fusedWeights.info.weights &&
        "OUT OF BOUNDS INDICES"
    );
    ML2_HARD_ASSERT(fusedWeights.data.device == ML2_DeviceCPU);
    return (ML2_LayerWeights){ML2_FusedWeightsInfoWeights(fusedWeights.info), {fusedWeights.data.device, .as.cpu = {ML2_CPU_FusedWeightsAt(fusedWeights, w, 0, 0)}}};
}

ML2_FN void ML2_CPU_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent) {
    ML2_HARD_ASSERT(fusedWeights.data.device == ML2_DeviceCPU);
    ML2_FusedWeightsInfoPrint(fusedWeights.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
            printf(ML2_INDENT("FusedWeights[%d]:\n", indent), w);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                for (ML2_Int o = 0; o < fusedWeights.info.outputs; o++) {
                    printf(ML2_INDENT("", indent));
                    for (ML2_Int i = 0; i < fusedWeights.info.inputs; i++) {
                        printf(ML2_SCALAR_FMT " ", *ML2_CPU_FusedWeightsAt(fusedWeights, w, o, i));
                    }
                    printf("\n");
                }
                indent -= ML2_Indentation;
            }
            printf(ML2_INDENT("}\n", indent));
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights) {
    ML2_HARD_ASSERT(fusedWeights.data.device == ML2_DeviceCPU);
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_CPU_WeightsClear(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsCopy(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int weights = dest.info.weights;

    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsCopy(ML2_CPU_FusedWeightsWeightsAt(dest, w), ML2_CPU_FusedWeightsWeightsAt(src, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int weights = dest.info.weights;

    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsSum(ML2_CPU_FusedWeightsWeightsAt(dest, w), ML2_CPU_FusedWeightsWeightsAt(src, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(fusedWeights.data.device == ML2_DeviceCPU);
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_CPU_WeightsRand(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w), low, high);
    }
}

// TODO(3/9/2026 17:22:04): this doesnt take into account the size of the entire structure, not sure if this is correct
ML2_FN void ML2_CPU_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights) {
    ML2_HARD_ASSERT(fusedWeights.data.device == ML2_DeviceCPU);
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_CPU_WeightsXavierInit(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w));
    }
}

// TODO(3/9/2026 17:22:04): this doesnt take into account the size of the entire structure, not sure if this is correct
ML2_FN void ML2_CPU_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights) {
    ML2_HARD_ASSERT(fusedWeights.data.device == ML2_DeviceCPU);
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_CPU_WeightsHeInit(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsForwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_FusedWeightsForwardCompatibleVectors(fusedWeights.info, input.info, output.info));
    ML2_HARD_ASSERT(
        fusedWeights.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = output.info.samples;
    ML2_Int matrices = output.info.matrices;
    ML2_Int vectors = output.info.vectors;
    ML2_Int outputs = fusedWeights.info.outputs;
    ML2_Int inputs = fusedWeights.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            ML2_Scalar *restrict inputVectorsPtr = ML2_CPU_VectorsAt(input, s, m, 0);
            for (ML2_Int v = 0; v < vectors; v++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    ML2_Scalar *restrict fusedWeightsPtr = ML2_CPU_FusedWeightsAt(fusedWeights, v, o, 0);
                    ML2_Scalar *restrict outputPtr = ML2_CPU_MatricesAt(output, s, m, v, 0);
                    ML2_Scalar acc = {};
                    for (ML2_Int i = 0; i < inputs; i++) {
                        acc += inputVectorsPtr[i] * fusedWeightsPtr[i];
                    }
                    outputPtr[o] += acc;
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FusedWeightsBackwardVectors(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsBackwardCompatibleVectors(fusedWeightsGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        fusedWeightsGradient.data.device == ML2_DeviceCPU &&
        input.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = outputGradient.info.samples;
    ML2_Int matrices = outputGradient.info.matrices;
    ML2_Int vectors = outputGradient.info.vectors;
    ML2_Int outputs = fusedWeightsGradient.info.outputs;
    ML2_Int inputs = fusedWeightsGradient.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            ML2_Scalar *restrict inputVectorsPtr = ML2_CPU_VectorsAt(input, s, m, 0);
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, m, v, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    ML2_Scalar *restrict fusedWeightsGradientPtr = ML2_CPU_FusedWeightsAt(fusedWeightsGradient, v, o, 0);
                    for (ML2_Int i = 0; i < inputs; i++) {
                        fusedWeightsGradientPtr[i] += inputVectorsPtr[i] * outputGradientPtr[o];
                    }
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FusedWeightsCacheBackwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsCacheBackwardCompatibleVectors(fusedWeights.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        fusedWeights.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = outputGradient.info.samples;
    ML2_Int matrices = outputGradient.info.matrices;
    ML2_Int vectors = outputGradient.info.vectors;
    ML2_Int outputs = fusedWeights.info.outputs;
    ML2_Int inputs = fusedWeights.info.inputs;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            ML2_Scalar *restrict inputGradientVectorsPtr = ML2_CPU_VectorsAt(inputGradient, s, m, 0);
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict outputGradientPtr = ML2_CPU_MatricesAt(outputGradient, s, m, v, 0);
                for (ML2_Int o = 0; o < outputs; o++) {
                    ML2_Scalar *restrict fusedWeightsPtr = ML2_CPU_FusedWeightsAt(fusedWeights, v, o, 0);
                    for (ML2_Int i = 0; i < inputs; i++) {
                        inputGradientVectorsPtr[i] += fusedWeightsPtr[i] * outputGradientPtr[o];
                    }
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsGradientDescentOptimize(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w), ML2_CPU_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate);
    }
}

ML2_FN void ML2_CPU_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, average.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsMomentumOptimize(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w), ML2_CPU_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, decayRate, ML2_CPU_FusedWeightsWeightsAt(average, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, squareSum.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsAdagradOptimize(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w), ML2_CPU_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, ML2_CPU_FusedWeightsWeightsAt(squareSum, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, squareAverage.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsRMSPropOptimize(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w), ML2_CPU_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, decayRate, ML2_CPU_FusedWeightsWeightsAt(squareAverage, w));
    }
}

ML2_FN void ML2_CPU_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, average.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, squareAverage.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_CPU_WeightsAdamOptimize(ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w), ML2_CPU_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, decayRate1, decayRate2, ML2_CPU_FusedWeightsWeightsAt(average, w), ML2_CPU_FusedWeightsWeightsAt(squareAverage, w), decayingWeight1, decayingWeight2);
    }
}

ML2_FN void ML2_CPU_AttentionForwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_AttentionForwardCompatibleMatrices(attention, input.info, output.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        output.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = output.info.samples;
    ML2_Int vectors = output.info.vectors;
    ML2_Int scalars = output.info.scalars;
    ML2_Int heads = attention.heads;
    ML2_Int headScalars = scalars / heads;
    const bool masked = attention.masked;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int qv = 0; qv < vectors; qv++) {
            ML2_Scalar *restrict outputPtr = ML2_CPU_VectorsAt(output, s, qv, 0);
            ML2_Scalar *restrict inputQueryPtr = ML2_CPU_MatricesAt(input, s, qv, ML2_AttentionInputIndexQuery, 0);
            // TODO(3/9/2026/ 21:29:05): find a better name, then find out if this impacts performance, the branch predictor should do fine because its always true/false, but maybe this impacts vectorization
            ML2_Int iterations = (masked) ? (qv + 1) : (vectors);

            for (ML2_Int h = 0; h < heads; h++) {
                // TODO(2/9/2026 00:59:43): this is a VLA, people say VLA bad, but its between this or heap memory
                ML2_Scalar outputVector[headScalars] = {};
                ML2_Scalar max = ML2_NegInf;
                ML2_Scalar sum = {};
                for (ML2_Int v = 0; v < iterations; v++) {
                    ML2_Scalar *restrict inputKeyPtr = ML2_CPU_MatricesAt(input, s, v, ML2_AttentionInputIndexKey, 0);
                    ML2_Scalar *restrict inputValuePtr = ML2_CPU_MatricesAt(input, s, v, ML2_AttentionInputIndexValue, 0);

                    ML2_Scalar scoreScalar = {};
                    for (ML2_Int hc = 0; hc < headScalars; hc++) {
                        ML2_Int c = h * headScalars + hc;
                        scoreScalar += inputQueryPtr[c] * inputKeyPtr[c];
                    }
                    scoreScalar /= ML2_SQRT(headScalars);
                    ML2_Scalar newMax = ML2_FMAX(max, scoreScalar);
                    ML2_Scalar oldCorrection = ML2_EXP(max - newMax);
                    ML2_Scalar newCorrection = ML2_EXP(scoreScalar - newMax);
                    for (ML2_Int hc = 0; hc < headScalars; hc++) {
                        ML2_Int c = h * headScalars + hc;
                        outputVector[hc] = outputVector[hc] * oldCorrection + inputValuePtr[c] * newCorrection;
                    }
                    sum = sum * oldCorrection + newCorrection;
                    max = newMax;
                }
                for (ML2_Int hc = 0; hc < headScalars; hc++) {
                    ML2_Int c = h * headScalars + hc;
                    outputPtr[c] += outputVector[hc] / sum;
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_AttentionCacheBackwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_AttentionCacheBackwardCompatibleMatrices(attention, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == ML2_DeviceCPU &&
        inputGradient.data.device == ML2_DeviceCPU &&
        outputGradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = outputGradient.info.samples;
    ML2_Int vectors = outputGradient.info.vectors;
    ML2_Int scalars = outputGradient.info.scalars;
    ML2_Int heads = attention.heads;
    ML2_Int headScalars = scalars / heads;
    const bool masked = attention.masked;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int qv = 0; qv < vectors; qv++) {
            ML2_Scalar *restrict outputGradientPtr = ML2_CPU_VectorsAt(outputGradient, s, qv, 0);
            ML2_Scalar *restrict inputQueryPtr = ML2_CPU_MatricesAt(input, s, qv, ML2_AttentionInputIndexQuery, 0);
            ML2_Scalar *restrict inputGradientQueryPtr = ML2_CPU_MatricesAt(inputGradient, s, qv, ML2_AttentionInputIndexQuery, 0);

            ML2_Int iterations = (masked) ? (qv + 1) : (vectors);

            for (ML2_Int h = 0; h < heads; h++) {
                ML2_Scalar scoreVector[iterations] = {};
                ML2_Scalar max = ML2_NegInf;
                ML2_Scalar sum = {};

                for (ML2_Int v = 0; v < iterations; v++) {
                    ML2_Scalar *restrict inputKeyPtr = ML2_CPU_MatricesAt(input, s, v, ML2_AttentionInputIndexKey, 0);

                    ML2_Scalar scoreScalar = {};
                    for (ML2_Int hc = 0; hc < headScalars; hc++) {
                        ML2_Int c = h * headScalars + hc;
                        scoreScalar += inputQueryPtr[c] * inputKeyPtr[c];
                    }
                    scoreScalar /= ML2_SQRT(headScalars);
                    scoreVector[v] = scoreScalar;

                    ML2_Scalar newMax = ML2_FMAX(max, scoreScalar);
                    ML2_Scalar oldCorrection = ML2_EXP(max - newMax);
                    ML2_Scalar newCorrection = ML2_EXP(scoreScalar - newMax);
                    sum = sum * oldCorrection + newCorrection;
                    max = newMax;
                }

                ML2_Scalar softmaxVector[iterations] = {};
                ML2_Scalar valueGradientVector[iterations] = {};
                ML2_Scalar correctionSum = {};

                for (ML2_Int v = 0; v < iterations; v++) {
                    ML2_Scalar *restrict inputValuePtr = ML2_CPU_MatricesAt(input, s, v, ML2_AttentionInputIndexValue, 0);
                    ML2_Scalar *restrict inputGradientValuePtr = ML2_CPU_MatricesAt(inputGradient, s, v, ML2_AttentionInputIndexValue, 0);

                    ML2_Scalar softmaxScore = ML2_EXP(scoreVector[v] - max) / sum;
                    softmaxVector[v] = softmaxScore;

                    ML2_Scalar valueGradientScalar = {};
                    for (ML2_Int hc = 0; hc < headScalars; hc++) {
                        ML2_Int c = h * headScalars + hc;
                        valueGradientScalar += outputGradientPtr[c] * inputValuePtr[c];
                        inputGradientValuePtr[c] += softmaxScore * outputGradientPtr[c];
                    }
                    valueGradientVector[v] = valueGradientScalar;
                    correctionSum += softmaxScore * valueGradientScalar;
                }

                ML2_Scalar softmaxGradientVector[headScalars] = {};

                for (ML2_Int v = 0; v < iterations; v++) {
                    ML2_Scalar *restrict inputKeyPtr = ML2_CPU_MatricesAt(input, s, v, ML2_AttentionInputIndexKey, 0);
                    ML2_Scalar *restrict inputGradientKeyPtr = ML2_CPU_MatricesAt(inputGradient, s, v, ML2_AttentionInputIndexKey, 0);

                    ML2_Scalar scoreGradientScalar = softmaxVector[v] * (valueGradientVector[v] - correctionSum) / ML2_SQRT(headScalars);
                    for (ML2_Int hc = 0; hc < headScalars; hc++) {
                        ML2_Int c = h * headScalars + hc;
                        softmaxGradientVector[hc] += scoreGradientScalar * inputKeyPtr[c];
                        inputGradientKeyPtr[c] += scoreGradientScalar * inputQueryPtr[c];
                    }
                }

                for (ML2_Int hc = 0; hc < headScalars; hc++) {
                    ML2_Int c = h * headScalars + hc;
                    inputGradientQueryPtr[c] += softmaxGradientVector[hc];
                }
            }
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int s, ML2_Int c) {
    ML2_SOFT_ASSERT(
        0 <= s && s < scalars.info.samples &&
        0 <= c && c < scalars.info.scalars &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(scalars.data, s * scalars.info.scalars + c);
}

ML2_FN void ML2_CPU_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent) {
    ML2_HARD_ASSERT(scalars.data.device == ML2_DeviceCPU);
    ML2_ScalarsInfoPrint(scalars.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int s = 0; s < scalars.info.samples; s++) {
            printf(ML2_INDENT("", indent));
            for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
                printf(ML2_SCALAR_FMT " ", *ML2_CPU_ScalarsAt(scalars, s, c));
            }
            printf("\n");
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_ScalarsClear(ML2_LayerCacheScalars scalars) {
    ML2_HARD_ASSERT(scalars.data.device == ML2_DeviceCPU);
    for (ML2_Int s = 0; s < scalars.info.samples; s++) {
        ML2_Scalar *restrict scalarsPtr = ML2_CPU_ScalarsAt(scalars, s, 0);
        for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
            scalarsPtr[c] = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_FN void ML2_CPU_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = dest.info.samples;
    ML2_Int scalars = dest.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict destPtr = ML2_CPU_ScalarsAt(dest, s, 0);
        ML2_Scalar *restrict srcPtr = ML2_CPU_ScalarsAt(src, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            destPtr[c] = srcPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = dest.info.samples;
    ML2_Int scalars = dest.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict destPtr = ML2_CPU_ScalarsAt(dest, s, 0);
        ML2_Scalar *restrict srcPtr = ML2_CPU_ScalarsAt(src, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            destPtr[c] += srcPtr[c];
        }
    }
}

ML2_FN void ML2_CPU_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(scalars.data.device == ML2_DeviceCPU);
    for (ML2_Int s = 0; s < scalars.info.samples; s++) {
        ML2_Scalar *restrict scalarsPtr = ML2_CPU_ScalarsAt(scalars, s, 0);
        for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
            scalarsPtr[c] = ML2_ScalarRand(low, high);
        }
    }
}

ML2_FN void ML2_CPU_ScalarsXavierInit(ML2_LayerCacheScalars scalars) {
    ML2_HARD_ASSERT(scalars.data.device == ML2_DeviceCPU);
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (scalars.info.samples + scalars.info.scalars));
    ML2_CPU_ScalarsRand(scalars, -limit, limit);
}

ML2_FN void ML2_CPU_ScalarsHeInit(ML2_LayerCacheScalars scalars) {
    ML2_HARD_ASSERT(scalars.data.device == ML2_DeviceCPU);
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / scalars.info.scalars);
    for (ML2_Int s = 0; s < scalars.info.samples; s++) {
        ML2_Scalar *restrict scalarsPtr = ML2_CPU_ScalarsAt(scalars, s, 0);
        for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
            scalarsPtr[c] = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expected.info, predicted.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar diff = predictedPtr[c] - expectedPtr[c];
            loss += diff * diff;
        }
    }
    // TODO(8/4/2026 10:15:17pm):
    //      all other losses divide only by samples, this causes a inconsistency in code,
    //      and the numbers users see for the loss changes significantly, this will also
    //      cause the learning rate to have to be smaller,
    //      all losses should use the same method, i dont know which though
    loss /= (samples * scalars);
    return loss;
}

ML2_FN void ML2_CPU_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        ML2_Scalar *restrict gradientPtr = ML2_CPU_ScalarsAt(gradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar diff = predictedPtr[c] - expectedPtr[c];
            gradientPtr[c] += ML2_SCALAR_LITERAL(2.0) * diff / (samples * scalars);
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMAX(predictedPtr[c], ML2_Epsilon);
            ML2_Scalar expectedScalar = expectedPtr[c];
            loss += -expectedScalar * ML2_LN(predictedScalar);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_CPU_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        ML2_Scalar *restrict gradientPtr = ML2_CPU_ScalarsAt(gradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMAX(predictedPtr[c], ML2_Epsilon);
            ML2_Scalar expectedScalar = expectedPtr[c];
            gradientPtr[c] += -(expectedScalar / predictedScalar) / samples;
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        ML2_Scalar max = ML2_NegInf;
        for (ML2_Int c = 0; c < scalars; c++) {
            max = ML2_FMAX(max, predictedPtr[c]);
        }
        ML2_Scalar sum = {};
        for (ML2_Int c = 0; c < scalars; c++) {
            sum += ML2_EXP(predictedPtr[c] - max);
        }
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMAX(ML2_EXP(predictedPtr[c] - max) / sum, ML2_Epsilon);
            ML2_Scalar expectedScalar = expectedPtr[c];
            loss += -expectedScalar * ML2_LN(predictedScalar);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_CPU_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        ML2_Scalar *restrict gradientPtr = ML2_CPU_ScalarsAt(gradient, s, 0);
        ML2_Scalar max = ML2_NegInf;
        for (ML2_Int c = 0; c < scalars; c++) {
            max = ML2_FMAX(max, predictedPtr[c]);
        }
        ML2_Scalar sum = {};
        for (ML2_Int c = 0; c < scalars; c++) {
            sum += ML2_EXP(predictedPtr[c] - max);
        }
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_EXP(predictedPtr[c] - max) / sum;
            ML2_Scalar expectedScalar = expectedPtr[c];
            gradientPtr[c] += (predictedScalar - expectedScalar) / samples;
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(predictedPtr[c], ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
            ML2_Scalar expectedScalar = expectedPtr[c];
            loss += -expectedScalar * ML2_LN(predictedScalar) - (ML2_SCALAR_LITERAL(1.0) - expectedScalar) * ML2_LN(ML2_SCALAR_LITERAL(1.0) - predictedScalar);
        }
    }
    loss /= (samples * scalars);
    return loss;
}

ML2_FN void ML2_CPU_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar *restrict predictedPtr = ML2_CPU_ScalarsAt(predicted, s, 0);
        ML2_Scalar *restrict expectedPtr = ML2_CPU_ScalarsAt(expected, s, 0);
        ML2_Scalar *restrict gradientPtr = ML2_CPU_ScalarsAt(gradient, s, 0);
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(predictedPtr[c], ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
            ML2_Scalar expectedScalar = expectedPtr[c];
            gradientPtr[c] += (predictedScalar - expectedScalar) / (predictedScalar * (ML2_SCALAR_LITERAL(1.0) - predictedScalar)) / (samples * scalars);
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int s, ML2_Int v, ML2_Int c) {
    ML2_SOFT_ASSERT(
        0 <= s && s < vectors.info.samples &&
        0 <= v && v < vectors.info.vectors &&
        0 <= c && c < vectors.info.scalars &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(vectors.data, (s * vectors.info.vectors + v) * vectors.info.scalars + c);
}

ML2_FN void ML2_CPU_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent) {
    ML2_HARD_ASSERT(vectors.data.device == ML2_DeviceCPU);
    ML2_VectorsInfoPrint(vectors.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int s = 0; s < vectors.info.samples; s++) {
            printf(ML2_INDENT("Samples[%d]:\n", indent), s);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
                    printf(ML2_INDENT("", indent));
                    for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                        printf(ML2_SCALAR_FMT " ", *ML2_CPU_VectorsAt(vectors, s, v, c));
                    }
                    printf("\n");
                }
                indent -= ML2_Indentation;
            }
            printf(ML2_INDENT("}\n", indent));
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_VectorsClear(ML2_LayerCacheVectors vectors) {
    ML2_HARD_ASSERT(vectors.data.device == ML2_DeviceCPU);
    for (ML2_Int s = 0; s < vectors.info.samples; s++) {
        for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
            ML2_Scalar *restrict vectorsPtr = ML2_CPU_VectorsAt(vectors, s, v, 0);
            for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                vectorsPtr[c] = ML2_SCALAR_LITERAL(0.0);
            }
        }
    }
}

ML2_FN void ML2_CPU_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = dest.info.samples;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict destPtr = ML2_CPU_VectorsAt(dest, s, v, 0);
            ML2_Scalar *restrict srcPtr = ML2_CPU_VectorsAt(src, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                destPtr[c] = srcPtr[c];
            }
        }
    }
}

ML2_FN void ML2_CPU_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = dest.info.samples;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict destPtr = ML2_CPU_VectorsAt(dest, s, v, 0);
            ML2_Scalar *restrict srcPtr = ML2_CPU_VectorsAt(src, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                destPtr[c] += srcPtr[c];
            }
        }
    }
}

ML2_FN void ML2_CPU_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(vectors.data.device == ML2_DeviceCPU);
    for (ML2_Int s = 0; s < vectors.info.samples; s++) {
        for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
            ML2_Scalar *restrict vectorsPtr = ML2_CPU_VectorsAt(vectors, s, v, 0);
            for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                vectorsPtr[c] = ML2_ScalarRand(low, high);
            }
        }
    }
}

ML2_FN void ML2_CPU_VectorsXavierInit(ML2_LayerCacheVectors vectors) {
    ML2_HARD_ASSERT(vectors.data.device == ML2_DeviceCPU);
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (vectors.info.samples + vectors.info.vectors + vectors.info.scalars));
    ML2_CPU_VectorsRand(vectors, -limit, limit);
}

ML2_FN void ML2_CPU_VectorsHeInit(ML2_LayerCacheVectors vectors) {
    ML2_HARD_ASSERT(vectors.data.device == ML2_DeviceCPU);
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (vectors.info.vectors * vectors.info.scalars));
    for (ML2_Int s = 0; s < vectors.info.samples; s++) {
        for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
            ML2_Scalar *restrict vectorsPtr = ML2_CPU_VectorsAt(vectors, s, v, 0);
            for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                vectorsPtr[c] = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar diff = predictedPtr[c] - expectedPtr[c];
                loss += diff * diff;
            }
        }
    }
    // TODO(8/4/2026 10:15:17pm):
    //      all other losses divide only by samples, this causes a inconsistency in code,
    //      and the numbers users see for the loss changes significantly, this will also
    //      cause the learning rate to have to be smaller,
    //      all losses should use the same method, i dont know which though
    loss /= (samples * vectors * scalars);
    return loss;
}

ML2_FN void ML2_CPU_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            ML2_Scalar *restrict gradientPtr = ML2_CPU_VectorsAt(gradient, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar diff = predictedPtr[c] - expectedPtr[c];
                gradientPtr[c] += ML2_SCALAR_LITERAL(2.0) * diff / (samples * vectors * scalars);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMAX(predictedPtr[c], ML2_Epsilon);
                ML2_Scalar expectedScalar = expectedPtr[c];
                loss += -expectedScalar * ML2_LN(predictedScalar);
            }
        }
    }
    loss /= (samples * vectors);
    return loss;
}

ML2_FN void ML2_CPU_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            ML2_Scalar *restrict gradientPtr = ML2_CPU_VectorsAt(gradient, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMAX(predictedPtr[c], ML2_Epsilon);
                ML2_Scalar expectedScalar = expectedPtr[c];
                gradientPtr[c] += -(expectedScalar / predictedScalar) / (samples * vectors);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            ML2_Scalar max = ML2_NegInf;
            for (ML2_Int c = 0; c < scalars; c++) {
                max = ML2_FMAX(max, predictedPtr[c]);
            }
            ML2_Scalar sum = {};
            for (ML2_Int c = 0; c < scalars; c++) {
                sum += ML2_EXP(predictedPtr[c] - max);
            }
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMAX(ML2_EXP(predictedPtr[c] - max) / sum, ML2_Epsilon);
                ML2_Scalar expectedScalar = expectedPtr[c];
                loss += -expectedScalar * ML2_LN(predictedScalar);
            }
        }
    }
    loss /= (samples * vectors);
    return loss;
}

ML2_FN void ML2_CPU_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            ML2_Scalar *restrict gradientPtr = ML2_CPU_VectorsAt(gradient, s, v, 0);
            ML2_Scalar max = ML2_NegInf;
            for (ML2_Int c = 0; c < scalars; c++) {
                max = ML2_FMAX(max, predictedPtr[c]);
            }
            ML2_Scalar sum = {};
            for (ML2_Int c = 0; c < scalars; c++) {
                sum += ML2_EXP(predictedPtr[c] - max);
            }
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_EXP(predictedPtr[c] - max) / sum;
                ML2_Scalar expectedScalar = expectedPtr[c];
                gradientPtr[c] += (predictedScalar - expectedScalar) / (samples * vectors);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(predictedPtr[c], ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
                ML2_Scalar expectedScalar = expectedPtr[c];
                loss += -expectedScalar * ML2_LN(predictedScalar) - (ML2_SCALAR_LITERAL(1.0) - expectedScalar) * ML2_LN(ML2_SCALAR_LITERAL(1.0) - predictedScalar);
            }
        }
    }
    loss /= (samples * vectors * scalars);
    return loss;
}

ML2_FN void ML2_CPU_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == ML2_DeviceCPU &&
        expected.data.device == ML2_DeviceCPU &&
        gradient.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar *restrict predictedPtr = ML2_CPU_VectorsAt(predicted, s, v, 0);
            ML2_Scalar *restrict expectedPtr = ML2_CPU_VectorsAt(expected, s, v, 0);
            ML2_Scalar *restrict gradientPtr = ML2_CPU_VectorsAt(gradient, s, v, 0);
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(predictedPtr[c], ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
                ML2_Scalar expectedScalar = expectedPtr[c];
                gradientPtr[c] += (predictedScalar - expectedScalar) / (predictedScalar * (ML2_SCALAR_LITERAL(1.0) - predictedScalar)) / (samples * vectors * scalars);
            }
        }
    }
}

ML2_FN ML2_Scalar *ML2_CPU_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c) {
    ML2_SOFT_ASSERT(
        0 <= s && s < matrices.info.samples &&
        0 <= m && m < matrices.info.matrices &&
        0 <= v && v < matrices.info.vectors &&
        0 <= c && c < matrices.info.scalars &&
        "OUT OF BOUNDS INDICES"
    );
    return ML2_CPU_DeviceDataAt(matrices.data, ((s * matrices.info.matrices + m) * matrices.info.vectors + v) * matrices.info.scalars + c);
}

ML2_FN void ML2_CPU_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent) {
    ML2_HARD_ASSERT(matrices.data.device == ML2_DeviceCPU);
    ML2_MatricesInfoPrint(matrices.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int s = 0; s < matrices.info.samples; s++) {
            printf(ML2_INDENT("Samples[%d]:\n", indent), s);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                for (ML2_Int m = 0; m < matrices.info.matrices; m++) {
                    printf(ML2_INDENT("", indent));
                    for (ML2_Int v = 0; v < matrices.info.vectors; v++) {
                        for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                            printf(ML2_INDENT("Channels[%d]:\n", indent), c);
                            printf(ML2_INDENT("{\n", indent));
                            {
                                indent += ML2_Indentation;
                                printf(ML2_SCALAR_FMT " ", *ML2_CPU_MatricesAt(matrices, s, m, v, c));
                                indent -= ML2_Indentation;
                            }
                            printf(ML2_INDENT("}\n", indent));
                        }
                    }
                    printf("\n");
                }
                indent -= ML2_Indentation;
            }
            printf(ML2_INDENT("}\n", indent));
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_CPU_MatricesClear(ML2_LayerCacheMatrices matrices) {
    ML2_HARD_ASSERT(matrices.data.device == ML2_DeviceCPU);
    for (ML2_Int s = 0; s < matrices.info.samples; s++) {
        for (ML2_Int m = 0; m < matrices.info.matrices; m++) {
            for (ML2_Int v = 0; v < matrices.info.vectors; v++) {
                ML2_Scalar *restrict matricesPtr = ML2_CPU_MatricesAt(matrices, s, m, v, 0);
                for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                    matricesPtr[c] = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = dest.info.samples;
    ML2_Int matrices = dest.info.matrices;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict destPtr = ML2_CPU_MatricesAt(dest, s, m, v, 0);
                ML2_Scalar *restrict srcPtr = ML2_CPU_MatricesAt(src, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    destPtr[c] = srcPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(
        dest.data.device == ML2_DeviceCPU &&
        src.data.device == ML2_DeviceCPU
    );

    ML2_Int samples = dest.info.samples;
    ML2_Int matrices = dest.info.matrices;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                ML2_Scalar *restrict destPtr = ML2_CPU_MatricesAt(dest, s, m, v, 0);
                ML2_Scalar *restrict srcPtr = ML2_CPU_MatricesAt(src, s, m, v, 0);
                for (ML2_Int c = 0; c < scalars; c++) {
                    destPtr[c] += srcPtr[c];
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high) {
    ML2_HARD_ASSERT(matrices.data.device == ML2_DeviceCPU);
    for (ML2_Int s = 0; s < matrices.info.samples; s++) {
        for (ML2_Int m = 0; m < matrices.info.matrices; m++) {
            for (ML2_Int v = 0; v < matrices.info.vectors; v++) {
                ML2_Scalar *restrict matricesPtr = ML2_CPU_MatricesAt(matrices, s, m, v, 0);
                for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                    matricesPtr[c] = ML2_ScalarRand(low, high);
                }
            }
        }
    }
}

ML2_FN void ML2_CPU_MatricesXavierInit(ML2_LayerCacheMatrices matrices) {
    ML2_HARD_ASSERT(matrices.data.device == ML2_DeviceCPU);
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (matrices.info.samples + matrices.info.matrices + matrices.info.vectors + matrices.info.scalars));
    ML2_CPU_MatricesRand(matrices, -limit, limit);
}

ML2_FN void ML2_CPU_MatricesHeInit(ML2_LayerCacheMatrices matrices) {
    ML2_HARD_ASSERT(matrices.data.device == ML2_DeviceCPU);
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (matrices.info.matrices * matrices.info.vectors * matrices.info.scalars));
    for (ML2_Int s = 0; s < matrices.info.samples; s++) {
        for (ML2_Int m = 0; m < matrices.info.matrices; m++) {
            for (ML2_Int v = 0; v < matrices.info.vectors; v++) {
                ML2_Scalar *restrict matricesPtr = ML2_CPU_MatricesAt(matrices, s, m, v, 0);
                for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                    matricesPtr[c] = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
                }
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    (void)predicted;
    (void)expected;
    ML2_TODO("ML2_CPU_MatricesLossForwardSquareAverage");
}

ML2_FN void ML2_CPU_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    (void)predicted;
    (void)expected;
    (void)gradient;
    ML2_TODO("ML2_CPU_MatricesLossBackwardSquareAverage");
}

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    (void)predicted;
    (void)expected;
    ML2_TODO("ML2_CPU_MatricesLossForwardCrossEntropy");
}

ML2_FN void ML2_CPU_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    (void)predicted;
    (void)expected;
    (void)gradient;
    ML2_TODO("ML2_CPU_MatricesLossBackwardCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    (void)predicted;
    (void)expected;
    ML2_TODO("ML2_CPU_MatricesLossForwardSoftmaxCrossEntropy");
}

ML2_FN void ML2_CPU_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    (void)predicted;
    (void)expected;
    (void)gradient;
    ML2_TODO("ML2_CPU_MatricesLossBackwardSoftmaxCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CPU_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    (void)predicted;
    (void)expected;
    ML2_TODO("ML2_CPU_MatricesLossForwardBinaryCrossEntropy");
}

ML2_FN void ML2_CPU_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    (void)predicted;
    (void)expected;
    (void)gradient;
    ML2_TODO("ML2_CPU_MatricesLossBackwardBinaryCrossEntropy");
}

// ML2_CPU ⬆️

// ML2_CUDA ⬇️

#ifdef ML2_CUDA_ENABLE

ML2_FN void ML2_CUDA_DeviceInit() {
    ML2_CUDA_CHECK_ERROR(cuInit(0), "Failed to initialize CUDA");
}

ML2_FN ML2_DeviceData ML2_CUDA_DeviceDataNew(ML2_Size size) {
    CUdeviceptr ptr = {};
    ML2_CUDA_CHECK_ERROR(cuMemAlloc(&ptr, size), "Failed to allocate memory");
    ML2_CUDA_CHECK_ERROR(cuMemsetD8(ptr, 0, size), "Failed to zero initialize memory");

    ML2_DeviceData data = {
        ML2_DeviceCUDA,
        .as.cuda = {
            .ptr = ptr,
        },
    };
    return data;
}

ML2_FN void ML2_CUDA_DeviceDataDestroy(ML2_DeviceData *data) {
    ML2_HARD_ASSERT(data->device == ML2_DeviceCUDA);
    ML2_CUDA_CHECK_ERROR(cuMemFree(ML2_DeviceDataPtrAsCUDA(data)->ptr), "Failed to free memory");
    *data = (ML2_DeviceData){};
}

ML2_FN ML2_Scalar *ML2_CUDA_WeightsAt(ML2_LayerWeights weights [[maybe_unused]], ML2_Int o [[maybe_unused]], ML2_Int i [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsAt");
}

ML2_FN void ML2_CUDA_WeightsPrint(ML2_LayerWeights weights [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsPrint");
}

ML2_FN void ML2_CUDA_WeightsClear(ML2_LayerWeights weights [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsClear");
}

ML2_FN void ML2_CUDA_WeightsCopy(ML2_LayerWeights dest [[maybe_unused]], ML2_LayerWeights src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsCopy");
}

ML2_FN void ML2_CUDA_WeightsSum(ML2_LayerWeights dest [[maybe_unused]], ML2_LayerWeights src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsSum");
}

ML2_FN void ML2_CUDA_WeightsRand(ML2_LayerWeights weights [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsRand");
}

ML2_FN void ML2_CUDA_WeightsXavierInit(ML2_LayerWeights weights [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsXavierInit");
}

ML2_FN void ML2_CUDA_WeightsHeInit(ML2_LayerWeights weights [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsHeInit");
}

ML2_FN void ML2_CUDA_WeightsCopyHostToDevice(ML2_LayerWeights destDevice, ML2_LayerWeights srcHost) {
    ML2_TODO("ML2_CUDA_WeightsCopyHostToDevice");
}

ML2_FN void ML2_CUDA_WeightsCopyDeviceToHost(ML2_LayerWeights destHost, ML2_LayerWeights srcDevice) {
    ML2_TODO("ML2_CUDA_WeightsCopyDeviceToHost");
}

ML2_FN void ML2_CUDA_WeightsForwardScalars(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsForwardScalars");
}

ML2_FN void ML2_CUDA_WeightsForwardVectors(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsForwardVectors");
}

ML2_FN void ML2_CUDA_WeightsBackwardScalars(ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsBackwardScalars");
}

ML2_FN void ML2_CUDA_WeightsBackwardVectors(ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsBackwardVectors");
}

ML2_FN void ML2_CUDA_WeightsCacheBackwardScalars(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerCacheScalars inputGradient [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsCacheBackwardScalars");
}

ML2_FN void ML2_CUDA_WeightsCacheBackwardVectors(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerCacheVectors inputGradient [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsCacheBackwardVectors");
}

ML2_FN void ML2_CUDA_WeightsGradientDescentOptimize(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsGradientDescentOptimize");
}

ML2_FN void ML2_CUDA_WeightsMomentumOptimize(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate [[maybe_unused]], ML2_LayerWeights average [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsMomentumOptimize");
}

ML2_FN void ML2_CUDA_WeightsAdagradOptimize(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_LayerWeights squareSum [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsAdagradOptimize");
}

ML2_FN void ML2_CUDA_WeightsRMSPropOptimize(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate [[maybe_unused]], ML2_LayerWeights squareAverage [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsRMSPropOptimize");
}

ML2_FN void ML2_CUDA_WeightsAdamOptimize(ML2_LayerWeights weights [[maybe_unused]], ML2_LayerWeights weightsGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate1 [[maybe_unused]], ML2_Scalar decayRate2 [[maybe_unused]], ML2_LayerWeights average [[maybe_unused]], ML2_LayerWeights squareAverage [[maybe_unused]], ML2_Scalar decayingWeight1 [[maybe_unused]], ML2_Scalar decayingWeight2 [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_WeightsAdamOptimize");
}

ML2_FN ML2_Scalar *ML2_CUDA_BiasesAt(ML2_LayerBiases biases [[maybe_unused]], ML2_Int i [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesAt");
}

ML2_FN void ML2_CUDA_BiasesPrint(ML2_LayerBiases biases [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesPrint");
}

ML2_FN void ML2_CUDA_BiasesClear(ML2_LayerBiases biases [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesClear");
}

ML2_FN void ML2_CUDA_BiasesCopy(ML2_LayerBiases dest [[maybe_unused]], ML2_LayerBiases src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesCopy");
}

ML2_FN void ML2_CUDA_BiasesSum(ML2_LayerBiases dest [[maybe_unused]], ML2_LayerBiases src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesSum");
}

ML2_FN void ML2_CUDA_BiasesRand(ML2_LayerBiases biases [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesRand");
}

ML2_FN void ML2_CUDA_BiasesXavierInit(ML2_LayerBiases biases) {
    ML2_TODO();
}

ML2_FN void ML2_CUDA_BiasesHeInit(ML2_LayerBiases biases) {
    ML2_TODO();
}

ML2_FN void ML2_CUDA_BiasesCopyHostToDevice(ML2_LayerBiases destDevice, ML2_LayerBiases srcHost) {
    ML2_TODO("ML2_CUDA_BiasesCopyHostToDevice");
}

ML2_FN void ML2_CUDA_BiasesCopyDeviceToHost(ML2_LayerBiases destHost, ML2_LayerBiases srcDevice) {
    ML2_TODO("ML2_CUDA_BiasesCopyDeviceToHost");
}

ML2_FN void ML2_CUDA_BiasesForwardScalars(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesForwardScalars");
}

ML2_FN void ML2_CUDA_BiasesForwardMatrices(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesForwardMatrices");
}

ML2_FN void ML2_CUDA_BiasesBackwardScalars(ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesBackwardScalars");
}

ML2_FN void ML2_CUDA_BiasesBackwardMatrices(ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesBackwardMatrices");
}

ML2_FN void ML2_CUDA_BiasesCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_TODO();
}

ML2_FN void ML2_CUDA_BiasesCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_TODO();
}

ML2_FN void ML2_CUDA_BiasesGradientDescentOptimize(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesGradientDescentOptimize");
}

ML2_FN void ML2_CUDA_BiasesMomentumOptimize(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate [[maybe_unused]], ML2_LayerBiases average [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesMomentumOptimize");
}

ML2_FN void ML2_CUDA_BiasesAdagradOptimize(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_LayerBiases squareSum [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesAdagradOptimize");
}

ML2_FN void ML2_CUDA_BiasesRMSPropOptimize(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate [[maybe_unused]], ML2_LayerBiases squareAverage [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesRMSPropOptimize");
}

ML2_FN void ML2_CUDA_BiasesAdamOptimize(ML2_LayerBiases biases [[maybe_unused]], ML2_LayerBiases biasesGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate1 [[maybe_unused]], ML2_Scalar decayRate2 [[maybe_unused]], ML2_LayerBiases average [[maybe_unused]], ML2_LayerBiases squareAverage [[maybe_unused]], ML2_Scalar decayingWeight1 [[maybe_unused]], ML2_Scalar decayingWeight2 [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_BiasesAdamOptimize");
}

ML2_FN void ML2_CUDA_ActivationForwardScalars(ML2_LayerInfoActivation activation [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationForwardScalars");
}

ML2_FN void ML2_CUDA_ActivationForwardVectors(ML2_LayerInfoActivation activation [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationForwardVectors");
}

ML2_FN void ML2_CUDA_ActivationForwardMatrices(ML2_LayerInfoActivation activation [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationForwardMatrices");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardInputScalars(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars inputGradient [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardInputScalars");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardInputVectors(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheVectors inputGradient [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardInputVectors");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardInputMatrices(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardInputMatrices");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardOutputScalars(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheScalars inputGradient [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardOutputScalars");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardOutputVectors(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheVectors inputGradient [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardOutputVectors");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardOutputMatrices(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardOutputMatrices");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardInputOutputScalars(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars inputGradient [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardInputOutputScalars");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardInputOutputVectors(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheVectors inputGradient [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardInputOutputVectors");
}

ML2_FN void ML2_CUDA_ActivationCacheBackwardInputOutputMatrices(ML2_ActivationType type [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ActivationCacheBackwardInputOutputMatrices");
}

ML2_FN void ML2_CUDA_SoftmaxForwardScalars(ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_SoftmaxForwardScalars");
}

ML2_FN void ML2_CUDA_SoftmaxForwardVectors(ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_SoftmaxForwardVectors");
}

ML2_FN void ML2_CUDA_SoftmaxCacheBackwardScalars(ML2_LayerCacheScalars inputGradient [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_SoftmaxCacheBackwardScalars");
}

ML2_FN void ML2_CUDA_SoftmaxCacheBackwardVectors(ML2_LayerCacheVectors inputGradient [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_SoftmaxCacheBackwardVectors");
}

ML2_FN ML2_Scalar *ML2_CUDA_LinearWeightsAt(ML2_LayerLinear linear [[maybe_unused]], ML2_Int o [[maybe_unused]], ML2_Int i [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_LinearWeightsAt");
}

ML2_FN ML2_Scalar *ML2_CUDA_LinearBiasesAt(ML2_LayerLinear linear [[maybe_unused]], ML2_Int o [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_LinearBiasesAt");
}

ML2_FN ML2_LayerWeights ML2_CUDA_LinearWeights(ML2_LayerLinear linear [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_LinearWeights");
}

ML2_FN void ML2_CUDA_LinearPrint(ML2_LayerLinear linear [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_LinearPrint");
}

ML2_FN void ML2_CUDA_LinearClear(ML2_LayerLinear linear) {
    ML2_TODO("ML2_CUDA_LinearClear");
}

ML2_FN void ML2_CUDA_LinearCopy(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_TODO("ML2_CUDA_LinearCopy");
}

ML2_FN void ML2_CUDA_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_TODO("ML2_CUDA_LinearSum");
}

ML2_FN void ML2_CUDA_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high) {
    ML2_TODO("ML2_CUDA_LinearRand");
}

ML2_FN void ML2_CUDA_LinearXavierInit(ML2_LayerLinear linear) {
    ML2_TODO("ML2_CUDA_LinearXavierInit");
}

ML2_FN void ML2_CUDA_LinearHeInit(ML2_LayerLinear linear) {
    ML2_TODO("ML2_CUDA_LinearHeInit");
}

ML2_FN void ML2_CUDA_LinearCopyHostToDevice(ML2_LayerLinear destDevice, ML2_LayerLinear srcHost) {
    ML2_TODO("ML2_CUDA_LinearCopyHostToDevice");
}

ML2_FN void ML2_CUDA_LinearCopyDeviceToHost(ML2_LayerLinear destHost, ML2_LayerLinear srcDevice) {
    ML2_TODO("ML2_CUDA_LinearCopyDeviceToHost");
}

ML2_FN ML2_LayerBiases ML2_CUDA_LinearBiases(ML2_LayerLinear linear [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_LinearBiases");
}

ML2_FN void ML2_CUDA_LinearForwardScalars(ML2_LayerLinear linear [[maybe_unused]], ML2_LayerCacheScalars input [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_LinearForwardScalars");
}

ML2_FN void ML2_CUDA_LinearBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_TODO("ML2_CUDA_LinearBackwardScalars");
}

ML2_FN void ML2_CUDA_LinearCacheBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_TODO("ML2_CUDA_LinearCacheBackwardScalars");
}

ML2_FN void ML2_CUDA_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate) {
    ML2_TODO("ML2_CUDA_LinearGradientDescentOptimize");
}

ML2_FN void ML2_CUDA_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average) {
    ML2_TODO("ML2_CUDA_LinearMomentumOptimize");
}

ML2_FN void ML2_CUDA_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum) {
    ML2_TODO("ML2_CUDA_LinearAdagradOptimize");
}

ML2_FN void ML2_CUDA_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage) {
    ML2_TODO("ML2_CUDA_LinearRMSPropOptimize");
}

ML2_FN void ML2_CUDA_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_TODO("ML2_CUDA_LinearAdamOptimize");
}

ML2_FN ML2_Scalar *ML2_CUDA_FiltersAt(ML2_LayerFilters filters [[maybe_unused]], ML2_Int y [[maybe_unused]], ML2_Int x [[maybe_unused]], ML2_Int i [[maybe_unused]], ML2_Int o [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersAt");
}

ML2_FN void ML2_CUDA_FiltersPrint(ML2_LayerFilters filters [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersPrint");
}

ML2_FN void ML2_CUDA_FiltersClear(ML2_LayerFilters filters [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersClear");
}

ML2_FN void ML2_CUDA_FiltersCopy(ML2_LayerFilters dest [[maybe_unused]], ML2_LayerFilters src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersCopy");
}

ML2_FN void ML2_CUDA_FiltersSum(ML2_LayerFilters dest [[maybe_unused]], ML2_LayerFilters src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersSum");
}

ML2_FN void ML2_CUDA_FiltersRand(ML2_LayerFilters filters [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersRand");
}

ML2_FN void ML2_CUDA_FiltersXavierInit(ML2_LayerFilters filters [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersXavierInit");
}

ML2_FN void ML2_CUDA_FiltersHeInit(ML2_LayerFilters filters [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersHeInit");
}

ML2_FN void ML2_CUDA_FiltersCopyHostToDevice(ML2_LayerFilters destDevice, ML2_LayerFilters srcHost) {
    ML2_TODO("ML2_CUDA_FiltersCopyHostToDevice");
}

ML2_FN void ML2_CUDA_FiltersCopyDeviceToHost(ML2_LayerFilters destHost, ML2_LayerFilters srcDevice) {
    ML2_TODO("ML2_CUDA_FiltersCopyDeviceToHost");
}

ML2_FN void ML2_CUDA_FiltersForwardMatrices(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersForwardMatrices");
}

ML2_FN void ML2_CUDA_FiltersBackwardMatrices(ML2_LayerFilters filtersGradient [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersBackwardMatrices");
}

ML2_FN void ML2_CUDA_FiltersCacheBackwardMatrices(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersCacheBackwardMatrices");
}

ML2_FN void ML2_CUDA_FiltersGradientDescentOptimize(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerFilters filtersGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersGradientDescentOptimize");
}

ML2_FN void ML2_CUDA_FiltersMomentumOptimize(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerFilters filtersGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate [[maybe_unused]], ML2_LayerFilters average [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersMomentumOptimize");
}

ML2_FN void ML2_CUDA_FiltersAdagradOptimize(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerFilters filtersGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_LayerFilters squareSum [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersAdagradOptimize");
}

ML2_FN void ML2_CUDA_FiltersRMSPropOptimize(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerFilters filtersGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate [[maybe_unused]], ML2_LayerFilters squareAverage [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersRMSPropOptimize");
}

ML2_FN void ML2_CUDA_FiltersAdamOptimize(ML2_LayerFilters filters [[maybe_unused]], ML2_LayerFilters filtersGradient [[maybe_unused]], ML2_Scalar learningRate [[maybe_unused]], ML2_Scalar decayRate1 [[maybe_unused]], ML2_Scalar decayRate2 [[maybe_unused]], ML2_LayerFilters average [[maybe_unused]], ML2_LayerFilters squareAverage [[maybe_unused]], ML2_Scalar decayingWeight1 [[maybe_unused]], ML2_Scalar decayingWeight2 [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FiltersAdamOptimize");
}

ML2_FN ML2_Scalar *ML2_CUDA_ConvFiltersAt(ML2_LayerConv conv [[maybe_unused]], ML2_Int y [[maybe_unused]], ML2_Int x [[maybe_unused]], ML2_Int i [[maybe_unused]], ML2_Int o [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ConvFiltersAt");
}

ML2_FN ML2_Scalar *ML2_CUDA_ConvBiasesAt(ML2_LayerConv conv [[maybe_unused]], ML2_Int o [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ConvBiasesAt");
}

ML2_FN ML2_LayerFilters ML2_CUDA_ConvFilters(ML2_LayerConv conv [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ConvFilters");
}

ML2_FN ML2_LayerBiases ML2_CUDA_ConvBiases(ML2_LayerConv conv [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ConvBiases");
}

ML2_FN void ML2_CUDA_ConvPrint(ML2_LayerConv conv [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ConvPrint");
}

ML2_FN void ML2_CUDA_ConvClear(ML2_LayerConv conv) {
    ML2_TODO("ML2_CUDA_ConvClear");
}

ML2_FN void ML2_CUDA_ConvCopy(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_TODO("ML2_CUDA_ConvCopy");
}

ML2_FN void ML2_CUDA_ConvSum(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_TODO("ML2_CUDA_ConvSum");
}

ML2_FN void ML2_CUDA_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high) {
    ML2_TODO("ML2_CUDA_ConvRand");
}

ML2_FN void ML2_CUDA_ConvXavierInit(ML2_LayerConv conv) {
    ML2_TODO("ML2_CUDA_ConvXavierInit");
}

ML2_FN void ML2_CUDA_ConvHeInit(ML2_LayerConv conv) {
    ML2_TODO("ML2_CUDA_ConvHeInit");
}

ML2_FN void ML2_CUDA_ConvCopyHostToDevice(ML2_LayerConv destDevice, ML2_LayerConv srcHost) {
    ML2_TODO("ML2_CUDA_ConvCopyHostToDevice");
}

ML2_FN void ML2_CUDA_ConvCopyDeviceToHost(ML2_LayerConv destHost, ML2_LayerConv srcDevice) {
    ML2_TODO("ML2_CUDA_ConvCopyDeviceToHost");
}

ML2_FN void ML2_CUDA_ConvForwardMatrices(ML2_LayerConv conv [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ConvForwardMatrices");
}

ML2_FN void ML2_CUDA_ConvBackwardMatrices(ML2_LayerConv convGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient) {
    ML2_TODO("ML2_CUDA_ConvBackwardMatrices");
}

ML2_FN void ML2_CUDA_ConvCacheBackwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_TODO("ML2_CUDA_ConvCacheBackwardMatrices");
}

ML2_FN void ML2_CUDA_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate) {
    ML2_TODO("ML2_CUDA_ConvGradientDescentOptimize");
}

ML2_FN void ML2_CUDA_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average) {
    ML2_TODO("ML2_CUDA_ConvMomentumOptimize");
}

ML2_FN void ML2_CUDA_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum) {
    ML2_TODO("ML2_CUDA_ConvAdagradOptimize");
}

ML2_FN void ML2_CUDA_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage) {
    ML2_TODO("ML2_CUDA_ConvRMSPropOptimize");
}

ML2_FN void ML2_CUDA_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_TODO("ML2_CUDA_ConvAdamOptimize");
}

ML2_FN void ML2_CUDA_FlattenForwardMatrices(ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheScalars output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FlattenForwardMatrices");
}

ML2_FN void ML2_CUDA_FlattenCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheScalars outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FlattenCacheBackwardMatrices");
}

ML2_FN void ML2_CUDA_PoolMaxForwardMatrices(ML2_LayerInfoPoolMax pool [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_PoolMaxForwardMatrices");
}

ML2_FN void ML2_CUDA_PoolMaxCacheBackwardMatrices(ML2_LayerInfoPoolMax pool [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_PoolMaxCacheBackwardMatrices");
}

ML2_FN void ML2_CUDA_PoolAverageForwardMatrices(ML2_LayerInfoPoolAverage pool [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_PoolAverageForwardMatrices");
}

ML2_FN void ML2_CUDA_PoolAverageCacheBackwardMatrices(ML2_LayerInfoPoolAverage pool [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_PoolAverageCacheBackwardMatrices");
}

ML2_FN void ML2_CUDA_PoolSoftmaxForwardMatrices(ML2_LayerInfoPoolSoftmax pool [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_PoolSoftmaxForwardMatrices");
}

ML2_FN void ML2_CUDA_PoolSoftmaxCacheBackwardMatrices(ML2_LayerInfoPoolSoftmax pool [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_PoolSoftmaxCacheBackwardMatrices");
}

ML2_FN ML2_Scalar *ML2_CUDA_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights [[maybe_unused]], ML2_Int w [[maybe_unused]], ML2_Int o [[maybe_unused]], ML2_Int i [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsAt");
}

ML2_FN ML2_LayerWeights ML2_CUDA_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights [[maybe_unused]], ML2_Int w [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsWeightsAt");
}

ML2_FN void ML2_CUDA_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsPrint");
}

ML2_FN void ML2_CUDA_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsClear");
}

ML2_FN void ML2_CUDA_FusedWeightsCopy(ML2_LayerFusedWeights dest [[maybe_unused]], ML2_LayerFusedWeights src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsCopy");
}

ML2_FN void ML2_CUDA_FusedWeightsSum(ML2_LayerFusedWeights dest [[maybe_unused]], ML2_LayerFusedWeights src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsSum");
}

ML2_FN void ML2_CUDA_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsRand");
}

ML2_FN void ML2_CUDA_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsXavierInit");
}

ML2_FN void ML2_CUDA_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsHeInit");
}

ML2_FN void ML2_CUDA_FusedWeightsCopyHostToDevice(ML2_LayerFusedWeights destDevice, ML2_LayerFusedWeights srcHost) {
    ML2_TODO("ML2_CUDA_FusedWeightsCopyHostToDevice");
}

ML2_FN void ML2_CUDA_FusedWeightsCopyDeviceToHost(ML2_LayerFusedWeights destHost, ML2_LayerFusedWeights srcDevice) {
    ML2_TODO("ML2_CUDA_FusedWeightsCopyDeviceToHost");
}

ML2_FN void ML2_CUDA_FusedWeightsForwardVectors(ML2_LayerFusedWeights fusedWeights [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheMatrices output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsForwardVectors");
}

ML2_FN void ML2_CUDA_FusedWeightsBackwardVectors(ML2_LayerFusedWeights fusedWeightsGradient [[maybe_unused]], ML2_LayerCacheVectors input [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsBackwardVectors");
}

ML2_FN void ML2_CUDA_FusedWeightsCacheBackwardVectors(ML2_LayerFusedWeights fusedWeights [[maybe_unused]], ML2_LayerCacheVectors inputGradient [[maybe_unused]], ML2_LayerCacheMatrices outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_FusedWeightsCacheBackwardVectors");
}

ML2_FN void ML2_CUDA_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate) {
    ML2_TODO("ML2_CUDA_FusedWeightsGradientDescentOptimize");
}

ML2_FN void ML2_CUDA_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average) {
    ML2_TODO("ML2_CUDA_FusedWeightsMomentumOptimize");
}

ML2_FN void ML2_CUDA_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum) {
    ML2_TODO("ML2_CUDA_FusedWeightsAdagradOptimize");
}

ML2_FN void ML2_CUDA_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage) {
    ML2_TODO("ML2_CUDA_FusedWeightsRMSPropOptimize");
}

ML2_FN void ML2_CUDA_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_TODO("ML2_CUDA_FusedWeightsAdamOptimize");
}

ML2_FN void ML2_CUDA_AttentionForwardMatrices(ML2_LayerInfoAttention attention [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheVectors output [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_AttentionForwardMatrices");
}

ML2_FN void ML2_CUDA_AttentionCacheBackwardMatrices(ML2_LayerInfoAttention attention [[maybe_unused]], ML2_LayerCacheMatrices input [[maybe_unused]], ML2_LayerCacheMatrices inputGradient [[maybe_unused]], ML2_LayerCacheVectors outputGradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_AttentionCacheBackwardMatrices");
}

ML2_FN ML2_Scalar *ML2_CUDA_ScalarsAt(ML2_LayerCacheScalars scalars [[maybe_unused]], ML2_Int s [[maybe_unused]], ML2_Int c [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsAt");
}

ML2_FN void ML2_CUDA_ScalarsPrint(ML2_LayerCacheScalars scalars [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsPrint");
}

ML2_FN void ML2_CUDA_ScalarsClear(ML2_LayerCacheScalars scalars [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsClear");
}

ML2_FN void ML2_CUDA_ScalarsCopy(ML2_LayerCacheScalars dest [[maybe_unused]], ML2_LayerCacheScalars src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsCopy");
}

ML2_FN void ML2_CUDA_ScalarsSum(ML2_LayerCacheScalars dest [[maybe_unused]], ML2_LayerCacheScalars src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsSum");
}

ML2_FN void ML2_CUDA_ScalarsRand(ML2_LayerCacheScalars scalars [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsRand");
}

ML2_FN void ML2_CUDA_ScalarsXavierInit(ML2_LayerCacheScalars scalars [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsXavierInit");
}

ML2_FN void ML2_CUDA_ScalarsHeInit(ML2_LayerCacheScalars scalars [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsHeInit");
}

ML2_FN void ML2_CUDA_ScalarsCopyHostToDevice(ML2_LayerCacheScalars destDevice, ML2_LayerCacheScalars srcHost) {
    ML2_TODO("ML2_CUDA_ScalarsCopyHostToDevice");
}

ML2_FN void ML2_CUDA_ScalarsCopyDeviceToHost(ML2_LayerCacheScalars destHost, ML2_LayerCacheScalars srcDevice) {
    ML2_TODO("ML2_CUDA_ScalarsCopyDeviceToHost");
}

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossForwardSquareAverage");
}

ML2_FN void ML2_CUDA_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]], ML2_LayerCacheScalars gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossBackwardSquareAverage");
}

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossForwardCrossEntropy");
}

ML2_FN void ML2_CUDA_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]], ML2_LayerCacheScalars gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossBackwardCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossForwardSoftmaxCrossEntropy");
}

ML2_FN void ML2_CUDA_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]], ML2_LayerCacheScalars gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossBackwardSoftmaxCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CUDA_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossForwardBinaryCrossEntropy");
}

ML2_FN void ML2_CUDA_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted [[maybe_unused]], ML2_LayerCacheScalars expected [[maybe_unused]], ML2_LayerCacheScalars gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_ScalarsLossBackwardBinaryCrossEntropy");
}

ML2_FN ML2_Scalar *ML2_CUDA_VectorsAt(ML2_LayerCacheVectors vectors [[maybe_unused]], ML2_Int s [[maybe_unused]], ML2_Int v [[maybe_unused]], ML2_Int c [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsAt");
}

ML2_FN void ML2_CUDA_VectorsPrint(ML2_LayerCacheVectors vectors [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsPrint");
}

ML2_FN void ML2_CUDA_VectorsClear(ML2_LayerCacheVectors vectors [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsClear");
}

ML2_FN void ML2_CUDA_VectorsCopy(ML2_LayerCacheVectors dest [[maybe_unused]], ML2_LayerCacheVectors src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsCopy");
}

ML2_FN void ML2_CUDA_VectorsSum(ML2_LayerCacheVectors dest [[maybe_unused]], ML2_LayerCacheVectors src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsSum");
}

ML2_FN void ML2_CUDA_VectorsRand(ML2_LayerCacheVectors vectors [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsRand");
}

ML2_FN void ML2_CUDA_VectorsXavierInit(ML2_LayerCacheVectors vectors [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsXavierInit");
}

ML2_FN void ML2_CUDA_VectorsHeInit(ML2_LayerCacheVectors vectors [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsHeInit");
}

ML2_FN void ML2_CUDA_VectorsCopyHostToDevice(ML2_LayerCacheVectors destDevice, ML2_LayerCacheVectors srcHost) {
    ML2_TODO("ML2_CUDA_VectorsCopyHostToDevice");
}

ML2_FN void ML2_CUDA_VectorsCopyDeviceToHost(ML2_LayerCacheVectors destHost, ML2_LayerCacheVectors srcDevice) {
    ML2_TODO("ML2_CUDA_VectorsCopyDeviceToHost");
}

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossForwardSquareAverage");
}

ML2_FN void ML2_CUDA_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]], ML2_LayerCacheVectors gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossBackwardSquareAverage");
}

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossForwardCrossEntropy");
}

ML2_FN void ML2_CUDA_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]], ML2_LayerCacheVectors gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossBackwardCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossForwardSoftmaxCrossEntropy");
}

ML2_FN void ML2_CUDA_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]], ML2_LayerCacheVectors gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossBackwardSoftmaxCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CUDA_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossForwardBinaryCrossEntropy");
}

ML2_FN void ML2_CUDA_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted [[maybe_unused]], ML2_LayerCacheVectors expected [[maybe_unused]], ML2_LayerCacheVectors gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_VectorsLossBackwardBinaryCrossEntropy");
}

ML2_FN ML2_Scalar *ML2_CUDA_MatricesAt(ML2_LayerCacheMatrices matrices [[maybe_unused]], ML2_Int s [[maybe_unused]], ML2_Int m [[maybe_unused]], ML2_Int v [[maybe_unused]], ML2_Int c [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesAt");
}

ML2_FN void ML2_CUDA_MatricesPrint(ML2_LayerCacheMatrices matrices [[maybe_unused]], ML2_Int indent [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesPrint");
}

ML2_FN void ML2_CUDA_MatricesClear(ML2_LayerCacheMatrices matrices [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesClear");
}

ML2_FN void ML2_CUDA_MatricesCopy(ML2_LayerCacheMatrices dest [[maybe_unused]], ML2_LayerCacheMatrices src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesCopy");
}

ML2_FN void ML2_CUDA_MatricesSum(ML2_LayerCacheMatrices dest [[maybe_unused]], ML2_LayerCacheMatrices src [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesSum");
}

ML2_FN void ML2_CUDA_MatricesRand(ML2_LayerCacheMatrices matrices [[maybe_unused]], ML2_Scalar low [[maybe_unused]], ML2_Scalar high [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesRand");
}

ML2_FN void ML2_CUDA_MatricesXavierInit(ML2_LayerCacheMatrices matrices [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesXavierInit");
}

ML2_FN void ML2_CUDA_MatricesHeInit(ML2_LayerCacheMatrices matrices [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesHeInit");
}

ML2_FN void ML2_CUDA_MatricesCopyHostToDevice(ML2_LayerCacheMatrices destDevice, ML2_LayerCacheMatrices srcHost) {
    ML2_TODO("ML2_CUDA_MatricesCopyHostToDevice");
}

ML2_FN void ML2_CUDA_MatricesCopyDeviceToHost(ML2_LayerCacheMatrices destHost, ML2_LayerCacheMatrices srcDevice) {
    ML2_TODO("ML2_CUDA_MatricesCopyDeviceToHost");
}

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossForwardSquareAverage");
}

ML2_FN void ML2_CUDA_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]], ML2_LayerCacheMatrices gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossBackwardSquareAverage");
}

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossForwardCrossEntropy");
}

ML2_FN void ML2_CUDA_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]], ML2_LayerCacheMatrices gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossBackwardCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossForwardSoftmaxCrossEntropy");
}

ML2_FN void ML2_CUDA_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]], ML2_LayerCacheMatrices gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossBackwardSoftmaxCrossEntropy");
}

ML2_FN ML2_Scalar ML2_CUDA_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossForwardBinaryCrossEntropy");
}

ML2_FN void ML2_CUDA_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted [[maybe_unused]], ML2_LayerCacheMatrices expected [[maybe_unused]], ML2_LayerCacheMatrices gradient [[maybe_unused]]) {
    ML2_TODO("ML2_CUDA_MatricesLossBackwardBinaryCrossEntropy");
}

#endif // ML2_CUDA_ENABLE

// ML2_CUDA ⬆️

// ML2_Device ⬆️

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_FN ML2_LayerInfo ML2_Weights(ML2_Int inputs, ML2_Int outputs) {
    return (ML2_LayerInfo){ML2_LayerTypeWeights, .as.weights = {outputs, inputs}};
}

ML2_FN ML2_LayerWeights ML2_WeightsNew(ML2_Device device, ML2_LayerInfoWeights info) {
    ML2_LayerWeights weights = {
        info,
        ML2_DeviceDataNew(device, info.outputs * info.inputs * sizeof(ML2_Scalar)),
    };
    return weights;
}

ML2_FN void ML2_WeightsDestroy(ML2_LayerWeights *weights) {
    ML2_DeviceDataDestroy(&weights->data);
    *weights = (ML2_LayerWeights){};
}

ML2_FN ML2_LayerInfoWeights ML2_LayerInfoAsWeights(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeWeights);
    return info.as.weights;
}

ML2_FN ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeWeights);
    return layer.as.weights;
}

ML2_FN void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, ML2_Int indent) {
    printf(ML2_INDENT("Weights(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_FN bool ML2_WeightsInfoForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheInfoScalars inputScalars = ML2_LayerCacheInfoAsScalars(input);
            return weights.inputs == inputScalars.scalars;
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheInfoVectors inputVectors = ML2_LayerCacheInfoAsVectors(input);
            return weights.inputs == inputVectors.scalars;
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_WeightsInfoForward(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_WeightsInfoForwardCompatible(weights, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheInfoScalars inputScalars = ML2_LayerCacheInfoAsScalars(input);
            return (ML2_LayerCacheInfo){input.type, .as.scalars = {inputScalars.samples, weights.outputs}};
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheInfoVectors inputVectors = ML2_LayerCacheInfoAsVectors(input);
            return (ML2_LayerCacheInfo){input.type, .as.vectors = {inputVectors.samples, inputVectors.vectors, weights.outputs}};
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b) {
    return a.outputs == b.outputs &&
           a.inputs == b.inputs;
}

ML2_FN ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i) {
    switch (weights.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_WeightsAt(weights, o, i);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_WeightsAt(weights, o, i);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent) {
    switch (weights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsPrint(weights, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsPrint(weights, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsClear(ML2_LayerWeights weights) {
    switch (weights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsClear(weights); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsClear(weights); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsCopy(ML2_LayerWeights dest, ML2_LayerWeights src) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high) {
    switch (weights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsRand(weights, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsRand(weights, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsXavierInit(ML2_LayerWeights weights) {
    switch (weights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsXavierInit(weights); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsXavierInit(weights); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsHeInit(ML2_LayerWeights weights) {
    switch (weights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsHeInit(weights); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsHeInit(weights); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsCopyHostToDevice(ML2_LayerWeights destDevice, ML2_LayerWeights srcHost) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsCopyDeviceToHost(ML2_LayerWeights destHost, ML2_LayerWeights srcDevice) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != output.type) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: return ML2_WeightsForwardCompatibleScalars(weights, ML2_LayerCacheInfoAsScalars(input), ML2_LayerCacheInfoAsScalars(output)); break;
        case ML2_LayerCacheTypeVectors: return ML2_WeightsForwardCompatibleVectors(weights, ML2_LayerCacheInfoAsVectors(input), ML2_LayerCacheInfoAsVectors(output)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatible(weights.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_WeightsForwardScalars(weights, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(output)); break;
        case ML2_LayerCacheTypeVectors: ML2_WeightsForwardVectors(weights, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsVectors(output)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsForwardCompatibleScalars(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output) {
    return input.scalars == weights.inputs &&
           output.samples == input.samples &&
           output.scalars == weights.outputs;
}

ML2_FN void ML2_WeightsForwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatibleScalars(weights.info, input.info, output.info));
    ML2_HARD_ASSERT(
        weights.data.device == input.data.device &&
        weights.data.device == output.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsForwardScalars(weights, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsForwardScalars(weights, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_WeightsForwardCompatibleVectors(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors output) {
    return input.scalars == weights.inputs &&
           output.samples == input.samples &&
           output.vectors == input.vectors &&
           output.scalars == weights.outputs;
}

ML2_FN void ML2_WeightsForwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatibleVectors(weights.info, input.info, output.info));
    ML2_HARD_ASSERT(
        weights.data.device == input.data.device &&
        weights.data.device == output.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsForwardVectors(weights, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsForwardVectors(weights, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_WeightsBackwardCompatible(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsForwardCompatible(weightsGradient, input, outputGradient);
}

ML2_FN void ML2_WeightsBackward(ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatible(weightsGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_WeightsBackwardScalars(weightsGradient, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_WeightsBackwardVectors(weightsGradient, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsVectors(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsBackwardCompatibleScalars(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_WeightsForwardCompatibleScalars(weightsGradient, input, outputGradient);
}

ML2_FN void ML2_WeightsBackwardScalars(ML2_LayerWeights weightsGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatibleScalars(weightsGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weightsGradient.data.device == input.data.device &&
        weightsGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = weightsGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsBackwardScalars(weightsGradient, input, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsBackwardScalars(weightsGradient, input, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_WeightsBackwardCompatibleVectors(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_WeightsForwardCompatibleVectors(weightsGradient, input, outputGradient);
}

ML2_FN void ML2_WeightsBackwardVectors(ML2_LayerWeights weightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatibleVectors(weightsGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weightsGradient.data.device == input.data.device &&
        weightsGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = weightsGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsBackwardVectors(weightsGradient, input, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsBackwardVectors(weightsGradient, input, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_WeightsCacheBackwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsForwardCompatible(weights, inputGradient, outputGradient);
}

ML2_FN void ML2_WeightsCacheBackward(ML2_LayerWeights weights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsCacheBackwardCompatible(weights.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_WeightsCacheBackwardScalars(weights, ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_WeightsCacheBackwardVectors(weights, ML2_LayerCacheAsVectors(inputGradient), ML2_LayerCacheAsVectors(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsCacheBackwardCompatibleScalars(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_WeightsForwardCompatibleScalars(weights, inputGradient, outputGradient);
}

ML2_FN void ML2_WeightsCacheBackwardScalars(ML2_LayerWeights weights, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsCacheBackwardCompatibleScalars(weights.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weights.data.device == inputGradient.data.device &&
        weights.data.device == outputGradient.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsCacheBackwardScalars(weights, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsCacheBackwardScalars(weights, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_WeightsCacheBackwardCompatibleVectors(ML2_LayerInfoWeights weights, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_WeightsForwardCompatibleVectors(weights, inputGradient, outputGradient);
}

ML2_FN void ML2_WeightsCacheBackwardVectors(ML2_LayerWeights weights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsCacheBackwardCompatibleVectors(weights.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        weights.data.device == inputGradient.data.device &&
        weights.data.device == outputGradient.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsCacheBackwardVectors(weights, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsCacheBackwardVectors(weights, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(
        weights.data.device == weightsGradient.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsGradientDescentOptimize(weights, weightsGradient, learningRate); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsGradientDescentOptimize(weights, weightsGradient, learningRate); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, average.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == weightsGradient.data.device &&
        weights.data.device == average.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsMomentumOptimize(weights, weightsGradient, learningRate, decayRate, average); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsMomentumOptimize(weights, weightsGradient, learningRate, decayRate, average); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == weightsGradient.data.device &&
        weights.data.device == squareSum.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsAdagradOptimize(weights, weightsGradient, learningRate, squareSum); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsAdagradOptimize(weights, weightsGradient, learningRate, squareSum); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == weightsGradient.data.device &&
        weights.data.device == squareAverage.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsRMSPropOptimize(weights, weightsGradient, learningRate, decayRate, squareAverage); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsRMSPropOptimize(weights, weightsGradient, learningRate, decayRate, squareAverage); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_WeightsInfoSame(weights.info, weightsGradient.info) &&
        ML2_WeightsInfoSame(weights.info, average.info) &&
        ML2_WeightsInfoSame(weights.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        weights.data.device == weightsGradient.data.device &&
        weights.data.device == average.data.device &&
        weights.data.device == squareAverage.data.device
    );
    ML2_Device device = weights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_WeightsAdamOptimize(weights, weightsGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_WeightsAdamOptimize(weights, weightsGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_WeightsSizeof(ML2_LayerWeights weights) {
    return weights.info.outputs * weights.info.inputs * sizeof(ML2_Scalar);
}

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_FN ML2_LayerInfo ML2_Biases(ML2_Int inputs) {
    return (ML2_LayerInfo){ML2_LayerTypeBiases, .as.biases = {inputs}};
}

ML2_FN ML2_LayerBiases ML2_BiasesNew(ML2_Device device, ML2_LayerInfoBiases info) {
    ML2_LayerBiases biases = {
        info,
        ML2_DeviceDataNew(device, info.inputs * sizeof(ML2_Scalar)),
    };
    return biases;
}

ML2_FN void ML2_BiasesDestroy(ML2_LayerBiases *biases) {
    ML2_DeviceDataDestroy(&biases->data);
    *biases = (ML2_LayerBiases){};
}

ML2_FN ML2_LayerInfoBiases ML2_LayerInfoAsBiases(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeBiases);
    return info.as.biases;
}

ML2_FN ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeBiases);
    return layer.as.biases;
}

ML2_FN void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, ML2_Int indent) {
    printf(ML2_INDENT("Biases(%d)\n", indent), info.inputs);
}

ML2_FN bool ML2_BiasesInfoForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheInfoScalars inputScalars = ML2_LayerCacheInfoAsScalars(input);
            return biases.inputs == inputScalars.scalars;
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Biases may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            return biases.inputs == inputMatrices.scalars;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_BiasesInfoForward(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_BiasesInfoForwardCompatible(biases, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return input;
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Biases may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            return input;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesInfoSame(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b) {
    return a.inputs == b.inputs;
}

ML2_FN ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, ML2_Int i) {
    switch (biases.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_BiasesAt(biases, i);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_BiasesAt(biases, i);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent) {
    switch (biases.data.device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesPrint(biases, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesPrint(biases, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesClear(ML2_LayerBiases biases) {
    switch (biases.data.device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesClear(biases); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesClear(biases); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesCopy(ML2_LayerBiases dest, ML2_LayerBiases src) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high) {
    switch (biases.data.device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesRand(biases, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesRand(biases, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesXavierInit(ML2_LayerBiases biases) {
    switch (biases.data.device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesXavierInit(biases); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesXavierInit(biases); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesHeInit(ML2_LayerBiases biases) {
    switch (biases.data.device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesHeInit(biases); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesHeInit(biases); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesCopyHostToDevice(ML2_LayerBiases destDevice, ML2_LayerBiases srcHost) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesCopyDeviceToHost(ML2_LayerBiases destHost, ML2_LayerBiases srcDevice) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != output.type) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: return ML2_BiasesForwardCompatibleScalars(biases, ML2_LayerCacheInfoAsScalars(input), ML2_LayerCacheInfoAsScalars(output));
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Biases may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: return ML2_BiasesForwardCompatibleMatrices(biases, ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsMatrices(output));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatible(biases.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_BiasesForwardScalars(biases, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(output)); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Biases may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: ML2_BiasesForwardMatrices(biases, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesForwardCompatibleScalars(ML2_LayerInfoBiases biases, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output) {
    return biases.inputs == output.scalars &&
           ML2_ScalarsInfoSame(input, output);
}

ML2_FN void ML2_BiasesForwardScalars(ML2_LayerBiases biases, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatibleScalars(biases.info, input.info, output.info));
    ML2_HARD_ASSERT(
        biases.data.device == input.data.device &&
        biases.data.device == output.data.device
    );
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesForwardScalars(biases, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesForwardScalars(biases, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_BiasesForwardCompatibleMatrices(ML2_LayerInfoBiases biases, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    return biases.inputs == output.scalars &&
           ML2_MatricesInfoSame(input, output);
}

ML2_FN void ML2_BiasesForwardMatrices(ML2_LayerBiases biases, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatibleMatrices(biases.info, input.info, output.info));
    ML2_HARD_ASSERT(
        biases.data.device == input.data.device &&
        biases.data.device == output.data.device
    );
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesForwardMatrices(biases, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesForwardMatrices(biases, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_BiasesBackwardCompatible(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfo outputGradient) {
    switch (outputGradient.type) {
        case ML2_LayerCacheTypeScalars: return ML2_BiasesBackwardCompatibleScalars(biasesGradient, ML2_LayerCacheInfoAsScalars(outputGradient));
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Biases may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: return ML2_BiasesBackwardCompatibleMatrices(biasesGradient, ML2_LayerCacheInfoAsMatrices(outputGradient));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_BiasesBackward(ML2_LayerBiases biasesGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatible(biasesGradient.info, ML2_LayerCacheAsInfo(outputGradient)));
    switch (outputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_BiasesBackwardScalars(biasesGradient, ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Biases may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: ML2_BiasesBackwardMatrices(biasesGradient, ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesBackwardCompatibleScalars(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfoScalars outputGradient) {
    return biasesGradient.inputs == outputGradient.scalars;
}

ML2_FN void ML2_BiasesBackwardScalars(ML2_LayerBiases biasesGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatibleScalars(biasesGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(biasesGradient.data.device == outputGradient.data.device);
    ML2_Device device = biasesGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesBackwardScalars(biasesGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesBackwardScalars(biasesGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_BiasesBackwardCompatibleMatrices(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return biasesGradient.inputs == outputGradient.scalars;
}

ML2_FN void ML2_BiasesBackwardMatrices(ML2_LayerBiases biasesGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatibleMatrices(biasesGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(biasesGradient.data.device == outputGradient.data.device);
    ML2_Device device = biasesGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesBackwardMatrices(biasesGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesBackwardMatrices(biasesGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_BiasesCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_BiasesCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesCacheBackwardCompatible(ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (outputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_BiasesCacheBackwardScalars(ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Biases may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: ML2_BiasesCacheBackwardMatrices(ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesCacheBackwardCompatibleScalars(ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_ScalarsInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_BiasesCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesCacheBackwardCompatibleScalars(inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(inputGradient.data.device == outputGradient.data.device);
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesCacheBackwardScalars(inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesCacheBackwardScalars(inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_BiasesCacheBackwardCompatibleMatrices(ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_MatricesInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_BiasesCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesCacheBackwardCompatibleMatrices(inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(inputGradient.data.device == outputGradient.data.device);
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesCacheBackwardMatrices(inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesCacheBackwardMatrices(inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(biases.data.device == biasesGradient.data.device);
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesGradientDescentOptimize(biases, biasesGradient, learningRate); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesGradientDescentOptimize(biases, biasesGradient, learningRate); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, average.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == biasesGradient.data.device &&
        biases.data.device == average.data.device
    );
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesMomentumOptimize(biases, biasesGradient, learningRate, decayRate, average); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesMomentumOptimize(biases, biasesGradient, learningRate, decayRate, average); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == biasesGradient.data.device &&
        biases.data.device == squareSum.data.device
    );
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesAdagradOptimize(biases, biasesGradient, learningRate, squareSum); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesAdagradOptimize(biases, biasesGradient, learningRate, squareSum); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == biasesGradient.data.device &&
        biases.data.device == squareAverage.data.device
    );
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesRMSPropOptimize(biases, biasesGradient, learningRate, decayRate, squareAverage); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesRMSPropOptimize(biases, biasesGradient, learningRate, decayRate, squareAverage); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_BiasesInfoSame(biases.info, biasesGradient.info) &&
        ML2_BiasesInfoSame(biases.info, average.info) &&
        ML2_BiasesInfoSame(biases.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        biases.data.device == biasesGradient.data.device &&
        biases.data.device == average.data.device &&
        biases.data.device == squareAverage.data.device
    );
    ML2_Device device = biases.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_BiasesAdamOptimize(biases, biasesGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_BiasesAdamOptimize(biases, biasesGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_BiasesSizeof(ML2_LayerBiases biases) {
    return biases.info.inputs * sizeof(ML2_Scalar);
}

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

ML2_FN ML2_LayerInfo ML2_Activation(ML2_ActivationType type) {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {type}};
}

ML2_FN ML2_LayerInfo ML2_ReLU() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeReLU}};
}

ML2_FN ML2_LayerInfo ML2_Sigmoid() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeSigmoid}};
}

ML2_FN ML2_LayerInfo ML2_Sin() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeSin}};
}

ML2_FN ML2_LayerInfo ML2_Cos() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeCos}};
}

ML2_FN ML2_LayerInfo ML2_Tanh() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeTanh}};
}

ML2_FN ML2_LayerInfo ML2_LeakyReLU() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeLeakyReLU}};
}

ML2_FN ML2_LayerInfo ML2_SiLU() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeSiLU}};
}

ML2_FN ML2_LayerInfoActivation ML2_LayerInfoAsActivation(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeActivation);
    return info.as.activation;
}

ML2_FN ML2_LayerInfoActivation ML2_LayerAsActivation(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeActivation);
    return layer.as.activation;
}

ML2_FN const char *ML2_ActivationNameOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return "ReLU";
        case ML2_ActivationTypeSigmoid: return "Sigmoid";
        case ML2_ActivationTypeSin: return "Sin";
        case ML2_ActivationTypeCos: return "Cos";
        case ML2_ActivationTypeTanh: return "Tanh";
        case ML2_ActivationTypeLeakyReLU: return "LeakyReLU";
        case ML2_ActivationTypeSiLU: return "SiLU";
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN void ML2_ActivationInfoPrint(ML2_LayerInfoActivation info, ML2_Int indent) {
    printf(ML2_INDENT("Activation(\"%s\")\n", indent), ML2_ActivationNameOf(info.type));
}

ML2_FN bool ML2_ActivationInfoSame(ML2_LayerInfoActivation a, ML2_LayerInfoActivation b) {
    return a.type == b.type;
}

ML2_FN ML2_ScalarActivationForward *ML2_ActivationForwardOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return ML2_ScalarReLUForward;
        case ML2_ActivationTypeSigmoid: return ML2_ScalarSigmoidForward;
        case ML2_ActivationTypeSin: return ML2_ScalarSinForward;
        case ML2_ActivationTypeCos: return ML2_ScalarCosForward;
        case ML2_ActivationTypeTanh: return ML2_ScalarTanhForward;
        case ML2_ActivationTypeLeakyReLU: return ML2_ScalarLeakyReLUForward;
        case ML2_ActivationTypeSiLU: return ML2_ScalarSiLUForward;
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN ML2_ScalarActivationBackwardInput *ML2_ActivationBackwardInputOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return ML2_ScalarReLUBackwardInput;
        case ML2_ActivationTypeSigmoid: return nullptr;
        case ML2_ActivationTypeSin: return ML2_ScalarSinBackwardInput;
        case ML2_ActivationTypeCos: return ML2_ScalarCosBackwardInput;
        case ML2_ActivationTypeTanh: return nullptr;
        case ML2_ActivationTypeLeakyReLU: return ML2_ScalarLeakyReLUBackwardInput;
        case ML2_ActivationTypeSiLU: return nullptr;
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN ML2_ScalarActivationBackwardOutput *ML2_ActivationBackwardOutputOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return nullptr;
        case ML2_ActivationTypeSigmoid: return ML2_ScalarSigmoidBackwardOutput;
        case ML2_ActivationTypeSin: return nullptr;
        case ML2_ActivationTypeCos: return nullptr;
        case ML2_ActivationTypeTanh: return ML2_ScalarTanhBackwardOutput;
        case ML2_ActivationTypeLeakyReLU: return nullptr;
        case ML2_ActivationTypeSiLU: return nullptr;
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN ML2_ScalarActivationBackwardInputOutput *ML2_ActivationBackwardInputOutputOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return nullptr;
        case ML2_ActivationTypeSigmoid: return nullptr;
        case ML2_ActivationTypeSin: return nullptr;
        case ML2_ActivationTypeCos: return nullptr;
        case ML2_ActivationTypeTanh: return nullptr;
        case ML2_ActivationTypeLeakyReLU: return nullptr;
        case ML2_ActivationTypeSiLU: return ML2_ScalarSiLUBackwardInputOutput;
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN bool ML2_ActivationForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_LayerCacheInfoSame(input, output);
}

ML2_FN void ML2_ActivationForward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_ActivationForwardScalars(activation, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(output)); break;
        case ML2_LayerCacheTypeVectors: ML2_ActivationForwardVectors(activation, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsVectors(output)); break;
        case ML2_LayerCacheTypeMatrices: ML2_ActivationForwardMatrices(activation, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationForwardCompatibleScalars(ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output) {
    return ML2_ScalarsInfoSame(input, output);
}

ML2_FN void ML2_ActivationForwardScalars(ML2_LayerInfoActivation activation, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatibleScalars(input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationForwardScalars(activation, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationForwardScalars(activation, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationForwardCompatibleVectors(ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors output) {
    return ML2_VectorsInfoSame(input, output);
}

ML2_FN void ML2_ActivationForwardVectors(ML2_LayerInfoActivation activation, ML2_LayerCacheVectors input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatibleVectors(input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationForwardVectors(activation, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationForwardVectors(activation, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationForwardCompatibleMatrices(ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    return ML2_MatricesInfoSame(input, output);
}

ML2_FN void ML2_ActivationForwardMatrices(ML2_LayerInfoActivation activation, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatibleMatrices(input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationForwardMatrices(activation, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationForwardMatrices(activation, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) &&
           ML2_LayerCacheInfoSame(input, output) &&
           ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (activation.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationCacheBackwardOutput(activation.type, inputGradient, output, outputGradient); break;
        case ML2_ActivationTypeSin: ML2_ActivationCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeCos: ML2_ActivationCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeTanh: ML2_ActivationCacheBackwardOutput(activation.type, inputGradient, output, outputGradient); break;
        case ML2_ActivationTypeLeakyReLU: ML2_ActivationCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeSiLU: ML2_ActivationCacheBackwardInputOutput(activation.type, input, inputGradient, output, outputGradient); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationIsBackwardInput(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return true;
        case ML2_ActivationTypeSigmoid: return false; // Sigmoid backward takes the output
        case ML2_ActivationTypeSin: return true;
        case ML2_ActivationTypeCos: return true;
        case ML2_ActivationTypeTanh: return false; // Tanh backward takes the output
        case ML2_ActivationTypeLeakyReLU: return true;
        case ML2_ActivationTypeSiLU: return false; // SiLU takes both the input and the output
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_ActivationIsBackwardInput(type) &&
           ML2_LayerCacheInfoSame(input, inputGradient) &&
           ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatible(type, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_ActivationCacheBackwardInputScalars(type, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_ActivationCacheBackwardInputVectors(type, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsVectors(inputGradient), ML2_LayerCacheAsVectors(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_ActivationCacheBackwardInputMatrices(type, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputCompatibleScalars(ML2_ActivationType type, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_ActivationIsBackwardInput(type) &&
           ML2_ScalarsInfoSame(input, inputGradient) &&
           ML2_ScalarsInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatibleScalars(type, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardInputScalars(type, input, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardInputScalars(type, input, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputCompatibleVectors(ML2_ActivationType type, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_ActivationIsBackwardInput(type) &&
           ML2_VectorsInfoSame(input, inputGradient) &&
           ML2_VectorsInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatibleVectors(type, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardInputVectors(type, input, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardInputVectors(type, input, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputCompatibleMatrices(ML2_ActivationType type, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_ActivationIsBackwardInput(type) &&
           ML2_MatricesInfoSame(input, inputGradient) &&
           ML2_MatricesInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputCompatibleMatrices(type, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardInputMatrices(type, input, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardInputMatrices(type, input, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationIsBackwardOutput(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return false; // ReLU backward takes the input
        case ML2_ActivationTypeSigmoid: return true;
        case ML2_ActivationTypeSin: return false; // Sin backward takes the input
        case ML2_ActivationTypeCos: return false; // Cos backward takes the input
        case ML2_ActivationTypeTanh: return true;
        case ML2_ActivationTypeLeakyReLU: return false; // LeakyReLU backward takes the input
        case ML2_ActivationTypeSiLU: return false;      // SiLU takes both the input and the output
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_ActivationIsBackwardOutput(type) &&
           ML2_LayerCacheInfoSame(inputGradient, output) &&
           ML2_LayerCacheInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardOutput(ML2_ActivationType type, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatible(type, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_ActivationCacheBackwardOutputScalars(type, ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(output), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_ActivationCacheBackwardOutputVectors(type, ML2_LayerCacheAsVectors(inputGradient), ML2_LayerCacheAsVectors(output), ML2_LayerCacheAsVectors(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_ActivationCacheBackwardOutputMatrices(type, ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(output), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardOutputCompatibleScalars(ML2_ActivationType type, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars output, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_ActivationIsBackwardOutput(type) &&
           ML2_ScalarsInfoSame(inputGradient, output) &&
           ML2_ScalarsInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatibleScalars(type, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == output.data.device &&
        inputGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardOutputScalars(type, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardOutputScalars(type, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardOutputCompatibleVectors(ML2_ActivationType type, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors output, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_ActivationIsBackwardOutput(type) &&
           ML2_VectorsInfoSame(inputGradient, output) &&
           ML2_VectorsInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatibleVectors(type, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == output.data.device &&
        inputGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardOutputVectors(type, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardOutputVectors(type, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardOutputCompatibleMatrices(ML2_ActivationType type, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices output, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_ActivationIsBackwardOutput(type) &&
           ML2_MatricesInfoSame(inputGradient, output) &&
           ML2_MatricesInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardOutputCompatibleMatrices(type, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == output.data.device &&
        inputGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardOutputMatrices(type, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardOutputMatrices(type, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationIsBackwardInputOutput(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return false;      // ReLU backward takes the input
        case ML2_ActivationTypeSigmoid: return false;   // Sigmoid backward takes the output
        case ML2_ActivationTypeSin: return false;       // Sin backward takes the input
        case ML2_ActivationTypeCos: return false;       // Cos backward takes the input
        case ML2_ActivationTypeTanh: return false;      // Tanh backward takes the output
        case ML2_ActivationTypeLeakyReLU: return false; // LeakyReLU backward takes the input
        case ML2_ActivationTypeSiLU: return true;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_ActivationIsBackwardInputOutput(type) &&
           ML2_LayerCacheInfoSame(input, inputGradient) &&
           ML2_LayerCacheInfoSame(input, output) &&
           ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputOutput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatible(type, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_ActivationCacheBackwardInputOutputScalars(type, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(output), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_ActivationCacheBackwardInputOutputVectors(type, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsVectors(inputGradient), ML2_LayerCacheAsVectors(output), ML2_LayerCacheAsVectors(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_ActivationCacheBackwardInputOutputMatrices(type, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(output), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatibleScalars(ML2_ActivationType type, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars output, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_ActivationIsBackwardInputOutput(type) &&
           ML2_ScalarsInfoSame(input, inputGradient) &&
           ML2_ScalarsInfoSame(input, output) &&
           ML2_ScalarsInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputOutputScalars(ML2_ActivationType type, ML2_LayerCacheScalars input, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatibleScalars(type, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == output.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardInputOutputScalars(type, input, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardInputOutputScalars(type, input, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatibleVectors(ML2_ActivationType type, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors output, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_ActivationIsBackwardInputOutput(type) &&
           ML2_VectorsInfoSame(input, inputGradient) &&
           ML2_VectorsInfoSame(input, output) &&
           ML2_VectorsInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputOutputVectors(ML2_ActivationType type, ML2_LayerCacheVectors input, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatibleVectors(type, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == output.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardInputOutputVectors(type, input, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardInputOutputVectors(type, input, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardInputOutputCompatibleMatrices(ML2_ActivationType type, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices output, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_ActivationIsBackwardInputOutput(type) &&
           ML2_MatricesInfoSame(input, inputGradient) &&
           ML2_MatricesInfoSame(input, output) &&
           ML2_MatricesInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackwardInputOutputMatrices(ML2_ActivationType type, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardInputOutputCompatibleMatrices(type, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == output.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ActivationCacheBackwardInputOutputMatrices(type, input, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ActivationCacheBackwardInputOutputMatrices(type, input, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerActivation ⬆️

// ML2_LayerSoftmax ⬇️

ML2_FN ML2_LayerInfo ML2_Softmax() {
    return (ML2_LayerInfo){ML2_LayerTypeSoftmax, {}};
}

ML2_FN void ML2_SoftmaxInfoPrint(ML2_Int indent) {
    printf(ML2_INDENT("Softmax()\n", indent));
}

ML2_FN bool ML2_SoftmaxForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_LayerCacheInfoSame(input, output);
}

ML2_FN void ML2_SoftmaxForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_SoftmaxForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_SoftmaxForwardScalars(ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(output)); break;
        case ML2_LayerCacheTypeVectors: ML2_SoftmaxForwardVectors(ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsVectors(output)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Softmax may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_SoftmaxForwardCompatibleScalars(ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output) {
    return ML2_ScalarsInfoSame(input, output);
}

ML2_FN void ML2_SoftmaxForwardScalars(ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_SoftmaxForwardCompatibleScalars(input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_SoftmaxForwardScalars(input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_SoftmaxForwardScalars(input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_SoftmaxForwardCompatibleVectors(ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoVectors output) {
    return ML2_VectorsInfoSame(input, output);
}

ML2_FN void ML2_SoftmaxForwardVectors(ML2_LayerCacheVectors input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_SoftmaxForwardCompatibleVectors(input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_SoftmaxForwardVectors(input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_SoftmaxForwardVectors(input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_SoftmaxCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(inputGradient, output) && ML2_LayerCacheInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_SoftmaxCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_SoftmaxCacheBackwardCompatible(ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_SoftmaxCacheBackwardScalars(ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(output), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_SoftmaxCacheBackwardVectors(ML2_LayerCacheAsVectors(inputGradient), ML2_LayerCacheAsVectors(output), ML2_LayerCacheAsVectors(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Softmax may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_SoftmaxCacheBackwardCompatibleScalars(ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars output, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_ScalarsInfoSame(inputGradient, output) && ML2_ScalarsInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_SoftmaxCacheBackwardScalars(ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars output, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_SoftmaxCacheBackwardCompatibleScalars(inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == output.data.device &&
        inputGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_SoftmaxCacheBackwardScalars(inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_SoftmaxCacheBackwardScalars(inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_SoftmaxCacheBackwardCompatibleVectors(ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoVectors output, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_VectorsInfoSame(inputGradient, output) && ML2_VectorsInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_SoftmaxCacheBackwardVectors(ML2_LayerCacheVectors inputGradient, ML2_LayerCacheVectors output, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_SoftmaxCacheBackwardCompatibleVectors(inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        inputGradient.data.device == output.data.device &&
        inputGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_SoftmaxCacheBackwardVectors(inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_SoftmaxCacheBackwardVectors(inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerSoftmax ⬆️

// ML2_LayerLinear ⬇️

ML2_FN ML2_LayerInfo ML2_Linear(ML2_Int inputs, ML2_Int outputs) {
    return (ML2_LayerInfo){ML2_LayerTypeLinear, .as.linear = {outputs, inputs}};
}

ML2_FN ML2_LayerLinear ML2_LinearNew(ML2_Device device, ML2_LayerInfoLinear info) {
    ML2_LayerLinear linear = {
        info,
        ML2_DeviceDataNew(device, (info.outputs * info.inputs + info.outputs) * sizeof(ML2_Scalar)),
    };
    return linear;
}

ML2_FN void ML2_LinearDestroy(ML2_LayerLinear *linear) {
    ML2_DeviceDataDestroy(&linear->data);
    *linear = (ML2_LayerLinear){};
}

ML2_FN ML2_LayerInfoLinear ML2_LayerInfoAsLinear(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeLinear);
    return info.as.linear;
}

ML2_FN ML2_LayerLinear ML2_LayerAsLinear(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeLinear);
    return layer.as.linear;
}

ML2_FN ML2_LayerInfoWeights ML2_LinearInfoWeights(ML2_LayerInfoLinear linear) {
    return ML2_LayerInfoAsWeights(ML2_Weights(linear.inputs, linear.outputs));
}

ML2_FN ML2_LayerInfoBiases ML2_LinearInfoBiases(ML2_LayerInfoLinear linear) {
    return ML2_LayerInfoAsBiases(ML2_Biases(linear.outputs));
}

ML2_FN void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, ML2_Int indent) {
    printf(ML2_INDENT("Linear(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_FN bool ML2_LinearInfoForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input) {
    return ML2_WeightsInfoForwardCompatible(ML2_LinearInfoWeights(linear), input);
}

ML2_FN ML2_LayerCacheInfo ML2_LinearInfoForward(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_LinearInfoForwardCompatible(linear, input));
    return ML2_WeightsInfoForward(ML2_LinearInfoWeights(linear), input);
}

ML2_FN bool ML2_LinearInfoSame(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b) {
    return ML2_WeightsInfoSame(ML2_LinearInfoWeights(a), ML2_LinearInfoWeights(b));
}

ML2_FN ML2_Scalar *ML2_LinearWeightsAt(ML2_LayerLinear linear, ML2_Int o, ML2_Int i) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_LinearWeightsAt(linear, o, i);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_LinearWeightsAt(linear, o, i);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar *ML2_LinearBiasesAt(ML2_LayerLinear linear, ML2_Int o) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_LinearBiasesAt(linear, o);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_LinearBiasesAt(linear, o);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_LinearWeights(linear);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_LinearWeights(linear);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_LinearBiases(linear);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_LinearBiases(linear);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearPrint(ML2_LayerLinear linear, ML2_Int indent) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: ML2_CPU_LinearPrint(linear, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearPrint(linear, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearClear(ML2_LayerLinear linear) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: ML2_CPU_LinearClear(linear); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearClear(linear); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearCopy(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: ML2_CPU_LinearRand(linear, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearRand(linear, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearXavierInit(ML2_LayerLinear linear) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: ML2_CPU_LinearXavierInit(linear); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearXavierInit(linear); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearHeInit(ML2_LayerLinear linear) {
    switch (linear.data.device) {
        case ML2_DeviceCPU: ML2_CPU_LinearHeInit(linear); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearHeInit(linear); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearCopyHostToDevice(ML2_LayerLinear destDevice, ML2_LayerLinear srcHost) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    ML2_Device device = destDevice.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearCopyDeviceToHost(ML2_LayerLinear destHost, ML2_LayerLinear srcDevice) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    ML2_Device device = srcDevice.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_WeightsForwardCompatible(ML2_LinearInfoWeights(linear), input, output);
}

ML2_FN void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_LinearForwardCompatible(linear.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_LinearForwardScalars(linear, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(output)); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Linear may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Linear may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LinearForwardCompatibleScalars(ML2_LayerInfoLinear linear, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars output) {
    return ML2_WeightsForwardCompatibleScalars(ML2_LinearInfoWeights(linear), input, output);
}

ML2_FN void ML2_LinearForwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_LinearForwardCompatibleScalars(linear.info, input.info, output.info));
    ML2_HARD_ASSERT(
        linear.data.device == input.data.device &&
        linear.data.device == output.data.device
    );
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearForwardScalars(linear, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearForwardScalars(linear, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_LinearBackwardCompatible(ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsBackwardCompatible(ML2_LinearInfoWeights(linearGradient), input, outputGradient);
}

ML2_FN void ML2_LinearBackward(ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearBackwardCompatible(linearGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_LinearBackwardScalars(linearGradient, ML2_LayerCacheAsScalars(input), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Linear may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Linear may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LinearBackwardCompatibleScalars(ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfoScalars input, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_WeightsBackwardCompatibleScalars(ML2_LinearInfoWeights(linearGradient), input, outputGradient);
}

ML2_FN void ML2_LinearBackwardScalars(ML2_LayerLinear linearGradient, ML2_LayerCacheScalars input, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearBackwardCompatibleScalars(linearGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        linearGradient.data.device == input.data.device &&
        linearGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = linearGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearBackwardScalars(linearGradient, input, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearBackwardScalars(linearGradient, input, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_LinearCacheBackwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsCacheBackwardCompatible(ML2_LinearInfoWeights(linear), inputGradient, outputGradient);
}

ML2_FN void ML2_LinearCacheBackward(ML2_LayerLinear linear, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearCacheBackwardCompatible(linear.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_LinearCacheBackwardScalars(linear, ML2_LayerCacheAsScalars(inputGradient), ML2_LayerCacheAsScalars(outputGradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Linear may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> Linear may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LinearCacheBackwardCompatibleScalars(ML2_LayerInfoLinear linear, ML2_LayerCacheInfoScalars inputGradient, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_WeightsCacheBackwardCompatibleScalars(ML2_LinearInfoWeights(linear), inputGradient, outputGradient);
}

ML2_FN void ML2_LinearCacheBackwardScalars(ML2_LayerLinear linear, ML2_LayerCacheScalars inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearCacheBackwardCompatibleScalars(linear.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        linear.data.device == inputGradient.data.device &&
        linear.data.device == outputGradient.data.device
    );
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearCacheBackwardScalars(linear, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearCacheBackwardScalars(linear, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(linear.data.device == linearGradient.data.device);
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearGradientDescentOptimize(linear, linearGradient, learningRate); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearGradientDescentOptimize(linear, linearGradient, learningRate); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average) {
    ML2_HARD_ASSERT(
        ML2_LinearInfoSame(linear.info, linearGradient.info) &&
        ML2_LinearInfoSame(linear.info, average.info)
    );
    ML2_HARD_ASSERT(
        linear.data.device == linearGradient.data.device &&
        linear.data.device == average.data.device
    );
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearMomentumOptimize(linear, linearGradient, learningRate, decayRate, average); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearMomentumOptimize(linear, linearGradient, learningRate, decayRate, average); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum) {
    ML2_HARD_ASSERT(
        ML2_LinearInfoSame(linear.info, linearGradient.info) &&
        ML2_LinearInfoSame(linear.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        linear.data.device == linearGradient.data.device &&
        linear.data.device == squareSum.data.device
    );
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearAdagradOptimize(linear, linearGradient, learningRate, squareSum); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearAdagradOptimize(linear, linearGradient, learningRate, squareSum); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage) {
    ML2_HARD_ASSERT(
        ML2_LinearInfoSame(linear.info, linearGradient.info) &&
        ML2_LinearInfoSame(linear.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        linear.data.device == linearGradient.data.device &&
        linear.data.device == squareAverage.data.device
    );
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearRMSPropOptimize(linear, linearGradient, learningRate, decayRate, squareAverage); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearRMSPropOptimize(linear, linearGradient, learningRate, decayRate, squareAverage); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_LinearInfoSame(linear.info, linearGradient.info) &&
        ML2_LinearInfoSame(linear.info, average.info) &&
        ML2_LinearInfoSame(linear.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        linear.data.device == linearGradient.data.device &&
        linear.data.device == average.data.device &&
        linear.data.device == squareAverage.data.device
    );
    ML2_Device device = linear.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_LinearAdamOptimize(linear, linearGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_LinearAdamOptimize(linear, linearGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_LinearSizeof(ML2_LayerLinear linear) {
    return (linear.info.outputs * linear.info.inputs + linear.info.outputs) * sizeof(ML2_Scalar);
}

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_FN ML2_LayerInfo ML2_Filters(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypeFilters, .as.filters = {height, width, inputs, outputs, paddingHeight, paddingWidth, strideHeight, strideWidth}};
}

ML2_FN ML2_LayerFilters ML2_FiltersNew(ML2_Device device, ML2_LayerInfoFilters info) {
    ML2_LayerFilters filters = {
        info,
        ML2_DeviceDataNew(device, info.height * info.width * info.inputs * info.outputs * sizeof(ML2_Scalar)),
    };
    return filters;
}

ML2_FN void ML2_FiltersDestroy(ML2_LayerFilters *filters) {
    ML2_DeviceDataDestroy(&filters->data);
    *filters = (ML2_LayerFilters){};
}

ML2_FN ML2_LayerInfoFilters ML2_LayerInfoAsFilters(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeFilters);
    return info.as.filters;
}

ML2_FN ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeFilters);
    return layer.as.filters;
}

ML2_FN void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, ML2_Int indent) {
    printf(ML2_INDENT("Filters(%dx%dx%dx%d)\n", indent), info.height, info.width, info.inputs, info.outputs);
}

ML2_FN ML2_Int ML2_FiltersInfoOutputHeight(ML2_LayerInfoFilters filters, ML2_Int inputHeight) {
    return (inputHeight + 2 * filters.paddingHeight - filters.height) / filters.strideHeight + 1;
}

ML2_FN ML2_Int ML2_FiltersInfoOutputWidth(ML2_LayerInfoFilters filters, ML2_Int inputWidth) {
    return (inputWidth + 2 * filters.paddingWidth - filters.width) / filters.strideWidth + 1;
}

ML2_FN bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_FiltersInfoOutputHeight(filters, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_FiltersInfoOutputWidth(filters, inputMatrices.vectors);
            return filters.inputs == inputMatrices.scalars &&
                   outputHeight > 0 &&
                   outputWidth > 0;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FiltersInfoForwardCompatible(filters, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_FiltersInfoOutputHeight(filters, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_FiltersInfoOutputWidth(filters, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = {inputMatrices.samples, outputHeight, outputWidth, filters.outputs}};
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b) {
    return a.height == b.height &&
           a.width == b.width &&
           a.inputs == b.inputs &&
           a.outputs == b.outputs &&
           a.paddingHeight == b.paddingHeight &&
           a.paddingWidth == b.paddingWidth &&
           a.strideHeight == b.strideHeight &&
           a.strideWidth == b.strideWidth;
}

ML2_FN ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o) {
    switch (filters.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_FiltersAt(filters, y, x, i, o);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_FiltersAt(filters, y, x, i, o);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent) {
    switch (filters.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersPrint(filters, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersPrint(filters, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersClear(ML2_LayerFilters filters) {
    switch (filters.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersClear(filters); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersClear(filters); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersCopy(ML2_LayerFilters dest, ML2_LayerFilters src) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high) {
    switch (filters.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersRand(filters, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersRand(filters, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersXavierInit(ML2_LayerFilters filters) {
    switch (filters.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersXavierInit(filters); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersXavierInit(filters); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersHeInit(ML2_LayerFilters filters) {
    switch (filters.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersHeInit(filters); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersHeInit(filters); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersCopyHostToDevice(ML2_LayerFilters destDevice, ML2_LayerFilters srcHost) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersCopyDeviceToHost(ML2_LayerFilters destHost, ML2_LayerFilters srcDevice) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != output.type) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: return ML2_FiltersForwardCompatibleMatrices(filters, ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsMatrices(output));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FiltersForwardCompatible(filters.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: ML2_FiltersForwardMatrices(filters, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersForwardCompatibleMatrices(ML2_LayerInfoFilters filters, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    ML2_Int outputHeight = ML2_FiltersInfoOutputHeight(filters, input.matrices);
    ML2_Int outputWidth = ML2_FiltersInfoOutputWidth(filters, input.vectors);
    return input.scalars == filters.inputs &&
           output.samples == input.samples &&
           output.matrices == outputHeight &&
           output.vectors == outputWidth &&
           output.scalars == filters.outputs;
}

ML2_FN void ML2_FiltersForwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_FiltersForwardCompatibleMatrices(filters.info, input.info, output.info));
    ML2_HARD_ASSERT(
        filters.data.device == input.data.device &&
        filters.data.device == output.data.device
    );
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersForwardMatrices(filters, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersForwardMatrices(filters, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FiltersBackwardCompatible(ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersForwardCompatible(filtersGradient, input, outputGradient);
}

ML2_FN void ML2_FiltersBackward(ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersBackwardCompatible(filtersGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: ML2_FiltersBackwardMatrices(filtersGradient, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersBackwardCompatibleMatrices(ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_FiltersForwardCompatibleMatrices(filtersGradient, input, outputGradient);
}

ML2_FN void ML2_FiltersBackwardMatrices(ML2_LayerFilters filtersGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersBackwardCompatibleMatrices(filtersGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        filtersGradient.data.device == input.data.device &&
        filtersGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = filtersGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersBackwardMatrices(filtersGradient, input, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersBackwardMatrices(filtersGradient, input, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FiltersCacheBackwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersForwardCompatible(filters, inputGradient, outputGradient);
}

ML2_FN void ML2_FiltersCacheBackward(ML2_LayerFilters filters, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersCacheBackwardCompatible(filters.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Filters may or may not be a feature in the future"); break;
        case ML2_LayerCacheTypeMatrices: ML2_FiltersCacheBackwardMatrices(filters, ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersCacheBackwardCompatibleMatrices(ML2_LayerInfoFilters filters, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_FiltersForwardCompatibleMatrices(filters, inputGradient, outputGradient);
}

ML2_FN void ML2_FiltersCacheBackwardMatrices(ML2_LayerFilters filters, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersCacheBackwardCompatibleMatrices(filters.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        filters.data.device == inputGradient.data.device &&
        filters.data.device == outputGradient.data.device
    );
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersCacheBackwardMatrices(filters, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersCacheBackwardMatrices(filters, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(filters.data.device == filtersGradient.data.device);
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersGradientDescentOptimize(filters, filtersGradient, learningRate); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersGradientDescentOptimize(filters, filtersGradient, learningRate); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, average.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == filtersGradient.data.device &&
        filters.data.device == average.data.device
    );
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersMomentumOptimize(filters, filtersGradient, learningRate, decayRate, average); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersMomentumOptimize(filters, filtersGradient, learningRate, decayRate, average); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == filtersGradient.data.device &&
        filters.data.device == squareSum.data.device
    );
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersAdagradOptimize(filters, filtersGradient, learningRate, squareSum); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersAdagradOptimize(filters, filtersGradient, learningRate, squareSum); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == filtersGradient.data.device &&
        filters.data.device == squareAverage.data.device
    );
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersRMSPropOptimize(filters, filtersGradient, learningRate, decayRate, squareAverage); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersRMSPropOptimize(filters, filtersGradient, learningRate, decayRate, squareAverage); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_FiltersInfoSame(filters.info, filtersGradient.info) &&
        ML2_FiltersInfoSame(filters.info, average.info) &&
        ML2_FiltersInfoSame(filters.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        filters.data.device == filtersGradient.data.device &&
        filters.data.device == average.data.device &&
        filters.data.device == squareAverage.data.device
    );
    ML2_Device device = filters.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FiltersAdamOptimize(filters, filtersGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FiltersAdamOptimize(filters, filtersGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_FiltersSizeof(ML2_LayerFilters filters) {
    return filters.info.height * filters.info.width * filters.info.inputs * filters.info.outputs * sizeof(ML2_Scalar);
}

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_FN ML2_LayerInfo ML2_Conv(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypeConv, .as.conv = {height, width, inputs, outputs, paddingHeight, paddingWidth, strideHeight, strideWidth}};
}

ML2_FN ML2_LayerConv ML2_ConvNew(ML2_Device device, ML2_LayerInfoConv info) {
    ML2_LayerConv conv = {
        info,
        ML2_DeviceDataNew(device, (info.height * info.width * info.inputs * info.outputs + info.outputs) * sizeof(ML2_Scalar)),
    };
    return conv;
}

ML2_FN void ML2_ConvDestroy(ML2_LayerConv *conv) {
    ML2_DeviceDataDestroy(&conv->data);
    *conv = (ML2_LayerConv){};
}

ML2_FN ML2_LayerInfoConv ML2_LayerInfoAsConv(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeConv);
    return info.as.conv;
}

ML2_FN ML2_LayerConv ML2_LayerAsConv(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeConv);
    return layer.as.conv;
}

ML2_FN ML2_LayerInfoFilters ML2_ConvInfoFilters(ML2_LayerInfoConv conv) {
    return ML2_LayerInfoAsFilters(ML2_Filters(conv.inputs, conv.outputs, conv.height, conv.width, conv.paddingHeight, conv.paddingWidth, conv.strideHeight, conv.strideWidth));
}

ML2_FN ML2_LayerInfoBiases ML2_ConvInfoBiases(ML2_LayerInfoConv conv) {
    return ML2_LayerInfoAsBiases(ML2_Biases(conv.outputs));
}

ML2_FN void ML2_ConvInfoPrint(ML2_LayerInfoConv info, ML2_Int indent) {
    printf(ML2_INDENT("Conv(%dx%dx%dx%d)\n", indent), info.height, info.width, info.inputs, info.outputs);
}

ML2_FN bool ML2_ConvInfoForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input) {
    return ML2_FiltersInfoForwardCompatible(ML2_ConvInfoFilters(conv), input);
}

ML2_FN ML2_LayerCacheInfo ML2_ConvInfoForward(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_ConvInfoForwardCompatible(conv, input));
    return ML2_FiltersInfoForward(ML2_ConvInfoFilters(conv), input);
}

ML2_FN bool ML2_ConvInfoSame(ML2_LayerInfoConv a, ML2_LayerInfoConv b) {
    return ML2_FiltersInfoSame(ML2_ConvInfoFilters(a), ML2_ConvInfoFilters(b));
}

ML2_FN ML2_Scalar *ML2_ConvFiltersAt(ML2_LayerConv conv, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_ConvFiltersAt(conv, y, x, i, o);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ConvFiltersAt(conv, y, x, i, o);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar *ML2_ConvBiasesAt(ML2_LayerConv conv, ML2_Int o) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_ConvBiasesAt(conv, o);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ConvBiasesAt(conv, o);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_ConvFilters(conv);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ConvFilters(conv);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_ConvBiases(conv);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ConvBiases(conv);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvPrint(ML2_LayerConv conv, ML2_Int indent) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ConvPrint(conv, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvPrint(conv, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvClear(ML2_LayerConv conv) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ConvClear(conv); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvClear(conv); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvCopy(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvSum(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ConvRand(conv, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvRand(conv, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvXavierInit(ML2_LayerConv conv) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ConvXavierInit(conv); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvXavierInit(conv); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvHeInit(ML2_LayerConv conv) {
    switch (conv.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ConvHeInit(conv); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvHeInit(conv); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvCopyHostToDevice(ML2_LayerConv destDevice, ML2_LayerConv srcHost) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    ML2_Device device = destDevice.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvCopyDeviceToHost(ML2_LayerConv destHost, ML2_LayerConv srcDevice) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    ML2_Device device = srcDevice.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_FiltersForwardCompatible(ML2_ConvInfoFilters(conv), input, output);
}

ML2_FN void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ConvForwardCompatible(conv.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Conv may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_ConvForwardMatrices(conv, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ConvForwardCompatibleMatrices(ML2_LayerInfoConv conv, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    return ML2_FiltersForwardCompatibleMatrices(ML2_ConvInfoFilters(conv), input, output);
}

ML2_FN void ML2_ConvForwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_ConvForwardCompatibleMatrices(conv.info, input.info, output.info));
    ML2_HARD_ASSERT(
        conv.data.device == input.data.device &&
        conv.data.device == output.data.device
    );
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvForwardMatrices(conv, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvForwardMatrices(conv, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ConvBackwardCompatible(ML2_LayerInfoConv convGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersBackwardCompatible(ML2_ConvInfoFilters(convGradient), input, outputGradient);
}

ML2_FN void ML2_ConvBackward(ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvBackwardCompatible(convGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Conv may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_ConvBackwardMatrices(convGradient, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ConvBackwardCompatibleMatrices(ML2_LayerInfoConv convGradient, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_FiltersBackwardCompatibleMatrices(ML2_ConvInfoFilters(convGradient), input, outputGradient);
}

ML2_FN void ML2_ConvBackwardMatrices(ML2_LayerConv convGradient, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvBackwardCompatibleMatrices(convGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        convGradient.data.device == input.data.device &&
        convGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = convGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvBackwardMatrices(convGradient, input, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvBackwardMatrices(convGradient, input, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_ConvCacheBackwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersCacheBackwardCompatible(ML2_ConvInfoFilters(conv), inputGradient, outputGradient);
}

ML2_FN void ML2_ConvCacheBackward(ML2_LayerConv conv, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvCacheBackwardCompatible(conv.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Conv may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_ConvCacheBackwardMatrices(conv, ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ConvCacheBackwardCompatibleMatrices(ML2_LayerInfoConv conv, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_FiltersCacheBackwardCompatibleMatrices(ML2_ConvInfoFilters(conv), inputGradient, outputGradient);
}

ML2_FN void ML2_ConvCacheBackwardMatrices(ML2_LayerConv conv, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvCacheBackwardCompatibleMatrices(conv.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        conv.data.device == inputGradient.data.device &&
        conv.data.device == outputGradient.data.device
    );
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvCacheBackwardMatrices(conv, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvCacheBackwardMatrices(conv, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(conv.data.device == convGradient.data.device);
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvGradientDescentOptimize(conv, convGradient, learningRate); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvGradientDescentOptimize(conv, convGradient, learningRate); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average) {
    ML2_HARD_ASSERT(
        ML2_ConvInfoSame(conv.info, convGradient.info) &&
        ML2_ConvInfoSame(conv.info, average.info)
    );
    ML2_HARD_ASSERT(
        conv.data.device == convGradient.data.device &&
        conv.data.device == average.data.device
    );
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvMomentumOptimize(conv, convGradient, learningRate, decayRate, average); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvMomentumOptimize(conv, convGradient, learningRate, decayRate, average); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum) {
    ML2_HARD_ASSERT(
        ML2_ConvInfoSame(conv.info, convGradient.info) &&
        ML2_ConvInfoSame(conv.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        conv.data.device == convGradient.data.device &&
        conv.data.device == squareSum.data.device
    );
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvAdagradOptimize(conv, convGradient, learningRate, squareSum); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvAdagradOptimize(conv, convGradient, learningRate, squareSum); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage) {
    ML2_HARD_ASSERT(
        ML2_ConvInfoSame(conv.info, convGradient.info) &&
        ML2_ConvInfoSame(conv.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        conv.data.device == convGradient.data.device &&
        conv.data.device == squareAverage.data.device
    );
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvRMSPropOptimize(conv, convGradient, learningRate, decayRate, squareAverage); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvRMSPropOptimize(conv, convGradient, learningRate, decayRate, squareAverage); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_ConvInfoSame(conv.info, convGradient.info) &&
        ML2_ConvInfoSame(conv.info, average.info) &&
        ML2_ConvInfoSame(conv.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        conv.data.device == convGradient.data.device &&
        conv.data.device == average.data.device &&
        conv.data.device == squareAverage.data.device
    );
    ML2_Device device = conv.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ConvAdamOptimize(conv, convGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ConvAdamOptimize(conv, convGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_ConvSizeof(ML2_LayerConv conv) {
    return (conv.info.height + conv.info.width + conv.info.inputs + conv.info.outputs + conv.info.outputs) * sizeof(ML2_Scalar);
}

// ML2_LayerConv ⬆️

// ML2_LayerFlatten ⬇️

ML2_FN ML2_LayerInfo ML2_Flatten() {
    return (ML2_LayerInfo){ML2_LayerTypeFlatten, {}};
}

ML2_FN void ML2_FlattenInfoPrint(ML2_Int indent) {
    printf(ML2_INDENT("Flatten()\n", indent));
}

ML2_FN bool ML2_FlattenInfoForwardCompatible(ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: return true;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FlattenInfoForwardCompatible(input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeScalars, .as.scalars = {inputMatrices.samples, inputMatrices.matrices * inputMatrices.vectors * inputMatrices.scalars}};
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FlattenForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    // TODO(27/9/2026 3:18:16): this rule might change in the future
    if (output.type != ML2_LayerCacheTypeScalars) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: return ML2_FlattenForwardCompatibleMatrices(ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsScalars(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FlattenForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FlattenForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_FlattenForwardMatrices(ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsScalars(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FlattenForwardCompatibleMatrices(ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoScalars output) {
    return input.samples == output.samples &&
           input.matrices * input.vectors * input.scalars == output.scalars;
}

ML2_FN void ML2_FlattenForwardMatrices(ML2_LayerCacheMatrices input, ML2_LayerCacheScalars output) {
    ML2_HARD_ASSERT(ML2_FlattenForwardCompatibleMatrices(input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FlattenForwardMatrices(input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FlattenForwardMatrices(input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FlattenCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FlattenForwardCompatible(inputGradient, outputGradient);
}

ML2_FN void ML2_FlattenCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FlattenCacheBackwardCompatible(ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Flatten may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_FlattenCacheBackwardMatrices(ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsScalars(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FlattenCacheBackwardCompatibleMatrices(ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoScalars outputGradient) {
    return ML2_FlattenForwardCompatibleMatrices(inputGradient, outputGradient);
}

ML2_FN void ML2_FlattenCacheBackwardMatrices(ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheScalars outputGradient) {
    ML2_HARD_ASSERT(ML2_FlattenCacheBackwardCompatibleMatrices(inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(inputGradient.data.device == outputGradient.data.device);
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FlattenCacheBackwardMatrices(inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FlattenCacheBackwardMatrices(inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerFlatten ⬆️

// ML2_LayerPool ⬇️

ML2_FN const char *ML2_PoolNameOf(ML2_PoolType type) {
    switch (type) {
        case ML2_PoolTypeMax: return "Max";
        case ML2_PoolTypeAverage: return "Average";
        case ML2_PoolTypeSoftmax: return "Softmax";
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

ML2_FN void ML2_PoolInfoPrint(ML2_LayerInfoPool pool, ML2_Int indent) {
    printf(ML2_INDENT("Pool(\"%s\")\n", indent), ML2_PoolNameOf(pool.type));
}

ML2_FN ML2_LayerInfoPool ML2_LayerInfoAsPool(ML2_LayerInfo layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypePool);
    return layer.as.pool;
}

ML2_FN ML2_LayerInfoPool ML2_LayerAsPool(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypePool);
    return layer.as.pool;
}

ML2_FN bool ML2_PoolInfoSame(ML2_LayerInfoPool a, ML2_LayerInfoPool b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ML2_PoolTypeMax: return ML2_PoolMaxInfoSame(ML2_PoolAsMax(a), ML2_PoolAsMax(b));
        case ML2_PoolTypeAverage: return ML2_PoolAverageInfoSame(ML2_PoolAsAverage(a), ML2_PoolAsAverage(b));
        case ML2_PoolTypeSoftmax: return ML2_PoolSoftmaxInfoSame(ML2_PoolAsSoftmax(a), ML2_PoolAsSoftmax(b));
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

ML2_FN bool ML2_PoolInfoForwardCompatible(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input) {
    switch (pool.type) {
        case ML2_PoolTypeMax: return ML2_PoolMaxInfoForwardCompatible(ML2_PoolAsMax(pool), input);
        case ML2_PoolTypeAverage: return ML2_PoolAverageInfoForwardCompatible(ML2_PoolAsAverage(pool), input);
        case ML2_PoolTypeSoftmax: return ML2_PoolSoftmaxInfoForwardCompatible(ML2_PoolAsSoftmax(pool), input);
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolInfoForward(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolInfoForwardCompatible(pool, input));
    switch (pool.type) {
        case ML2_PoolTypeMax: return ML2_PoolMaxInfoForward(ML2_PoolAsMax(pool), input);
        case ML2_PoolTypeAverage: return ML2_PoolAverageInfoForward(ML2_PoolAsAverage(pool), input);
        case ML2_PoolTypeSoftmax: return ML2_PoolSoftmaxInfoForward(ML2_PoolAsSoftmax(pool), input);
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

ML2_FN bool ML2_PoolForwardCompatible(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (pool.type) {
        case ML2_PoolTypeMax: return ML2_PoolMaxForwardCompatible(ML2_PoolAsMax(pool), input, output);
        case ML2_PoolTypeAverage: return ML2_PoolAverageForwardCompatible(ML2_PoolAsAverage(pool), input, output);
        case ML2_PoolTypeSoftmax: return ML2_PoolSoftmaxForwardCompatible(ML2_PoolAsSoftmax(pool), input, output);
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

ML2_FN void ML2_PoolForward(ML2_LayerInfoPool pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (pool.type) {
        case ML2_PoolTypeMax: ML2_PoolMaxForward(ML2_PoolAsMax(pool), input, output); break;
        case ML2_PoolTypeAverage: ML2_PoolAverageForward(ML2_PoolAsAverage(pool), input, output); break;
        case ML2_PoolTypeSoftmax: ML2_PoolSoftmaxForward(ML2_PoolAsSoftmax(pool), input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

ML2_FN bool ML2_PoolCacheBackwardCompatible(ML2_LayerInfoPool pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) &&
           ML2_LayerCacheInfoSame(output, outputGradient) &&
           ML2_PoolForwardCompatible(pool, input, output);
}

ML2_FN void ML2_PoolCacheBackward(ML2_LayerInfoPool pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (pool.type) {
        case ML2_PoolTypeMax: ML2_PoolMaxCacheBackward(ML2_PoolAsMax(pool), input, inputGradient, outputGradient); break;
        case ML2_PoolTypeAverage: ML2_PoolAverageCacheBackward(ML2_PoolAsAverage(pool), inputGradient, outputGradient); break;
        case ML2_PoolTypeSoftmax: ML2_PoolSoftmaxCacheBackward(ML2_PoolAsSoftmax(pool), input, inputGradient, output, outputGradient); break;
        default: ML2_UNREACHABLE("Unknown ML2_PoolType");
    }
}

// ML2_LayerPoolMax ⬇️

ML2_FN ML2_LayerInfo ML2_PoolMax(ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypePool, .as.pool = {ML2_PoolTypeMax, .as.max = {height, width, paddingHeight, paddingWidth, strideHeight, strideWidth}}};
}

ML2_FN ML2_LayerInfoPoolMax ML2_PoolAsMax(ML2_LayerInfoPool pool) {
    ML2_HARD_ASSERT(pool.type == ML2_PoolTypeMax);
    return pool.as.max;
}

ML2_FN bool ML2_PoolMaxInfoSame(ML2_LayerInfoPoolMax a, ML2_LayerInfoPoolMax b) {
    return a.height == b.height &&
           a.width == b.width &&
           a.paddingHeight == b.paddingHeight &&
           a.paddingWidth == b.paddingWidth &&
           a.strideHeight == b.strideHeight && a.strideWidth == b.strideWidth;
}

ML2_FN ML2_Int ML2_PoolMaxInfoOutputHeight(ML2_LayerInfoPoolMax pool, ML2_Int inputHeight) {
    return (inputHeight + 2 * pool.paddingHeight - pool.height) / pool.strideHeight + 1;
}

ML2_FN ML2_Int ML2_PoolMaxInfoOutputWidth(ML2_LayerInfoPoolMax pool, ML2_Int inputWidth) {
    return (inputWidth + 2 * pool.paddingWidth - pool.width) / pool.strideWidth + 1;
}

ML2_FN bool ML2_PoolMaxInfoForwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolMaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolMaxInfoOutputWidth(pool, inputMatrices.vectors);
            return outputHeight > 0 && outputWidth > 0;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolMaxInfoForward(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolMaxInfoForwardCompatible(pool, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolMaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolMaxInfoOutputWidth(pool, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputMatrices.samples, outputHeight, outputWidth, inputMatrices.scalars))};
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolMaxForwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != output.type) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: return ML2_PoolMaxForwardCompatibleMatrices(pool, ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsMatrices(output));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_PoolMaxForward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolMaxForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_PoolMaxForwardMatrices(pool, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolMaxForwardCompatibleMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    ML2_Int outputHeight = ML2_PoolMaxInfoOutputHeight(pool, input.matrices);
    ML2_Int outputWidth = ML2_PoolMaxInfoOutputWidth(pool, input.vectors);
    return output.samples == input.samples &&
           output.matrices == outputHeight &&
           output.vectors == outputWidth &&
           output.scalars == input.scalars;
}

ML2_FN void ML2_PoolMaxForwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_PoolMaxForwardCompatibleMatrices(pool, input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_PoolMaxForwardMatrices(pool, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_PoolMaxForwardMatrices(pool, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_PoolMaxCacheBackwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_PoolMaxForwardCompatible(pool, input, outputGradient);
}

ML2_FN void ML2_PoolMaxCacheBackward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolMaxCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_PoolMaxCacheBackwardMatrices(pool, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolMaxCacheBackwardCompatibleMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_MatricesInfoSame(input, inputGradient) && ML2_PoolMaxForwardCompatibleMatrices(pool, input, outputGradient);
}

ML2_FN void ML2_PoolMaxCacheBackwardMatrices(ML2_LayerInfoPoolMax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolMaxCacheBackwardCompatibleMatrices(pool, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_PoolMaxCacheBackwardMatrices(pool, input, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_PoolMaxCacheBackwardMatrices(pool, input, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerPoolMax ⬆️

// ML2_LayerPoolAverage ⬇️

ML2_FN ML2_LayerInfo ML2_PoolAverage(ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypePool, .as.pool = {ML2_PoolTypeAverage, .as.average = {height, width, paddingHeight, paddingWidth, strideHeight, strideWidth}}};
}

ML2_FN ML2_LayerInfoPoolAverage ML2_PoolAsAverage(ML2_LayerInfoPool pool) {
    ML2_HARD_ASSERT(pool.type == ML2_PoolTypeAverage);
    return pool.as.average;
}

ML2_FN bool ML2_PoolAverageInfoSame(ML2_LayerInfoPoolAverage a, ML2_LayerInfoPoolAverage b) {
    return a.height == b.height &&
           a.width == b.width &&
           a.paddingHeight == b.paddingHeight &&
           a.paddingWidth == b.paddingWidth &&
           a.strideHeight == b.strideHeight &&
           a.strideWidth == b.strideWidth;
}

ML2_FN ML2_Int ML2_PoolAverageInfoOutputHeight(ML2_LayerInfoPoolAverage pool, ML2_Int inputHeight) {
    return (inputHeight + 2 * pool.paddingHeight - pool.height) / pool.strideHeight + 1;
}

ML2_FN ML2_Int ML2_PoolAverageInfoOutputWidth(ML2_LayerInfoPoolAverage pool, ML2_Int inputWidth) {
    return (inputWidth + 2 * pool.paddingWidth - pool.width) / pool.strideWidth + 1;
}

ML2_FN bool ML2_PoolAverageInfoForwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolAverageInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolAverageInfoOutputWidth(pool, inputMatrices.vectors);
            return outputHeight > 0 && outputWidth > 0;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolAverageInfoForward(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolAverageInfoForwardCompatible(pool, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolAverageInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolAverageInfoOutputWidth(pool, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputMatrices.samples, outputHeight, outputWidth, inputMatrices.scalars))};
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolAverageForwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != output.type) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: return ML2_PoolAverageForwardCompatibleMatrices(pool, ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsMatrices(output));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_PoolAverageForward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolAverageForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_PoolAverageForwardMatrices(pool, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolAverageForwardCompatibleMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    ML2_Int outputHeight = ML2_PoolAverageInfoOutputHeight(pool, input.matrices);
    ML2_Int outputWidth = ML2_PoolAverageInfoOutputWidth(pool, input.vectors);
    return output.samples == input.samples &&
           output.matrices == outputHeight &&
           output.vectors == outputWidth &&
           output.scalars == input.scalars;
}

ML2_FN void ML2_PoolAverageForwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_PoolAverageForwardCompatibleMatrices(pool, input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_PoolAverageForwardMatrices(pool, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_PoolAverageForwardMatrices(pool, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_PoolAverageCacheBackwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_PoolAverageForwardCompatible(pool, inputGradient, outputGradient);
}

ML2_FN void ML2_PoolAverageCacheBackward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolAverageCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_PoolAverageCacheBackwardMatrices(pool, ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolAverageCacheBackwardCompatibleMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_PoolAverageForwardCompatibleMatrices(pool, inputGradient, outputGradient);
}

ML2_FN void ML2_PoolAverageCacheBackwardMatrices(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolAverageCacheBackwardCompatibleMatrices(pool, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(inputGradient.data.device == outputGradient.data.device);
    ML2_Device device = inputGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_PoolAverageCacheBackwardMatrices(pool, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_PoolAverageCacheBackwardMatrices(pool, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerPoolAverage ⬆️

// ML2_LayerPoolSoftmax ⬇️

ML2_FN ML2_LayerInfo ML2_PoolSoftmax(ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypePool, .as.pool = {ML2_PoolTypeSoftmax, .as.softmax = {height, width, paddingHeight, paddingWidth, strideHeight, strideWidth}}};
}

ML2_FN ML2_LayerInfoPoolSoftmax ML2_PoolAsSoftmax(ML2_LayerInfoPool pool) {
    ML2_HARD_ASSERT(pool.type == ML2_PoolTypeSoftmax);
    return pool.as.softmax;
}

ML2_FN bool ML2_PoolSoftmaxInfoSame(ML2_LayerInfoPoolSoftmax a, ML2_LayerInfoPoolSoftmax b) {
    return a.height == b.height &&
           a.width == b.width &&
           a.paddingHeight == b.paddingHeight &&
           a.paddingWidth == b.paddingWidth &&
           a.strideHeight == b.strideHeight &&
           a.strideWidth == b.strideWidth;
}

ML2_FN ML2_Int ML2_PoolSoftmaxInfoOutputHeight(ML2_LayerInfoPoolSoftmax pool, ML2_Int inputHeight) {
    return (inputHeight + 2 * pool.paddingHeight - pool.height) / pool.strideHeight + 1;
}

ML2_FN ML2_Int ML2_PoolSoftmaxInfoOutputWidth(ML2_LayerInfoPoolSoftmax pool, ML2_Int inputWidth) {
    return (inputWidth + 2 * pool.paddingWidth - pool.width) / pool.strideWidth + 1;
}

ML2_FN bool ML2_PoolSoftmaxInfoForwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolSoftmaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolSoftmaxInfoOutputWidth(pool, inputMatrices.vectors);
            return outputHeight > 0 && outputWidth > 0;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolSoftmaxInfoForward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxInfoForwardCompatible(pool, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolSoftmaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolSoftmaxInfoOutputWidth(pool, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputMatrices.samples, outputHeight, outputWidth, inputMatrices.scalars))};
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolSoftmaxForwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != output.type) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: return ML2_PoolSoftmaxForwardCompatibleMatrices(pool, ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsMatrices(output));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_PoolSoftmaxForward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_PoolSoftmaxForwardMatrices(pool, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolSoftmaxForwardCompatibleMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices output) {
    ML2_Int outputHeight = ML2_PoolSoftmaxInfoOutputHeight(pool, input.matrices);
    ML2_Int outputWidth = ML2_PoolSoftmaxInfoOutputWidth(pool, input.vectors);
    return output.samples == input.samples &&
           output.matrices == outputHeight &&
           output.vectors == outputWidth &&
           output.scalars == input.scalars;
}

ML2_FN void ML2_PoolSoftmaxForwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxForwardCompatibleMatrices(pool, input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_PoolSoftmaxForwardMatrices(pool, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_PoolSoftmaxForwardMatrices(pool, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_PoolSoftmaxCacheBackwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) &&
           ML2_LayerCacheInfoSame(output, outputGradient) &&
           ML2_PoolSoftmaxForwardCompatible(pool, inputGradient, output);
}

ML2_FN void ML2_PoolSoftmaxCacheBackward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_PoolSoftmaxCacheBackwardMatrices(pool, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsMatrices(output), ML2_LayerCacheAsMatrices(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolSoftmaxCacheBackwardCompatibleMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoMatrices output, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_MatricesInfoSame(input, inputGradient) &&
           ML2_MatricesInfoSame(output, outputGradient) &&
           ML2_PoolSoftmaxForwardCompatibleMatrices(pool, inputGradient, output);
}

ML2_FN void ML2_PoolSoftmaxCacheBackwardMatrices(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheMatrices output, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxCacheBackwardCompatibleMatrices(pool, input.info, inputGradient.info, output.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == output.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_PoolSoftmaxCacheBackwardMatrices(pool, input, inputGradient, output, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_PoolSoftmaxCacheBackwardMatrices(pool, input, inputGradient, output, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerPoolSoftmax ⬆️

// ML2_LayerPool ⬆️

// ML2_LayerFusedWeights ⬇️

ML2_FN ML2_LayerInfo ML2_FusedWeights(ML2_Int weights, ML2_Int inputs, ML2_Int outputs) {
    return (ML2_LayerInfo){ML2_LayerTypeFusedWeights, .as.fusedWeights = {weights, outputs, inputs}};
}

ML2_FN ML2_LayerFusedWeights ML2_FusedWeightsNew(ML2_Device device, ML2_LayerInfoFusedWeights info) {
    ML2_LayerFusedWeights fusedWeights = {
        info,
        ML2_DeviceDataNew(device, info.weights * info.outputs * info.inputs * sizeof(ML2_Scalar)),
    };
    return fusedWeights;
}

ML2_FN void ML2_FusedWeightsDestroy(ML2_LayerFusedWeights *fusedWeights) {
    ML2_DeviceDataDestroy(&fusedWeights->data);
    *fusedWeights = (ML2_LayerFusedWeights){};
}

ML2_FN ML2_LayerInfoFusedWeights ML2_LayerInfoAsFusedWeights(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeFusedWeights);
    return info.as.fusedWeights;
}

ML2_FN ML2_LayerFusedWeights ML2_LayerAsFusedWeights(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeFusedWeights);
    return layer.as.fusedWeights;
}

ML2_FN ML2_LayerInfoWeights ML2_FusedWeightsInfoWeights(ML2_LayerInfoFusedWeights fusedWeights) {
    return ML2_LayerInfoAsWeights(ML2_Weights(fusedWeights.inputs, fusedWeights.outputs));
}

ML2_FN void ML2_FusedWeightsInfoPrint(ML2_LayerInfoFusedWeights info, ML2_Int indent) {
    printf(ML2_INDENT("FusedWeights(%dx%dx%d)\n", indent), info.weights, info.outputs, info.inputs);
}

ML2_FN bool ML2_FusedWeightsInfoForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheInfoVectors inputVectors = ML2_LayerCacheInfoAsVectors(input);
            return fusedWeights.inputs == inputVectors.scalars;
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_FusedWeightsInfoForward(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoForwardCompatible(fusedWeights, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheInfoVectors inputVectors = ML2_LayerCacheInfoAsVectors(input);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputVectors.samples, inputVectors.vectors, fusedWeights.weights, fusedWeights.outputs))};
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FusedWeightsInfoSame(ML2_LayerInfoFusedWeights a, ML2_LayerInfoFusedWeights b) {
    return a.weights == b.weights &&
           a.outputs == b.outputs &&
           a.inputs == b.inputs;
}

ML2_FN ML2_Scalar *ML2_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_FusedWeightsAt(fusedWeights, w, o, i);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_FusedWeightsAt(fusedWeights, w, o, i);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_LayerWeights ML2_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_FusedWeightsWeightsAt(fusedWeights, w);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_FusedWeightsWeightsAt(fusedWeights, w);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsPrint(fusedWeights, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsPrint(fusedWeights, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsClear(fusedWeights); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsClear(fusedWeights); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsCopy(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsRand(fusedWeights, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsRand(fusedWeights, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsXavierInit(fusedWeights); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsXavierInit(fusedWeights); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights) {
    switch (fusedWeights.data.device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsHeInit(fusedWeights); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsHeInit(fusedWeights); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsCopyHostToDevice(ML2_LayerFusedWeights destDevice, ML2_LayerFusedWeights srcHost) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsCopyDeviceToHost(ML2_LayerFusedWeights destHost, ML2_LayerFusedWeights srcDevice) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FusedWeightsForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    // TODO(27/9/2026 3:18:16): this rule might change in the future
    if (output.type != ML2_LayerCacheTypeMatrices) return false;
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: return ML2_FusedWeightsForwardCompatibleVectors(fusedWeights, ML2_LayerCacheInfoAsVectors(input), ML2_LayerCacheInfoAsMatrices(output));
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FusedWeightsForward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FusedWeightsForwardCompatible(fusedWeights.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_FusedWeightsForwardVectors(fusedWeights, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsMatrices(output)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FusedWeightsForwardCompatibleVectors(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoMatrices output) {
    return fusedWeights.inputs == input.scalars &&
           output.samples == input.samples &&
           output.matrices == input.vectors &&
           output.vectors == fusedWeights.weights &&
           output.scalars == fusedWeights.outputs;
}

ML2_FN void ML2_FusedWeightsForwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices output) {
    ML2_HARD_ASSERT(ML2_FusedWeightsForwardCompatibleVectors(fusedWeights.info, input.info, output.info));
    ML2_HARD_ASSERT(
        fusedWeights.data.device == input.data.device &&
        fusedWeights.data.device == output.data.device
    );
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsForwardVectors(fusedWeights, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsForwardVectors(fusedWeights, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FusedWeightsBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_FusedWeightsForwardCompatible(fusedWeightsGradient, input, outputGradient);
}

ML2_FN void ML2_FusedWeightsBackward(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsBackwardCompatible(fusedWeightsGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_FusedWeightsBackwardVectors(fusedWeightsGradient, ML2_LayerCacheAsVectors(input), ML2_LayerCacheAsMatrices(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FusedWeightsBackwardCompatibleVectors(ML2_LayerInfoFusedWeights fusedWeightsGradient, ML2_LayerCacheInfoVectors input, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_FusedWeightsForwardCompatibleVectors(fusedWeightsGradient, input, outputGradient);
}

ML2_FN void ML2_FusedWeightsBackwardVectors(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCacheVectors input, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsBackwardCompatibleVectors(fusedWeightsGradient.info, input.info, outputGradient.info));
    ML2_HARD_ASSERT(
        fusedWeightsGradient.data.device == input.data.device &&
        fusedWeightsGradient.data.device == outputGradient.data.device
    );
    ML2_Device device = fusedWeightsGradient.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsBackwardVectors(fusedWeightsGradient, input, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsBackwardVectors(fusedWeightsGradient, input, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_FusedWeightsCacheBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FusedWeightsForwardCompatible(fusedWeights, inputGradient, outputGradient);
}

ML2_FN void ML2_FusedWeightsCacheBackward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsCacheBackwardCompatible(fusedWeights.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_FusedWeightsCacheBackwardVectors(fusedWeights, ML2_LayerCacheAsVectors(inputGradient), ML2_LayerCacheAsMatrices(outputGradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FusedWeightsCacheBackwardCompatibleVectors(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfoVectors inputGradient, ML2_LayerCacheInfoMatrices outputGradient) {
    return ML2_FusedWeightsForwardCompatibleVectors(fusedWeights, inputGradient, outputGradient);
}

ML2_FN void ML2_FusedWeightsCacheBackwardVectors(ML2_LayerFusedWeights fusedWeights, ML2_LayerCacheVectors inputGradient, ML2_LayerCacheMatrices outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsCacheBackwardCompatibleVectors(fusedWeights.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        fusedWeights.data.device == inputGradient.data.device &&
        fusedWeights.data.device == outputGradient.data.device
    );
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsCacheBackwardVectors(fusedWeights, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsCacheBackwardVectors(fusedWeights, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(fusedWeights.data.device == fusedWeightsGradient.data.device);
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsGradientDescentOptimize(fusedWeights, fusedWeightsGradient, learningRate); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsGradientDescentOptimize(fusedWeights, fusedWeightsGradient, learningRate); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average) {
    ML2_HARD_ASSERT(
        ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info) &&
        ML2_FusedWeightsInfoSame(fusedWeights.info, average.info)
    );
    ML2_HARD_ASSERT(
        fusedWeights.data.device == fusedWeightsGradient.data.device &&
        fusedWeights.data.device == average.data.device
    );
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsMomentumOptimize(fusedWeights, fusedWeightsGradient, learningRate, decayRate, average); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsMomentumOptimize(fusedWeights, fusedWeightsGradient, learningRate, decayRate, average); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum) {
    ML2_HARD_ASSERT(
        ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info) &&
        ML2_FusedWeightsInfoSame(fusedWeights.info, squareSum.info)
    );
    ML2_HARD_ASSERT(
        fusedWeights.data.device == fusedWeightsGradient.data.device &&
        fusedWeights.data.device == squareSum.data.device
    );
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsAdagradOptimize(fusedWeights, fusedWeightsGradient, learningRate, squareSum); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsAdagradOptimize(fusedWeights, fusedWeightsGradient, learningRate, squareSum); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage) {
    ML2_HARD_ASSERT(
        ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info) &&
        ML2_FusedWeightsInfoSame(fusedWeights.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        fusedWeights.data.device == fusedWeightsGradient.data.device &&
        fusedWeights.data.device == squareAverage.data.device
    );
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsRMSPropOptimize(fusedWeights, fusedWeightsGradient, learningRate, decayRate, squareAverage); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsRMSPropOptimize(fusedWeights, fusedWeightsGradient, learningRate, decayRate, squareAverage); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(
        ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info) &&
        ML2_FusedWeightsInfoSame(fusedWeights.info, average.info) &&
        ML2_FusedWeightsInfoSame(fusedWeights.info, squareAverage.info)
    );
    ML2_HARD_ASSERT(
        fusedWeights.data.device == fusedWeightsGradient.data.device &&
        fusedWeights.data.device == average.data.device &&
        fusedWeights.data.device == squareAverage.data.device
    );
    ML2_Device device = fusedWeights.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_FusedWeightsAdamOptimize(fusedWeights, fusedWeightsGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_FusedWeightsAdamOptimize(fusedWeights, fusedWeightsGradient, learningRate, decayRate1, decayRate2, average, squareAverage, decayingWeight1, decayingWeight2); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_FusedWeightsSizeof(ML2_LayerFusedWeights fusedWeights) {
    return fusedWeights.info.weights * fusedWeights.info.outputs * fusedWeights.info.inputs * sizeof(ML2_Scalar);
}

// ML2_LayerFusedWeights ⬆️

// ML2_LayerAttention ⬇️

ML2_FN ML2_LayerInfo ML2_Attention(ML2_Int heads, bool masked) {
    return (ML2_LayerInfo){.type = ML2_LayerTypeAttention, .as.attention = {heads, masked}};
}

ML2_FN ML2_LayerInfoAttention ML2_LayerInfoAsAttention(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeAttention);
    return info.as.attention;
}

ML2_FN ML2_LayerInfoAttention ML2_LayerAsAttention(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeAttention);
    return layer.as.attention;
}

ML2_FN void ML2_AttentionInfoPrint(ML2_LayerInfoAttention info, ML2_Int indent) {
    printf(ML2_INDENT("Attention(%d, %s)\n", indent), info.heads, ML2_BOOL_TO_STR(info.masked));
}

ML2_FN bool ML2_AttentionInfoForwardCompatible(ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            return inputMatrices.vectors == ML2_AttentionVectorsCount;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_AttentionInfoForward(ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_AttentionInfoForwardCompatible(input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            return ML2_Vectors(inputMatrices.samples, inputMatrices.matrices, inputMatrices.scalars);
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_AttentionInfoSame(ML2_LayerInfoAttention a, ML2_LayerInfoAttention b) {
    return a.heads == b.heads &&
           a.masked == b.masked;
}

ML2_FN bool ML2_AttentionForwardCompatible(ML2_LayerInfoAttention attention, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: return ML2_AttentionForwardCompatibleMatrices(attention, ML2_LayerCacheInfoAsMatrices(input), ML2_LayerCacheInfoAsVectors(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_AttentionForward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_AttentionForwardCompatible(attention, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_AttentionForwardMatrices(attention, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsVectors(output)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_AttentionForwardCompatibleMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoVectors output) {
    return input.vectors == ML2_AttentionVectorsCount &&
           output.samples == input.samples &&
           output.vectors == input.matrices &&
           output.scalars == input.scalars &&
           output.scalars % attention.heads == 0;
}

ML2_FN void ML2_AttentionForwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheVectors output) {
    ML2_HARD_ASSERT(ML2_AttentionForwardCompatibleMatrices(attention, input.info, output.info));
    ML2_HARD_ASSERT(input.data.device == output.data.device);
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_AttentionForwardMatrices(attention, input, output); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_AttentionForwardMatrices(attention, input, output); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN bool ML2_AttentionCacheBackwardCompatible(ML2_LayerInfoAttention attention, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_AttentionForwardCompatible(attention, inputGradient, outputGradient);
}

ML2_FN void ML2_AttentionCacheBackward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_AttentionCacheBackwardCompatible(attention, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        case ML2_LayerCacheTypeVectors: ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        case ML2_LayerCacheTypeMatrices: ML2_AttentionCacheBackwardMatrices(attention, ML2_LayerCacheAsMatrices(input), ML2_LayerCacheAsMatrices(inputGradient), ML2_LayerCacheAsVectors(outputGradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_AttentionCacheBackwardCompatibleMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheInfoMatrices input, ML2_LayerCacheInfoMatrices inputGradient, ML2_LayerCacheInfoVectors outputGradient) {
    return ML2_MatricesInfoSame(input, inputGradient) && ML2_AttentionForwardCompatibleMatrices(attention, inputGradient, outputGradient);
}

ML2_FN void ML2_AttentionCacheBackwardMatrices(ML2_LayerInfoAttention attention, ML2_LayerCacheMatrices input, ML2_LayerCacheMatrices inputGradient, ML2_LayerCacheVectors outputGradient) {
    ML2_HARD_ASSERT(ML2_AttentionCacheBackwardCompatibleMatrices(attention, input.info, inputGradient.info, outputGradient.info));
    ML2_HARD_ASSERT(
        input.data.device == inputGradient.data.device &&
        input.data.device == outputGradient.data.device
    );
    ML2_Device device = input.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_AttentionCacheBackwardMatrices(attention, input, inputGradient, outputGradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_AttentionCacheBackwardMatrices(attention, input, inputGradient, outputGradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerAttention ⬆️

ML2_FN ML2_Layer ML2_LayerNew(ML2_Device device, ML2_LayerInfo info) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return (ML2_Layer){info.type, .as.weights = ML2_WeightsNew(device, ML2_LayerInfoAsWeights(info))};
        case ML2_LayerTypeBiases: return (ML2_Layer){info.type, .as.biases = ML2_BiasesNew(device, ML2_LayerInfoAsBiases(info))};
        case ML2_LayerTypeActivation: return (ML2_Layer){info.type, .as.activation = ML2_LayerInfoAsActivation(info)};
        case ML2_LayerTypeSoftmax: return (ML2_Layer){info.type, {}};
        case ML2_LayerTypeLinear: return (ML2_Layer){info.type, .as.linear = ML2_LinearNew(device, ML2_LayerInfoAsLinear(info))};
        case ML2_LayerTypeFilters: return (ML2_Layer){info.type, .as.filters = ML2_FiltersNew(device, ML2_LayerInfoAsFilters(info))};
        case ML2_LayerTypeConv: return (ML2_Layer){info.type, .as.conv = ML2_ConvNew(device, ML2_LayerInfoAsConv(info))};
        case ML2_LayerTypeFlatten: return (ML2_Layer){info.type, {}};
        case ML2_LayerTypePool: return (ML2_Layer){info.type, .as.pool = ML2_LayerInfoAsPool(info)};
        case ML2_LayerTypeFusedWeights: return (ML2_Layer){info.type, .as.fusedWeights = ML2_FusedWeightsNew(device, ML2_LayerInfoAsFusedWeights(info))};
        case ML2_LayerTypeAttention: return (ML2_Layer){info.type, .as.attention = ML2_LayerInfoAsAttention(info)};
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerDestroy(ML2_Layer *layer) {
    switch (layer->type) {
        case ML2_LayerTypeWeights: ML2_WeightsDestroy(&layer->as.weights); break;
        case ML2_LayerTypeBiases: ML2_BiasesDestroy(&layer->as.biases); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearDestroy(&layer->as.linear); break;
        case ML2_LayerTypeFilters: ML2_FiltersDestroy(&layer->as.filters); break;
        case ML2_LayerTypeConv: ML2_ConvDestroy(&layer->as.conv); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsDestroy(&layer->as.fusedWeights); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
    *layer = (ML2_Layer){};
}

ML2_FN ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return (ML2_LayerInfo){layer.type, .as.weights = ML2_LayerAsWeights(layer).info};
        case ML2_LayerTypeBiases: return (ML2_LayerInfo){layer.type, .as.biases = ML2_LayerAsBiases(layer).info};
        case ML2_LayerTypeActivation: return (ML2_LayerInfo){layer.type, .as.activation = ML2_LayerAsActivation(layer)};
        case ML2_LayerTypeSoftmax: return (ML2_LayerInfo){layer.type, {}};
        case ML2_LayerTypeLinear: return (ML2_LayerInfo){layer.type, .as.linear = ML2_LayerAsLinear(layer).info};
        case ML2_LayerTypeFilters: return (ML2_LayerInfo){layer.type, .as.filters = ML2_LayerAsFilters(layer).info};
        case ML2_LayerTypeConv: return (ML2_LayerInfo){layer.type, .as.conv = ML2_LayerAsConv(layer).info};
        case ML2_LayerTypeFlatten: return (ML2_LayerInfo){layer.type, {}};
        case ML2_LayerTypePool: return (ML2_LayerInfo){layer.type, .as.pool = ML2_LayerAsPool(layer)};
        case ML2_LayerTypeFusedWeights: return (ML2_LayerInfo){layer.type, .as.fusedWeights = ML2_LayerAsFusedWeights(layer).info};
        case ML2_LayerTypeAttention: return (ML2_LayerInfo){layer.type, .as.attention = ML2_LayerAsAttention(layer)};
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerInfoPrint(ML2_LayerInfo layer, ML2_Int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsInfoPrint(ML2_LayerInfoAsWeights(layer), indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesInfoPrint(ML2_LayerInfoAsBiases(layer), indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationInfoPrint(ML2_LayerInfoAsActivation(layer), indent); break;
        case ML2_LayerTypeSoftmax: ML2_SoftmaxInfoPrint(indent); break;
        case ML2_LayerTypeLinear: ML2_LinearInfoPrint(ML2_LayerInfoAsLinear(layer), indent); break;
        case ML2_LayerTypeFilters: ML2_FiltersInfoPrint(ML2_LayerInfoAsFilters(layer), indent); break;
        case ML2_LayerTypeConv: ML2_ConvInfoPrint(ML2_LayerInfoAsConv(layer), indent); break;
        case ML2_LayerTypeFlatten: ML2_FlattenInfoPrint(indent); break;
        case ML2_LayerTypePool: ML2_PoolInfoPrint(ML2_LayerInfoAsPool(layer), indent); break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsInfoPrint(ML2_LayerInfoAsFusedWeights(layer), indent); break;
        case ML2_LayerTypeAttention: ML2_AttentionInfoPrint(ML2_LayerInfoAsAttention(layer), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN bool ML2_LayerInfoForwardCompatible(ML2_LayerInfo info, ML2_LayerCacheInfo input) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsInfoForwardCompatible(ML2_LayerInfoAsWeights(info), input);
        case ML2_LayerTypeBiases: return ML2_BiasesInfoForwardCompatible(ML2_LayerInfoAsBiases(info), input);
        case ML2_LayerTypeActivation: return true;
        case ML2_LayerTypeSoftmax: return true;
        case ML2_LayerTypeLinear: return ML2_LinearInfoForwardCompatible(ML2_LayerInfoAsLinear(info), input);
        case ML2_LayerTypeFilters: return ML2_FiltersInfoForwardCompatible(ML2_LayerInfoAsFilters(info), input);
        case ML2_LayerTypeConv: return ML2_ConvInfoForwardCompatible(ML2_LayerInfoAsConv(info), input);
        case ML2_LayerTypeFlatten: return ML2_FlattenInfoForwardCompatible(input);
        case ML2_LayerTypePool: return ML2_PoolInfoForwardCompatible(ML2_LayerInfoAsPool(info), input);
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsInfoForwardCompatible(ML2_LayerInfoAsFusedWeights(info), input);
        case ML2_LayerTypeAttention: return ML2_AttentionInfoForwardCompatible(input);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_LayerInfoForwardCompatible(info, input));
    switch (info.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsInfoForward(ML2_LayerInfoAsWeights(info), input);
        case ML2_LayerTypeBiases: return ML2_BiasesInfoForward(ML2_LayerInfoAsBiases(info), input);
        case ML2_LayerTypeActivation: return input;
        case ML2_LayerTypeSoftmax: return input;
        case ML2_LayerTypeLinear: return ML2_LinearInfoForward(ML2_LayerInfoAsLinear(info), input);
        case ML2_LayerTypeFilters: return ML2_FiltersInfoForward(ML2_LayerInfoAsFilters(info), input);
        case ML2_LayerTypeConv: return ML2_ConvInfoForward(ML2_LayerInfoAsConv(info), input);
        case ML2_LayerTypeFlatten: return ML2_FlattenInfoForward(input);
        case ML2_LayerTypePool: return ML2_PoolInfoForward(ML2_LayerInfoAsPool(info), input);
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsInfoForward(ML2_LayerInfoAsFusedWeights(info), input);
        case ML2_LayerTypeAttention: return ML2_AttentionInfoForward(input);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN bool ML2_LayerInfoSame(ML2_LayerInfo a, ML2_LayerInfo b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsInfoSame(ML2_LayerInfoAsWeights(a), ML2_LayerInfoAsWeights(b));
        case ML2_LayerTypeBiases: return ML2_BiasesInfoSame(ML2_LayerInfoAsBiases(a), ML2_LayerInfoAsBiases(b));
        case ML2_LayerTypeActivation: return ML2_ActivationInfoSame(ML2_LayerInfoAsActivation(a), ML2_LayerInfoAsActivation(b));
        case ML2_LayerTypeSoftmax: return true;
        case ML2_LayerTypeLinear: return ML2_LinearInfoSame(ML2_LayerInfoAsLinear(a), ML2_LayerInfoAsLinear(b));
        case ML2_LayerTypeFilters: return ML2_FiltersInfoSame(ML2_LayerInfoAsFilters(a), ML2_LayerInfoAsFilters(b));
        case ML2_LayerTypeConv: return ML2_ConvInfoSame(ML2_LayerInfoAsConv(a), ML2_LayerInfoAsConv(b));
        case ML2_LayerTypeFlatten: return true;
        case ML2_LayerTypePool: return ML2_PoolInfoSame(ML2_LayerInfoAsPool(a), ML2_LayerInfoAsPool(b));
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsInfoSame(ML2_LayerInfoAsFusedWeights(a), ML2_LayerInfoAsFusedWeights(b));
        case ML2_LayerTypeAttention: return ML2_AttentionInfoSame(ML2_LayerInfoAsAttention(a), ML2_LayerInfoAsAttention(b));
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerPrint(ML2_Layer layer, ML2_Int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsPrint(ML2_LayerAsWeights(layer), indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesPrint(ML2_LayerAsBiases(layer), indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationInfoPrint(ML2_LayerAsActivation(layer), indent); break;
        case ML2_LayerTypeSoftmax: ML2_SoftmaxInfoPrint(indent); break;
        case ML2_LayerTypeLinear: ML2_LinearPrint(ML2_LayerAsLinear(layer), indent); break;
        case ML2_LayerTypeFilters: ML2_FiltersPrint(ML2_LayerAsFilters(layer), indent); break;
        case ML2_LayerTypeConv: ML2_ConvPrint(ML2_LayerAsConv(layer), indent); break;
        case ML2_LayerTypeFlatten: ML2_FlattenInfoPrint(indent); break;
        case ML2_LayerTypePool: ML2_PoolInfoPrint(ML2_LayerAsPool(layer), indent); break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsPrint(ML2_LayerAsFusedWeights(layer), indent); break;
        case ML2_LayerTypeAttention: ML2_AttentionInfoPrint(ML2_LayerAsAttention(layer), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerClear(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsClear(ML2_LayerAsWeights(layer)); break;
        case ML2_LayerTypeBiases: ML2_BiasesClear(ML2_LayerAsBiases(layer)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearClear(ML2_LayerAsLinear(layer)); break;
        case ML2_LayerTypeFilters: ML2_FiltersClear(ML2_LayerAsFilters(layer)); break;
        case ML2_LayerTypeConv: ML2_ConvClear(ML2_LayerAsConv(layer)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsClear(ML2_LayerAsFusedWeights(layer)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerCopy(ML2_Layer dest, ML2_Layer src) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(dest), ML2_LayerAsInfo(src)));
    switch (dest.type) {
        case ML2_LayerTypeWeights: ML2_WeightsCopy(ML2_LayerAsWeights(dest), ML2_LayerAsWeights(src)); break;
        case ML2_LayerTypeBiases: ML2_BiasesCopy(ML2_LayerAsBiases(dest), ML2_LayerAsBiases(src)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearCopy(ML2_LayerAsLinear(dest), ML2_LayerAsLinear(src)); break;
        case ML2_LayerTypeFilters: ML2_FiltersCopy(ML2_LayerAsFilters(dest), ML2_LayerAsFilters(src)); break;
        case ML2_LayerTypeConv: ML2_ConvCopy(ML2_LayerAsConv(dest), ML2_LayerAsConv(src)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsCopy(ML2_LayerAsFusedWeights(dest), ML2_LayerAsFusedWeights(src)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerSum(ML2_Layer dest, ML2_Layer src) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(dest), ML2_LayerAsInfo(src)));
    switch (dest.type) {
        case ML2_LayerTypeWeights: ML2_WeightsSum(ML2_LayerAsWeights(dest), ML2_LayerAsWeights(src)); break;
        case ML2_LayerTypeBiases: ML2_BiasesSum(ML2_LayerAsBiases(dest), ML2_LayerAsBiases(src)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearSum(ML2_LayerAsLinear(dest), ML2_LayerAsLinear(src)); break;
        case ML2_LayerTypeFilters: ML2_FiltersSum(ML2_LayerAsFilters(dest), ML2_LayerAsFilters(src)); break;
        case ML2_LayerTypeConv: ML2_ConvSum(ML2_LayerAsConv(dest), ML2_LayerAsConv(src)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsSum(ML2_LayerAsFusedWeights(dest), ML2_LayerAsFusedWeights(src)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerRand(ML2_Layer layer, ML2_Scalar low, ML2_Scalar high) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsRand(ML2_LayerAsWeights(layer), low, high); break;
        case ML2_LayerTypeBiases: ML2_BiasesRand(ML2_LayerAsBiases(layer), low, high); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearRand(ML2_LayerAsLinear(layer), low, high); break;
        case ML2_LayerTypeFilters: ML2_FiltersRand(ML2_LayerAsFilters(layer), low, high); break;
        case ML2_LayerTypeConv: ML2_ConvRand(ML2_LayerAsConv(layer), low, high); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsRand(ML2_LayerAsFusedWeights(layer), low, high); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerXavierInit(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsXavierInit(ML2_LayerAsWeights(layer)); break;
        case ML2_LayerTypeBiases: ML2_BiasesXavierInit(ML2_LayerAsBiases(layer)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearXavierInit(ML2_LayerAsLinear(layer)); break;
        case ML2_LayerTypeFilters: ML2_FiltersXavierInit(ML2_LayerAsFilters(layer)); break;
        case ML2_LayerTypeConv: ML2_ConvXavierInit(ML2_LayerAsConv(layer)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsXavierInit(ML2_LayerAsFusedWeights(layer)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerHeInit(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsHeInit(ML2_LayerAsWeights(layer)); break;
        case ML2_LayerTypeBiases: ML2_BiasesHeInit(ML2_LayerAsBiases(layer)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearHeInit(ML2_LayerAsLinear(layer)); break;
        case ML2_LayerTypeFilters: ML2_FiltersHeInit(ML2_LayerAsFilters(layer)); break;
        case ML2_LayerTypeConv: ML2_ConvHeInit(ML2_LayerAsConv(layer)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsHeInit(ML2_LayerAsFusedWeights(layer)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerCopyHostToDevice(ML2_Layer destDevice, ML2_Layer srcHost) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(destDevice), ML2_LayerAsInfo(srcHost)));
    switch (destDevice.type) {
        case ML2_LayerTypeWeights: ML2_WeightsCopyHostToDevice(ML2_LayerAsWeights(destDevice), ML2_LayerAsWeights(srcHost)); break;
        case ML2_LayerTypeBiases: ML2_BiasesCopyHostToDevice(ML2_LayerAsBiases(destDevice), ML2_LayerAsBiases(srcHost)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearCopyHostToDevice(ML2_LayerAsLinear(destDevice), ML2_LayerAsLinear(srcHost)); break;
        case ML2_LayerTypeFilters: ML2_FiltersCopyHostToDevice(ML2_LayerAsFilters(destDevice), ML2_LayerAsFilters(srcHost)); break;
        case ML2_LayerTypeConv: ML2_ConvCopyHostToDevice(ML2_LayerAsConv(destDevice), ML2_LayerAsConv(srcHost)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsCopyHostToDevice(ML2_LayerAsFusedWeights(destDevice), ML2_LayerAsFusedWeights(srcHost)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerCopyDeviceToHost(ML2_Layer destHost, ML2_Layer srcDevice) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(destHost), ML2_LayerAsInfo(srcDevice)));
    switch (destHost.type) {
        case ML2_LayerTypeWeights: ML2_WeightsCopyDeviceToHost(ML2_LayerAsWeights(destHost), ML2_LayerAsWeights(srcDevice)); break;
        case ML2_LayerTypeBiases: ML2_BiasesCopyDeviceToHost(ML2_LayerAsBiases(destHost), ML2_LayerAsBiases(srcDevice)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearCopyDeviceToHost(ML2_LayerAsLinear(destHost), ML2_LayerAsLinear(srcDevice)); break;
        case ML2_LayerTypeFilters: ML2_FiltersCopyDeviceToHost(ML2_LayerAsFilters(destHost), ML2_LayerAsFilters(srcDevice)); break;
        case ML2_LayerTypeConv: ML2_ConvCopyDeviceToHost(ML2_LayerAsConv(destHost), ML2_LayerAsConv(srcDevice)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsCopyDeviceToHost(ML2_LayerAsFusedWeights(destHost), ML2_LayerAsFusedWeights(srcDevice)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN bool ML2_LayerForwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsForwardCompatible(ML2_LayerInfoAsWeights(layer), input, output);
        case ML2_LayerTypeBiases: return ML2_BiasesForwardCompatible(ML2_LayerInfoAsBiases(layer), input, output);
        case ML2_LayerTypeActivation: return ML2_ActivationForwardCompatible(input, output);
        case ML2_LayerTypeSoftmax: return ML2_SoftmaxForwardCompatible(input, output);
        case ML2_LayerTypeLinear: return ML2_LinearForwardCompatible(ML2_LayerInfoAsLinear(layer), input, output);
        case ML2_LayerTypeFilters: return ML2_FiltersForwardCompatible(ML2_LayerInfoAsFilters(layer), input, output);
        case ML2_LayerTypeConv: return ML2_ConvForwardCompatible(ML2_LayerInfoAsConv(layer), input, output);
        case ML2_LayerTypeFlatten: return ML2_FlattenForwardCompatible(input, output);
        case ML2_LayerTypePool: return ML2_PoolForwardCompatible(ML2_LayerInfoAsPool(layer), input, output);
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsForwardCompatible(ML2_LayerInfoAsFusedWeights(layer), input, output);
        case ML2_LayerTypeAttention: return ML2_AttentionForwardCompatible(ML2_LayerInfoAsAttention(layer), input, output);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_LayerForwardCompatible(ML2_LayerAsInfo(layer), ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsForward(ML2_LayerAsWeights(layer), input, output); break;
        case ML2_LayerTypeBiases: ML2_BiasesForward(ML2_LayerAsBiases(layer), input, output); break;
        case ML2_LayerTypeActivation: ML2_ActivationForward(ML2_LayerAsActivation(layer), input, output); break;
        case ML2_LayerTypeSoftmax: ML2_SoftmaxForward(input, output); break;
        case ML2_LayerTypeLinear: ML2_LinearForward(ML2_LayerAsLinear(layer), input, output); break;
        case ML2_LayerTypeFilters: ML2_FiltersForward(ML2_LayerAsFilters(layer), input, output); break;
        case ML2_LayerTypeConv: ML2_ConvForward(ML2_LayerAsConv(layer), input, output); break;
        case ML2_LayerTypeFlatten: ML2_FlattenForward(input, output); break;
        case ML2_LayerTypePool: ML2_PoolForward(ML2_LayerAsPool(layer), input, output); break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsForward(ML2_LayerAsFusedWeights(layer), input, output); break;
        case ML2_LayerTypeAttention: ML2_AttentionForward(ML2_LayerAsAttention(layer), input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN bool ML2_LayerBackwardCompatible(ML2_LayerInfo gradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    switch (gradient.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsBackwardCompatible(ML2_LayerInfoAsWeights(gradient), input, outputGradient);
        case ML2_LayerTypeBiases: return ML2_BiasesBackwardCompatible(ML2_LayerInfoAsBiases(gradient), outputGradient);
        case ML2_LayerTypeActivation: return true;
        case ML2_LayerTypeSoftmax: return true;
        case ML2_LayerTypeLinear: return ML2_LinearBackwardCompatible(ML2_LayerInfoAsLinear(gradient), input, outputGradient);
        case ML2_LayerTypeFilters: return ML2_FiltersBackwardCompatible(ML2_LayerInfoAsFilters(gradient), input, outputGradient);
        case ML2_LayerTypeConv: return ML2_ConvBackwardCompatible(ML2_LayerInfoAsConv(gradient), input, outputGradient);
        case ML2_LayerTypeFlatten: return true;
        case ML2_LayerTypePool: return true;
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsBackwardCompatible(ML2_LayerInfoAsFusedWeights(gradient), input, outputGradient);
        case ML2_LayerTypeAttention: return true;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerBackward(ML2_Layer gradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LayerBackwardCompatible(ML2_LayerAsInfo(gradient), ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (gradient.type) {
        case ML2_LayerTypeWeights: ML2_WeightsBackward(ML2_LayerAsWeights(gradient), input, outputGradient); break;
        case ML2_LayerTypeBiases: ML2_BiasesBackward(ML2_LayerAsBiases(gradient), outputGradient); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearBackward(ML2_LayerAsLinear(gradient), input, outputGradient); break;
        case ML2_LayerTypeFilters: ML2_FiltersBackward(ML2_LayerAsFilters(gradient), input, outputGradient); break;
        case ML2_LayerTypeConv: ML2_ConvBackward(ML2_LayerAsConv(gradient), input, outputGradient); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsBackward(ML2_LayerAsFusedWeights(gradient), input, outputGradient); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN bool ML2_LayerCacheBackwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsCacheBackwardCompatible(ML2_LayerInfoAsWeights(layer), inputGradient, outputGradient);
        case ML2_LayerTypeBiases: return ML2_BiasesCacheBackwardCompatible(inputGradient, outputGradient);
        case ML2_LayerTypeActivation: return ML2_ActivationCacheBackwardCompatible(input, inputGradient, output, outputGradient);
        case ML2_LayerTypeSoftmax: return ML2_SoftmaxCacheBackwardCompatible(inputGradient, output, outputGradient);
        case ML2_LayerTypeLinear: return ML2_LinearCacheBackwardCompatible(ML2_LayerInfoAsLinear(layer), inputGradient, outputGradient);
        case ML2_LayerTypeFilters: return ML2_FiltersCacheBackwardCompatible(ML2_LayerInfoAsFilters(layer), inputGradient, outputGradient);
        case ML2_LayerTypeConv: return ML2_ConvCacheBackwardCompatible(ML2_LayerInfoAsConv(layer), inputGradient, outputGradient);
        case ML2_LayerTypeFlatten: return ML2_FlattenCacheBackwardCompatible(inputGradient, outputGradient);
        case ML2_LayerTypePool: return ML2_PoolCacheBackwardCompatible(ML2_LayerInfoAsPool(layer), input, inputGradient, output, outputGradient);
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsCacheBackwardCompatible(ML2_LayerInfoAsFusedWeights(layer), inputGradient, outputGradient);
        case ML2_LayerTypeAttention: return ML2_AttentionCacheBackwardCompatible(ML2_LayerInfoAsAttention(layer), input, inputGradient, outputGradient);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerCacheBackward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LayerCacheBackwardCompatible(ML2_LayerAsInfo(layer), ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsCacheBackward(ML2_LayerAsWeights(layer), inputGradient, outputGradient); break;
        case ML2_LayerTypeBiases: ML2_BiasesCacheBackward(inputGradient, outputGradient); break;
        case ML2_LayerTypeActivation: ML2_ActivationCacheBackward(ML2_LayerAsActivation(layer), input, inputGradient, output, outputGradient); break;
        case ML2_LayerTypeSoftmax: ML2_SoftmaxCacheBackward(inputGradient, output, outputGradient); break;
        case ML2_LayerTypeLinear: ML2_LinearCacheBackward(ML2_LayerAsLinear(layer), inputGradient, outputGradient); break;
        case ML2_LayerTypeFilters: ML2_FiltersCacheBackward(ML2_LayerAsFilters(layer), inputGradient, outputGradient); break;
        case ML2_LayerTypeConv: ML2_ConvCacheBackward(ML2_LayerAsConv(layer), inputGradient, outputGradient); break;
        case ML2_LayerTypeFlatten: ML2_FlattenCacheBackward(inputGradient, outputGradient); break;
        case ML2_LayerTypePool: ML2_PoolCacheBackward(ML2_LayerAsPool(layer), input, inputGradient, output, outputGradient); break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsCacheBackward(ML2_LayerAsFusedWeights(layer), inputGradient, outputGradient); break;
        case ML2_LayerTypeAttention: ML2_AttentionCacheBackward(ML2_LayerAsAttention(layer), input, inputGradient, outputGradient); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerGradientDescentOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(gradient)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsGradientDescentOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate); break;
        case ML2_LayerTypeBiases: ML2_BiasesGradientDescentOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearGradientDescentOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate); break;
        case ML2_LayerTypeFilters: ML2_FiltersGradientDescentOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate); break;
        case ML2_LayerTypeConv: ML2_ConvGradientDescentOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsGradientDescentOptimize(ML2_LayerAsFusedWeights(layer), ML2_LayerAsFusedWeights(gradient), learningRate); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerMomentumOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer average) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(gradient)));
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(average)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsMomentumOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, decayRate, ML2_LayerAsWeights(average)); break;
        case ML2_LayerTypeBiases: ML2_BiasesMomentumOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, decayRate, ML2_LayerAsBiases(average)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearMomentumOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, decayRate, ML2_LayerAsLinear(average)); break;
        case ML2_LayerTypeFilters: ML2_FiltersMomentumOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, decayRate, ML2_LayerAsFilters(average)); break;
        case ML2_LayerTypeConv: ML2_ConvMomentumOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, decayRate, ML2_LayerAsConv(average)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsMomentumOptimize(ML2_LayerAsFusedWeights(layer), ML2_LayerAsFusedWeights(gradient), learningRate, decayRate, ML2_LayerAsFusedWeights(average)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerAdagradOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Layer squareSum) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(gradient)));
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(squareSum)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsAdagradOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, ML2_LayerAsWeights(squareSum)); break;
        case ML2_LayerTypeBiases: ML2_BiasesAdagradOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, ML2_LayerAsBiases(squareSum)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearAdagradOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, ML2_LayerAsLinear(squareSum)); break;
        case ML2_LayerTypeFilters: ML2_FiltersAdagradOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, ML2_LayerAsFilters(squareSum)); break;
        case ML2_LayerTypeConv: ML2_ConvAdagradOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, ML2_LayerAsConv(squareSum)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsAdagradOptimize(ML2_LayerAsFusedWeights(layer), ML2_LayerAsFusedWeights(gradient), learningRate, ML2_LayerAsFusedWeights(squareSum)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerRMSPropOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer squareAverage) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(gradient)));
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(squareAverage)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsRMSPropOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, decayRate, ML2_LayerAsWeights(squareAverage)); break;
        case ML2_LayerTypeBiases: ML2_BiasesRMSPropOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, decayRate, ML2_LayerAsBiases(squareAverage)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearRMSPropOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, decayRate, ML2_LayerAsLinear(squareAverage)); break;
        case ML2_LayerTypeFilters: ML2_FiltersRMSPropOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, decayRate, ML2_LayerAsFilters(squareAverage)); break;
        case ML2_LayerTypeConv: ML2_ConvRMSPropOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, decayRate, ML2_LayerAsConv(squareAverage)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsRMSPropOptimize(ML2_LayerAsFusedWeights(layer), ML2_LayerAsFusedWeights(gradient), learningRate, decayRate, ML2_LayerAsFusedWeights(squareAverage)); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerAdamOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_Layer average, ML2_Layer squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(gradient)));
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(average)));
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(layer), ML2_LayerAsInfo(squareAverage)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsAdamOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsWeights(average), ML2_LayerAsWeights(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeBiases: ML2_BiasesAdamOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsBiases(average), ML2_LayerAsBiases(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeSoftmax: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearAdamOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsLinear(average), ML2_LayerAsLinear(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeFilters: ML2_FiltersAdamOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsFilters(average), ML2_LayerAsFilters(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeConv: ML2_ConvAdamOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsConv(average), ML2_LayerAsConv(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        case ML2_LayerTypePool: /* No action needed */ break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsAdamOptimize(ML2_LayerAsFusedWeights(layer), ML2_LayerAsFusedWeights(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsFusedWeights(average), ML2_LayerAsFusedWeights(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeAttention: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN ML2_Size ML2_LayerSizeof(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsSizeof(ML2_LayerAsWeights(layer));
        case ML2_LayerTypeBiases: return ML2_BiasesSizeof(ML2_LayerAsBiases(layer));
        case ML2_LayerTypeActivation: return 0;
        case ML2_LayerTypeSoftmax: return 0;
        case ML2_LayerTypeLinear: return ML2_LinearSizeof(ML2_LayerAsLinear(layer));
        case ML2_LayerTypeFilters: return ML2_FiltersSizeof(ML2_LayerAsFilters(layer));
        case ML2_LayerTypeConv: return ML2_ConvSizeof(ML2_LayerAsConv(layer));
        case ML2_LayerTypeFlatten: return 0;
        case ML2_LayerTypePool: return 0;
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsSizeof(ML2_LayerAsFusedWeights(layer));
        case ML2_LayerTypeAttention: return 0;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

// ML2_LayerCacheScalars ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Scalars(ML2_Int samples, ML2_Int scalars) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeScalars, .as.scalars = {samples, scalars}};
}

ML2_FN ML2_LayerCacheScalars ML2_ScalarsNew(ML2_Device device, ML2_LayerCacheInfoScalars info) {
    ML2_LayerCacheScalars scalars = {
        info,
        ML2_DeviceDataNew(device, info.samples * info.scalars * sizeof(ML2_Scalar)),
    };
    return scalars;
}

ML2_FN void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars) {
    ML2_DeviceDataDestroy(&scalars->data);
    *scalars = (ML2_LayerCacheScalars){};
}

ML2_FN ML2_LayerCacheInfoScalars ML2_LayerCacheInfoAsScalars(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeScalars);
    return info.as.scalars;
}

ML2_FN ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeScalars);
    return cache.as.scalars;
}

ML2_FN void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, ML2_Int indent) {
    printf(ML2_INDENT("Scalars(%dx%d)\n", indent), info.samples, info.scalars);
}

ML2_FN bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b) {
    return a.samples == b.samples && a.scalars == b.scalars;
}

ML2_FN ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int s, ML2_Int c) {
    switch (scalars.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_ScalarsAt(scalars, s, c);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ScalarsAt(scalars, s, c);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent) {
    switch (scalars.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsPrint(scalars, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsPrint(scalars, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsClear(ML2_LayerCacheScalars scalars) {
    switch (scalars.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsClear(scalars); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsClear(scalars); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high) {
    switch (scalars.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsRand(scalars, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsRand(scalars, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsXavierInit(ML2_LayerCacheScalars scalars) {
    switch (scalars.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsXavierInit(scalars); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsXavierInit(scalars); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsHeInit(ML2_LayerCacheScalars scalars) {
    switch (scalars.data.device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsHeInit(scalars); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsHeInit(scalars); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsCopyHostToDevice(ML2_LayerCacheScalars destDevice, ML2_LayerCacheScalars srcHost) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsCopyDeviceToHost(ML2_LayerCacheScalars destHost, ML2_LayerCacheScalars srcDevice) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_ScalarsSizeof(ML2_LayerCacheScalars scalars) {
    return scalars.info.samples * scalars.info.scalars * sizeof(ML2_Scalar);
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expected.info, predicted.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_ScalarsLossForwardSquareAverage(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ScalarsLossForwardSquareAverage(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsLossBackwardSquareAverage(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsLossBackwardSquareAverage(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_ScalarsLossForwardCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ScalarsLossForwardCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsLossBackwardCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsLossBackwardCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_ScalarsLossForwardSoftmaxCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ScalarsLossForwardSoftmaxCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsLossBackwardSoftmaxCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsLossBackwardSoftmaxCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_ScalarsLossForwardBinaryCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_ScalarsLossForwardBinaryCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(
        ML2_ScalarsInfoSame(predicted.info, expected.info) &&
        ML2_ScalarsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_ScalarsLossBackwardBinaryCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_ScalarsLossBackwardBinaryCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheVectors ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Vectors(ML2_Int samples, ML2_Int vectors, ML2_Int scalars) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeVectors, .as.vectors = {samples, vectors, scalars}};
}

ML2_FN ML2_LayerCacheVectors ML2_VectorsNew(ML2_Device device, ML2_LayerCacheInfoVectors info) {
    ML2_LayerCacheVectors vectors = {
        info,
        ML2_DeviceDataNew(device, info.samples * info.vectors * info.scalars * sizeof(ML2_Scalar)),
    };
    return vectors;
}

ML2_FN void ML2_VectorsDestroy(ML2_LayerCacheVectors *vectors) {
    ML2_DeviceDataDestroy(&vectors->data);
    *vectors = (ML2_LayerCacheVectors){};
}

ML2_FN ML2_LayerCacheInfoVectors ML2_LayerCacheInfoAsVectors(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeVectors);
    return info.as.vectors;
}

ML2_FN ML2_LayerCacheVectors ML2_LayerCacheAsVectors(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeVectors);
    return cache.as.vectors;
}

ML2_FN void ML2_VectorsInfoPrint(ML2_LayerCacheInfoVectors info, ML2_Int indent) {
    printf(ML2_INDENT("Vectors(%dx%dx%d)\n", indent), info.samples, info.vectors, info.scalars);
}

ML2_FN bool ML2_VectorsInfoSame(ML2_LayerCacheInfoVectors a, ML2_LayerCacheInfoVectors b) {
    return a.samples == b.samples &&
           a.vectors == b.vectors &&
           a.scalars == b.scalars;
}

ML2_FN ML2_Scalar *ML2_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int s, ML2_Int v, ML2_Int c) {
    switch (vectors.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_VectorsAt(vectors, s, v, c);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_VectorsAt(vectors, s, v, c);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent) {
    switch (vectors.data.device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsPrint(vectors, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsPrint(vectors, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsClear(ML2_LayerCacheVectors vectors) {
    switch (vectors.data.device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsClear(vectors); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsClear(vectors); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high) {
    switch (vectors.data.device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsRand(vectors, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsRand(vectors, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsXavierInit(ML2_LayerCacheVectors vectors) {
    switch (vectors.data.device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsXavierInit(vectors); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsXavierInit(vectors); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsHeInit(ML2_LayerCacheVectors vectors) {
    switch (vectors.data.device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsHeInit(vectors); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsHeInit(vectors); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsCopyHostToDevice(ML2_LayerCacheVectors destDevice, ML2_LayerCacheVectors srcHost) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsCopyDeviceToHost(ML2_LayerCacheVectors destHost, ML2_LayerCacheVectors srcDevice) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_VectorsSizeof(ML2_LayerCacheVectors vectors) {
    return vectors.info.samples * vectors.info.vectors * vectors.info.scalars * sizeof(ML2_Scalar);
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_VectorsLossForwardSquareAverage(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_VectorsLossForwardSquareAverage(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsLossBackwardSquareAverage(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsLossBackwardSquareAverage(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_VectorsLossForwardCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_VectorsLossForwardCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsLossBackwardCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsLossBackwardCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_VectorsLossForwardSoftmaxCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_VectorsLossForwardSoftmaxCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsLossBackwardSoftmaxCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsLossBackwardSoftmaxCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_VectorsLossForwardBinaryCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_VectorsLossForwardBinaryCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(
        ML2_VectorsInfoSame(predicted.info, expected.info) &&
        ML2_VectorsInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_VectorsLossBackwardBinaryCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_VectorsLossBackwardBinaryCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerCacheVectors ⬆️

// ML2_LayerCacheMatrices ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Matrices(ML2_Int samples, ML2_Int height, ML2_Int width, ML2_Int channels) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = {samples, height, width, channels}};
}

ML2_FN ML2_LayerCacheMatrices ML2_MatricesNew(ML2_Device device, ML2_LayerCacheInfoMatrices info) {
    ML2_LayerCacheMatrices matrices = {
        info,
        ML2_DeviceDataNew(device, info.samples * info.matrices * info.vectors * info.scalars * sizeof(ML2_Scalar)),
    };
    return matrices;
}

ML2_FN void ML2_MatricesDestroy(ML2_LayerCacheMatrices *matrices) {
    ML2_DeviceDataDestroy(&matrices->data);
    *matrices = (ML2_LayerCacheMatrices){};
}

ML2_FN ML2_LayerCacheInfoMatrices ML2_LayerCacheInfoAsMatrices(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeMatrices);
    return info.as.matrices;
}

ML2_FN ML2_LayerCacheMatrices ML2_LayerCacheAsMatrices(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeMatrices);
    return cache.as.matrices;
}

ML2_FN void ML2_MatricesInfoPrint(ML2_LayerCacheInfoMatrices info, ML2_Int indent) {
    printf(ML2_INDENT("Matrices(%dx%dx%dx%d)\n", indent), info.samples, info.matrices, info.vectors, info.scalars);
}

ML2_FN bool ML2_MatricesInfoSame(ML2_LayerCacheInfoMatrices a, ML2_LayerCacheInfoMatrices b) {
    return a.samples == b.samples &&
           a.matrices == b.matrices &&
           a.vectors == b.vectors &&
           a.scalars == b.scalars;
}

ML2_FN ML2_Scalar *ML2_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c) {
    switch (matrices.data.device) {
        case ML2_DeviceCPU: return ML2_CPU_MatricesAt(matrices, s, m, v, c);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_MatricesAt(matrices, s, m, v, c);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent) {
    switch (matrices.data.device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesPrint(matrices, indent); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesPrint(matrices, indent); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesClear(ML2_LayerCacheMatrices matrices) {
    switch (matrices.data.device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesClear(matrices); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesClear(matrices); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesCopy(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesCopy(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(dest.info, src.info));
    ML2_HARD_ASSERT(dest.data.device == src.data.device);
    ML2_Device device = dest.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesSum(dest, src); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesSum(dest, src); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high) {
    switch (matrices.data.device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesRand(matrices, low, high); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesRand(matrices, low, high); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesXavierInit(ML2_LayerCacheMatrices matrices) {
    switch (matrices.data.device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesXavierInit(matrices); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesXavierInit(matrices); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesHeInit(ML2_LayerCacheMatrices matrices) {
    switch (matrices.data.device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesHeInit(matrices); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesHeInit(matrices); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesCopyHostToDevice(ML2_LayerCacheMatrices destDevice, ML2_LayerCacheMatrices srcHost) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(destDevice.info, srcHost.info));
    ML2_HARD_ASSERT(
        destDevice.data.device != ML2_DeviceCPU &&
        srcHost.data.device == ML2_DeviceCPU
    );
    switch (destDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesCopyHostToDevice(destDevice, srcHost); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesCopyDeviceToHost(ML2_LayerCacheMatrices destHost, ML2_LayerCacheMatrices srcDevice) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(destHost.info, srcDevice.info));
    ML2_HARD_ASSERT(
        destHost.data.device == ML2_DeviceCPU &&
        srcDevice.data.device != ML2_DeviceCPU
    );
    switch (srcDevice.data.device) {
        case ML2_DeviceCPU: ML2_UNREACHABLE("ML2_DeviceCPU is the host");
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesCopyDeviceToHost(destHost, srcDevice); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Size ML2_MatricesSizeof(ML2_LayerCacheMatrices matrices) {
    return matrices.info.samples * matrices.info.matrices * matrices.info.vectors * matrices.info.scalars * sizeof(ML2_Scalar);
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_MatricesLossForwardSquareAverage(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_MatricesLossForwardSquareAverage(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_HARD_ASSERT(
        ML2_MatricesInfoSame(predicted.info, expected.info) &&
        ML2_MatricesInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesLossBackwardSquareAverage(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesLossBackwardSquareAverage(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_MatricesLossForwardCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_MatricesLossForwardCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_HARD_ASSERT(
        ML2_MatricesInfoSame(predicted.info, expected.info) &&
        ML2_MatricesInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesLossBackwardCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesLossBackwardCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_MatricesLossForwardSoftmaxCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_MatricesLossForwardSoftmaxCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_HARD_ASSERT(
        ML2_MatricesInfoSame(predicted.info, expected.info) &&
        ML2_MatricesInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesLossBackwardSoftmaxCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesLossBackwardSoftmaxCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(predicted.data.device == expected.data.device);
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: return ML2_CPU_MatricesLossForwardBinaryCrossEntropy(predicted, expected);
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: return ML2_CUDA_MatricesLossForwardBinaryCrossEntropy(predicted, expected);
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

ML2_FN void ML2_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_HARD_ASSERT(
        ML2_MatricesInfoSame(predicted.info, expected.info) &&
        ML2_MatricesInfoSame(predicted.info, gradient.info)
    );
    ML2_HARD_ASSERT(
        predicted.data.device == expected.data.device &&
        predicted.data.device == gradient.data.device
    );
    ML2_Device device = predicted.data.device;
    switch (device) {
        case ML2_DeviceCPU: ML2_CPU_MatricesLossBackwardBinaryCrossEntropy(predicted, expected, gradient); break;
#ifdef ML2_CUDA_ENABLE
        case ML2_DeviceCUDA: ML2_CUDA_MatricesLossBackwardBinaryCrossEntropy(predicted, expected, gradient); break;
#endif // ML2_CUDA_ENABLE
        default: ML2_UNREACHABLE("Unknown ML2_Device");
    }
}

// ML2_LayerCacheMatrices ⬆️

ML2_FN ML2_LayerCache ML2_LayerCacheNew(ML2_Device device, ML2_LayerCacheInfo info) {
    switch (info.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCache){info.type, .as.scalars = ML2_ScalarsNew(device, ML2_LayerCacheInfoAsScalars(info))};
        case ML2_LayerCacheTypeVectors: return (ML2_LayerCache){info.type, .as.vectors = ML2_VectorsNew(device, ML2_LayerCacheInfoAsVectors(info))};
        case ML2_LayerCacheTypeMatrices: return (ML2_LayerCache){info.type, .as.matrices = ML2_MatricesNew(device, ML2_LayerCacheInfoAsMatrices(info))};
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// TODO(30/8/2026 18:57:30): i dont like how its not using As functions, but its also too much work to have AsPtr functions
ML2_FN void ML2_LayerCacheDestroy(ML2_LayerCache *cache) {
    switch (cache->type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsDestroy(&cache->as.scalars); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsDestroy(&cache->as.vectors); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesDestroy(&cache->as.matrices); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
    *cache = (ML2_LayerCache){};
}

ML2_FN ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCacheInfo){cache.type, .as.scalars = ML2_LayerCacheAsScalars(cache).info};
        case ML2_LayerCacheTypeVectors: return (ML2_LayerCacheInfo){cache.type, .as.vectors = ML2_LayerCacheAsVectors(cache).info};
        case ML2_LayerCacheTypeMatrices: return (ML2_LayerCacheInfo){cache.type, .as.matrices = ML2_LayerCacheAsMatrices(cache).info};
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, ML2_Int indent) {
    switch (layer.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsInfoPrint(ML2_LayerCacheInfoAsScalars(layer), indent); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsInfoPrint(ML2_LayerCacheInfoAsVectors(layer), indent); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesInfoPrint(ML2_LayerCacheInfoAsMatrices(layer), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsInfoSame(ML2_LayerCacheInfoAsScalars(a), ML2_LayerCacheInfoAsScalars(b));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsInfoSame(ML2_LayerCacheInfoAsVectors(a), ML2_LayerCacheInfoAsVectors(b));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesInfoSame(ML2_LayerCacheInfoAsMatrices(a), ML2_LayerCacheInfoAsMatrices(b));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCachePrint(ML2_LayerCache cache, ML2_Int indent) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsPrint(ML2_LayerCacheAsScalars(cache), indent); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsPrint(ML2_LayerCacheAsVectors(cache), indent); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesPrint(ML2_LayerCacheAsMatrices(cache), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheClear(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsClear(ML2_LayerCacheAsScalars(cache)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsClear(ML2_LayerCacheAsVectors(cache)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesClear(ML2_LayerCacheAsMatrices(cache)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(dest), ML2_LayerCacheAsInfo(src)));
    switch (dest.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsCopy(ML2_LayerCacheAsScalars(dest), ML2_LayerCacheAsScalars(src)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsCopy(ML2_LayerCacheAsVectors(dest), ML2_LayerCacheAsVectors(src)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesCopy(ML2_LayerCacheAsMatrices(dest), ML2_LayerCacheAsMatrices(src)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheSum(ML2_LayerCache dest, ML2_LayerCache src) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(dest), ML2_LayerCacheAsInfo(src)));
    switch (dest.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsSum(ML2_LayerCacheAsScalars(dest), ML2_LayerCacheAsScalars(src)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsSum(ML2_LayerCacheAsVectors(dest), ML2_LayerCacheAsVectors(src)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesSum(ML2_LayerCacheAsMatrices(dest), ML2_LayerCacheAsMatrices(src)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheRand(ML2_LayerCache cache, ML2_Scalar low, ML2_Scalar high) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsRand(ML2_LayerCacheAsScalars(cache), low, high); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsRand(ML2_LayerCacheAsVectors(cache), low, high); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesRand(ML2_LayerCacheAsMatrices(cache), low, high); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheXavierInit(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsXavierInit(ML2_LayerCacheAsScalars(cache)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsXavierInit(ML2_LayerCacheAsVectors(cache)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesXavierInit(ML2_LayerCacheAsMatrices(cache)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheHeInit(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsHeInit(ML2_LayerCacheAsScalars(cache)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsHeInit(ML2_LayerCacheAsVectors(cache)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesHeInit(ML2_LayerCacheAsMatrices(cache)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheCopyHostToDevice(ML2_LayerCache destDevice, ML2_LayerCache srcHost) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(destDevice), ML2_LayerCacheAsInfo(srcHost)));
    switch (destDevice.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsCopyHostToDevice(ML2_LayerCacheAsScalars(destDevice), ML2_LayerCacheAsScalars(srcHost)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsCopyHostToDevice(ML2_LayerCacheAsVectors(destDevice), ML2_LayerCacheAsVectors(srcHost)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesCopyHostToDevice(ML2_LayerCacheAsMatrices(destDevice), ML2_LayerCacheAsMatrices(srcHost)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheCopyDeviceToHost(ML2_LayerCache destHost, ML2_LayerCache srcDevice) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(destHost), ML2_LayerCacheAsInfo(srcDevice)));
    switch (destHost.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsCopyDeviceToHost(ML2_LayerCacheAsScalars(destHost), ML2_LayerCacheAsScalars(srcDevice)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsCopyDeviceToHost(ML2_LayerCacheAsVectors(destHost), ML2_LayerCacheAsVectors(srcDevice)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesCopyDeviceToHost(ML2_LayerCacheAsMatrices(destHost), ML2_LayerCacheAsMatrices(srcDevice)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Size ML2_LayerCacheSizeof(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsSizeof(ML2_LayerCacheAsScalars(cache));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsSizeof(ML2_LayerCacheAsVectors(cache));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesSizeof(ML2_LayerCacheAsMatrices(cache));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

ML2_FN ML2_Arch ML2_ArchNew(ML2_Int layers, ML2_LayerInfo infos[static layers]) {
    ML2_Arch arch = {layers, ML2_Calloc(layers * sizeof(*arch.infos))};
    for (ML2_Int i = 0; i < layers; i++) {
        arch.infos[i] = infos[i];
    }
    return arch;
}

ML2_FN void ML2_ArchDestroy(ML2_Arch *arch) {
    ML2_FREE(arch->infos);
    *arch = (ML2_Arch){};
}

// ML2_Arch ⬆️

// ML2_Model ⬇️

ML2_FN ML2_Model ML2_ModelNew(ML2_Device device, ML2_Arch arch) {
    ML2_Model model = {
        .layerCount = arch.layerCount,
        .layers = ML2_Calloc(arch.layerCount * sizeof(*model.layers)),
    };
    for (ML2_Int i = 0; i < arch.layerCount; i++) {
        model.layers[i] = ML2_LayerNew(device, arch.infos[i]);
    }
    return model;
}

ML2_FN void ML2_ModelDestroy(ML2_Model *model) {
    for (ML2_Int i = 0; i < model->layerCount; i++) {
        ML2_LayerDestroy(&model->layers[i]);
    }
    ML2_FREE(model->layers);
    *model = (ML2_Model){};
}

ML2_FN void ML2_ModelInfoPrint(ML2_Model model, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerInfoPrint(ML2_LayerAsInfo(model.layers[i]), indent + ML2_Indentation);
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN bool ML2_ModelInfoSame(ML2_Model a, ML2_Model b) {
    if (a.layerCount != b.layerCount) return false;
    ML2_Int layerCount = a.layerCount;
    for (ML2_Int i = 0; i < layerCount; i++) {
        if (!ML2_LayerInfoSame(ML2_LayerAsInfo(a.layers[i]), ML2_LayerAsInfo(b.layers[i]))) return false;
    }
    return true;
}

// TODO(30/9/2026 1:31:53): this looks nothing like how ModelCachePrint does it, perhaps it should but idk yet
ML2_FN void ML2_ModelPrint(ML2_Model model, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerPrint(model.layers[i], indent + ML2_Indentation);
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_ModelClear(ML2_Model model) {
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerClear(model.layers[i]);
    }
}

ML2_FN void ML2_ModelCopy(ML2_Model dest, ML2_Model src) {
    ML2_HARD_ASSERT(ML2_ModelInfoSame(dest, src));
    for (ML2_Int i = 0; i < dest.layerCount; i++) {
        ML2_LayerCopy(dest.layers[i], src.layers[i]);
    }
}

ML2_FN void ML2_ModelSum(ML2_Model dest, ML2_Model src) {
    ML2_HARD_ASSERT(ML2_ModelInfoSame(dest, src));
    for (ML2_Int i = 0; i < dest.layerCount; i++) {
        ML2_LayerSum(dest.layers[i], src.layers[i]);
    }
}

ML2_FN void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerRand(model.layers[i], low, high);
    }
}

ML2_FN void ML2_ModelXavierInit(ML2_Model model) {
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerXavierInit(model.layers[i]);
    }
}

ML2_FN void ML2_ModelHeInit(ML2_Model model) {
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerHeInit(model.layers[i]);
    }
}

ML2_FN void ML2_ModelCopyHostToDevice(ML2_Model destDevice, ML2_Model srcHost) {
    ML2_HARD_ASSERT(ML2_ModelInfoSame(destDevice, srcHost));
    for (ML2_Int i = 0; i < destDevice.layerCount; i++) {
        ML2_LayerCopyHostToDevice(destDevice.layers[i], srcHost.layers[i]);
    }
}

ML2_FN void ML2_ModelCopyDeviceToHost(ML2_Model destHost, ML2_Model srcDevice) {
    ML2_HARD_ASSERT(ML2_ModelInfoSame(destHost, srcDevice));
    for (ML2_Int i = 0; i < destHost.layerCount; i++) {
        ML2_LayerCopyDeviceToHost(destHost.layers[i], srcDevice.layers[i]);
    }
}

ML2_FN bool ML2_ModelForwardCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_LayerCache input) {
    if (model.layerCount != modelCache.layerCount) return false;
    if (!ML2_LayerForwardCompatible(ML2_LayerAsInfo(model.layers[0]), ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(modelCache.layers[0]))) return false;
    for (ML2_Int i = 1; i < model.layerCount; i++) {
        if (!ML2_LayerForwardCompatible(ML2_LayerAsInfo(model.layers[i]), ML2_LayerCacheAsInfo(modelCache.layers[i - 1]), ML2_LayerCacheAsInfo(modelCache.layers[i]))) return false;
    }
    return true;
}

ML2_FN void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache, ML2_LayerCache input) {
    ML2_HARD_ASSERT(ML2_ModelForwardCompatible(model, modelCache, input));
    ML2_LayerForward(model.layers[0], input, modelCache.layers[0]);
    for (ML2_Int i = 1; i < model.layerCount; i++) {
        ML2_LayerForward(model.layers[i], modelCache.layers[i - 1], modelCache.layers[i]);
    }
}

ML2_FN bool ML2_ModelBackwardCompatible(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient, ML2_Batch batch) {
    if (
        !ML2_ModelInfoSame(model, modelGradient) ||
        model.layerCount != modelCache.layerCount ||
        !ML2_ModelCacheInfoSame(modelCache, modelCacheGradient) ||
        !ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(ML2_ModelCacheOutput(modelCache)), ML2_LayerCacheAsInfo(batch.output))
    ) return false;
    for (ML2_Int i = modelCache.layerCount - 1; i > 0; i--) {
        if (
            !ML2_LayerBackwardCompatible(ML2_LayerAsInfo(modelGradient.layers[i]), ML2_LayerCacheAsInfo(modelCache.layers[i - 1]), ML2_LayerCacheAsInfo(modelCacheGradient.layers[i])) ||
            !ML2_LayerCacheBackwardCompatible(ML2_LayerAsInfo(model.layers[i]), ML2_LayerCacheAsInfo(modelCache.layers[i - 1]), ML2_LayerCacheAsInfo(modelCacheGradient.layers[i - 1]), ML2_LayerCacheAsInfo(modelCache.layers[i]), ML2_LayerCacheAsInfo(modelCacheGradient.layers[i]))
        ) return false;
    }
    if (!ML2_LayerBackwardCompatible(ML2_LayerAsInfo(modelGradient.layers[0]), ML2_LayerCacheAsInfo(batch.input), ML2_LayerCacheAsInfo(modelCacheGradient.layers[0]))) return false;
    return true;
}

ML2_FN void ML2_ModelBackward(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient, ML2_Batch batch, ML2_LossBackward lossBackward) {
    ML2_HARD_ASSERT(ML2_ModelBackwardCompatible(model, modelGradient, modelCache, modelCacheGradient, batch));
    lossBackward(ML2_ModelCacheOutput(modelCache), batch.output, ML2_ModelCacheOutput(modelCacheGradient));
    for (ML2_Int i = modelCache.layerCount - 1; i > 0; i--) {
        ML2_LayerBackward(modelGradient.layers[i], modelCache.layers[i - 1], modelCacheGradient.layers[i]);
        ML2_LayerCacheBackward(model.layers[i], modelCache.layers[i - 1], modelCacheGradient.layers[i - 1], modelCache.layers[i], modelCacheGradient.layers[i]);
    }
    ML2_LayerBackward(modelGradient.layers[0], batch.input, modelCacheGradient.layers[0]);
}

ML2_FN bool ML2_ModelGradientDescentOptimizeCompatible(ML2_Model model, ML2_Model modelGradient) {
    return ML2_ModelInfoSame(model, modelGradient);
}

ML2_FN void ML2_ModelGradientDescentOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerGradientDescent gradientDescent) {
    ML2_HARD_ASSERT(ML2_ModelGradientDescentOptimizeCompatible(model, modelGradient));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerGradientDescentOptimize(model.layers[i], modelGradient.layers[i], gradientDescent.parameters.learningRate);
    }
}

ML2_FN bool ML2_ModelMomentumOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerMomentum momentum) {
    return ML2_ModelInfoSame(model, modelGradient) && model.layerCount == momentum.state.layerCount;
}

ML2_FN void ML2_ModelMomentumOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerMomentum momentum) {
    ML2_HARD_ASSERT(ML2_ModelMomentumOptimizeCompatible(model, modelGradient, momentum));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerMomentumOptimize(model.layers[i], modelGradient.layers[i], momentum.parameters.learningRate, momentum.parameters.decayRate, momentum.state.averageLayers[i]);
    }
}

ML2_FN bool ML2_ModelAdagradOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdagrad adagrad) {
    return ML2_ModelInfoSame(model, modelGradient) && model.layerCount == adagrad.state.layerCount;
}

ML2_FN void ML2_ModelAdagradOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdagrad adagrad) {
    ML2_HARD_ASSERT(ML2_ModelAdagradOptimizeCompatible(model, modelGradient, adagrad));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerAdagradOptimize(model.layers[i], modelGradient.layers[i], adagrad.parameters.learningRate, adagrad.state.squareSumLayers[i]);
    }
}

ML2_FN bool ML2_ModelRMSPropOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerRMSProp RMSProp) {
    return ML2_ModelInfoSame(model, modelGradient) && model.layerCount == RMSProp.state.layerCount;
}

ML2_FN void ML2_ModelRMSPropOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerRMSProp RMSProp) {
    ML2_HARD_ASSERT(ML2_ModelRMSPropOptimizeCompatible(model, modelGradient, RMSProp));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerRMSPropOptimize(model.layers[i], modelGradient.layers[i], RMSProp.parameters.learningRate, RMSProp.parameters.decayRate, RMSProp.state.squareAverageLayers[i]);
    }
}

ML2_FN bool ML2_ModelAdamOptimizeCompatible(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdam adam) {
    return ML2_ModelInfoSame(model, modelGradient) && model.layerCount == adam.state.layerCount;
}

ML2_FN void ML2_ModelAdamOptimize(ML2_Model model, ML2_Model modelGradient, ML2_OptimizerAdam *adam) {
    ML2_HARD_ASSERT(ML2_ModelAdamOptimizeCompatible(model, modelGradient, *adam));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerAdamOptimize(model.layers[i], modelGradient.layers[i], adam->parameters.learningRate, adam->parameters.decayRate1, adam->parameters.decayRate2, adam->state.averageLayers[i], adam->state.squareAverageLayers[i], adam->state.decayingWeight1, adam->state.decayingWeight2);
    }
    adam->state.decayingWeight1 *= adam->parameters.decayRate1;
    adam->state.decayingWeight2 *= adam->parameters.decayRate2;
}

ML2_FN void ML2_ModelOptimize(ML2_Model model, ML2_Model modelGradient, ML2_Optimizer *optimizer) {
    switch (optimizer->type) {
        case ML2_OptimizerTypeGradientDescent: ML2_ModelGradientDescentOptimize(model, modelGradient, optimizer->as.gradientDescent); break;
        case ML2_OptimizerTypeMomentum: ML2_ModelMomentumOptimize(model, modelGradient, optimizer->as.momentum); break;
        case ML2_OptimizerTypeAdagrad: ML2_ModelAdagradOptimize(model, modelGradient, optimizer->as.adagrad); break;
        case ML2_OptimizerTypeRMSProp: ML2_ModelRMSPropOptimize(model, modelGradient, optimizer->as.RMSProp); break;
        case ML2_OptimizerTypeAdam: ML2_ModelAdamOptimize(model, modelGradient, &optimizer->as.adam); break;
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

ML2_FN ML2_Size ML2_ModelSizeof(ML2_Model model) {
    ML2_Size size = {};
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        size += ML2_LayerSizeof(model.layers[i]);
    }
    return size;
}

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_FN ML2_ModelCache ML2_ModelCacheNew(ML2_Device device, ML2_Arch arch, ML2_BatchInfo info) {
    ML2_ModelCache modelCache = {arch.layerCount, ML2_Calloc(arch.layerCount * sizeof(*modelCache.layers))};

    ML2_LayerCacheInfo prevInfo = info.input;
    for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
        ML2_LayerCacheInfo nextInfo = ML2_LayerInfoForward(arch.infos[i], prevInfo);

        modelCache.layers[i] = ML2_LayerCacheNew(device, nextInfo);

        prevInfo = nextInfo;
    }
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(info.output, prevInfo));
    return modelCache;
}

ML2_FN void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache) {
    for (ML2_Int i = 0; i < modelCache->layerCount; i++) {
        ML2_LayerCacheDestroy(&modelCache->layers[i]);
    }
    ML2_FREE(modelCache->layers);
    *modelCache = (ML2_ModelCache){};
}

ML2_FN void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
            printf(ML2_INDENT("%s[%d]: ", indent), "Cache", i);
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(modelCache.layers[i]), 0);
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN bool ML2_ModelCacheInfoSame(ML2_ModelCache a, ML2_ModelCache b) {
    if (a.layerCount != b.layerCount) return false;
    ML2_Int layerCount = a.layerCount;
    for (ML2_Int i = 0; i < layerCount; i++) {
        if (!ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(a.layers[i]), ML2_LayerCacheAsInfo(b.layers[i]))) return false;
    }
    return true;
}

ML2_FN void ML2_ModelCachePrint(ML2_ModelCache modelCache, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
            printf(ML2_INDENT("Cache[%d]\n", indent), i);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                ML2_LayerCachePrint(modelCache.layers[i], indent);
                indent -= ML2_Indentation;
            }
            printf(ML2_INDENT("}\n", indent));
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_ModelCacheClear(ML2_ModelCache modelCache) {
    for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
        ML2_LayerCacheClear(modelCache.layers[i]);
    }
}

ML2_FN void ML2_ModelCacheCopyHostToDevice(ML2_ModelCache destDevice, ML2_ModelCache srcHost) {
    ML2_HARD_ASSERT(ML2_ModelCacheInfoSame(destDevice, srcHost));
    for (ML2_Int i = 0; i < destDevice.layerCount; i++) {
        ML2_LayerCacheCopyHostToDevice(destDevice.layers[i], srcHost.layers[i]);
    }
}

ML2_FN void ML2_ModelCacheCopyDeviceToHost(ML2_ModelCache destHost, ML2_ModelCache srcDevice) {
    ML2_HARD_ASSERT(ML2_ModelCacheInfoSame(destHost, srcDevice));
    for (ML2_Int i = 0; i < destHost.layerCount; i++) {
        ML2_LayerCacheCopyDeviceToHost(destHost.layers[i], srcDevice.layers[i]);
    }
}

ML2_FN ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache) {
    return modelCache.layers[modelCache.layerCount - 1];
}

ML2_FN ML2_Scalar ML2_ModelCacheLoss(ML2_ModelCache modelCache, ML2_LayerCache expected, ML2_LossForward lossForward) {
    return lossForward(ML2_ModelCacheOutput(modelCache), expected);
}

ML2_FN ML2_Size ML2_ModelCacheSizeof(ML2_ModelCache modelCache) {
    ML2_Size size = {};
    for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
        size += ML2_LayerCacheSizeof(modelCache.layers[i]);
    }
    return size;
}

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_FN ML2_BatchInfo ML2_BatchInfoMake(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return (ML2_BatchInfo){input, output};
}

ML2_FN ML2_Batch ML2_BatchNew(ML2_Device device, ML2_BatchInfo info) {
    return (ML2_Batch){ML2_LayerCacheNew(device, info.input), ML2_LayerCacheNew(device, info.output)};
}

ML2_FN void ML2_BatchDestroy(ML2_Batch *batch) {
    ML2_LayerCacheDestroy(&batch->input);
    ML2_LayerCacheDestroy(&batch->output);
    // NOTE: this is not required because LayerCacheDestroy will zero it out, but whatever
    *batch = (ML2_Batch){};
}

ML2_FN void ML2_BatchInfoPrint(ML2_Batch batch, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Input:\n", indent));
        printf(ML2_INDENT("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(batch.input), indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_INDENT("}\n", indent));

        printf(ML2_INDENT("Output:\n", indent));
        printf(ML2_INDENT("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(batch.output), indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_INDENT("}\n", indent));
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_BatchPrint(ML2_Batch batch, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Input:\n", indent));
        printf(ML2_INDENT("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCachePrint(batch.input, indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_INDENT("}\n", indent));

        printf(ML2_INDENT("Output:\n", indent));
        printf(ML2_INDENT("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCachePrint(batch.output, indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_INDENT("}\n", indent));
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN ML2_Size ML2_BatchSizeof(ML2_Batch batch) {
    return ML2_LayerCacheSizeof(batch.input) + ML2_LayerCacheSizeof(batch.output);
}

// ML2_Batch ⬆️

// ML2_Optimizer ⬇️

// ML2_OptimizerGradientDescent ⬇️

ML2_FN ML2_OptimizerParameters ML2_GradientDescent(ML2_Scalar learningRate) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeGradientDescent, .as.gradientDescent = {learningRate}};
}

ML2_FN ML2_OptimizerParametersGradientDescent ML2_OptimizerParametersAsGradientDescent(ML2_OptimizerParameters parameters) {
    ML2_HARD_ASSERT(parameters.type == ML2_OptimizerTypeGradientDescent);
    return parameters.as.gradientDescent;
}

ML2_FN ML2_OptimizerGradientDescent ML2_OptimizerAsGradientDescent(ML2_Optimizer optimizer) {
    ML2_HARD_ASSERT(optimizer.type == ML2_OptimizerTypeGradientDescent);
    return optimizer.as.gradientDescent;
}

ML2_FN void ML2_GradientDescentInfoPrint(ML2_OptimizerParametersGradientDescent parameters, ML2_Int indent) {
    printf(ML2_INDENT("Optimizer(\"GradientDescent\")\n", indent));
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Learning Rate = " ML2_SCALAR_FMT "\n", indent), parameters.learningRate);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

// ML2_OptimizerGradientDescent ⬆️

// ML2_OptimizerMomentum ⬇️

ML2_FN ML2_OptimizerParameters ML2_Momentum(ML2_Scalar learningRate, ML2_Scalar decayRate) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeMomentum, .as.momentum = {learningRate, decayRate}};
}

ML2_FN ML2_OptimizerMomentum ML2_MomentumNew(ML2_Device device, ML2_OptimizerParametersMomentum parameters, ML2_Arch arch) {
    ML2_OptimizerMomentum momentum = {
        parameters,
        {
            arch.layerCount,
            ML2_Calloc(arch.layerCount * sizeof(*momentum.state.averageLayers)),
        },
    };
    for (ML2_Int i = 0; i < momentum.state.layerCount; i++) {
        momentum.state.averageLayers[i] = ML2_LayerNew(device, arch.infos[i]);
    }
    return momentum;
}

ML2_FN void ML2_MomentumDestroy(ML2_OptimizerMomentum *momentum) {
    for (ML2_Int i = 0; i < momentum->state.layerCount; i++) {
        ML2_LayerDestroy(&momentum->state.averageLayers[i]);
    }
    ML2_FREE(momentum->state.averageLayers);
    *momentum = (ML2_OptimizerMomentum){};
}

ML2_FN void ML2_MomentumReset(ML2_OptimizerMomentum momentum) {
    for (ML2_Int i = 0; i < momentum.state.layerCount; i++) {
        ML2_LayerClear(momentum.state.averageLayers[i]);
    }
}

ML2_FN ML2_OptimizerParametersMomentum ML2_OptimizerParametersAsMomentum(ML2_OptimizerParameters parameters) {
    ML2_HARD_ASSERT(parameters.type == ML2_OptimizerTypeMomentum);
    return parameters.as.momentum;
}

ML2_FN ML2_OptimizerMomentum ML2_OptimizerAsMomentum(ML2_Optimizer optimizer) {
    ML2_HARD_ASSERT(optimizer.type == ML2_OptimizerTypeMomentum);
    return optimizer.as.momentum;
}

ML2_FN void ML2_MomentumInfoPrint(ML2_OptimizerParametersMomentum parameters, ML2_Int indent) {
    printf(ML2_INDENT("Optimizer(\"Momentum\")\n", indent));
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Learning Rate = " ML2_SCALAR_FMT "\n", indent), parameters.learningRate);
        printf(ML2_INDENT("Decay Rate    = " ML2_SCALAR_FMT "\n", indent), parameters.decayRate);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN ML2_Size ML2_MomentumSizeof(ML2_OptimizerMomentum momentum) {
    ML2_Size size = {};
    for (ML2_Int i = 0; i < momentum.state.layerCount; i++) {
        size += ML2_LayerSizeof(momentum.state.averageLayers[i]);
    }
    return size;
}

// ML2_OptimizerMomentum ⬆️

// ML2_OptimizerAdagrad ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adagrad(ML2_Scalar learningRate) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeAdagrad, .as.adagrad = {learningRate}};
}

ML2_FN ML2_OptimizerAdagrad ML2_AdagradNew(ML2_Device device, ML2_OptimizerParametersAdagrad parameters, ML2_Arch arch) {
    ML2_OptimizerAdagrad adagrad = {
        parameters,
        {
            arch.layerCount,
            ML2_Calloc(arch.layerCount * sizeof(*adagrad.state.squareSumLayers)),
        },
    };
    for (ML2_Int i = 0; i < adagrad.state.layerCount; i++) {
        adagrad.state.squareSumLayers[i] = ML2_LayerNew(device, arch.infos[i]);
    }
    return adagrad;
}

ML2_FN void ML2_AdagradDestroy(ML2_OptimizerAdagrad *adagrad) {
    for (ML2_Int i = 0; i < adagrad->state.layerCount; i++) {
        ML2_LayerDestroy(&adagrad->state.squareSumLayers[i]);
    }
    ML2_FREE(adagrad->state.squareSumLayers);
    *adagrad = (ML2_OptimizerAdagrad){};
}

ML2_FN void ML2_AdagradReset(ML2_OptimizerAdagrad adagrad) {
    for (ML2_Int i = 0; i < adagrad.state.layerCount; i++) {
        ML2_LayerClear(adagrad.state.squareSumLayers[i]);
    }
}

ML2_FN ML2_OptimizerParametersAdagrad ML2_OptimizerParametersAsAdagrad(ML2_OptimizerParameters parameters) {
    ML2_HARD_ASSERT(parameters.type == ML2_OptimizerTypeAdagrad);
    return parameters.as.adagrad;
}

ML2_FN ML2_OptimizerAdagrad ML2_OptimizerAsAdagrad(ML2_Optimizer optimizer) {
    ML2_HARD_ASSERT(optimizer.type == ML2_OptimizerTypeAdagrad);
    return optimizer.as.adagrad;
}

ML2_FN void ML2_AdagradInfoPrint(ML2_OptimizerParametersAdagrad parameters, ML2_Int indent) {
    printf(ML2_INDENT("Optimizer(\"Adagrad\")\n", indent));
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Learning Rate = " ML2_SCALAR_FMT "\n", indent), parameters.learningRate);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN ML2_Size ML2_AdagradSizeof(ML2_OptimizerAdagrad adagrad) {
    ML2_Size size = {};
    for (ML2_Int i = 0; i < adagrad.state.layerCount; i++) {
        size += ML2_LayerSizeof(adagrad.state.squareSumLayers[i]);
    }
    return size;
}

// ML2_OptimizerAdagrad ⬆️

// ML2_OptimizerRMSProp ⬇️

ML2_FN ML2_OptimizerParameters ML2_RMSProp(ML2_Scalar learningRate, ML2_Scalar decayRate) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeRMSProp, .as.RMSProp = {learningRate, decayRate}};
}

ML2_FN ML2_OptimizerRMSProp ML2_RMSPropNew(ML2_Device device, ML2_OptimizerParametersRMSProp parameters, ML2_Arch arch) {
    ML2_OptimizerRMSProp RMSProp = {
        parameters,
        {
            arch.layerCount,
            ML2_Calloc(arch.layerCount * sizeof(*RMSProp.state.squareAverageLayers)),
        },
    };
    for (ML2_Int i = 0; i < RMSProp.state.layerCount; i++) {
        RMSProp.state.squareAverageLayers[i] = ML2_LayerNew(device, arch.infos[i]);
    }
    return RMSProp;
}

ML2_FN void ML2_RMSPropDestroy(ML2_OptimizerRMSProp *RMSProp) {
    for (ML2_Int i = 0; i < RMSProp->state.layerCount; i++) {
        ML2_LayerDestroy(&RMSProp->state.squareAverageLayers[i]);
    }
    ML2_FREE(RMSProp->state.squareAverageLayers);
    *RMSProp = (ML2_OptimizerRMSProp){};
}

ML2_FN void ML2_RMSPropReset(ML2_OptimizerRMSProp RMSProp) {
    for (ML2_Int i = 0; i < RMSProp.state.layerCount; i++) {
        ML2_LayerClear(RMSProp.state.squareAverageLayers[i]);
    }
}

ML2_FN ML2_OptimizerParametersRMSProp ML2_OptimizerParametersAsRMSProp(ML2_OptimizerParameters parameters) {
    ML2_HARD_ASSERT(parameters.type == ML2_OptimizerTypeRMSProp);
    return parameters.as.RMSProp;
}

ML2_FN ML2_OptimizerRMSProp ML2_OptimizerAsRMSProp(ML2_Optimizer optimizer) {
    ML2_HARD_ASSERT(optimizer.type == ML2_OptimizerTypeRMSProp);
    return optimizer.as.RMSProp;
}

ML2_FN void ML2_RMSPropInfoPrint(ML2_OptimizerParametersRMSProp parameters, ML2_Int indent) {
    printf(ML2_INDENT("Optimizer(\"RMSProp\")\n", indent));
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Learning Rate = " ML2_SCALAR_FMT "\n", indent), parameters.learningRate);
        printf(ML2_INDENT("Decay Rate    = " ML2_SCALAR_FMT "\n", indent), parameters.decayRate);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN ML2_Size ML2_RMSPropSizeof(ML2_OptimizerRMSProp RMSProp) {
    ML2_Size size = {};
    for (ML2_Int i = 0; i < RMSProp.state.layerCount; i++) {
        size += ML2_LayerSizeof(RMSProp.state.squareAverageLayers[i]);
    }
    return size;
}

// ML2_OptimizerRMSProp ⬆️

// ML2_OptimizerAdam ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adam(ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeAdam, .as.adam = {learningRate, decayRate1, decayRate2}};
}

ML2_FN ML2_OptimizerAdam ML2_AdamNew(ML2_Device device, ML2_OptimizerParametersAdam parameters, ML2_Arch arch) {
    ML2_OptimizerAdam adam = {
        parameters,
        {
            arch.layerCount,
            ML2_Calloc(arch.layerCount * sizeof(*adam.state.averageLayers)),
            ML2_Calloc(arch.layerCount * sizeof(*adam.state.squareAverageLayers)),
            parameters.decayRate1,
            parameters.decayRate2,
        },
    };
    for (ML2_Int i = 0; i < adam.state.layerCount; i++) {
        adam.state.averageLayers[i] = ML2_LayerNew(device, arch.infos[i]);
        adam.state.squareAverageLayers[i] = ML2_LayerNew(device, arch.infos[i]);
    }
    return adam;
}

ML2_FN void ML2_AdamDestroy(ML2_OptimizerAdam *adam) {
    for (ML2_Int i = 0; i < adam->state.layerCount; i++) {
        ML2_LayerDestroy(&adam->state.averageLayers[i]);
        ML2_LayerDestroy(&adam->state.squareAverageLayers[i]);
    }
    ML2_FREE(adam->state.averageLayers);
    ML2_FREE(adam->state.squareAverageLayers);
    *adam = (ML2_OptimizerAdam){};
}

ML2_FN void ML2_AdamReset(ML2_OptimizerAdam *adam) {
    adam->state.decayingWeight1 = adam->parameters.decayRate1;
    adam->state.decayingWeight2 = adam->parameters.decayRate2;
    for (ML2_Int i = 0; i < adam->state.layerCount; i++) {
        ML2_LayerClear(adam->state.averageLayers[i]);
        ML2_LayerClear(adam->state.squareAverageLayers[i]);
    }
}

ML2_FN ML2_OptimizerParametersAdam ML2_OptimizerParametersAsAdam(ML2_OptimizerParameters parameters) {
    ML2_HARD_ASSERT(parameters.type == ML2_OptimizerTypeAdam);
    return parameters.as.adam;
}

ML2_FN ML2_OptimizerAdam ML2_OptimizerAsAdam(ML2_Optimizer optimizer) {
    ML2_HARD_ASSERT(optimizer.type == ML2_OptimizerTypeAdam);
    return optimizer.as.adam;
}

ML2_FN void ML2_AdamInfoPrint(ML2_OptimizerParametersAdam parameters, ML2_Int indent) {
    printf(ML2_INDENT("Optimizer(\"Adam\")\n", indent));
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("Learning Rate = " ML2_SCALAR_FMT "\n", indent), parameters.learningRate);
        printf(ML2_INDENT("Decay Rate 1  = " ML2_SCALAR_FMT "\n", indent), parameters.decayRate1);
        printf(ML2_INDENT("Decay Rate 2  = " ML2_SCALAR_FMT "\n", indent), parameters.decayRate2);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN ML2_Size ML2_AdamSizeof(ML2_OptimizerAdam adam) {
    ML2_Size size = {};
    for (ML2_Int i = 0; i < adam.state.layerCount; i++) {
        size += ML2_LayerSizeof(adam.state.averageLayers[i]);
        size += ML2_LayerSizeof(adam.state.squareAverageLayers[i]);
    }
    return size;
}

// ML2_OptimizerAdam ⬆️

ML2_FN ML2_Optimizer ML2_OptimizerNew(ML2_Device device, ML2_OptimizerParameters parameters, ML2_Arch arch) {
    switch (parameters.type) {
        case ML2_OptimizerTypeGradientDescent: return (ML2_Optimizer){parameters.type, .as.gradientDescent = {ML2_OptimizerParametersAsGradientDescent(parameters)}};
        case ML2_OptimizerTypeMomentum: return (ML2_Optimizer){parameters.type, .as.momentum = ML2_MomentumNew(device, ML2_OptimizerParametersAsMomentum(parameters), arch)};
        case ML2_OptimizerTypeAdagrad: return (ML2_Optimizer){parameters.type, .as.adagrad = ML2_AdagradNew(device, ML2_OptimizerParametersAsAdagrad(parameters), arch)};
        case ML2_OptimizerTypeRMSProp: return (ML2_Optimizer){parameters.type, .as.RMSProp = ML2_RMSPropNew(device, ML2_OptimizerParametersAsRMSProp(parameters), arch)};
        case ML2_OptimizerTypeAdam: return (ML2_Optimizer){parameters.type, .as.adam = ML2_AdamNew(device, ML2_OptimizerParametersAsAdam(parameters), arch)};
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

ML2_FN void ML2_OptimizerDestroy(ML2_Optimizer *optimizer) {
    switch (optimizer->type) {
        case ML2_OptimizerTypeGradientDescent: /* No action needed */ break;
        case ML2_OptimizerTypeMomentum: ML2_MomentumDestroy(&optimizer->as.momentum); break;
        case ML2_OptimizerTypeAdagrad: ML2_AdagradDestroy(&optimizer->as.adagrad); break;
        case ML2_OptimizerTypeRMSProp: ML2_RMSPropDestroy(&optimizer->as.RMSProp); break;
        case ML2_OptimizerTypeAdam: ML2_AdamDestroy(&optimizer->as.adam); break;
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
    *optimizer = (ML2_Optimizer){};
}

ML2_FN void ML2_OptimizerReset(ML2_Optimizer *optimizer) {
    switch (optimizer->type) {
        case ML2_OptimizerTypeGradientDescent: /* No action needed */ break;
        case ML2_OptimizerTypeMomentum: ML2_MomentumReset(optimizer->as.momentum); break;
        case ML2_OptimizerTypeAdagrad: ML2_AdagradReset(optimizer->as.adagrad); break;
        case ML2_OptimizerTypeRMSProp: ML2_RMSPropReset(optimizer->as.RMSProp); break;
        case ML2_OptimizerTypeAdam: ML2_AdamReset(&optimizer->as.adam); break;
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

ML2_FN ML2_OptimizerParameters ML2_OptimizerAsParameters(ML2_Optimizer optimizer) {
    switch (optimizer.type) {
        case ML2_OptimizerTypeGradientDescent: return (ML2_OptimizerParameters){optimizer.type, .as.gradientDescent = optimizer.as.gradientDescent.parameters};
        case ML2_OptimizerTypeMomentum: return (ML2_OptimizerParameters){optimizer.type, .as.momentum = optimizer.as.momentum.parameters};
        case ML2_OptimizerTypeAdagrad: return (ML2_OptimizerParameters){optimizer.type, .as.adagrad = optimizer.as.adagrad.parameters};
        case ML2_OptimizerTypeRMSProp: return (ML2_OptimizerParameters){optimizer.type, .as.RMSProp = optimizer.as.RMSProp.parameters};
        case ML2_OptimizerTypeAdam: return (ML2_OptimizerParameters){optimizer.type, .as.adam = optimizer.as.adam.parameters};
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

ML2_FN void ML2_OptimizerInfoPrint(ML2_OptimizerParameters parameters, ML2_Int indent) {
    switch (parameters.type) {
        case ML2_OptimizerTypeGradientDescent: ML2_GradientDescentInfoPrint(ML2_OptimizerParametersAsGradientDescent(parameters), indent); break;
        case ML2_OptimizerTypeMomentum: ML2_MomentumInfoPrint(ML2_OptimizerParametersAsMomentum(parameters), indent); break;
        case ML2_OptimizerTypeAdagrad: ML2_AdagradInfoPrint(ML2_OptimizerParametersAsAdagrad(parameters), indent); break;
        case ML2_OptimizerTypeRMSProp: ML2_RMSPropInfoPrint(ML2_OptimizerParametersAsRMSProp(parameters), indent); break;
        case ML2_OptimizerTypeAdam: ML2_AdamInfoPrint(ML2_OptimizerParametersAsAdam(parameters), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

ML2_FN ML2_Size ML2_OptimizerSizeof(ML2_Optimizer optimizer) {
    switch (optimizer.type) {
        case ML2_OptimizerTypeGradientDescent: return 0;
        case ML2_OptimizerTypeMomentum: return ML2_MomentumSizeof(ML2_OptimizerAsMomentum(optimizer));
        case ML2_OptimizerTypeAdagrad: return ML2_AdagradSizeof(ML2_OptimizerAsAdagrad(optimizer));
        case ML2_OptimizerTypeRMSProp: return ML2_RMSPropSizeof(ML2_OptimizerAsRMSProp(optimizer));
        case ML2_OptimizerTypeAdam: return ML2_AdamSizeof(ML2_OptimizerAsAdam(optimizer));
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

// ML2_Optimizer ⬆️

// ML2_Loss ⬇️

ML2_FN ML2_Scalar ML2_LossForwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardSquareAverage(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardSquareAverage(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardSquareAverage(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsLossBackwardSquareAverage(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsLossBackwardSquareAverage(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesLossBackwardSquareAverage(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Scalar ML2_LossForwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsLossBackwardCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesLossBackwardCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Scalar ML2_LossForwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Scalar ML2_LossForwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient)); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient)); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_Loss ⬆️

#endif // ML2_IMPLEMENTATION

#ifdef ML2_STRIP_PREFIX

#ifndef ML2_H_STRIP_PREFIX_INCLUDE_GUARD
    #define ML2_H_STRIP_PREFIX_INCLUDE_GUARD

    // ML2_Scalar ⬇️

    #define Scalar ML2_Scalar

    #define RandScalar ML2_RandScalar

    // ML2_Scalar ⬆️

    // ML2_Layer ⬇️

    #define LayerType ML2_LayerType
    #define LayerTypeWeights ML2_LayerTypeWeights
    #define LayerTypeBiases ML2_LayerTypeBiases
    #define LayerTypeActivation ML2_LayerTypeActivation
    #define LayerTypeLinear ML2_LayerTypeLinear
    #define LayerTypeConv ML2_LayerTypeConv

    // ML2_LayerWeights ⬇️

    #define LayerInfoWeights ML2_LayerInfoWeights
    #define LayerWeights ML2_LayerWeights

    #define Weights ML2_Weights
    #define WeightsNew ML2_WeightsNew
    #define WeightsDestroy ML2_WeightsDestroy
    #define WeightsAt ML2_WeightsAt
    #define LayerAsWeights ML2_LayerAsWeights
    #define WeightsClear ML2_WeightsClear
    #define WeightsRand ML2_WeightsRand
    #define WeightsInfoPrint ML2_WeightsInfoPrint
    #define WeightsPrint ML2_WeightsPrint
    #define WeightsInfoForward ML2_WeightsInfoForward
    #define WeightsInfoSame ML2_WeightsInfoSame
    #define WeightsInfoForwardCompatible ML2_WeightsInfoForwardCompatible
    #define WeightsForward ML2_WeightsForward
    #define WeightsBackward ML2_WeightsBackward
    #define WeightsGradientDescent ML2_WeightsGradientDescent

    // ML2_LayerWeights ⬆️

    // ML2_LayerBiases ⬇️

    #define LayerInfoBiases ML2_LayerInfoBiases
    #define LayerBiases ML2_LayerBiases

    #define Biases ML2_Biases
    #define BiasesNew ML2_BiasesNew
    #define BiasesDestroy ML2_BiasesDestroy
    #define BiasesAt ML2_BiasesAt
    #define LayerAsBiases ML2_LayerAsBiases
    #define BiasesClear ML2_BiasesClear
    #define BiasesRand ML2_BiasesRand
    #define BiasesInfoPrint ML2_BiasesInfoPrint
    #define BiasesPrint ML2_BiasesPrint
    #define BiasesInfoForward ML2_BiasesInfoForward
    #define BiasesInfoSame ML2_BiasesInfoSame
    #define BiasesInfoForwardCompatible ML2_BiasesInfoForwardCompatible
    #define BiasesForward ML2_BiasesForward
    #define BiasesBackward ML2_BiasesBackward
    #define BiasesGradientDescent ML2_BiasesGradientDescent

    // ML2_LayerBiases ⬆️

    // ML2_LayerActivation ⬇️

    #define ActivationType ML2_ActivationType
    #define ActivationTypeReLU ML2_ActivationTypeReLU
    #define ActivationTypeSigmoid ML2_ActivationTypeSigmoid
    #define ActivationTypeSoftmax ML2_ActivationTypeSoftmax

    #define LayerInfoActivation ML2_LayerInfoActivation
    #define LayerActivation ML2_LayerActivation

    #define Activation ML2_Activation
    #define ReLU ML2_ReLU
    #define Sigmoid ML2_Sigmoid
    #define Softmax ML2_Softmax
    #define LayerAsActivation ML2_LayerAsActivation
    #define ActivationNameOf ML2_ActivationNameOf
    #define ActivationInfoPrint ML2_ActivationInfoPrint
    #define ActivationPrint ML2_ActivationPrint
    #define ActivationInfoForward ML2_ActivationInfoForward
    #define ActivationInfoSame ML2_ActivationInfoSame
    #define ActivationForwardReLU ML2_ActivationForwardReLU
    #define ActivationBackwardReLU ML2_ActivationBackwardReLU
    #define ActivationForwardSigmoid ML2_ActivationForwardSigmoid
    #define ActivationBackwardSigmoid ML2_ActivationBackwardSigmoid
    #define ActivationForwardSoftmax ML2_ActivationForwardSoftmax
    #define ActivationBackwardSoftmax ML2_ActivationBackwardSoftmax
    #define ActivationForward ML2_ActivationForward
    #define ActivationBackward ML2_ActivationBackward

    // ML2_LayerActivation ⬆️

    // ML2_LayerLinear ⬇️

    #define LayerInfoLinear ML2_LayerInfoLinear
    #define LayerLinear ML2_LayerLinear

    #define Linear ML2_Linear
    #define LinearNew ML2_LinearNew
    #define LinearDestroy ML2_LinearDestroy
    #define LayerAsLinear ML2_LayerAsLinear
    #define LinearWeights ML2_LinearWeights
    #define LinearBiases ML2_LinearBiases
    #define LinearClear ML2_LinearClear
    #define LinearRand ML2_LinearRand
    #define LinearInfoPrint ML2_LinearInfoPrint
    #define LinearPrint ML2_LinearPrint
    #define LinearInfoForward ML2_LinearInfoForward
    #define LinearInfoSame ML2_LinearInfoSame
    #define LinearInfoForwardCompatible ML2_LinearInfoForwardCompatible
    #define LinearForward ML2_LinearForward
    #define LinearBackward ML2_LinearBackward
    #define LinearGradientDescent ML2_LinearGradientDescent

    // ML2_LayerLinear ⬆️

    // ML2_LayerConv ⬇️

    #define LayerInfoConv ML2_LayerInfoConv
    #define LayerConv ML2_LayerConv

    // ML2_LayerConv ⬆️

    #define LayerInfo ML2_LayerInfo
    #define Layer ML2_Layer

    #define LayerNew ML2_LayerNew
    #define LayerDestroy ML2_LayerDestroy
    #define LayerAsInfo ML2_LayerAsInfo
    #define LayerClear ML2_LayerClear
    #define LayerRand ML2_LayerRand
    #define LayerInfoPrint ML2_LayerInfoPrint
    #define LayerPrint ML2_LayerPrint
    #define LayerInfoForward ML2_LayerInfoForward
    #define LayerInfoSame ML2_LayerInfoSame
    #define LayerInfoForwardCompatible ML2_LayerInfoForwardCompatible
    #define LayerForward ML2_LayerForward
    #define LayerBackward ML2_LayerBackward
    #define LayerGradientDescent ML2_LayerGradientDescent

    // ML2_Layer ⬆️

    // ML2_LayerCache ⬇️

    #define LayerCacheType ML2_LayerCacheType
    #define LayerCacheTypeScalars ML2_LayerCacheTypeScalars

    // ML2_LayerCacheScalars ⬇️

    #define LayerCacheInfoScalars ML2_LayerCacheInfoScalars
    #define LayerCacheScalars ML2_LayerCacheScalars

    #define Scalars ML2_Scalars
    #define ScalarsNew ML2_ScalarsNew
    #define ScalarsDestroy ML2_ScalarsDestroy
    #define ScalarsAt ML2_ScalarsAt
    #define LayerCacheAsScalars ML2_LayerCacheAsScalars
    #define ScalarsClear ML2_ScalarsClear
    #define ScalarsInfoPrint ML2_ScalarsInfoPrint
    #define ScalarsPrint ML2_ScalarsPrint
    #define ScalarsInfoSame ML2_ScalarsInfoSame
    #define ScalarsCopy ML2_ScalarsCopy

    // ML2_LayerCacheScalars ⬆️

    #define LayerCacheInfo ML2_LayerCacheInfo
    #define LayerCache ML2_LayerCache

    #define LayerCacheNew ML2_LayerCacheNew
    #define LayerCacheDestroy ML2_LayerCacheDestroy
    #define LayerCacheAsInfo ML2_LayerCacheAsInfo
    #define LayerCacheClear ML2_LayerCacheClear
    #define LayerCacheInfoPrint ML2_LayerCacheInfoPrint
    #define LayerCachePrint ML2_LayerCachePrint
    #define LayerCacheInfoSame ML2_LayerCacheInfoSame
    #define LayerCacheCopy ML2_LayerCacheCopy

    // ML2_LayerCache ⬆️

    // ML2_Arch ⬇️

    #define Arch ML2_Arch

    #define ArchMake ML2_ArchMake
    #define ArchNew ML2_ArchNew
    #define ArchDestroy ML2_ArchDestroy

    // ML2_Arch ⬆️

    // ML2_Model ⬇️

    #define Model ML2_Model

    #define ModelNew ML2_ModelNew
    #define ModelDestroy ML2_ModelDestroy
    #define ModelRand ML2_ModelRand
    #define ModelInfoPrint ML2_ModelInfoPrint
    #define ModelPrint ML2_ModelPrint
    #define ModelForward ML2_ModelForward
    #define ModelCacheLossForward ML2_ModelCacheLossForward
    #define ModelBackward ML2_ModelBackward
    #define ModelGradientDescent ML2_ModelGradientDescent

    // ML2_Model ⬆️

    // ML2_ModelCache ⬇️

    #define ModelCache ML2_ModelCache

    #define ModelCacheNew ML2_ModelCacheNew
    #define ModelCacheDestroy ML2_ModelCacheDestroy
    #define ModelCacheInfoPrint ML2_ModelCacheInfoPrint
    #define ModelCachePrint ML2_ModelCachePrint
    #define ModelCacheInput ML2_ModelCacheInput
    #define ModelCacheOutput ML2_ModelCacheOutput
    #define ModelCacheOutputGradient ML2_ModelCacheOutputGradient
    #define ModelCacheCopyBatchInput ML2_ModelCacheCopyBatchInput
    #define ModelCacheLossBackward ML2_ModelCacheLossBackward

    // ML2_ModelCache ⬆️

    // ML2_Batch ⬇️

    #define BatchInfo ML2_BatchInfo
    #define Batch ML2_Batch

    #define BatchInfoMake ML2_BatchInfoMake
    #define BatchNew ML2_BatchNew
    #define BatchDestroy ML2_BatchDestroy
    #define BatchInfoPrint ML2_BatchInfoPrint
    #define BatchPrint ML2_BatchPrint

    // ML2_Batch ⬆️

    // ML2_Loss ⬇️

    #define LossForward ML2_LossForward
    #define LossBackward ML2_LossBackward

    #define LossForwardSquareAverage ML2_LossForwardSquareAverage
    #define LossBackwardSquareAverage ML2_LossBackwardSquareAverage
    #define LossForwardCrossEntropy ML2_LossForwardCrossEntropy
    #define LossBackwardCrossEntropy ML2_LossBackwardCrossEntropy

    // ML2_Loss ⬆️

#endif // ML2_H_STRIP_PREFIX_INCLUDE_GUARD

#endif // ML2_STRIP_PREFIX

/*
    TODOLIST:
    - a diverse pool of loss functions
    - He, Xavier, etc initialization
    - Tanh, and just have a bigger pool of choices for activations
    - Conv layer, and in general everything needed for CNN
    - Batching, find a way to split the model cache and the batch to smaller parts
    - Adam optimizer, and other optimizers that use gradients in unique ways
    - find a way to implement the Softmax and Cross Entropy optimization
*/

/*
    previous TODOLIST is done i think
    TODOLIST(9/4/2026 10:00:00pm):
    generated by claude

    [X] Pooling
        MaxPool and AvgPool layers
        CNNs are basically unusable without them

    [X] Padding
        Conv currently only does valid convolution (output shrinks)
        need same/full padding so output can match input size

    [ ] BatchNorm
        normalizes each layer's output during training
        makes training significantly more stable on deep networks

    [ ] Dropout
        randomly zeros activations during training
        prevents overfitting, ~10 lines to implement

    [X] Leaky ReLU
        ReLU kills neurons that go negative permanently
        Leaky ReLU lets a small gradient through, fixes dying ReLU problem

    [ ] Model save/load
        currently a trained model lives only in memory
        need to write/read weights to disk

    [ ] Metrics
        accuracy for classification
        nothing else needed, loss alone is not enough to evaluate a model

    (3/9/2026 17:39:31): marked Pooling Padding and Leaky ReLU, they were done months ago just not documented
*/

/*
    TODOLIST(3/9/2026 17:34:41):

    1.  most functions do .as.layerType = {...}
        this can be changed to
        .as.layerType = ML2_LayerInfoAsLayerType(ML2_LayerType(...))
        its nicer because the intention is more clear
        this appears in slightly different patterns too

    2.  a lot of functions dont handle Vectors yet, there is a warning in every place, it will be very long to write handlers

    3.  im not sure if all the XavierInit HeInit functions are correct,
        because the underlying structure muts know about the bigger structure,
        some functions ignore this, not critical, but a correctness issue

    (3/9/2026 18:13:05):
    4.  most *Compatible functions dont check in a specific order,
        i just realised an order that makes sense,
        check that the layer matches the input,
        then check the output dimensions against everything else,
        an example exists in ML2_FusedWeightsForwardCompatible() as of right now
        also the checks should be in multiple lines like this
        return cond1 &&
               cond2;

    5.  i neglected the ML2_STRIP_PREFIX,
        thats because i didnt use it a lot and it sometimes causes issues,
        those issues need to be fixed eventually and the macro needs to work properly

    (3/9/2026 18:55:08):
    6.  now that Vectors exists Flatten can support it,
        Flatten could be made to Flatten to Scalars,
        However maybe controlling to what you are flattening would be more useful,
        for example (Matrices -> Vectors) or higher level tensors in the future to lower tensors

    (3/9/2026 19:40:49):
    7.  optimizers update their parameters on ModelOptimize,
        this doesnt let you use optimizers with graphed models where execution isnt linear,
        there should be a ML2_OptimizerUpdate() but its not as simple as it seems if you try

    (4/9/2026 23:07:33):
    8.  a lot of switch cases sometimes dont have break;
        for better consistency every case should have a break;
        it will make copy pasting more viable
*/