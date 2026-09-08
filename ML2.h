#ifndef _ML2_H
#define _ML2_H

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
#include <stdint.h>
#include <math.h>

#define ML2_UNREACHABLE(msg, ...)                                       \
    do {                                                                \
        fprintf(stderr, "UNREACHABLE: \"" msg "\":\n"                   \
                        "    file:     |%s|\n"                          \
                        "    function: |%s|\n"                          \
                        "    line:     |%d|\n",                         \
                __VA_OPT__(__VA_ARGS__,) __FILE__, __func__, __LINE__); \
        abort();                                                        \
    } while (0)

#define ML2_TODO(msg, ...)                                              \
    do {                                                                \
        fprintf(stderr, "TODO: \"" msg "\":\n"                          \
                        "    file:     |%s|\n"                          \
                        "    function: |%s|\n"                          \
                        "    line:     |%d|\n",                         \
                __VA_OPT__(__VA_ARGS__,) __FILE__, __func__, __LINE__); \
        abort();                                                        \
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
    #define ML2_SOFT_ASSERT(...)
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
    #define ML2_CALLOC calloc
#endif // ML2_CALLOC

#ifndef ML2_FREE
    #include <stdlib.h>
    #define ML2_FREE free
#endif // ML2_FREE

#ifndef ML2_RELIABLE_CALLOC
    static inline void *ML2_ReliableCalloc(ML2_Size count, ML2_Size size) {
        #ifdef ML2_PRINT_ALLOC
        {
            constexpr double measure = 1000.0;
            static double total = 0;
            double bytes = count * size;
            total += bytes;
            enum {B, KB, MB, GB, TB, PB, _count} p = 0;
            const char *units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
            while (bytes >= measure && p < _count - 1) {
                p++;
                bytes /= measure;
            }
            printf("ML2_ReliableCalloc: %.3lf %s\n", bytes, units[p]);
            double totalBytes = total;
            p = 0;
            while (totalBytes >= measure && p < _count - 1) {
                p++;
                totalBytes /= measure;
            }
            printf("ML2_ReliableCalloc: Total: %.3lf %s\n", totalBytes, units[p]);
        }
        #endif // ML2_PRINT_ALLOC
        void *ptr = ML2_CALLOC(count, size);
        if (!ptr) {
            // NOTE: fputs does not have internal allocations
            fputs("ML2_ReliableCalloc: Failed to allocate memory\n", stderr);
            abort();
        }
        return ptr;
    }
    #define ML2_RELIABLE_CALLOC ML2_ReliableCalloc
#endif // ML2_RELIABLE_CALLOC

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

ML2_VAR constexpr ML2_Scalar ML2_Epsilon = ML2_SCALAR_LITERAL(1e-8);
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

// ML2_Layer ⬇️

typedef enum : ML2_Int {
    ML2_LayerTypeWeights = 1,
    ML2_LayerTypeBiases,
    ML2_LayerTypeActivation,
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
    struct {
        ML2_Scalar *weights;
    } data;
} ML2_LayerWeights;

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

typedef struct {
    ML2_Int inputs;
} ML2_LayerInfoBiases;

typedef struct {
    ML2_LayerInfoBiases info;
    struct {
        ML2_Scalar *biases;
    } data;
} ML2_LayerBiases;

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

typedef enum : ML2_Int {
    ML2_ActivationTypeReLU = 1,
    ML2_ActivationTypeSigmoid,
    ML2_ActivationTypeSoftmax,
    ML2_ActivationTypeSin,
    ML2_ActivationTypeCos,
    ML2_ActivationTypeTanh,
    ML2_ActivationTypeLeakyReLU,
    ML2_ActivationTypeSiLU,
} ML2_ActivationType;

typedef struct {
    ML2_ActivationType type;
} ML2_LayerInfoActivation;

typedef ML2_Scalar (ML2_ScalarActivationForward)(ML2_Scalar input);
typedef ML2_Scalar (ML2_ScalarActivationBackwardInput)(ML2_Scalar input);
typedef ML2_Scalar (ML2_ScalarActivationBackwardOutput)(ML2_Scalar output);
typedef ML2_Scalar (ML2_ScalarActivationBackwardInputOutput)(ML2_Scalar input, ML2_Scalar output);

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

typedef struct {
    ML2_Int outputs;
    ML2_Int inputs;
} ML2_LayerInfoLinear;

typedef struct {
    ML2_LayerInfoLinear info;
    struct {
        ML2_Scalar *weights;
        ML2_Scalar *biases;
    } data;
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
    struct {
        ML2_Scalar *weights;
    } data;
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
    struct {
        ML2_Scalar *weights;
        ML2_Scalar *biases;
    } data;
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
    struct {
        ML2_Scalar *weights;
    } data;
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
constexpr ML2_Int ML2_AttentionVectorsCount = 3;
constexpr ML2_Int ML2_AttentionInputIndexQuery = 0;
constexpr ML2_Int ML2_AttentionInputIndexKey = 1;
constexpr ML2_Int ML2_AttentionInputIndexValue = 2;

typedef struct {
    bool masked;
} ML2_LayerInfoAttention;

// TODO(3/9/2026 19:45:41): for now Attention has no data, and i dont know if it will
// typedef struct {
//     ML2_LayerInfoAttention info;
//     struct {
//     } data;
// } ML2_LayerAttention;

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
    struct {
        ML2_Scalar *scalars;
    } data;
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
    struct {
        ML2_Scalar *scalars;
    } data;
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
    struct {
        ML2_Scalar *matrices;
    } data;
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

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_FN ML2_LayerInfo ML2_Weights(ML2_Int inputs, ML2_Int outputs);
ML2_FN ML2_LayerWeights ML2_WeightsNew(ML2_LayerInfoWeights info);
ML2_FN void ML2_WeightsDestroy(ML2_LayerWeights *weights);
ML2_FN ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i);
ML2_FN ML2_LayerInfoWeights ML2_LayerInfoAsWeights(ML2_LayerInfo info);
ML2_FN ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer);
ML2_FN void ML2_WeightsClear(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src);
ML2_FN void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_WeightsXavierInit(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsHeInit(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, ML2_Int indent);
ML2_FN void ML2_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent);
ML2_FN bool ML2_WeightsInfoForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_WeightsInfoForward(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_FN bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b);
ML2_FN bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_WeightsBackwardCompatible(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_WeightsBackward(ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_WeightsCacheBackwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_WeightsCacheBackward(ML2_LayerWeights weights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN void ML2_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average);
ML2_FN void ML2_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum);
ML2_FN void ML2_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage);
ML2_FN void ML2_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);
ML2_FN ML2_Size ML2_WeightsSizeof(ML2_LayerWeights weights);

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_FN ML2_LayerInfo ML2_Biases(ML2_Int inputs);
ML2_FN ML2_LayerBiases ML2_BiasesNew(ML2_LayerInfoBiases info);
ML2_FN void ML2_BiasesDestroy(ML2_LayerBiases *biases);
ML2_FN ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, ML2_Int i);
ML2_FN ML2_LayerInfoBiases ML2_LayerInfoAsBiases(ML2_LayerInfo info);
ML2_FN ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer);
ML2_FN void ML2_BiasesClear(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src);
ML2_FN void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_BiasesXavierInit(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesHeInit(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, ML2_Int indent);
ML2_FN void ML2_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent);
ML2_FN bool ML2_BiasesInfoForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_BiasesInfoForward(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_FN bool ML2_BiasesInfoSame(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b);
ML2_FN bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_BiasesBackwardCompatible(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_BiasesBackward(ML2_LayerBiases biasesGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_BiasesCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_BiasesCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
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
ML2_FN ML2_LayerInfo ML2_Softmax();
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
ML2_FN bool ML2_ActivationInfoForwardCompatible(ML2_LayerInfoActivation info, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_ActivationInfoForward(ML2_LayerInfoActivation info, ML2_LayerCacheInfo input);
ML2_FN bool ML2_ActivationForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_ActivationForward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ActivationUnaryForwardCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_ActivationUnaryForward(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ActivationCacheBackwardCompatible(ML2_LayerInfoActivation activation, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationCacheBackward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationUnaryCacheBackwardInputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationUnaryCacheBackwardInput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationUnaryCacheBackwardOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationUnaryCacheBackwardOutput(ML2_ActivationType type, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationUnaryCacheBackwardInputOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationUnaryCacheBackwardInputOutput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ActivationSoftmaxInfoForwardCompatible(ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_ActivationSoftmaxInfoForward(ML2_LayerCacheInfo input);
ML2_FN void ML2_ActivationSoftmaxForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ActivationSoftmaxCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationSoftmaxCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

ML2_FN ML2_LayerInfo ML2_Linear(ML2_Int inputs, ML2_Int outputs);
ML2_FN ML2_LayerLinear ML2_LinearNew(ML2_LayerInfoLinear info);
ML2_FN void ML2_LinearDestroy(ML2_LayerLinear *linear);
ML2_FN ML2_LayerInfoLinear ML2_LayerInfoAsLinear(ML2_LayerInfo info);
ML2_FN ML2_LayerLinear ML2_LayerAsLinear(ML2_Layer layer);
ML2_FN ML2_LayerInfoWeights ML2_LinearInfoWeights(ML2_LayerInfoLinear linear);
ML2_FN ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear);
ML2_FN ML2_LayerInfoBiases ML2_LinearInfoBiases(ML2_LayerInfoLinear linear);
ML2_FN ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear);
ML2_FN void ML2_LinearClear(ML2_LayerLinear linear);
ML2_FN void ML2_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src);
ML2_FN void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LinearXavierInit(ML2_LayerLinear linear);
ML2_FN void ML2_LinearHeInit(ML2_LayerLinear linear);
ML2_FN void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, ML2_Int indent);
ML2_FN void ML2_LinearPrint(ML2_LayerLinear linear, ML2_Int indent);
ML2_FN bool ML2_LinearInfoForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_LinearInfoForward(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_FN bool ML2_LinearInfoSame(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b);
ML2_FN bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_LinearBackwardCompatible(ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LinearBackward(ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_LinearCacheBackwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LinearCacheBackward(ML2_LayerLinear linear, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN void ML2_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate);
ML2_FN void ML2_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average);
ML2_FN void ML2_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum);
ML2_FN void ML2_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage);
ML2_FN void ML2_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);
ML2_FN ML2_Size ML2_LinearSizeof(ML2_LayerLinear linear);

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_FN ML2_LayerInfo ML2_Filters(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerFilters ML2_FiltersNew(ML2_LayerInfoFilters info);
ML2_FN void ML2_FiltersDestroy(ML2_LayerFilters *filters);
ML2_FN ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o);
ML2_FN ML2_LayerInfoFilters ML2_LayerInfoAsFilters(ML2_LayerInfo info);
ML2_FN ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer);
ML2_FN void ML2_FiltersClear(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src);
ML2_FN void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_FiltersXavierInit(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersHeInit(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, ML2_Int indent);
ML2_FN void ML2_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent);
ML2_FN ML2_Int ML2_FiltersInfoOutputHeight(ML2_LayerInfoFilters filters, ML2_Int inputHeight);
ML2_FN ML2_Int ML2_FiltersInfoOutputWidth(ML2_LayerInfoFilters filters, ML2_Int inputWidth);
ML2_FN bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_FN bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b);
ML2_FN bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FiltersBackwardCompatible(ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FiltersBackward(ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FiltersCacheBackwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FiltersCacheBackward(ML2_LayerFilters filters, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN void ML2_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate);
ML2_FN void ML2_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average);
ML2_FN void ML2_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum);
ML2_FN void ML2_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage);
ML2_FN void ML2_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);
ML2_FN ML2_Size ML2_FiltersSizeof(ML2_LayerFilters filters);

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_FN ML2_LayerInfo ML2_Conv(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth);
ML2_FN ML2_LayerConv ML2_ConvNew(ML2_LayerInfoConv info);
ML2_FN void ML2_ConvDestroy(ML2_LayerConv *conv);
ML2_FN ML2_LayerInfoConv ML2_LayerInfoAsConv(ML2_LayerInfo info);
ML2_FN ML2_LayerConv ML2_LayerAsConv(ML2_Layer layer);
ML2_FN ML2_LayerInfoFilters ML2_ConvInfoFilters(ML2_LayerInfoConv conv);
ML2_FN ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv);
ML2_FN ML2_LayerInfoBiases ML2_ConvInfoBiases(ML2_LayerInfoConv conv);
ML2_FN ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv);
ML2_FN void ML2_ConvClear(ML2_LayerConv conv);
ML2_FN void ML2_ConvSum(ML2_LayerConv dest, ML2_LayerConv src);
ML2_FN void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ConvXavierInit(ML2_LayerConv conv);
ML2_FN void ML2_ConvHeInit(ML2_LayerConv conv);
ML2_FN void ML2_ConvInfoPrint(ML2_LayerInfoConv info, ML2_Int indent);
ML2_FN void ML2_ConvPrint(ML2_LayerConv conv, ML2_Int indent);
ML2_FN bool ML2_ConvInfoForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_ConvInfoForward(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_FN bool ML2_ConvInfoSame(ML2_LayerInfoConv a, ML2_LayerInfoConv b);
ML2_FN bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ConvBackwardCompatible(ML2_LayerInfoConv convGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ConvBackward(ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_ConvCacheBackwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ConvCacheBackward(ML2_LayerConv conv, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
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
ML2_FN bool ML2_FlattenCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FlattenCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);

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
ML2_FN bool ML2_PoolMaxCacheBackwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolMaxCacheBackward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);

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
ML2_FN bool ML2_PoolAverageCacheBackwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolAverageCacheBackward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);

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
ML2_FN bool ML2_PoolSoftmaxCacheBackwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_PoolSoftmaxCacheBackward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);

// ML2_LayerPoolSoftmax ⬆️

// ML2_LayerPool ⬆️

// ML2_LayerFusedWeights ⬇️

ML2_FN ML2_LayerInfo ML2_FusedWeights(ML2_Int weights, ML2_Int inputs, ML2_Int outputs);
ML2_FN ML2_LayerFusedWeights ML2_FusedWeightsNew(ML2_LayerInfoFusedWeights info);
ML2_FN void ML2_FusedWeightsDestroy(ML2_LayerFusedWeights *fusedWeights);
ML2_FN ML2_Scalar *ML2_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i);
ML2_FN ML2_LayerInfoFusedWeights ML2_LayerInfoAsFusedWeights(ML2_LayerInfo info);
ML2_FN ML2_LayerFusedWeights ML2_LayerAsFusedWeights(ML2_Layer layer);
ML2_FN ML2_LayerInfoWeights ML2_FusedWeightsInfoWeightsAt(ML2_LayerInfoFusedWeights fusedWeights, ML2_Int w);
ML2_FN ML2_LayerWeights ML2_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w);
ML2_FN void ML2_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src);
ML2_FN void ML2_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights);
ML2_FN void ML2_FusedWeightsInfoPrint(ML2_LayerInfoFusedWeights info, ML2_Int indent);
ML2_FN void ML2_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent);
ML2_FN bool ML2_FusedWeightsInfoForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FusedWeightsInfoForward(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input);
ML2_FN bool ML2_FusedWeightsInfoSame(ML2_LayerInfoFusedWeights a, ML2_LayerInfoFusedWeights b);
ML2_FN bool ML2_FusedWeightsForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FusedWeightsForward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FusedWeightsBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FusedWeightsBackward(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_FusedWeightsCacheBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FusedWeightsCacheBackward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_FN void ML2_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average);
ML2_FN void ML2_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum);
ML2_FN void ML2_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage);
ML2_FN void ML2_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);
ML2_FN ML2_Size ML2_FusedWeightsSizeof(ML2_LayerFusedWeights fusedWeights);

// ML2_LayerFusedWeights ⬆️

// ML2_LayerAttention ⬇️

ML2_FN ML2_LayerInfo ML2_Attention(bool masked);
// ML2_FN ML2_LayerAttention ML2_AttentionNew(ML2_LayerInfoAttention info);
// ML2_FN void ML2_AttentionDestroy(ML2_LayerAttention *attention);
ML2_FN ML2_LayerInfoAttention ML2_LayerInfoAsAttention(ML2_LayerInfo info);
ML2_FN ML2_LayerInfoAttention ML2_LayerAsAttention(ML2_Layer layer);
// ML2_FN ML2_LayerInfoWeights ML2_AttentionInfoWeightsQuery(ML2_LayerInfoAttention attention);
// ML2_FN ML2_LayerInfoWeights ML2_AttentionInfoWeightsKey(ML2_LayerInfoAttention attention);
// ML2_FN ML2_LayerInfoWeights ML2_AttentionInfoWeightsValue(ML2_LayerInfoAttention attention);
// ML2_FN ML2_LayerWeights ML2_AttentionWeightsQuery(ML2_LayerAttention attention);
// ML2_FN ML2_LayerWeights ML2_AttentionWeightsKey(ML2_LayerAttention attention);
// ML2_FN ML2_LayerWeights ML2_AttentionWeightsValue(ML2_LayerAttention attention);
// ML2_FN void ML2_AttentionClear(ML2_LayerAttention attention);
// ML2_FN void ML2_AttentionSum(ML2_LayerAttention dest, ML2_LayerAttention src);
// ML2_FN void ML2_AttentionRand(ML2_LayerAttention attention, ML2_Scalar low, ML2_Scalar high);
// ML2_FN void ML2_AttentionXavierInit(ML2_LayerAttention attention);
// ML2_FN void ML2_AttentionHeInit(ML2_LayerAttention attention);
ML2_FN void ML2_AttentionInfoPrint(ML2_LayerInfoAttention info, ML2_Int indent);
// ML2_FN void ML2_AttentionPrint(ML2_LayerAttention attention, ML2_Int indent);
ML2_FN bool ML2_AttentionInfoForwardCompatible(ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_AttentionInfoForward(ML2_LayerCacheInfo input);
ML2_FN bool ML2_AttentionInfoSame(ML2_LayerInfoAttention a, ML2_LayerInfoAttention b);
ML2_FN bool ML2_AttentionForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_AttentionForward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache output);
// ML2_FN bool ML2_AttentionBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient);
// ML2_FN void ML2_AttentionBackward(ML2_LayerInfoAttention attentionGradient, ML2_LayerCache input, ML2_LayerCache outputGradient);
ML2_FN bool ML2_AttentionCacheBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_AttentionCacheBackward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
// ML2_FN void ML2_AttentionGradientDescentOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate);
// ML2_FN void ML2_AttentionMomentumOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerAttention average);
// ML2_FN void ML2_AttentionAdagradOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_LayerAttention squareSum);
// ML2_FN void ML2_AttentionRMSPropOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerAttention squareAverage);
// ML2_FN void ML2_AttentionAdamOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerAttention average, ML2_LayerAttention squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);
// ML2_FN ML2_Size ML2_AttentionSizeof(ML2_LayerAttention attention);

// ML2_LayerAttention ⬆️

ML2_FN ML2_Layer ML2_LayerNew(ML2_LayerInfo info);
ML2_FN void ML2_LayerDestroy(ML2_Layer *layer);
ML2_FN ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer);
ML2_FN void ML2_LayerClear(ML2_Layer layer);
ML2_FN void ML2_LayerSum(ML2_Layer dest, ML2_Layer src);
ML2_FN void ML2_LayerRand(ML2_Layer layer, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LayerXavierInit(ML2_Layer layer);
ML2_FN void ML2_LayerHeInit(ML2_Layer layer);
ML2_FN void ML2_LayerInfoPrint(ML2_LayerInfo layer, ML2_Int indent);
ML2_FN void ML2_LayerPrint(ML2_Layer layer, ML2_Int indent);
ML2_FN bool ML2_LayerInfoForwardCompatible(ML2_LayerInfo info, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input);
ML2_FN bool ML2_LayerInfoSame(ML2_LayerInfo a, ML2_LayerInfo b);
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
ML2_FN ML2_LayerCacheScalars ML2_ScalarsNew(ML2_LayerCacheInfoScalars info);
ML2_FN void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars);
ML2_FN ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int sample, ML2_Int scalar);
ML2_FN ML2_LayerCacheInfoScalars ML2_LayerCacheInfoAsScalars(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache);
ML2_FN void ML2_ScalarsClear(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ScalarsXavierInit(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_ScalarsHeInit(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, ML2_Int indent);
ML2_FN void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent);
ML2_FN bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b);
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
ML2_FN ML2_LayerCacheVectors ML2_VectorsNew(ML2_LayerCacheInfoVectors info);
ML2_FN void ML2_VectorsDestroy(ML2_LayerCacheVectors *vectors);
ML2_FN ML2_Scalar *ML2_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int sample, ML2_Int vector, ML2_Int scalar);
ML2_FN ML2_LayerCacheInfoVectors ML2_LayerCacheInfoAsVectors(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheVectors ML2_LayerCacheAsVectors(ML2_LayerCache cache);
ML2_FN void ML2_VectorsClear(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src);
ML2_FN void ML2_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_VectorsXavierInit(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_VectorsHeInit(ML2_LayerCacheVectors vectors);
ML2_FN void ML2_VectorsInfoPrint(ML2_LayerCacheInfoVectors info, ML2_Int indent);
ML2_FN void ML2_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent);
ML2_FN bool ML2_VectorsInfoSame(ML2_LayerCacheInfoVectors a, ML2_LayerCacheInfoVectors b);
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
ML2_FN ML2_LayerCacheMatrices ML2_MatricesNew(ML2_LayerCacheInfoMatrices info);
ML2_FN void ML2_MatricesDestroy(ML2_LayerCacheMatrices *matrices);
ML2_FN ML2_Scalar *ML2_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c);
ML2_FN ML2_LayerCacheInfoMatrices ML2_LayerCacheInfoAsMatrices(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheMatrices ML2_LayerCacheAsMatrices(ML2_LayerCache cache);
ML2_FN void ML2_MatricesClear(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src);
ML2_FN void ML2_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_MatricesXavierInit(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_MatricesHeInit(ML2_LayerCacheMatrices matrices);
ML2_FN void ML2_MatricesInfoPrint(ML2_LayerCacheInfoMatrices info, ML2_Int indent);
ML2_FN void ML2_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent);
ML2_FN bool ML2_MatricesInfoSame(ML2_LayerCacheInfoMatrices a, ML2_LayerCacheInfoMatrices b);
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

ML2_FN ML2_LayerCache ML2_LayerCacheNew(ML2_LayerCacheInfo info);
ML2_FN void ML2_LayerCacheDestroy(ML2_LayerCache *cache);
ML2_FN ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheClear(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src);
ML2_FN void ML2_LayerCacheSum(ML2_LayerCache dest, ML2_LayerCache src);
ML2_FN void ML2_LayerCacheRand(ML2_LayerCache cache, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LayerCacheXavierInit(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheHeInit(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, ML2_Int indent);
ML2_FN void ML2_LayerCachePrint(ML2_LayerCache cache, ML2_Int indent);
ML2_FN bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b);
ML2_FN ML2_Size ML2_LayerCacheSizeof(ML2_LayerCache cache);

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

#define ML2_ArchMake(...) ((ML2_Arch){sizeof((ML2_LayerInfo[]){__VA_ARGS__}) / sizeof(ML2_LayerInfo), (ML2_LayerInfo[]){__VA_ARGS__}})
ML2_FN ML2_Arch ML2_ArchNew(ML2_Int layers, ML2_LayerInfo infos[layers]);
ML2_FN void ML2_ArchDestroy(ML2_Arch *arch);

// ML2_Arch ⬆️

// ML2_Model ⬇️

ML2_FN ML2_Model ML2_ModelNew(ML2_Arch arch);
ML2_FN void ML2_ModelDestroy(ML2_Model *model);
ML2_FN void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ModelXavierInit(ML2_Model model);
ML2_FN void ML2_ModelHeInit(ML2_Model model);
ML2_FN void ML2_ModelClear(ML2_Model model);
ML2_FN void ML2_ModelSum(ML2_Model dest, ML2_Model src);
ML2_FN void ML2_ModelInfoPrint(ML2_Model model, ML2_Int indent);
ML2_FN void ML2_ModelPrint(ML2_Model model, ML2_Int indent);
ML2_FN bool ML2_ModelInfoSame(ML2_Model a, ML2_Model b);
ML2_FN bool ML2_ModelForwardCompatible(ML2_Model model, ML2_ModelCache modelCache);
ML2_FN void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache, ML2_LayerCache input);
ML2_FN bool ML2_ModelBackwardCompatible(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient);
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

ML2_FN ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, ML2_BatchInfo info);
ML2_FN void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache);
ML2_FN void ML2_ModelCacheClear(ML2_ModelCache modelCache);
ML2_FN void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, ML2_Int indent);
ML2_FN void ML2_ModelCachePrint(ML2_ModelCache modelCache, ML2_Int indent);
ML2_FN bool ML2_ModelCacheInfoSame(ML2_ModelCache a, ML2_ModelCache b);
ML2_FN ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache);
ML2_FN ML2_Scalar ML2_ModelCacheLoss(ML2_ModelCache modelCache, ML2_LayerCache expected, ML2_LossForward lossForward);
ML2_FN ML2_Size ML2_ModelCacheSizeof(ML2_ModelCache modelCache);

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_FN ML2_BatchInfo ML2_BatchInfoMake(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN ML2_Batch ML2_BatchNew(ML2_BatchInfo info);
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
ML2_FN ML2_OptimizerMomentum ML2_MomentumNew(ML2_OptimizerParametersMomentum parameters, ML2_Arch arch);
ML2_FN void ML2_MomentumDestroy(ML2_OptimizerMomentum *momentum);
ML2_FN void ML2_MomentumReset(ML2_OptimizerMomentum momentum);
ML2_FN ML2_OptimizerParametersMomentum ML2_OptimizerParametersAsMomentum(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerMomentum ML2_OptimizerAsMomentum(ML2_Optimizer optimizer);
ML2_FN void ML2_MomentumInfoPrint(ML2_OptimizerParametersMomentum parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_MomentumSizeof(ML2_OptimizerMomentum momentum);

// ML2_OptimizerMomentum ⬆️

// ML2_OptimizerAdagrad ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adagrad(ML2_Scalar learningRate);
ML2_FN ML2_OptimizerAdagrad ML2_AdagradNew(ML2_OptimizerParametersAdagrad parameters, ML2_Arch arch);
ML2_FN void ML2_AdagradDestroy(ML2_OptimizerAdagrad *adagrad);
ML2_FN void ML2_AdagradReset(ML2_OptimizerAdagrad adagrad);
ML2_FN ML2_OptimizerParametersAdagrad ML2_OptimizerParametersAsAdagrad(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerAdagrad ML2_OptimizerAsAdagrad(ML2_Optimizer optimizer);
ML2_FN void ML2_AdagradInfoPrint(ML2_OptimizerParametersAdagrad parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_AdagradSizeof(ML2_OptimizerAdagrad adagrad);

// ML2_OptimizerAdagrad ⬆️

// ML2_OptimizerRMSProp ⬇️

ML2_FN ML2_OptimizerParameters ML2_RMSProp(ML2_Scalar learningRate, ML2_Scalar decayRate);
ML2_FN ML2_OptimizerRMSProp ML2_RMSPropNew(ML2_OptimizerParametersRMSProp parameters, ML2_Arch arch);
ML2_FN void ML2_RMSPropDestroy(ML2_OptimizerRMSProp *RMSProp);
ML2_FN void ML2_RMSPropReset(ML2_OptimizerRMSProp RMSProp);
ML2_FN ML2_OptimizerParametersRMSProp ML2_OptimizerParametersAsRMSProp(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerRMSProp ML2_OptimizerAsRMSProp(ML2_Optimizer optimizer);
ML2_FN void ML2_RMSPropInfoPrint(ML2_OptimizerParametersRMSProp parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_RMSPropSizeof(ML2_OptimizerRMSProp RMSProp);

// ML2_OptimizerRMSProp ⬆️

// ML2_OptimizerAdam ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adam(ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2);
ML2_FN ML2_OptimizerAdam ML2_AdamNew(ML2_OptimizerParametersAdam parameters, ML2_Arch arch);
ML2_FN void ML2_AdamDestroy(ML2_OptimizerAdam *adam);
ML2_FN void ML2_AdamReset(ML2_OptimizerAdam *adam);
ML2_FN ML2_OptimizerParametersAdam ML2_OptimizerParametersAsAdam(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerAdam ML2_OptimizerAsAdam(ML2_Optimizer optimizer);
ML2_FN void ML2_AdamInfoPrint(ML2_OptimizerParametersAdam parameters, ML2_Int indent);
ML2_FN ML2_Size ML2_AdamSizeof(ML2_OptimizerAdam adam);

// ML2_OptimizerAdam ⬆️

ML2_FN ML2_Optimizer ML2_OptimizerNew(ML2_OptimizerParameters parameters, ML2_Arch arch);
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

#endif // _ML2_H

#ifdef ML2_IMPLEMENTATION

// ML2_ ⬇️
// ML2_ ⬆️

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

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_FN ML2_LayerInfo ML2_Weights(ML2_Int inputs, ML2_Int outputs) {
    return (ML2_LayerInfo){ML2_LayerTypeWeights, .as.weights = {outputs, inputs}};
}

ML2_FN ML2_LayerWeights ML2_WeightsNew(ML2_LayerInfoWeights info) {
    ML2_LayerWeights weights = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs, sizeof(*weights.data.weights))}
    };
    return weights;
}

ML2_FN void ML2_WeightsDestroy(ML2_LayerWeights *weights) {
    ML2_FREE(weights->data.weights);
    *weights = (ML2_LayerWeights){};
}

ML2_FN ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, ML2_Int o, ML2_Int i) {
    ML2_SOFT_ASSERT(0 <= o && o < weights.info.outputs && 0 <= i && i < weights.info.inputs && "OUT OF BOUNDS INDICES");
    return &weights.data.weights[o * weights.info.inputs + i];
}

ML2_FN ML2_LayerInfoWeights ML2_LayerInfoAsWeights(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeWeights);
    return info.as.weights;
}

ML2_FN ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeWeights);
    return layer.as.weights;
}

ML2_FN void ML2_WeightsClear(ML2_LayerWeights weights) {
    for (ML2_Int o = 0; o < weights.info.outputs; o++) {
        for (ML2_Int i = 0; i < weights.info.inputs; i++) {
            *ML2_WeightsAt(weights, o, i) = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_FN void ML2_WeightsSum(ML2_LayerWeights dest, ML2_LayerWeights src) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(dest.info, src.info));
    ML2_Int outputs = dest.info.outputs;
    ML2_Int inputs = dest.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        for (ML2_Int i = 0; i < inputs; i++) {
            *ML2_WeightsAt(dest, o, i) += *ML2_WeightsAt(src, o, i);
        }
    }
}

ML2_FN void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int o = 0; o < weights.info.outputs; o++) {
        for (ML2_Int i = 0; i < weights.info.inputs; i++) {
            *ML2_WeightsAt(weights, o, i) = ML2_ScalarRand(low, high);
        }
    }
}

ML2_FN void ML2_WeightsXavierInit(ML2_LayerWeights weights) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (weights.info.inputs + weights.info.outputs));
    ML2_WeightsRand(weights, -limit, limit);
}

ML2_FN void ML2_WeightsHeInit(ML2_LayerWeights weights) {
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / weights.info.inputs);
    for (ML2_Int o = 0; o < weights.info.outputs; o++) {
        for (ML2_Int i = 0; i < weights.info.inputs; i++) {
            *ML2_WeightsAt(weights, o, i) = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
        }
    }
}

ML2_FN void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, ML2_Int indent) {
    printf(ML2_INDENT("Weights(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_FN void ML2_WeightsPrint(ML2_LayerWeights weights, ML2_Int indent) {
    ML2_WeightsInfoPrint(weights.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int o = 0; o < weights.info.outputs; o++) {
            printf(ML2_INDENT("", indent));
            for (ML2_Int i = 0; i < weights.info.inputs; i++) {
                printf(ML2_SCALAR_FMT " ", *ML2_WeightsAt(weights, o, i));
            }
            printf("\n");
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
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
    return a.outputs == b.outputs && a.inputs == b.inputs;
}

ML2_FN bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoScalars inputScalars = ML2_LayerCacheInfoAsScalars(input);
            ML2_LayerCacheInfoScalars outputScalars = ML2_LayerCacheInfoAsScalars(output);
            return weights.inputs == inputScalars.scalars && weights.outputs == outputScalars.scalars && inputScalars.samples == outputScalars.samples;
        } break;
        case ML2_LayerCacheTypeVectors: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoVectors inputVectors = ML2_LayerCacheInfoAsVectors(input);
            ML2_LayerCacheInfoVectors outputVectors = ML2_LayerCacheInfoAsVectors(output);
            return weights.inputs == inputVectors.scalars && weights.outputs == outputVectors.scalars && inputVectors.samples == outputVectors.samples && inputVectors.vectors == outputVectors.vectors;
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatible(weights.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_Int samples = inputScalars.info.samples;
            ML2_Int outputs = weights.info.outputs;
            ML2_Int inputs = weights.info.inputs;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    ML2_Scalar acc = {};
                    for (ML2_Int i = 0; i < inputs; i++) {
                        acc += *ML2_ScalarsAt(inputScalars, s, i) * *ML2_WeightsAt(weights, o, i);
                    }
                    *ML2_ScalarsAt(outputScalars, s, o) += acc;
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheVectors outputVectors = ML2_LayerCacheAsVectors(output);
            ML2_Int samples = inputVectors.info.samples;
            ML2_Int vectors = inputVectors.info.vectors;
            ML2_Int outputs = weights.info.outputs;
            ML2_Int inputs = weights.info.inputs;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    for (ML2_Int o = 0; o < outputs; o++) {
                        ML2_Scalar acc = {};
                        for (ML2_Int i = 0; i < inputs; i++) {
                            acc += *ML2_VectorsAt(inputVectors, s, v, i) * *ML2_WeightsAt(weights, o, i);
                        }
                        *ML2_VectorsAt(outputVectors, s, v, o) += acc;
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsBackwardCompatible(ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return  ML2_WeightsForwardCompatible(weightsGradient, input, outputGradient);
}

ML2_FN void ML2_WeightsBackward(ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatible(weightsGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = inputScalars.info.samples;
            ML2_Int outputs = weightsGradient.info.outputs;
            ML2_Int inputs = weightsGradient.info.inputs;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    for (ML2_Int i = 0; i < inputs; i++) {
                        *ML2_WeightsAt(weightsGradient, o, i) += *ML2_ScalarsAt(inputScalars, s, i) * *ML2_ScalarsAt(outputGradientScalars, s, o);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            ML2_Int samples = inputVectors.info.samples;
            ML2_Int vectors = inputVectors.info.vectors;
            ML2_Int outputs = weightsGradient.info.outputs;
            ML2_Int inputs = weightsGradient.info.inputs;

            // TODO(31/8/2026 18:34:20): 99% sure its correct but gotta test
            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    for (ML2_Int o = 0; o < outputs; o++) {
                        for (ML2_Int i = 0; i < inputs; i++) {
                            *ML2_WeightsAt(weightsGradient, o, i) += *ML2_VectorsAt(inputVectors, s, v, i) * *ML2_VectorsAt(outputGradientVectors, s, v, o);
                        }
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsCacheBackwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsForwardCompatible(weights, inputGradient, outputGradient);
}

ML2_FN void ML2_WeightsCacheBackward(ML2_LayerWeights weights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_WeightsCacheBackwardCompatible(weights.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = inputGradientScalars.info.samples;
            ML2_Int outputs = weights.info.outputs;
            ML2_Int inputs = weights.info.inputs;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    for (ML2_Int i = 0; i < inputs; i++) {
                        *ML2_ScalarsAt(inputGradientScalars, s, i) += *ML2_WeightsAt(weights, o, i) * *ML2_ScalarsAt(outputGradientScalars, s, o);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputGradientVectors = ML2_LayerCacheAsVectors(inputGradient);
            ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            ML2_Int samples = inputGradientVectors.info.samples;
            ML2_Int vectors = inputGradientVectors.info.vectors;
            ML2_Int outputs = weights.info.outputs;
            ML2_Int inputs = weights.info.inputs;

            // TODO(31/8/2026 18:34:20): 99% sure its correct but gotta test
            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    for (ML2_Int o = 0; o < outputs; o++) {
                        for (ML2_Int i = 0; i < inputs; i++) {
                            *ML2_VectorsAt(inputGradientVectors, s, v, i) += *ML2_WeightsAt(weights, o, i) * *ML2_VectorsAt(outputGradientVectors, s, v, o);
                        }
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        for (ML2_Int i = 0; i < inputs; i++) {
            *ML2_WeightsAt(weights, o, i) -= *ML2_WeightsAt(weightsGradient, o, i) * learningRate;
        }
    }
}

ML2_FN void ML2_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, average.info));
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        for (ML2_Int i = 0; i < inputs; i++) {
            *ML2_WeightsAt(average, o, i) = decayRate * *ML2_WeightsAt(average, o, i) - learningRate * *ML2_WeightsAt(weightsGradient, o, i);
            *ML2_WeightsAt(weights, o, i) += *ML2_WeightsAt(average, o, i);
        }
    }
}

ML2_FN void ML2_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, squareSum.info));
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        for (ML2_Int i = 0; i < inputs; i++) {
            *ML2_WeightsAt(squareSum, o, i) += *ML2_WeightsAt(weightsGradient, o, i) * *ML2_WeightsAt(weightsGradient, o, i);
            *ML2_WeightsAt(weights, o, i) -= *ML2_WeightsAt(weightsGradient, o, i) * learningRate / ML2_SQRT(*ML2_WeightsAt(squareSum, o, i) + ML2_Epsilon);
        }
    }
}

ML2_FN void ML2_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, squareAverage.info));
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        for (ML2_Int i = 0; i < inputs; i++) {
            *ML2_WeightsAt(squareAverage, o, i) = decayRate * *ML2_WeightsAt(squareAverage, o, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate) * *ML2_WeightsAt(weightsGradient, o, i) * *ML2_WeightsAt(weightsGradient, o, i);
            *ML2_WeightsAt(weights, o, i) -= *ML2_WeightsAt(weightsGradient, o, i) * learningRate / ML2_SQRT(*ML2_WeightsAt(squareAverage, o, i) + ML2_Epsilon);
        }
    }
}

ML2_FN void ML2_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, average.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, squareAverage.info));
    ML2_Int outputs = weights.info.outputs;
    ML2_Int inputs = weights.info.inputs;
    for (ML2_Int o = 0; o < outputs; o++) {
        for (ML2_Int i = 0; i < inputs; i++) {
            *ML2_WeightsAt(average, o, i) = decayRate1 * *ML2_WeightsAt(average, o, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * *ML2_WeightsAt(weightsGradient, o, i);
            *ML2_WeightsAt(squareAverage, o, i) = decayRate2 * *ML2_WeightsAt(squareAverage, o, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * *ML2_WeightsAt(weightsGradient, o, i) * *ML2_WeightsAt(weightsGradient, o, i);
            ML2_Scalar averageWeighted = *ML2_WeightsAt(average, o, i) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
            ML2_Scalar squareAverageWeighed = *ML2_WeightsAt(squareAverage, o, i) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
            *ML2_WeightsAt(weights, o, i) -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
        }
    }
}

ML2_FN ML2_Size ML2_WeightsSizeof(ML2_LayerWeights weights) {
    return weights.info.outputs * weights.info.inputs * sizeof(*weights.data.weights);
}

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_FN ML2_LayerInfo ML2_Biases(ML2_Int inputs) {
    return (ML2_LayerInfo){ML2_LayerTypeBiases, .as.biases = {inputs}};
}

ML2_FN ML2_LayerBiases ML2_BiasesNew(ML2_LayerInfoBiases info) {
    ML2_LayerBiases biases = {
        info,
        {ML2_RELIABLE_CALLOC(info.inputs, sizeof(*biases.data.biases))}
    };
    return biases;
}

ML2_FN void ML2_BiasesDestroy(ML2_LayerBiases *biases) {
    ML2_FREE(biases->data.biases);
    *biases = (ML2_LayerBiases){};
}

ML2_FN ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, ML2_Int i) {
    ML2_SOFT_ASSERT(0 <= i && i < biases.info.inputs && "OUT OF BOUNDS INDICES");
    return &biases.data.biases[i];
}

ML2_FN ML2_LayerInfoBiases ML2_LayerInfoAsBiases(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeBiases);
    return info.as.biases;
}

ML2_FN ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeBiases);
    return layer.as.biases;
}

ML2_FN void ML2_BiasesClear(ML2_LayerBiases biases) {
    for (ML2_Int i = 0; i < biases.info.inputs; i++) {
        *ML2_BiasesAt(biases, i) = ML2_SCALAR_LITERAL(0.0);
    }
}

ML2_FN void ML2_BiasesSum(ML2_LayerBiases dest, ML2_LayerBiases src) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(dest.info, src.info));
    ML2_Int inputs = dest.info.inputs;
    for (ML2_Int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(dest, i) += *ML2_BiasesAt(src, i);
    }
}

ML2_FN void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int i = 0; i < biases.info.inputs; i++) {
        biases.data.biases[i] = ML2_ScalarRand(low, high);
    }
}

ML2_FN void ML2_BiasesXavierInit(ML2_LayerBiases biases) {
    ML2_BiasesClear(biases);
}

ML2_FN void ML2_BiasesHeInit(ML2_LayerBiases biases) {
    ML2_BiasesClear(biases);
}

ML2_FN void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, ML2_Int indent) {
    printf(ML2_INDENT("Biases(%d)\n", indent), info.inputs);
}

ML2_FN void ML2_BiasesPrint(ML2_LayerBiases biases, ML2_Int indent) {
    ML2_BiasesInfoPrint(biases.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("", indent));
        for (ML2_Int i = 0; i < biases.info.inputs; i++) {
            printf(ML2_SCALAR_FMT " ", *ML2_BiasesAt(biases, i));
        }
        printf("\n");
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
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

ML2_FN bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoScalars inputScalars = ML2_LayerCacheInfoAsScalars(input);
            ML2_LayerCacheInfoScalars outputScalars = ML2_LayerCacheInfoAsScalars(output);
            return biases.inputs == outputScalars.scalars && ML2_ScalarsInfoSame(inputScalars, outputScalars);
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Biases may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_LayerCacheInfoMatrices outputMatrices = ML2_LayerCacheInfoAsMatrices(output);
            return biases.inputs == outputMatrices.scalars && ML2_MatricesInfoSame(inputMatrices, outputMatrices);
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatible(biases.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_Int samples = inputScalars.info.samples;
            ML2_Int inputs = biases.info.inputs;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int i = 0; i < inputs; i++) {
                    *ML2_ScalarsAt(outputScalars, s, i) += *ML2_ScalarsAt(inputScalars, s, i) + *ML2_BiasesAt(biases, i);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Biases may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int matrices = inputMatrices.info.matrices;
            ML2_Int vectors = inputMatrices.info.vectors;
            ML2_Int scalars = inputMatrices.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(outputMatrices, s, m, v, c) += *ML2_MatricesAt(inputMatrices, s, m, v, c) + *ML2_BiasesAt(biases, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesBackwardCompatible(ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_BiasesInfoForwardCompatible(biasesGradient, outputGradient);
}

ML2_FN void ML2_BiasesBackward(ML2_LayerBiases biasesGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatible(biasesGradient.info, ML2_LayerCacheAsInfo(outputGradient)));
    switch (outputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = outputGradientScalars.info.samples;
            ML2_Int inputs = outputGradientScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int i = 0; i < inputs; i++) {
                    *ML2_BiasesAt(biasesGradient, i) += *ML2_ScalarsAt(outputGradientScalars, s, i);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Biases may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = outputGradientMatrices.info.samples;
            ML2_Int matrices = outputGradientMatrices.info.matrices;
            ML2_Int vectors = outputGradientMatrices.info.vectors;
            ML2_Int scalars = outputGradientMatrices.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                            for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_BiasesAt(biasesGradient, c) += *ML2_MatricesAt(outputGradientMatrices, s, m, v, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_BiasesCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_BiasesCacheBackwardCompatible(ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (outputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsSum(inputGradientScalars, outputGradientScalars);
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Biases may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_MatricesSum(inputGradientMatrices, outputGradientMatrices);
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_Int inputs = biases.info.inputs;
    for (ML2_Int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(biasesGradient, i) * learningRate;
    }
}

ML2_FN void ML2_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, average.info));
    ML2_Int inputs = biases.info.inputs;
    for (ML2_Int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(average, i) = decayRate * *ML2_BiasesAt(average, i) - learningRate * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(biases, i) += *ML2_BiasesAt(average, i);
    }
}

ML2_FN void ML2_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, squareSum.info));
    ML2_Int inputs = biases.info.inputs;
    for (ML2_Int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(squareSum, i) += *ML2_BiasesAt(biasesGradient, i) * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(biasesGradient, i) * learningRate / ML2_SQRT(*ML2_BiasesAt(squareSum, i) + ML2_Epsilon);
    }
}

ML2_FN void ML2_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, squareAverage.info));
    ML2_Int inputs = biases.info.inputs;
    for (ML2_Int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(squareAverage, i) = decayRate * *ML2_BiasesAt(squareAverage, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate) * *ML2_BiasesAt(biasesGradient, i) * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(biasesGradient, i) * learningRate / ML2_SQRT(*ML2_BiasesAt(squareAverage, i) + ML2_Epsilon);
    }
}

ML2_FN void ML2_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, average.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, squareAverage.info));
    ML2_Int inputs = biases.info.inputs;
    for (ML2_Int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(average, i) = decayRate1 * *ML2_BiasesAt(average, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(squareAverage, i) = decayRate2 * *ML2_BiasesAt(squareAverage, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * *ML2_BiasesAt(biasesGradient, i) * *ML2_BiasesAt(biasesGradient, i);
        ML2_Scalar averageWeighted = *ML2_BiasesAt(average, i) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
        ML2_Scalar squareAverageWeighed = *ML2_BiasesAt(squareAverage, i) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
        *ML2_BiasesAt(biases, i) -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
    }
}

ML2_FN ML2_Size ML2_BiasesSizeof(ML2_LayerBiases biases) {
    return biases.info.inputs * sizeof(*biases.data.biases);
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

ML2_FN ML2_LayerInfo ML2_Softmax() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeSoftmax}};
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
        case ML2_ActivationTypeSoftmax: return "Softmax";
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

// TODO(22/4/2026 11:51:20pm): is returning nullptr correct?
ML2_FN ML2_ScalarActivationForward *ML2_ActivationForwardOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return ML2_ScalarReLUForward;
        case ML2_ActivationTypeSigmoid: return ML2_ScalarSigmoidForward;
        case ML2_ActivationTypeSoftmax: return nullptr;
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
        case ML2_ActivationTypeSoftmax: return nullptr;
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
        case ML2_ActivationTypeSoftmax: return nullptr;
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
        case ML2_ActivationTypeSoftmax: return nullptr;
        case ML2_ActivationTypeSin: return nullptr;
        case ML2_ActivationTypeCos: return nullptr;
        case ML2_ActivationTypeTanh: return nullptr;
        case ML2_ActivationTypeLeakyReLU: return nullptr;
        case ML2_ActivationTypeSiLU: return ML2_ScalarSiLUBackwardInputOutput;
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN bool ML2_ActivationInfoForwardCompatible(ML2_LayerInfoActivation info, ML2_LayerCacheInfo input) {
    switch (info.type) {
        case ML2_ActivationTypeReLU: return true;
        case ML2_ActivationTypeSigmoid: return true;
        case ML2_ActivationTypeSoftmax: return ML2_ActivationSoftmaxInfoForwardCompatible(input);
        case ML2_ActivationTypeSin: return true;
        case ML2_ActivationTypeCos: return true;
        case ML2_ActivationTypeTanh: return true;
        case ML2_ActivationTypeLeakyReLU: return true;
        case ML2_ActivationTypeSiLU: return true;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_ActivationInfoForward(ML2_LayerInfoActivation info, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_ActivationInfoForwardCompatible(info, input));
    switch (info.type) {
        case ML2_ActivationTypeReLU: return input;
        case ML2_ActivationTypeSigmoid: return input;
        case ML2_ActivationTypeSoftmax: return ML2_ActivationSoftmaxInfoForward(input);
        case ML2_ActivationTypeSin: return input;
        case ML2_ActivationTypeCos: return input;
        case ML2_ActivationTypeTanh: return input;
        case ML2_ActivationTypeLeakyReLU: return input;
        case ML2_ActivationTypeSiLU: return input;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_LayerCacheInfoSame(input, output);
}

ML2_FN void ML2_ActivationForward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (activation.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationUnaryForward(activation.type, input, output); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationUnaryForward(activation.type, input, output); break;
        case ML2_ActivationTypeSoftmax: ML2_ActivationSoftmaxForward(input, output); break;
        case ML2_ActivationTypeSin: ML2_ActivationUnaryForward(activation.type, input, output); break;
        case ML2_ActivationTypeCos: ML2_ActivationUnaryForward(activation.type, input, output); break;
        case ML2_ActivationTypeTanh: ML2_ActivationUnaryForward(activation.type, input, output); break;
        case ML2_ActivationTypeLeakyReLU: ML2_ActivationUnaryForward(activation.type, input, output); break;
        case ML2_ActivationTypeSiLU: ML2_ActivationUnaryForward(activation.type, input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationUnaryForwardCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (type) {
        case ML2_ActivationTypeReLU: break;
        case ML2_ActivationTypeSigmoid: break;
        case ML2_ActivationTypeSoftmax: return false; // Softmax is not a unary function
        case ML2_ActivationTypeSin: break;
        case ML2_ActivationTypeCos: break;
        case ML2_ActivationTypeTanh: break;
        case ML2_ActivationTypeLeakyReLU: break;
        case ML2_ActivationTypeSiLU: break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
    return ML2_LayerCacheInfoSame(input, output);
}

ML2_FN void ML2_ActivationUnaryForward(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationUnaryForwardCompatible(type, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    ML2_ScalarActivationForward *forward = ML2_ActivationForwardOf(type);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_Int samples = inputScalars.info.samples;
            ML2_Int scalars = inputScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(outputScalars, s, c) += forward(*ML2_ScalarsAt(inputScalars, s, c));
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheVectors outputVectors = ML2_LayerCacheAsVectors(output);
            ML2_Int samples = inputVectors.info.samples;
            ML2_Int vectors = inputVectors.info.vectors;
            ML2_Int scalars = inputVectors.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    for (ML2_Int c = 0; c < scalars; c++) {
                        *ML2_VectorsAt(outputVectors, s, v, c) += forward(*ML2_VectorsAt(inputVectors, s, v, c));
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int matrices = inputMatrices.info.matrices;
            ML2_Int vectors = inputMatrices.info.vectors;
            ML2_Int scalars = inputMatrices.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(outputMatrices, s, m, v, c) += forward(*ML2_MatricesAt(inputMatrices, s, m, v, c));
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationCacheBackwardCompatible(ML2_LayerInfoActivation activation, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    switch (activation.type) {
        case ML2_ActivationTypeReLU: break;
        case ML2_ActivationTypeSigmoid: break;
        case ML2_ActivationTypeSoftmax: if (!ML2_ActivationSoftmaxCacheBackwardCompatible(inputGradient, output, outputGradient)) return false; break;
        case ML2_ActivationTypeSin: break;
        case ML2_ActivationTypeCos: break;
        case ML2_ActivationTypeTanh: break;
        case ML2_ActivationTypeLeakyReLU:  break;
        case ML2_ActivationTypeSiLU: if (!ML2_LayerCacheInfoSame(input, output)) return false; break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_LayerCacheInfoSame(input, output) && ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationCacheBackward(ML2_LayerInfoActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationCacheBackwardCompatible(activation, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (activation.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationUnaryCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationUnaryCacheBackwardOutput(activation.type, inputGradient, output, outputGradient); break;
        case ML2_ActivationTypeSoftmax: ML2_ActivationSoftmaxCacheBackward(inputGradient, output, outputGradient); break;
        case ML2_ActivationTypeSin: ML2_ActivationUnaryCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeCos: ML2_ActivationUnaryCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeTanh: ML2_ActivationUnaryCacheBackwardOutput(activation.type, inputGradient, output, outputGradient); break;
        case ML2_ActivationTypeLeakyReLU: ML2_ActivationUnaryCacheBackwardInput(activation.type, input, inputGradient, outputGradient); break;
        case ML2_ActivationTypeSiLU: ML2_ActivationUnaryCacheBackwardInputOutput(activation.type, input, inputGradient, output, outputGradient); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN bool ML2_ActivationUnaryCacheBackwardInputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    switch (type) {
        case ML2_ActivationTypeReLU: break;
        case ML2_ActivationTypeSigmoid: return false; // Sigmoid backward takes the output
        case ML2_ActivationTypeSoftmax: return false; // Softmax is not a unary function
        case ML2_ActivationTypeSin: break;
        case ML2_ActivationTypeCos: break;
        case ML2_ActivationTypeTanh: return false; // Tanh backward takes the output
        case ML2_ActivationTypeLeakyReLU: break;
        case ML2_ActivationTypeSiLU: return false; // SiLU takes both the input and the output
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationUnaryCacheBackwardInput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationUnaryCacheBackwardInputCompatible(type, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    ML2_ScalarActivationBackwardInput *backwardInput = ML2_ActivationBackwardInputOf(type);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = inputScalars.info.samples;
            ML2_Int scalars = inputScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(inputGradientScalars, s, c) += backwardInput(*ML2_ScalarsAt(inputScalars, s, c)) * *ML2_ScalarsAt(outputGradientScalars, s, c);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheVectors inputGradientVectors = ML2_LayerCacheAsVectors(inputGradient);
            ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            ML2_Int samples = inputVectors.info.samples;
            ML2_Int vectors = inputVectors.info.vectors;
            ML2_Int scalars = inputVectors.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    for (ML2_Int c = 0; c < scalars; c++) {
                        *ML2_VectorsAt(inputGradientVectors, s, v, c) += backwardInput(*ML2_VectorsAt(inputVectors, s, v, c)) * *ML2_VectorsAt(outputGradientVectors, s, v, c);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int matrices = inputMatrices.info.matrices;
            ML2_Int vectors = inputMatrices.info.vectors;
            ML2_Int scalars = inputMatrices.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(inputGradientMatrices, s, m, v, c) += backwardInput(*ML2_MatricesAt(inputMatrices, s, m, v, c)) * *ML2_MatricesAt(outputGradientMatrices, s, m, v, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationUnaryCacheBackwardOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    switch (type) {
        case ML2_ActivationTypeReLU: return false; // ReLU backward takes the input
        case ML2_ActivationTypeSigmoid: break;
        case ML2_ActivationTypeSoftmax: return false; // Softmax is not a unary function
        case ML2_ActivationTypeSin: return false; // Sin backward takes the input
        case ML2_ActivationTypeCos: return false; // Cos backward takes the input
        case ML2_ActivationTypeTanh: break;
        case ML2_ActivationTypeLeakyReLU: return false; // LeakyReLU backward takes the input
        case ML2_ActivationTypeSiLU: return false; // SiLU takes both the input and the output
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
    return ML2_LayerCacheInfoSame(inputGradient, output) && ML2_LayerCacheInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_ActivationUnaryCacheBackwardOutput(ML2_ActivationType type, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationUnaryCacheBackwardOutputCompatible(type, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    ML2_ScalarActivationBackwardOutput *backwardOutput = ML2_ActivationBackwardOutputOf(type);
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = inputGradientScalars.info.samples;
            ML2_Int scalars = inputGradientScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(inputGradientScalars, s, c) += backwardOutput(*ML2_ScalarsAt(outputScalars, s, c)) * *ML2_ScalarsAt(outputGradientScalars, s, c);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Activation may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputGradientMatrices.info.samples;
            ML2_Int matrices = inputGradientMatrices.info.matrices;
            ML2_Int vectors = inputGradientMatrices.info.vectors;
            ML2_Int scalars = inputGradientMatrices.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(inputGradientMatrices, s, m, v, c) += backwardOutput(*ML2_MatricesAt(outputMatrices, s, m, v, c)) * *ML2_MatricesAt(outputGradientMatrices, s, m, v, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationUnaryCacheBackwardInputOutputCompatible(ML2_ActivationType type, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    switch (type) {
        case ML2_ActivationTypeReLU: return false; // ReLU backward takes the input
        case ML2_ActivationTypeSigmoid: return false; // Sigmoid backward takes the output
        case ML2_ActivationTypeSoftmax: return false; // Softmax is not a unary function
        case ML2_ActivationTypeSin: return false; // Sin backward takes the input
        case ML2_ActivationTypeCos: return false; // Cos backward takes the input
        case ML2_ActivationTypeTanh: return false; // Tanh backward takes the output
        case ML2_ActivationTypeLeakyReLU: return false; // LeakyReLU backward takes the input
        case ML2_ActivationTypeSiLU: break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_LayerCacheInfoSame(input, output) && ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationUnaryCacheBackwardInputOutput(ML2_ActivationType type, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationUnaryCacheBackwardInputOutputCompatible(type, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    ML2_ScalarActivationBackwardInputOutput *backwardInputOutput = ML2_ActivationBackwardInputOutputOf(type);
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = inputGradientScalars.info.samples;
            ML2_Int scalars = inputGradientScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(inputGradientScalars, s, c) += backwardInputOutput(*ML2_ScalarsAt(inputScalars, s, c), *ML2_ScalarsAt(outputScalars, s, c)) * *ML2_ScalarsAt(outputGradientScalars, s, c);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheVectors inputGradientVectors = ML2_LayerCacheAsVectors(inputGradient);
            ML2_LayerCacheVectors outputVectors = ML2_LayerCacheAsVectors(output);
            ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            ML2_Int samples = inputGradientVectors.info.samples;
            ML2_Int vectors = inputGradientVectors.info.vectors;
            ML2_Int scalars = inputGradientVectors.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    for (ML2_Int c = 0; c < scalars; c++) {
                        *ML2_VectorsAt(inputGradientVectors, s, v, c) += backwardInputOutput(*ML2_VectorsAt(inputVectors, s, v, c), *ML2_VectorsAt(outputVectors, s, v, c)) * *ML2_VectorsAt(outputGradientVectors, s, v, c);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputGradientMatrices.info.samples;
            ML2_Int matrices = inputGradientMatrices.info.matrices;
            ML2_Int vectors = inputGradientMatrices.info.vectors;
            ML2_Int scalars = inputGradientMatrices.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(inputGradientMatrices, s, m, v, c) += backwardInputOutput(*ML2_MatricesAt(inputMatrices, s, m, v, c), *ML2_MatricesAt(outputMatrices, s, m, v, c)) * *ML2_MatricesAt(outputGradientMatrices, s, m, v, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationSoftmaxInfoForwardCompatible(ML2_LayerCacheInfo input) {
    // return input.type == ML2_LayerCacheTypeScalars; // TODO(23/4/2026 12:50:43am): maybe i can allow softmax on Matrices, or maybe there will be another LayerCache in the future
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return true;
        } break;
        case ML2_LayerCacheTypeVectors: {
            return true;
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Softmax may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_ActivationSoftmaxInfoForward(ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_ActivationSoftmaxInfoForwardCompatible(input));
    return input;
}

ML2_FN void ML2_ActivationSoftmaxForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_Int samples = outputScalars.info.samples;
            ML2_Int scalars = outputScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                ML2_Scalar max = ML2_NegInf;
                for (ML2_Int c = 0; c < scalars; c++) {
                    max = ML2_FMAX(max, *ML2_ScalarsAt(inputScalars, s, c));
                }
                ML2_Scalar sum = {};
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar e = ML2_EXP(*ML2_ScalarsAt(inputScalars, s, c) - max);
                    sum += e;
                }
                for (ML2_Int c = 0; c < scalars; c++) {
                    ML2_Scalar e = ML2_EXP(*ML2_ScalarsAt(inputScalars, s, c) - max);
                    *ML2_ScalarsAt(outputScalars, s, c) += e / sum;
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheVectors outputVectors = ML2_LayerCacheAsVectors(output);
            ML2_Int samples = outputVectors.info.samples;
            ML2_Int vectors = outputVectors.info.vectors;
            ML2_Int scalars = outputVectors.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    ML2_Scalar max = ML2_NegInf;
                    for (ML2_Int c = 0; c < scalars; c++) {
                        max = ML2_FMAX(max, *ML2_VectorsAt(inputVectors, s, v, c));
                    }
                    ML2_Scalar sum = {};
                    for (ML2_Int c = 0; c < scalars; c++) {
                        ML2_Scalar e = ML2_EXP(*ML2_VectorsAt(inputVectors, s, v, c) - max);
                        sum += e;
                    }
                    for (ML2_Int c = 0; c < scalars; c++) {
                        ML2_Scalar e = ML2_EXP(*ML2_VectorsAt(inputVectors, s, v, c) - max);
                        *ML2_VectorsAt(outputVectors, s, v, c) += e / sum;
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Softmax may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ActivationSoftmaxCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(inputGradient, output) && ML2_LayerCacheInfoSame(inputGradient, outputGradient);
}

ML2_FN void ML2_ActivationSoftmaxCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ActivationSoftmaxCacheBackwardCompatible(ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_Int samples = inputGradientScalars.info.samples;
            ML2_Int scalars = inputGradientScalars.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                ML2_Scalar sum = {};
                for (ML2_Int c = 0; c < scalars; c++) {
                    sum += *ML2_ScalarsAt(outputScalars, s, c) * *ML2_ScalarsAt(outputGradientScalars, s, c);
                }
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(inputGradientScalars, s, c) += *ML2_ScalarsAt(outputScalars, s, c) * (*ML2_ScalarsAt(outputGradientScalars, s, c) - sum);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputGradientVectors = ML2_LayerCacheAsVectors(inputGradient);
            ML2_LayerCacheVectors outputVectors = ML2_LayerCacheAsVectors(output);
            ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            ML2_Int samples = inputGradientVectors.info.samples;
            ML2_Int vectors = inputGradientVectors.info.vectors;
            ML2_Int scalars = inputGradientVectors.info.scalars;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int v = 0; v < vectors; v++) {
                    ML2_Scalar sum = {};
                    for (ML2_Int c = 0; c < scalars; c++) {
                        sum += *ML2_VectorsAt(outputVectors, s, v, c) * *ML2_VectorsAt(outputGradientVectors, s, v, c);
                    }
                    for (ML2_Int c = 0; c < scalars; c++) {
                        *ML2_VectorsAt(inputGradientVectors, s, v, c) += *ML2_VectorsAt(outputVectors, s, v, c) * (*ML2_VectorsAt(outputGradientVectors, s, v, c) - sum);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Softmax may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

ML2_FN ML2_LayerInfo ML2_Linear(ML2_Int inputs, ML2_Int outputs) {
    return (ML2_LayerInfo){ML2_LayerTypeLinear, .as.linear = {outputs, inputs}};
}

ML2_FN ML2_LayerLinear ML2_LinearNew(ML2_LayerInfoLinear info) {
    ML2_LayerLinear linear = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs, sizeof(*linear.data.weights)),
         ML2_RELIABLE_CALLOC(info.outputs, sizeof(*linear.data.biases))}
    };
    return linear;
}

ML2_FN void ML2_LinearDestroy(ML2_LayerLinear *linear) {
    ML2_FREE(linear->data.weights);
    ML2_FREE(linear->data.biases);
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

ML2_FN ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear) {
    return (ML2_LayerWeights){ML2_LinearInfoWeights(linear.info), {linear.data.weights}};
}

ML2_FN ML2_LayerInfoBiases ML2_LinearInfoBiases(ML2_LayerInfoLinear linear) {
    return ML2_LayerInfoAsBiases(ML2_Biases(linear.outputs));
}

ML2_FN ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear) {
    return (ML2_LayerBiases){ML2_LinearInfoBiases(linear.info), {linear.data.biases}};
}

ML2_FN void ML2_LinearClear(ML2_LayerLinear linear) {
    ML2_WeightsClear(ML2_LinearWeights(linear));
    ML2_BiasesClear(ML2_LinearBiases(linear));
}

ML2_FN void ML2_LinearSum(ML2_LayerLinear dest, ML2_LayerLinear src) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(dest.info, src.info));
    ML2_WeightsSum(ML2_LinearWeights(dest), ML2_LinearWeights(src));
    ML2_BiasesSum(ML2_LinearBiases(dest), ML2_LinearBiases(src));
}

ML2_FN void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high) {
    ML2_WeightsRand(ML2_LinearWeights(linear), low, high);
    ML2_BiasesRand(ML2_LinearBiases(linear), low, high);
}

ML2_FN void ML2_LinearXavierInit(ML2_LayerLinear linear) {
    ML2_WeightsXavierInit(ML2_LinearWeights(linear));
    ML2_BiasesXavierInit(ML2_LinearBiases(linear));
}

ML2_FN void ML2_LinearHeInit(ML2_LayerLinear linear) {
    ML2_WeightsHeInit(ML2_LinearWeights(linear));
    ML2_BiasesHeInit(ML2_LinearBiases(linear));
}

ML2_FN void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, ML2_Int indent) {
    printf(ML2_INDENT("Linear(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_FN void ML2_LinearPrint(ML2_LayerLinear linear, ML2_Int indent) {
    ML2_LinearInfoPrint(linear.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        ML2_WeightsPrint(ML2_LinearWeights(linear), indent);
        ML2_BiasesPrint(ML2_LinearBiases(linear), indent);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
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

ML2_FN bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_WeightsForwardCompatible(ML2_LinearInfoWeights(linear), input, output);
}

ML2_FN void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_LinearForwardCompatible(linear.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_WeightsForward(ML2_LinearWeights(linear), input, output);
            ML2_LayerBiases biases = ML2_LinearBiases(linear);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_Int samples = outputScalars.info.samples;
            ML2_Int scalars = outputScalars.info.scalars;
            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(outputScalars, s, c) += *ML2_BiasesAt(biases, c);
                }
            }
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Linear may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LinearBackwardCompatible(ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsBackwardCompatible(ML2_LinearInfoWeights(linearGradient), input, outputGradient);
}

ML2_FN void ML2_LinearBackward(ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearBackwardCompatible(linearGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_WeightsBackward(ML2_LinearWeights(linearGradient), input, outputGradient);
            ML2_BiasesBackward(ML2_LinearBiases(linearGradient), outputGradient);
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Linear may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LinearCacheBackwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsCacheBackwardCompatible(ML2_LinearInfoWeights(linear), inputGradient, outputGradient);
}

ML2_FN void ML2_LinearCacheBackward(ML2_LayerLinear linear, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LinearCacheBackwardCompatible(linear.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_WeightsCacheBackward(ML2_LinearWeights(linear), inputGradient, outputGradient);
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Linear may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> Linear may or may not be a feature in the future");
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_WeightsGradientDescentOptimize(ML2_LinearWeights(linear), ML2_LinearWeights(linearGradient), learningRate);
    ML2_BiasesGradientDescentOptimize(ML2_LinearBiases(linear), ML2_LinearBiases(linearGradient), learningRate);
}

ML2_FN void ML2_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, average.info));
    ML2_WeightsMomentumOptimize(ML2_LinearWeights(linear), ML2_LinearWeights(linearGradient), learningRate, decayRate, ML2_LinearWeights(average));
    ML2_BiasesMomentumOptimize(ML2_LinearBiases(linear), ML2_LinearBiases(linearGradient), learningRate, decayRate, ML2_LinearBiases(average));
}

ML2_FN void ML2_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, squareSum.info));
    ML2_WeightsAdagradOptimize(ML2_LinearWeights(linear), ML2_LinearWeights(linearGradient), learningRate, ML2_LinearWeights(squareSum));
    ML2_BiasesAdagradOptimize(ML2_LinearBiases(linear), ML2_LinearBiases(linearGradient), learningRate, ML2_LinearBiases(squareSum));
}

ML2_FN void ML2_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, squareAverage.info));
    ML2_WeightsRMSPropOptimize(ML2_LinearWeights(linear), ML2_LinearWeights(linearGradient), learningRate, decayRate, ML2_LinearWeights(squareAverage));
    ML2_BiasesRMSPropOptimize(ML2_LinearBiases(linear), ML2_LinearBiases(linearGradient), learningRate, decayRate, ML2_LinearBiases(squareAverage));
}

ML2_FN void ML2_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, linearGradient.info));
    ML2_HARD_ASSERT(ML2_LinearInfoSame(linear.info, average.info));
    ML2_WeightsAdamOptimize(ML2_LinearWeights(linear), ML2_LinearWeights(linearGradient), learningRate, decayRate1, decayRate2, ML2_LinearWeights(average), ML2_LinearWeights(squareAverage), decayingWeight1, decayingWeight2);
    ML2_BiasesAdamOptimize(ML2_LinearBiases(linear), ML2_LinearBiases(linearGradient), learningRate, decayRate1, decayRate2, ML2_LinearBiases(average), ML2_LinearBiases(squareAverage), decayingWeight1, decayingWeight2);
}

ML2_FN ML2_Size ML2_LinearSizeof(ML2_LayerLinear linear) {
    return ML2_WeightsSizeof(ML2_LinearWeights(linear)) + ML2_BiasesSizeof(ML2_LinearBiases(linear));
}

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_FN ML2_LayerInfo ML2_Filters(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypeFilters, .as.filters = {height, width, inputs, outputs, paddingHeight, paddingWidth, strideHeight, strideWidth}};
}

ML2_FN ML2_LayerFilters ML2_FiltersNew(ML2_LayerInfoFilters info) {
    ML2_LayerFilters filters = {
        info,
        {ML2_RELIABLE_CALLOC(info.height * info.width * info.inputs * info.outputs, sizeof(*filters.data.weights))}
    };
    return filters;
}

ML2_FN void ML2_FiltersDestroy(ML2_LayerFilters *filters) {
    ML2_FREE(filters->data.weights);
    *filters = (ML2_LayerFilters){};
}

ML2_FN ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, ML2_Int y, ML2_Int x, ML2_Int i, ML2_Int o) {
    ML2_SOFT_ASSERT(0 <= y && y < filters.info.height && 0 <= x && x < filters.info.width && 0 <= i && i < filters.info.inputs && 0 <= o && o < filters.info.outputs && "OUT OF BOUNDS INDICES");
    return &filters.data.weights[((y * filters.info.width + x) * filters.info.inputs + i) * filters.info.outputs + o];
}

ML2_FN ML2_LayerInfoFilters ML2_LayerInfoAsFilters(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeFilters);
    return info.as.filters;
}

ML2_FN ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeFilters);
    return layer.as.filters;
}

ML2_FN void ML2_FiltersClear(ML2_LayerFilters filters) {
    for (ML2_Int y = 0; y < filters.info.height; y++) {
        for (ML2_Int x = 0; x < filters.info.width; x++) {
            for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                    *ML2_FiltersAt(filters, y, x, i, o) = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersSum(ML2_LayerFilters dest, ML2_LayerFilters src) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(dest.info, src.info));
    ML2_Int height = dest.info.height;
    ML2_Int width = dest.info.width;
    ML2_Int inputs = dest.info.inputs;
    ML2_Int outputs = dest.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    *ML2_FiltersAt(dest, y, x, i, o) += *ML2_FiltersAt(src, y, x, i, o);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int y = 0; y < filters.info.height; y++) {
        for (ML2_Int x = 0; x < filters.info.width; x++) {
            for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                    *ML2_FiltersAt(filters, y, x, i, o) = ML2_ScalarRand(low, high);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersXavierInit(ML2_LayerFilters filters) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / ((filters.info.inputs + filters.info.outputs) * filters.info.height * filters.info.width));
    ML2_FiltersRand(filters, -limit, limit);
}

ML2_FN void ML2_FiltersHeInit(ML2_LayerFilters filters) {
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (filters.info.height * filters.info.width * filters.info.inputs));
    for (ML2_Int y = 0; y < filters.info.height; y++) {
        for (ML2_Int x = 0; x < filters.info.width; x++) {
            for (ML2_Int i = 0; i < filters.info.inputs; i++) {
                for (ML2_Int o = 0; o < filters.info.outputs; o++) {
                    *ML2_FiltersAt(filters, y, x, i, o) = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, ML2_Int indent) {
    printf(ML2_INDENT("Filters(%dx%dx%dx%d)\n", indent), info.height, info.width, info.inputs, info.outputs);
}

ML2_FN void ML2_FiltersPrint(ML2_LayerFilters filters, ML2_Int indent) {
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
                            printf(ML2_SCALAR_FMT " ", *ML2_FiltersAt(filters, y, x, i, o));
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
            return filters.inputs == inputMatrices.scalars && outputHeight > 0 && outputWidth > 0;
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
    return a.height == b.height && a.width == b.width && a.paddingHeight == b.paddingHeight && a.paddingWidth == b.paddingWidth && a.strideHeight == b.strideHeight && a.strideWidth == b.strideWidth;
}

ML2_FN bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_LayerCacheInfoMatrices outputMatrices = ML2_LayerCacheInfoAsMatrices(output);
            ML2_Int outputHeight = ML2_FiltersInfoOutputHeight(filters, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_FiltersInfoOutputWidth(filters, inputMatrices.vectors);
            return inputMatrices.scalars   == filters.inputs &&
                   outputMatrices.samples  == inputMatrices.samples &&
                   outputMatrices.matrices == outputHeight &&
                   outputMatrices.vectors  == outputWidth &&
                   outputMatrices.scalars  == filters.outputs;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FiltersForwardCompatible(filters.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices  = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int outputs = filters.info.outputs;
            ML2_Int inputs = filters.info.inputs;

            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputMatrices.info.matrices;
            ML2_Int outputWidth = outputMatrices.info.vectors;
            ML2_Int filtersHeight = filters.info.height;
            ML2_Int filtersWidth = filters.info.width;
            ML2_Int paddingHeight = filters.info.paddingHeight;
            ML2_Int paddingWidth = filters.info.paddingWidth;
            ML2_Int strideHeight = filters.info.strideHeight;
            ML2_Int strideWidth = filters.info.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                        ML2_Int yfMin = ML2_IntMax(paddingHeight - yo * strideHeight, 0);
                        ML2_Int yfMax = ML2_IntMin(inputHeight - 1 + paddingHeight - yo * strideHeight, filtersHeight - 1);
                        ML2_Int xfMin = ML2_IntMax(paddingWidth - xo * strideWidth, 0);
                        ML2_Int xfMax = ML2_IntMin(inputWidth - 1 + paddingWidth - xo * strideWidth, filtersWidth - 1);
                        for (ML2_Int yf = yfMin; yf <= yfMax; yf++) {
                            for (ML2_Int xf = xfMin; xf <= xfMax; xf++) {
                                ML2_Int yi = yo * strideHeight + yf - paddingHeight;
                                ML2_Int xi = xo * strideWidth + xf - paddingWidth;
                                for (ML2_Int i = 0; i < inputs; i++) {
                                    for (ML2_Int o = 0; o < outputs; o++) {
                                        *ML2_MatricesAt(outputMatrices, s, yo, xo, o) += *ML2_MatricesAt(inputMatrices, s, yi, xi, i) * *ML2_FiltersAt(filters, yf, xf, i, o);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersBackwardCompatible(ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersForwardCompatible(filtersGradient, input, outputGradient);
}

ML2_FN void ML2_FiltersBackward(ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersBackwardCompatible(filtersGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int outputs = filtersGradient.info.outputs;
            ML2_Int inputs = filtersGradient.info.inputs;

            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputGradientMatrices.info.matrices;
            ML2_Int outputWidth = outputGradientMatrices.info.vectors;
            ML2_Int filtersHeight = filtersGradient.info.height;
            ML2_Int filtersWidth = filtersGradient.info.width;
            ML2_Int paddingHeight = filtersGradient.info.paddingHeight;
            ML2_Int paddingWidth = filtersGradient.info.paddingWidth;
            ML2_Int strideHeight = filtersGradient.info.strideHeight;
            ML2_Int strideWidth = filtersGradient.info.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                        ML2_Int yfMin = ML2_IntMax(paddingHeight - yo * strideHeight, 0);
                        ML2_Int yfMax = ML2_IntMin(inputHeight - 1 + paddingHeight - yo * strideHeight, filtersHeight - 1);
                        ML2_Int xfMin = ML2_IntMax(paddingWidth - xo * strideWidth, 0);
                        ML2_Int xfMax = ML2_IntMin(inputWidth - 1 + paddingWidth - xo * strideWidth, filtersWidth - 1);
                        for (ML2_Int yf = yfMin; yf <= yfMax; yf++) {
                            for (ML2_Int xf = xfMin; xf <= xfMax; xf++) {
                                ML2_Int yi = yo * strideHeight + yf - paddingHeight;
                                ML2_Int xi = xo * strideWidth + xf - paddingWidth;
                                for (ML2_Int i = 0; i < inputs; i++) {
                                    for (ML2_Int o = 0; o < outputs; o++) {
                                        // PERF(25/4/2026 1:13:27am): will factoring the outputGradient out make it faster?
                                        *ML2_FiltersAt(filtersGradient, yf, xf, i, o) += *ML2_MatricesAt(inputMatrices, s, yi, xi, i) * *ML2_MatricesAt(outputGradientMatrices, s, yo, xo, o);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersCacheBackwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersForwardCompatible(filters, inputGradient, outputGradient);
}

ML2_FN void ML2_FiltersCacheBackward(ML2_LayerFilters filters, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FiltersCacheBackwardCompatible(filters.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Filters may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputGradientMatrices.info.samples;
            ML2_Int outputs = filters.info.outputs;
            ML2_Int inputs = filters.info.inputs;

            ML2_Int inputHeight = inputGradientMatrices.info.matrices;
            ML2_Int inputWidth = inputGradientMatrices.info.vectors;
            ML2_Int outputHeight = outputGradientMatrices.info.matrices;
            ML2_Int outputWidth = outputGradientMatrices.info.vectors;
            ML2_Int filtersHeight = filters.info.height;
            ML2_Int filtersWidth = filters.info.width;
            ML2_Int paddingHeight = filters.info.paddingHeight;
            ML2_Int paddingWidth = filters.info.paddingWidth;
            ML2_Int strideHeight = filters.info.strideHeight;
            ML2_Int strideWidth = filters.info.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                        ML2_Int yfMin = ML2_IntMax(paddingHeight - yo * strideHeight, 0);
                        ML2_Int yfMax = ML2_IntMin(inputHeight - 1 + paddingHeight - yo * strideHeight, filtersHeight - 1);
                        ML2_Int xfMin = ML2_IntMax(paddingWidth - xo * strideWidth, 0);
                        ML2_Int xfMax = ML2_IntMin(inputWidth - 1 + paddingWidth - xo * strideWidth, filtersWidth - 1);
                        for (ML2_Int yf = yfMin; yf <= yfMax; yf++) {
                            for (ML2_Int xf = xfMin; xf <= xfMax; xf++) {
                                ML2_Int yi = yo * strideHeight + yf - paddingHeight;
                                ML2_Int xi = xo * strideWidth + xf - paddingWidth;
                                for (ML2_Int i = 0; i < inputs; i++) {
                                    ML2_Scalar acc = {};
                                    for (ML2_Int o = 0; o < outputs; o++) {
                                        acc += *ML2_FiltersAt(filters, yf, xf, i, o) * *ML2_MatricesAt(outputGradientMatrices, s, yo, xo, o);
                                    }
                                    *ML2_MatricesAt(inputGradientMatrices, s, yi, xi, i) += acc;
                                }
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    *ML2_FiltersAt(filters, y, x, i, o) -= *ML2_FiltersAt(filtersGradient, y, x, i, o) * learningRate;
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, average.info));
    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    *ML2_FiltersAt(average, y, x, i, o) = decayRate * *ML2_FiltersAt(average, y, x, i, o) - learningRate * *ML2_FiltersAt(filtersGradient, y, x, i, o);
                    *ML2_FiltersAt(filters, y, x, i, o) += *ML2_FiltersAt(average, y, x, i, o);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, squareSum.info));
    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    *ML2_FiltersAt(squareSum, y, x, i, o) += *ML2_FiltersAt(filtersGradient, y, x, i, o) * *ML2_FiltersAt(filtersGradient, y, x, i, o);
                    *ML2_FiltersAt(filters, y, x, i, o) -= *ML2_FiltersAt(filtersGradient, y, x, i, o) * learningRate / ML2_SQRT(*ML2_FiltersAt(squareSum, y, x, i, o) + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, squareAverage.info));
    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    *ML2_FiltersAt(squareAverage, y, x, i, o) = decayRate * *ML2_FiltersAt(squareAverage, y, x, i, o) + (ML2_SCALAR_LITERAL(1.0) - decayRate) * *ML2_FiltersAt(filtersGradient, y, x, i, o) * *ML2_FiltersAt(filtersGradient, y, x, i, o);
                    *ML2_FiltersAt(filters, y, x, i, o) -= *ML2_FiltersAt(filtersGradient, y, x, i, o) * learningRate / ML2_SQRT(*ML2_FiltersAt(squareAverage, y, x, i, o) + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, average.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, squareAverage.info));
    ML2_Int height = filters.info.height;
    ML2_Int width = filters.info.width;
    ML2_Int inputs = filters.info.inputs;
    ML2_Int outputs = filters.info.outputs;
    for (ML2_Int y = 0; y < height; y++) {
        for (ML2_Int x = 0; x < width; x++) {
            for (ML2_Int i = 0; i < inputs; i++) {
                for (ML2_Int o = 0; o < outputs; o++) {
                    *ML2_FiltersAt(average, y, x, i, o) = decayRate1 * *ML2_FiltersAt(average, y, x, i, o) + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * *ML2_FiltersAt(filtersGradient, y, x, i, o);
                    *ML2_FiltersAt(squareAverage, y, x, i, o) = decayRate2 * *ML2_FiltersAt(squareAverage, y, x, i, o) + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * *ML2_FiltersAt(filtersGradient, y, x, i, o) * *ML2_FiltersAt(filtersGradient, y, x, i, o);
                    ML2_Scalar averageWeighted = *ML2_FiltersAt(average, y, x, i, o) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
                    ML2_Scalar squareAverageWeighed = *ML2_FiltersAt(squareAverage, y, x, i, o) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
                    *ML2_FiltersAt(filters, y, x, i, o) -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN ML2_Size ML2_FiltersSizeof(ML2_LayerFilters filters) {
    return filters.info.height * filters.info.width * filters.info.inputs * filters.info.outputs * sizeof(*filters.data.weights);
}

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_FN ML2_LayerInfo ML2_Conv(ML2_Int inputs, ML2_Int outputs, ML2_Int height, ML2_Int width, ML2_Int paddingHeight, ML2_Int paddingWidth, ML2_Int strideHeight, ML2_Int strideWidth) {
    return (ML2_LayerInfo){ML2_LayerTypeConv, .as.conv = {height, width, inputs, outputs, paddingHeight, paddingWidth, strideHeight, strideWidth}};
}

ML2_FN ML2_LayerConv ML2_ConvNew(ML2_LayerInfoConv info) {
    ML2_LayerConv conv = {
        info,
        {ML2_RELIABLE_CALLOC(info.height * info.width * info.inputs * info.outputs, sizeof(*conv.data.weights)),
         ML2_RELIABLE_CALLOC(info.outputs, sizeof(*conv.data.biases))}
    };
    return conv;
}

ML2_FN void ML2_ConvDestroy(ML2_LayerConv *conv) {
    ML2_FREE(conv->data.weights);
    ML2_FREE(conv->data.biases);
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

ML2_FN ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv) {
    return (ML2_LayerFilters){ML2_ConvInfoFilters(conv.info), {conv.data.weights}};
}

ML2_FN ML2_LayerInfoBiases ML2_ConvInfoBiases(ML2_LayerInfoConv conv) {
    return ML2_LayerInfoAsBiases(ML2_Biases(conv.outputs));
}

ML2_FN ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv) {
    return (ML2_LayerBiases){ML2_ConvInfoBiases(conv.info), {conv.data.biases}};
}

ML2_FN void ML2_ConvClear(ML2_LayerConv conv) {
    ML2_FiltersClear(ML2_ConvFilters(conv));
    ML2_BiasesClear(ML2_ConvBiases(conv));
}

ML2_FN void ML2_ConvSum(ML2_LayerConv dest, ML2_LayerConv src) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(dest.info, src.info));
    ML2_FiltersSum(ML2_ConvFilters(dest), ML2_ConvFilters(src));
    ML2_BiasesSum(ML2_ConvBiases(dest), ML2_ConvBiases(src));
}

ML2_FN void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high) {
    ML2_FiltersRand(ML2_ConvFilters(conv), low, high);
    ML2_BiasesRand(ML2_ConvBiases(conv), low, high);
}

ML2_FN void ML2_ConvXavierInit(ML2_LayerConv conv) {
    ML2_FiltersXavierInit(ML2_ConvFilters(conv));
    ML2_BiasesXavierInit(ML2_ConvBiases(conv));
}

ML2_FN void ML2_ConvHeInit(ML2_LayerConv conv) {
    ML2_FiltersHeInit(ML2_ConvFilters(conv));
    ML2_BiasesHeInit(ML2_ConvBiases(conv));
}

ML2_FN void ML2_ConvInfoPrint(ML2_LayerInfoConv info, ML2_Int indent) {
    printf(ML2_INDENT("Conv(%dx%dx%dx%d)\n", indent), info.height, info.width, info.inputs, info.outputs);
}

ML2_FN void ML2_ConvPrint(ML2_LayerConv conv, ML2_Int indent) {
    ML2_ConvInfoPrint(conv.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        ML2_FiltersPrint(ML2_ConvFilters(conv), indent);
        ML2_BiasesPrint(ML2_ConvBiases(conv), indent);
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
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

ML2_FN bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_FiltersForwardCompatible(ML2_ConvInfoFilters(conv), input, output);
}

ML2_FN void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ConvForwardCompatible(conv.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Conv may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_FiltersForward(ML2_ConvFilters(conv), input, output);
            // TODO(3/9/2026 18:51:27): why doesnt this call ML2_BiasesForward(), i remember there was a reason but didnt i fix it? i dont remember, should be investigated
            ML2_LayerBiases biases = ML2_ConvBiases(conv);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = outputMatrices.info.samples;
            ML2_Int matrices = outputMatrices.info.matrices;
            ML2_Int vectors = outputMatrices.info.vectors;
            ML2_Int scalars = outputMatrices.info.scalars;
            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(outputMatrices, s, m, v, c) += *ML2_BiasesAt(biases, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ConvBackwardCompatible(ML2_LayerInfoConv convGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersBackwardCompatible(ML2_ConvInfoFilters(convGradient), input, outputGradient);
}

ML2_FN void ML2_ConvBackward(ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvBackwardCompatible(convGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Conv may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_FiltersBackward(ML2_ConvFilters(convGradient), input, outputGradient);
            ML2_BiasesBackward(ML2_ConvBiases(convGradient), outputGradient);
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ConvCacheBackwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersCacheBackwardCompatible(ML2_ConvInfoFilters(conv), inputGradient, outputGradient);
}

ML2_FN void ML2_ConvCacheBackward(ML2_LayerConv conv, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_ConvCacheBackwardCompatible(conv.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Conv may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_FiltersCacheBackward(ML2_ConvFilters(conv), inputGradient, outputGradient);
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_FiltersGradientDescentOptimize(ML2_ConvFilters(conv), ML2_ConvFilters(convGradient), learningRate);
    ML2_BiasesGradientDescentOptimize(ML2_ConvBiases(conv), ML2_ConvBiases(convGradient), learningRate);
}

ML2_FN void ML2_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, average.info));
    ML2_FiltersMomentumOptimize(ML2_ConvFilters(conv), ML2_ConvFilters(convGradient), learningRate, decayRate, ML2_ConvFilters(average));
    ML2_BiasesMomentumOptimize(ML2_ConvBiases(conv), ML2_ConvBiases(convGradient), learningRate, decayRate, ML2_ConvBiases(average));
}

ML2_FN void ML2_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, squareSum.info));
    ML2_FiltersAdagradOptimize(ML2_ConvFilters(conv), ML2_ConvFilters(convGradient), learningRate, ML2_ConvFilters(squareSum));
    ML2_BiasesAdagradOptimize(ML2_ConvBiases(conv), ML2_ConvBiases(convGradient), learningRate, ML2_ConvBiases(squareSum));
}

ML2_FN void ML2_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, squareAverage.info));
    ML2_FiltersRMSPropOptimize(ML2_ConvFilters(conv), ML2_ConvFilters(convGradient), learningRate, decayRate, ML2_ConvFilters(squareAverage));
    ML2_BiasesRMSPropOptimize(ML2_ConvBiases(conv), ML2_ConvBiases(convGradient), learningRate, decayRate, ML2_ConvBiases(squareAverage));
}

ML2_FN void ML2_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, convGradient.info));
    ML2_HARD_ASSERT(ML2_ConvInfoSame(conv.info, average.info));
    ML2_FiltersAdamOptimize(ML2_ConvFilters(conv), ML2_ConvFilters(convGradient), learningRate, decayRate1, decayRate2, ML2_ConvFilters(average), ML2_ConvFilters(squareAverage), decayingWeight1, decayingWeight2);
    ML2_BiasesAdamOptimize(ML2_ConvBiases(conv), ML2_ConvBiases(convGradient), learningRate, decayRate1, decayRate2, ML2_ConvBiases(average), ML2_ConvBiases(squareAverage), decayingWeight1, decayingWeight2);
}

ML2_FN ML2_Size ML2_ConvSizeof(ML2_LayerConv conv) {
    return ML2_FiltersSizeof(ML2_ConvFilters(conv)) + ML2_BiasesSizeof(ML2_ConvBiases(conv));
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
    return input.type == ML2_LayerCacheTypeMatrices;
}

ML2_FN ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FlattenInfoForwardCompatible(input));
    ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeScalars, .as.scalars = {inputMatrices.samples, inputMatrices.matrices * inputMatrices.vectors * inputMatrices.scalars}};
}

ML2_FN bool ML2_FlattenForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != ML2_LayerCacheTypeMatrices || output.type != ML2_LayerCacheTypeScalars) return false;
    ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
    ML2_LayerCacheInfoScalars outputScalars = ML2_LayerCacheInfoAsScalars(output);
    return inputMatrices.samples == outputScalars.samples && inputMatrices.matrices * inputMatrices.vectors * inputMatrices.scalars == outputScalars.scalars;
}

ML2_FN void ML2_FlattenForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FlattenForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
    ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
    ML2_Int samples = inputMatrices.info.samples;
    ML2_Int matrices = inputMatrices.info.matrices;
    ML2_Int vectors = inputMatrices.info.vectors;
    ML2_Int scalars = inputMatrices.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_ScalarsAt(outputScalars, s, (m * vectors + v) * scalars + c) += *ML2_MatricesAt(inputMatrices, s, m, v, c);
                }
            }
        }
    }
}

ML2_FN bool ML2_FlattenCacheBackwardCompatible(ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FlattenForwardCompatible(inputGradient, outputGradient);
}

ML2_FN void ML2_FlattenCacheBackward(ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FlattenCacheBackwardCompatible(ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
    ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
    ML2_Int samples = inputGradientMatrices.info.samples;
    ML2_Int matrices = inputGradientMatrices.info.matrices;
    ML2_Int vectors = inputGradientMatrices.info.vectors;
    ML2_Int scalars = inputGradientMatrices.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_MatricesAt(inputGradientMatrices, s, m, v, c) += *ML2_ScalarsAt(outputGradientScalars, s, (m * vectors + v) * scalars + c);
                }
            }
        }
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
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_LayerCacheInfoSame(output, outputGradient) && ML2_PoolForwardCompatible(pool, input, output);
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
    return a.height == b.height && a.width == b.width && a.paddingHeight == b.paddingHeight && a.paddingWidth == b.paddingWidth && a.strideHeight == b.strideHeight && a.strideWidth == b.strideWidth;
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
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolMaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolMaxInfoOutputWidth(pool, inputMatrices.vectors);
            return outputHeight > 0 && outputWidth > 0;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolMaxInfoForward(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolMaxInfoForwardCompatible(pool, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolMaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolMaxInfoOutputWidth(pool, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputMatrices.samples, outputHeight, outputWidth, inputMatrices.scalars))};
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolMaxForwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_LayerCacheInfoMatrices outputMatrices = ML2_LayerCacheInfoAsMatrices(output);
            ML2_Int outputHeight = ML2_PoolMaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolMaxInfoOutputWidth(pool, inputMatrices.vectors);
            return outputMatrices.samples  == inputMatrices.samples &&
                   outputMatrices.matrices == outputHeight &&
                   outputMatrices.vectors  == outputWidth &&
                   outputMatrices.scalars  == inputMatrices.scalars;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_PoolMaxForward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolMaxForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int scalars = inputMatrices.info.scalars;

            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputMatrices.info.matrices;
            ML2_Int outputWidth = outputMatrices.info.vectors;
            ML2_Int poolHeight = pool.height;
            ML2_Int poolWidth = pool.width;
            ML2_Int paddingHeight = pool.paddingHeight;
            ML2_Int paddingWidth = pool.paddingWidth;
            ML2_Int strideHeight = pool.strideHeight;
            ML2_Int strideWidth = pool.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
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
                                    max[c] = ML2_FMAX(max[c], *ML2_MatricesAt(inputMatrices, s, yi, xi, c));
                                }
                            }
                        }
                        for (ML2_Int c = 0; c < scalars; c++) {
                            *ML2_MatricesAt(outputMatrices, s, yo, xo, c) += max[c];
                        }
                        */
                        for (ML2_Int c = 0; c < scalars; c++) {
                            ML2_Scalar max = ML2_NegInf;
                            for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                                for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                                    max = ML2_FMAX(max, *ML2_MatricesAt(inputMatrices, s, yi, xi, c));
                                }
                            }
                            *ML2_MatricesAt(outputMatrices, s, yo, xo, c) += max;
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolMaxCacheBackwardCompatible(ML2_LayerInfoPoolMax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_PoolMaxForwardCompatible(pool, input, outputGradient);
}

ML2_FN void ML2_PoolMaxCacheBackward(ML2_LayerInfoPoolMax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolMaxCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int scalars = inputMatrices.info.scalars;

            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputGradientMatrices.info.matrices;
            ML2_Int outputWidth = outputGradientMatrices.info.vectors;
            ML2_Int poolHeight = pool.height;
            ML2_Int poolWidth = pool.width;
            ML2_Int paddingHeight = pool.paddingHeight;
            ML2_Int paddingWidth = pool.paddingWidth;
            ML2_Int strideHeight = pool.strideHeight;
            ML2_Int strideWidth = pool.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
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
                                    if (*ML2_MatricesAt(inputMatrices, s, yi, xi, c) > max) {
                                        max = *ML2_MatricesAt(inputMatrices, s, yi, xi, c);
                                        yIdx = yi;
                                        xIdx = xi;
                                    }
                                }
                            }
                            *ML2_MatricesAt(inputGradientMatrices, s, yIdx, xIdx, c) += *ML2_MatricesAt(outputGradientMatrices, s, yo, xo, c);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
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
    return a.height == b.height && a.width == b.width && a.paddingHeight == b.paddingHeight && a.paddingWidth == b.paddingWidth && a.strideHeight == b.strideHeight && a.strideWidth == b.strideWidth;
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
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolAverageInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolAverageInfoOutputWidth(pool, inputMatrices.vectors);
            return outputHeight > 0 && outputWidth > 0;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolAverageInfoForward(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolAverageInfoForwardCompatible(pool, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolAverageInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolAverageInfoOutputWidth(pool, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputMatrices.samples, outputHeight, outputWidth, inputMatrices.scalars))};
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolAverageForwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_LayerCacheInfoMatrices outputMatrices = ML2_LayerCacheInfoAsMatrices(output);
            ML2_Int outputHeight = ML2_PoolAverageInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolAverageInfoOutputWidth(pool, inputMatrices.vectors);
            return outputMatrices.samples  == inputMatrices.samples &&
                   outputMatrices.matrices == outputHeight &&
                   outputMatrices.vectors  == outputWidth &&
                   outputMatrices.scalars  == inputMatrices.scalars;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_PoolAverageForward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolAverageForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int scalars = inputMatrices.info.scalars;

            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputMatrices.info.matrices;
            ML2_Int outputWidth = outputMatrices.info.vectors;
            ML2_Int poolHeight = pool.height;
            ML2_Int poolWidth = pool.width;
            ML2_Int paddingHeight = pool.paddingHeight;
            ML2_Int paddingWidth = pool.paddingWidth;
            ML2_Int strideHeight = pool.strideHeight;
            ML2_Int strideWidth = pool.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                        ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                        ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                        ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                        ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                        ML2_Scalar poolArea = (yiMax - yiMin + ML2_SCALAR_LITERAL(1.0)) * (xiMax - xiMin + ML2_SCALAR_LITERAL(1.0));
                        // TODO(25/4/2026 1:32:54am): another implementation is to just divide by the poolArea in the loop
                        //                            but i dont know how slow that would be, needs to be tested
                        for (ML2_Int c = 0; c < scalars; c++) {
                            ML2_Scalar sum = {};
                            for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                                for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                                    sum += *ML2_MatricesAt(inputMatrices, s, yi, xi, c);
                                }
                            }
                            *ML2_MatricesAt(outputMatrices, s, yo, xo, c) += sum / poolArea;
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolAverageCacheBackwardCompatible(ML2_LayerInfoPoolAverage pool, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_PoolAverageForwardCompatible(pool, inputGradient, outputGradient);
}

ML2_FN void ML2_PoolAverageCacheBackward(ML2_LayerInfoPoolAverage pool, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolAverageCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputGradientMatrices.info.samples;
            ML2_Int scalars = inputGradientMatrices.info.scalars;

            ML2_Int inputHeight = inputGradientMatrices.info.matrices;
            ML2_Int inputWidth = inputGradientMatrices.info.vectors;
            ML2_Int outputHeight = outputGradientMatrices.info.matrices;
            ML2_Int outputWidth = outputGradientMatrices.info.vectors;
            ML2_Int poolHeight = pool.height;
            ML2_Int poolWidth = pool.width;
            ML2_Int paddingHeight = pool.paddingHeight;
            ML2_Int paddingWidth = pool.paddingWidth;
            ML2_Int strideHeight = pool.strideHeight;
            ML2_Int strideWidth = pool.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
                        ML2_Int yiMin = ML2_IntMax(yo * strideHeight - paddingHeight, 0);
                        ML2_Int yiMax = ML2_IntMin(yo * strideHeight - paddingHeight + poolHeight - 1, inputHeight - 1);
                        ML2_Int xiMin = ML2_IntMax(xo * strideWidth - paddingWidth, 0);
                        ML2_Int xiMax = ML2_IntMin(xo * strideWidth - paddingWidth + poolWidth - 1, inputWidth - 1);
                        ML2_Scalar poolArea = (yiMax - yiMin + ML2_SCALAR_LITERAL(1.0)) * (xiMax - xiMin + ML2_SCALAR_LITERAL(1.0));
                        for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                            for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                                for (ML2_Int c = 0; c < scalars; c++) {
                                    *ML2_MatricesAt(inputGradientMatrices, s, yi, xi, c) += *ML2_MatricesAt(outputGradientMatrices, s, yo, xo, c) / poolArea;
                                }
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
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
    return a.height == b.height && a.width == b.width && a.paddingHeight == b.paddingHeight && a.paddingWidth == b.paddingWidth && a.strideHeight == b.strideHeight && a.strideWidth == b.strideWidth;
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
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolSoftmaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolSoftmaxInfoOutputWidth(pool, inputMatrices.vectors);
            return outputHeight > 0 && outputWidth > 0;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_PoolSoftmaxInfoForward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxInfoForwardCompatible(pool, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_Int outputHeight = ML2_PoolSoftmaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolSoftmaxInfoOutputWidth(pool, inputMatrices.vectors);
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = ML2_LayerCacheInfoAsMatrices(ML2_Matrices(inputMatrices.samples, outputHeight, outputWidth, inputMatrices.scalars))};
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolSoftmaxForwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeMatrices: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_LayerCacheInfoMatrices outputMatrices = ML2_LayerCacheInfoAsMatrices(output);
            ML2_Int outputHeight = ML2_PoolSoftmaxInfoOutputHeight(pool, inputMatrices.matrices);
            ML2_Int outputWidth = ML2_PoolSoftmaxInfoOutputWidth(pool, inputMatrices.vectors);
            return outputMatrices.samples  == inputMatrices.samples &&
                   outputMatrices.matrices == outputHeight &&
                   outputMatrices.vectors  == outputWidth &&
                   outputMatrices.scalars  == inputMatrices.scalars;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_PoolSoftmaxForward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxForwardCompatible(pool, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int scalars = inputMatrices.info.scalars;
            
            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputMatrices.info.matrices;
            ML2_Int outputWidth = outputMatrices.info.vectors;
            ML2_Int poolHeight = pool.height;
            ML2_Int poolWidth = pool.width;
            ML2_Int paddingHeight = pool.paddingHeight;
            ML2_Int paddingWidth = pool.paddingWidth;
            ML2_Int strideHeight = pool.strideHeight;
            ML2_Int strideWidth = pool.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
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
                                    max = ML2_FMAX(max, *ML2_MatricesAt(inputMatrices, s, yi, xi, c));
                                }
                            }
                            ML2_Scalar sum = {};
                            ML2_Scalar weightedSum = {};
                            for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                                for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                                    ML2_Scalar inputScalar = *ML2_MatricesAt(inputMatrices, s, yi, xi, c) - max;
                                    ML2_Scalar e = ML2_EXP(inputScalar);
                                    sum += e;
                                    weightedSum += e * inputScalar;
                                }
                            }
                            *ML2_MatricesAt(outputMatrices, s, yo, xo, c) += weightedSum / sum;
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_PoolSoftmaxCacheBackwardCompatible(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo output, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(output, outputGradient) && ML2_PoolSoftmaxForwardCompatible(pool, inputGradient, output);
}

ML2_FN void ML2_PoolSoftmaxCacheBackward(ML2_LayerInfoPoolSoftmax pool, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_PoolSoftmaxCacheBackwardCompatible(pool, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Pool may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = inputMatrices.info.samples;
            ML2_Int scalars = inputMatrices.info.scalars;

            ML2_Int inputHeight = inputMatrices.info.matrices;
            ML2_Int inputWidth = inputMatrices.info.vectors;
            ML2_Int outputHeight = outputMatrices.info.matrices;
            ML2_Int outputWidth = outputMatrices.info.vectors;
            ML2_Int poolHeight = pool.height;
            ML2_Int poolWidth = pool.width;
            ML2_Int paddingHeight = pool.paddingHeight;
            ML2_Int paddingWidth = pool.paddingWidth;
            ML2_Int strideHeight = pool.strideHeight;
            ML2_Int strideWidth = pool.strideWidth;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int yo = 0; yo < outputHeight; yo++) {
                    for (ML2_Int xo = 0; xo < outputWidth; xo++) {
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
                                    max = ML2_FMAX(max, *ML2_MatricesAt(inputMatrices, s, yi, xi, c));
                                }
                            }
                            ML2_Scalar sum = {};
                            for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                                for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                                    ML2_Scalar e = ML2_EXP(*ML2_MatricesAt(inputMatrices, s, yi, xi, c) - max);
                                    sum += e;
                                }
                            }
                            for (ML2_Int yi = yiMin; yi <= yiMax; yi++) {
                                for (ML2_Int xi = xiMin; xi <= xiMax; xi++) {
                                    ML2_Scalar inputScalar = *ML2_MatricesAt(inputMatrices, s, yi, xi, c);
                                    ML2_Scalar outputScalar = *ML2_MatricesAt(outputMatrices, s, yo, xo, c);
                                    ML2_Scalar e = ML2_EXP(inputScalar - max);
                                    *ML2_MatricesAt(inputGradientMatrices, s, yi, xi, c) += e / sum * (inputScalar + ML2_SCALAR_LITERAL(1.0) - outputScalar - max) * *ML2_MatricesAt(outputGradientMatrices, s, yo, xo, c);
                                }
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_LayerPoolSoftmax ⬆️

// ML2_LayerPool ⬆️

// ML2_LayerFusedWeights ⬇️

ML2_FN ML2_LayerInfo ML2_FusedWeights(ML2_Int weights, ML2_Int inputs, ML2_Int outputs) {
    return (ML2_LayerInfo){ML2_LayerTypeFusedWeights, .as.fusedWeights = {weights, outputs, inputs}};
}

ML2_FN ML2_LayerFusedWeights ML2_FusedWeightsNew(ML2_LayerInfoFusedWeights info) {
    ML2_LayerFusedWeights fusedWeights = {
        info,
        {ML2_RELIABLE_CALLOC(info.weights * info.outputs * info.inputs, sizeof(*fusedWeights.data.weights))}
    };
    return fusedWeights;
}

ML2_FN void ML2_FusedWeightsDestroy(ML2_LayerFusedWeights *fusedWeights) {
    ML2_FREE(fusedWeights->data.weights);
    *fusedWeights = (ML2_LayerFusedWeights){};
}

ML2_FN ML2_Scalar *ML2_FusedWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w, ML2_Int o, ML2_Int i) {
    ML2_SOFT_ASSERT(0 <= w && w < fusedWeights.info.weights && 0 <= o && o < fusedWeights.info.outputs && 0 <= i && i < fusedWeights.info.inputs && "OUT OF BOUNDS INDICES");
    return &fusedWeights.data.weights[(w * fusedWeights.info.outputs + o) * fusedWeights.info.inputs + i];
}

ML2_FN ML2_LayerInfoFusedWeights ML2_LayerInfoAsFusedWeights(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeFusedWeights);
    return info.as.fusedWeights;
}

ML2_FN ML2_LayerFusedWeights ML2_LayerAsFusedWeights(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeFusedWeights);
    return layer.as.fusedWeights;
}

ML2_FN ML2_LayerInfoWeights ML2_FusedWeightsInfoWeightsAt(ML2_LayerInfoFusedWeights fusedWeights, ML2_Int w) {
    ML2_SOFT_ASSERT(0 <= w && w < fusedWeights.weights && "OUT OF BOUNDS INDICES");
    return ML2_LayerInfoAsWeights(ML2_Weights(fusedWeights.inputs, fusedWeights.outputs));
}

ML2_FN ML2_LayerWeights ML2_FusedWeightsWeightsAt(ML2_LayerFusedWeights fusedWeights, ML2_Int w) {
    ML2_SOFT_ASSERT(0 <= w && w < fusedWeights.info.weights && "OUT OF BOUNDS INDICES");
    return (ML2_LayerWeights){ML2_FusedWeightsInfoWeightsAt(fusedWeights.info, w), {ML2_FusedWeightsAt(fusedWeights, w, 0, 0)}};
}

ML2_FN void ML2_FusedWeightsClear(ML2_LayerFusedWeights fusedWeights) {
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_WeightsClear(ML2_FusedWeightsWeightsAt(fusedWeights, w));
    }
}

ML2_FN void ML2_FusedWeightsSum(ML2_LayerFusedWeights dest, ML2_LayerFusedWeights src) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(dest.info, src.info));
    ML2_Int weights = dest.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_WeightsSum(ML2_FusedWeightsWeightsAt(dest, w), ML2_FusedWeightsWeightsAt(src, w));
    }
}

ML2_FN void ML2_FusedWeightsRand(ML2_LayerFusedWeights fusedWeights, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_WeightsRand(ML2_FusedWeightsWeightsAt(fusedWeights, w), low, high);
    }
}

// TODO(3/9/2026 17:22:04): this doesnt take into account the size of the entire structure, not sure if this is correct
ML2_FN void ML2_FusedWeightsXavierInit(ML2_LayerFusedWeights fusedWeights) {
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_WeightsXavierInit(ML2_FusedWeightsWeightsAt(fusedWeights, w));
    }
}

// TODO(3/9/2026 17:22:04): this doesnt take into account the size of the entire structure, not sure if this is correct
ML2_FN void ML2_FusedWeightsHeInit(ML2_LayerFusedWeights fusedWeights) {
    for (ML2_Int w = 0; w < fusedWeights.info.weights; w++) {
        ML2_WeightsHeInit(ML2_FusedWeightsWeightsAt(fusedWeights, w));
    }
}

ML2_FN void ML2_FusedWeightsInfoPrint(ML2_LayerInfoFusedWeights info, ML2_Int indent) {
    printf(ML2_INDENT("FusedWeights(%dx%dx%d)\n", indent), info.weights, info.outputs, info.inputs);
}

ML2_FN void ML2_FusedWeightsPrint(ML2_LayerFusedWeights fusedWeights, ML2_Int indent) {
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
                        printf(ML2_SCALAR_FMT " ", *ML2_FusedWeightsAt(fusedWeights, w, o, i));
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
    return a.weights == b.weights && a.outputs == b.outputs && a.inputs == b.inputs;
}

ML2_FN bool ML2_FusedWeightsForwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeVectors: {
            if (output.type != ML2_LayerCacheTypeMatrices) return false;
            ML2_LayerCacheInfoVectors inputVectors = ML2_LayerCacheInfoAsVectors(input);
            ML2_LayerCacheInfoMatrices outputMatrices = ML2_LayerCacheInfoAsMatrices(output);
            return fusedWeights.inputs     == inputVectors.scalars &&
                   outputMatrices.samples  == inputVectors.samples &&
                   outputMatrices.matrices == inputVectors.vectors &&
                   outputMatrices.vectors  == fusedWeights.weights &&
                   outputMatrices.scalars  == fusedWeights.outputs;
        }
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FusedWeightsForward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FusedWeightsForwardCompatible(fusedWeights.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheMatrices outputMatrices = ML2_LayerCacheAsMatrices(output);
            ML2_Int samples = outputMatrices.info.samples;
            ML2_Int matrices = outputMatrices.info.matrices;
            ML2_Int vectors = outputMatrices.info.vectors;
            ML2_Int outputs = fusedWeights.info.outputs;
            ML2_Int inputs = fusedWeights.info.inputs;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int o = 0; o < outputs; o++) {
                            ML2_Scalar acc = {};
                            for (ML2_Int i = 0; i < inputs; i++) {
                                acc += *ML2_VectorsAt(inputVectors, s, m, i) * *ML2_FusedWeightsAt(fusedWeights, v, o, i);
                            }
                            *ML2_MatricesAt(outputMatrices, s, m, v, o) += acc;
                        }
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FusedWeightsBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
    return ML2_FusedWeightsForwardCompatible(fusedWeightsGradient, input, outputGradient);
}

ML2_FN void ML2_FusedWeightsBackward(ML2_LayerFusedWeights fusedWeightsGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsBackwardCompatible(fusedWeightsGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = outputGradientMatrices.info.samples;
            ML2_Int matrices = outputGradientMatrices.info.matrices;
            ML2_Int vectors = outputGradientMatrices.info.vectors;
            ML2_Int outputs = fusedWeightsGradient.info.outputs;
            ML2_Int inputs = fusedWeightsGradient.info.inputs;

            // TODO(3/9/2026 21:14:05): 99% sure its correct but gotta test
            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int o = 0; o < outputs; o++) {
                            for (ML2_Int i = 0; i < inputs; i++) {
                                *ML2_FusedWeightsAt(fusedWeightsGradient, v, o, i) += *ML2_VectorsAt(inputVectors, s, m, i) * *ML2_MatricesAt(outputGradientMatrices, s, m, v, o);
                            }
                        }
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FusedWeightsCacheBackwardCompatible(ML2_LayerInfoFusedWeights fusedWeights, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FusedWeightsForwardCompatible(fusedWeights, inputGradient, outputGradient);
}

ML2_FN void ML2_FusedWeightsCacheBackward(ML2_LayerFusedWeights fusedWeights, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_FusedWeightsCacheBackwardCompatible(fusedWeights.info, ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> FusedWeights may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_LayerCacheVectors inputGradientVectors = ML2_LayerCacheAsVectors(inputGradient);
            ML2_LayerCacheMatrices outputGradientMatrices = ML2_LayerCacheAsMatrices(outputGradient);
            ML2_Int samples = outputGradientMatrices.info.samples;
            ML2_Int matrices = outputGradientMatrices.info.matrices;
            ML2_Int vectors = outputGradientMatrices.info.vectors;
            ML2_Int outputs = fusedWeights.info.outputs;
            ML2_Int inputs = fusedWeights.info.inputs;

            // TODO(3/9/2026 21:14:05): 99% sure its correct but gotta test
            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int m = 0; m < matrices; m++) {
                    for (ML2_Int v = 0; v < vectors; v++) {
                        for (ML2_Int o = 0; o < outputs; o++) {
                            for (ML2_Int i = 0; i < inputs; i++) {
                                *ML2_VectorsAt(inputGradientVectors, s, m, i) += *ML2_FusedWeightsAt(fusedWeights, v, o, i) * *ML2_MatricesAt(outputGradientMatrices, s, m, v, o);
                            }
                        }
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_TODO("Matrices -> FusedWeights may or may not be a feature in the future");
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FusedWeightsGradientDescentOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_WeightsGradientDescentOptimize(ML2_FusedWeightsWeightsAt(fusedWeights, w), ML2_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate);
    }
}

ML2_FN void ML2_FusedWeightsMomentumOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights average) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, average.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_WeightsMomentumOptimize(ML2_FusedWeightsWeightsAt(fusedWeights, w), ML2_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, decayRate, ML2_FusedWeightsWeightsAt(average, w));
    }
}

ML2_FN void ML2_FusedWeightsAdagradOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_LayerFusedWeights squareSum) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, squareSum.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_WeightsAdagradOptimize(ML2_FusedWeightsWeightsAt(fusedWeights, w), ML2_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, ML2_FusedWeightsWeightsAt(squareSum, w));
    }
}

ML2_FN void ML2_FusedWeightsRMSPropOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFusedWeights squareAverage) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, squareAverage.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_WeightsRMSPropOptimize(ML2_FusedWeightsWeightsAt(fusedWeights, w), ML2_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, decayRate, ML2_FusedWeightsWeightsAt(squareAverage, w));
    }
}

ML2_FN void ML2_FusedWeightsAdamOptimize(ML2_LayerFusedWeights fusedWeights, ML2_LayerFusedWeights fusedWeightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFusedWeights average, ML2_LayerFusedWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, fusedWeightsGradient.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, average.info));
    ML2_HARD_ASSERT(ML2_FusedWeightsInfoSame(fusedWeights.info, squareAverage.info));
    ML2_Int weights = fusedWeights.info.weights;
    for (ML2_Int w = 0; w < weights; w++) {
        ML2_WeightsAdamOptimize(ML2_FusedWeightsWeightsAt(fusedWeights, w), ML2_FusedWeightsWeightsAt(fusedWeightsGradient, w), learningRate, decayRate1, decayRate2, ML2_FusedWeightsWeightsAt(average, w), ML2_FusedWeightsWeightsAt(squareAverage, w), decayingWeight1, decayingWeight2);
    }
}

ML2_FN ML2_Size ML2_FusedWeightsSizeof(ML2_LayerFusedWeights fusedWeights) {
    return fusedWeights.info.weights * fusedWeights.info.outputs * fusedWeights.info.inputs * sizeof(*fusedWeights.data.weights);
}

// ML2_LayerFusedWeights ⬆️

// ML2_LayerAttention ⬇️

ML2_FN ML2_LayerInfo ML2_Attention(bool masked) {
    return (ML2_LayerInfo){.type = ML2_LayerTypeAttention, .as.attention = {masked}};
}

// ML2_FN ML2_LayerAttention ML2_AttentionNew(ML2_LayerInfoAttention info) {
//     ML2_LayerAttention attention = {
//         info,
//         {ML2_RELIABLE_CALLOC(info.inputs * info.inputs, sizeof(*attention.data.weightsQuery)),
//          ML2_RELIABLE_CALLOC(info.inputs * info.inputs, sizeof(*attention.data.weightsKey)),
//          ML2_RELIABLE_CALLOC(info.inputs * info.inputs, sizeof(*attention.data.weightsValue))}
//     };
//     return attention;
// }

// ML2_FN void ML2_AttentionDestroy(ML2_LayerAttention *attention) {
//     ML2_FREE(attention->data.weightsQuery);
//     ML2_FREE(attention->data.weightsKey);
//     ML2_FREE(attention->data.weightsValue);
//     *attention = (ML2_LayerAttention){};
// }

ML2_FN ML2_LayerInfoAttention ML2_LayerInfoAsAttention(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeAttention);
    return info.as.attention;
}

ML2_FN ML2_LayerInfoAttention ML2_LayerAsAttention(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeAttention);
    return layer.as.attention;
}

// ML2_FN ML2_LayerInfoWeights ML2_AttentionInfoWeightsQuery(ML2_LayerInfoAttention attention) {
//     return (ML2_LayerInfoWeights){attention.inputs, attention.inputs};
// }

// ML2_FN ML2_LayerInfoWeights ML2_AttentionInfoWeightsKey(ML2_LayerInfoAttention attention) {
//     return (ML2_LayerInfoWeights){attention.inputs, attention.inputs};
// }

// ML2_FN ML2_LayerInfoWeights ML2_AttentionInfoWeightsValue(ML2_LayerInfoAttention attention) {
//     return (ML2_LayerInfoWeights){attention.inputs, attention.inputs};
// }

// ML2_FN ML2_LayerWeights ML2_AttentionWeightsQuery(ML2_LayerAttention attention) {
//     return (ML2_LayerWeights){ML2_AttentionInfoWeightsQuery(attention.info), {attention.data.weightsQuery}};
// }

// ML2_FN ML2_LayerWeights ML2_AttentionWeightsKey(ML2_LayerAttention attention) {
//     return (ML2_LayerWeights){ML2_AttentionInfoWeightsKey(attention.info), {attention.data.weightsKey}};
// }

// ML2_FN ML2_LayerWeights ML2_AttentionWeightsValue(ML2_LayerAttention attention) {
//     return (ML2_LayerWeights){ML2_AttentionInfoWeightsValue(attention.info), {attention.data.weightsValue}};
// }

// ML2_FN void ML2_AttentionClear(ML2_LayerAttention attention) {
//     ML2_WeightsClear(ML2_AttentionWeightsQuery(attention));
//     ML2_WeightsClear(ML2_AttentionWeightsKey(attention));
//     ML2_WeightsClear(ML2_AttentionWeightsValue(attention));
// }

// ML2_FN void ML2_AttentionSum(ML2_LayerAttention dest, ML2_LayerAttention src) {
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(dest.info, src.info));
//     ML2_WeightsSum(ML2_AttentionWeightsQuery(dest), ML2_AttentionWeightsQuery(src));
//     ML2_WeightsSum(ML2_AttentionWeightsKey(dest), ML2_AttentionWeightsKey(src));
//     ML2_WeightsSum(ML2_AttentionWeightsValue(dest), ML2_AttentionWeightsValue(src));
// }

// ML2_FN void ML2_AttentionRand(ML2_LayerAttention attention, ML2_Scalar low, ML2_Scalar high) {
//     ML2_WeightsRand(ML2_AttentionWeightsQuery(attention), low, high);
//     ML2_WeightsRand(ML2_AttentionWeightsKey(attention), low, high);
//     ML2_WeightsRand(ML2_AttentionWeightsValue(attention), low, high);
// }

// ML2_FN void ML2_AttentionXavierInit(ML2_LayerAttention attention) {
//     ML2_WeightsXavierInit(ML2_AttentionWeightsQuery(attention));
//     ML2_WeightsXavierInit(ML2_AttentionWeightsKey(attention));
//     ML2_WeightsXavierInit(ML2_AttentionWeightsValue(attention));
// }

// ML2_FN void ML2_AttentionHeInit(ML2_LayerAttention attention) {
//     ML2_WeightsHeInit(ML2_AttentionWeightsQuery(attention));
//     ML2_WeightsHeInit(ML2_AttentionWeightsKey(attention));
//     ML2_WeightsHeInit(ML2_AttentionWeightsValue(attention));
// }

ML2_FN void ML2_AttentionInfoPrint(ML2_LayerInfoAttention info, ML2_Int indent) {
    printf(ML2_INDENT("Attention(%s)\n", indent), ML2_BOOL_TO_STR(info.masked));
}

// ML2_FN void ML2_AttentionPrint(ML2_LayerAttention attention, ML2_Int indent) {
//     ML2_AttentionInfoPrint(attention.info, indent);
//     printf(ML2_INDENT("{\n", indent));
//     {
//         indent += ML2_Indentation;
//         printf(ML2_INDENT("Query(%dx%d):\n", indent), attention.info.inputs, attention.info.inputs);
//         printf(ML2_INDENT("{\n", indent));
//         {
//             indent += ML2_Indentation;
//             ML2_WeightsPrint(ML2_AttentionWeightsQuery(attention), indent);
//             indent -= ML2_Indentation;
//         }
//         printf(ML2_INDENT("}\n", indent));
//         printf(ML2_INDENT("Key(%dx%d):\n", indent), attention.info.inputs, attention.info.inputs);
//         printf(ML2_INDENT("{\n", indent));
//         {
//             indent += ML2_Indentation;
//             ML2_WeightsPrint(ML2_AttentionWeightsKey(attention), indent);
//             indent -= ML2_Indentation;
//         }
//         printf(ML2_INDENT("}\n", indent));
//         printf(ML2_INDENT("Value(%dx%d):\n", indent), attention.info.inputs, attention.info.inputs);
//         printf(ML2_INDENT("{\n", indent));
//         {
//             indent += ML2_Indentation;
//             ML2_WeightsPrint(ML2_AttentionWeightsValue(attention), indent);
//             indent -= ML2_Indentation;
//         }
//         printf(ML2_INDENT("}\n", indent));
//         indent -= ML2_Indentation;
//     }
//     printf(ML2_INDENT("}\n", indent));
// }

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
    return a.masked == b.masked;
}

ML2_FN bool ML2_AttentionForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheInfoMatrices inputMatrices = ML2_LayerCacheInfoAsMatrices(input);
            ML2_LayerCacheInfoVectors outputVectors = ML2_LayerCacheInfoAsVectors(output);
            return inputMatrices.vectors == ML2_AttentionVectorsCount &&
                   outputVectors.samples == inputMatrices.samples &&
                   outputVectors.vectors == inputMatrices.matrices &&
                   outputVectors.scalars == inputMatrices.scalars;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_AttentionForward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_AttentionForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheVectors outputVectors = ML2_LayerCacheAsVectors(output);
            
            ML2_Int samples = outputVectors.info.samples;
            ML2_Int vectors = outputVectors.info.vectors;
            ML2_Int scalars = outputVectors.info.scalars;
            const bool masked = attention.masked;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int qv = 0; qv < vectors; qv++) {
                    // TODO(2/9/2026 00:59:43): this is a VLA, people say VLA bad, but its between this or heap memory
                    ML2_Scalar softmaxVector[scalars] = {};
                    ML2_Scalar max = ML2_NegInf;
                    ML2_Scalar sum = {};
                    // TODO(3/9/2026/ 21:29:05): find a better name, then find out if this impacts performance, the branch predictor should do fine because its always true/false, but maybe this impacts vectorization
                    ML2_Int iterations = (masked) ? (qv + 1) : (vectors);

                    for (ML2_Int v = 0; v < iterations; v++) {
                        ML2_Scalar score = {};
                        for (ML2_Int c = 0; c < scalars; c++) {
                            score += *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c) * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
                        }
                        score /= ML2_SQRT(scalars);
                        ML2_Scalar newMax = ML2_FMAX(max, score);
                        ML2_Scalar oldCorrection = ML2_EXP(max - newMax);
                        ML2_Scalar newCorrection = ML2_EXP(score - newMax);
                        for (ML2_Int c = 0; c < scalars; c++) {
                            softmaxVector[c] = softmaxVector[c] * oldCorrection + *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexValue, c) * newCorrection;
                        }
                        sum = sum * oldCorrection + newCorrection;
                        max = newMax;
                    }
                    for (ML2_Int c = 0; c < scalars; c++) {
                        *ML2_VectorsAt(outputVectors, s, qv, c) += softmaxVector[c] / sum;
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_FN bool ML2_AttentionBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo outputGradient) {
//     return ML2_AttentionForwardCompatible(input, outputGradient);
// }

// ML2_FN void ML2_AttentionBackward(ML2_LayerInfoAttention attentionGradient, ML2_LayerCache input, ML2_LayerCache outputGradient) {
//     ML2_HARD_ASSERT(ML2_AttentionBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient)));
//     switch (input.type) {
//         case ML2_LayerCacheTypeScalars: {
//             ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
//         } break;
//         case ML2_LayerCacheTypeVectors: {
//             ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
//         } break;
//         case ML2_LayerCacheTypeMatrices: {
//             ML2_LayerCacheVectors inputVectors = ML2_LayerCacheAsVectors(input);
//             ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            
//             ML2_Int samples = inputVectors.info.samples;
//             ML2_Int vectors = inputVectors.info.vectors;
//             ML2_Int scalars = inputVectors.info.scalars;

            
//         }
//         default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
//     }
// }

ML2_FN bool ML2_AttentionCacheBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_AttentionForwardCompatible(inputGradient, outputGradient);
}

// ML2_FN void ML2_AttentionCacheBackward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
//     ML2_HARD_ASSERT(ML2_AttentionCacheBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
//     switch (inputGradient.type) {
//         case ML2_LayerCacheTypeScalars: {
//             ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
//         } break;
//         case ML2_LayerCacheTypeVectors: {
//             ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
//         } break;
//         case ML2_LayerCacheTypeMatrices: {
//             ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
//             ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
//             ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);
            
//             ML2_Int samples = outputGradientVectors.info.samples;
//             ML2_Int vectors = outputGradientVectors.info.vectors;
//             ML2_Int scalars = outputGradientVectors.info.scalars;

//             for (ML2_Int s = 0; s < samples; s++) {
//                 for (ML2_Int qv = 0; qv < vectors; qv++) {
//                     // TODO(3/9/2026/ 21:29:05): find a better name, then find out if this impacts performance, the branch predictor should do fine because its always true/false, but maybe this impacts vectorization
//                     ML2_Int iterations = (attention.masked) ? (qv + 1) : (vectors);
//                     ML2_Scalar max = ML2_NegInf;
//                     ML2_Scalar sum = {};

//                     for (ML2_Int v = 0; v < iterations; v++) {
//                         ML2_Scalar scoreScalar = {};
//                         for (ML2_Int c = 0; c < scalars; c++) {
//                             scoreScalar += *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c) * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
//                         }
//                         scoreScalar /= ML2_SQRT(scalars);
//                         ML2_Scalar newMax = ML2_FMAX(max, scoreScalar);
//                         ML2_Scalar oldCorrection = ML2_EXP(max - newMax);
//                         ML2_Scalar newCorrection = ML2_EXP(scoreScalar - newMax);
//                         sum = sum * oldCorrection + newCorrection;
//                         max = newMax;
//                     }

//                     // TODO(3/9/2026 23:00:27): this is a VLA, people say VLA bad, but its between this or heap memory
//                     ML2_Scalar softmaxGradientVector[scalars] = {};
//                     ML2_Scalar correctionSum = {};

//                     for (ML2_Int v = 0; v < iterations; v++) {
//                         ML2_Scalar scoreScalar = {};
//                         for (ML2_Int c = 0; c < scalars; c++) {
//                             scoreScalar += *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c) * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
//                         }
//                         scoreScalar /= ML2_SQRT(scalars);
//                         ML2_Scalar softmaxScore = ML2_EXP(scoreScalar - max) / sum;

//                         ML2_Scalar valueGradientScalar = {};
//                         for (ML2_Int c = 0; c < scalars; c++) {
//                             ML2_Scalar outputGradientScalar = *ML2_VectorsAt(outputGradientVectors, s, qv, c);
//                             valueGradientScalar += outputGradientScalar * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexValue, c);
//                             *ML2_MatricesAt(inputGradientMatrices, s, v, ML2_AttentionInputIndexValue, c) += softmaxScore * outputGradientScalar;
//                         }
//                         correctionSum += softmaxScore * valueGradientScalar;
//                     }

//                     for (ML2_Int v = 0; v < iterations; v++) {
//                         ML2_Scalar score = {};
//                         for (ML2_Int c = 0; c < scalars; c++) {
//                             score += *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c) * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
//                         }
//                         score /= ML2_SQRT(scalars);
//                         ML2_Scalar softmaxScore = ML2_EXP(score - max) / sum;

//                         ML2_Scalar valueGradientScalar = {};
//                         for (ML2_Int c = 0; c < scalars; c++) {
//                             valueGradientScalar += *ML2_VectorsAt(outputGradientVectors, s, qv, c) * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexValue, c);
//                         }

//                         ML2_Scalar scoreGradientScalar = softmaxScore * (valueGradientScalar - correctionSum) / ML2_SQRT(scalars);
//                         for (ML2_Int c = 0; c < scalars; c++) {
//                             softmaxGradientVector[c] += scoreGradientScalar * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
//                             *ML2_MatricesAt(inputGradientMatrices, s, v, ML2_AttentionInputIndexKey, c) += scoreGradientScalar * *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c);
//                         }
//                     }

//                     for (ML2_Int c = 0; c < scalars; c++) {
//                         *ML2_MatricesAt(inputGradientMatrices, s, qv, ML2_AttentionInputIndexQuery, c) += softmaxGradientVector[c];
//                     }
//                 }
//             }
//         } break;
//         default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
//     }
// }
ML2_FN void ML2_AttentionCacheBackward(ML2_LayerInfoAttention attention, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_AttentionCacheBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (inputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeVectors: {
            ML2_TODO("Vectors -> Attention may or may not be a feature in the future");
        } break;
        case ML2_LayerCacheTypeMatrices: {
            ML2_LayerCacheMatrices inputMatrices = ML2_LayerCacheAsMatrices(input);
            ML2_LayerCacheMatrices inputGradientMatrices = ML2_LayerCacheAsMatrices(inputGradient);
            ML2_LayerCacheVectors outputGradientVectors = ML2_LayerCacheAsVectors(outputGradient);

            ML2_Int samples = outputGradientVectors.info.samples;
            ML2_Int vectors = outputGradientVectors.info.vectors;
            ML2_Int scalars = outputGradientVectors.info.scalars;
            const bool masked = attention.masked;

            for (ML2_Int s = 0; s < samples; s++) {
                for (ML2_Int qv = 0; qv < vectors; qv++) {
                    ML2_Int iterations = (masked) ? (qv + 1) : (vectors);

                    ML2_Scalar scoreVector[iterations];
                    ML2_Scalar softmaxVector[iterations];
                    ML2_Scalar valueGradientVector[iterations];

                    ML2_Scalar max = ML2_NegInf;
                    ML2_Scalar sum = {};

                    for (ML2_Int v = 0; v < iterations; v++) {
                        ML2_Scalar scoreScalar = {};
                        for (ML2_Int c = 0; c < scalars; c++) {
                            scoreScalar += *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c) * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
                        }
                        scoreScalar /= ML2_SQRT(scalars);
                        scoreVector[v] = scoreScalar;

                        ML2_Scalar newMax = ML2_FMAX(max, scoreScalar);
                        ML2_Scalar oldCorrection = ML2_EXP(max - newMax);
                        ML2_Scalar newCorrection = ML2_EXP(scoreScalar - newMax);
                        sum = sum * oldCorrection + newCorrection;
                        max = newMax;
                    }

                    ML2_Scalar softmaxGradientVector[scalars] = {};
                    ML2_Scalar correctionSum = {};

                    for (ML2_Int v = 0; v < iterations; v++) {
                        ML2_Scalar softmaxScore = ML2_EXP(scoreVector[v] - max) / sum;
                        softmaxVector[v] = softmaxScore;

                        ML2_Scalar valueGradientScalar = {};
                        for (ML2_Int c = 0; c < scalars; c++) {
                            ML2_Scalar outputGradientScalar = *ML2_VectorsAt(outputGradientVectors, s, qv, c);
                            valueGradientScalar += outputGradientScalar * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexValue, c);
                            *ML2_MatricesAt(inputGradientMatrices, s, v, ML2_AttentionInputIndexValue, c) += softmaxScore * outputGradientScalar;
                        }
                        valueGradientVector[v] = valueGradientScalar;
                        correctionSum += softmaxScore * valueGradientScalar;
                    }

                    for (ML2_Int v = 0; v < iterations; v++) {
                        ML2_Scalar scoreGradientScalar = softmaxVector[v] * (valueGradientVector[v] - correctionSum) / ML2_SQRT(scalars);
                        for (ML2_Int c = 0; c < scalars; c++) {
                            softmaxGradientVector[c] += scoreGradientScalar * *ML2_MatricesAt(inputMatrices, s, v, ML2_AttentionInputIndexKey, c);
                            *ML2_MatricesAt(inputGradientMatrices, s, v, ML2_AttentionInputIndexKey, c) += scoreGradientScalar * *ML2_MatricesAt(inputMatrices, s, qv, ML2_AttentionInputIndexQuery, c);
                        }
                    }

                    for (ML2_Int c = 0; c < scalars; c++) {
                        *ML2_MatricesAt(inputGradientMatrices, s, qv, ML2_AttentionInputIndexQuery, c) += softmaxGradientVector[c];
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_FN void ML2_AttentionGradientDescentOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate) {
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, attentionGradient.info));
//     ML2_WeightsGradientDescentOptimize(ML2_AttentionWeightsQuery(attention), ML2_AttentionWeightsQuery(attentionGradient), learningRate);
//     ML2_WeightsGradientDescentOptimize(ML2_AttentionWeightsKey(attention), ML2_AttentionWeightsKey(attentionGradient), learningRate);
//     ML2_WeightsGradientDescentOptimize(ML2_AttentionWeightsValue(attention), ML2_AttentionWeightsValue(attentionGradient), learningRate);
// }

// ML2_FN void ML2_AttentionMomentumOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerAttention average) {
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, attentionGradient.info));
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, average.info));
//     ML2_WeightsMomentumOptimize(ML2_AttentionWeightsQuery(attention), ML2_AttentionWeightsQuery(attentionGradient), learningRate, decayRate, ML2_AttentionWeightsQuery(average));
//     ML2_WeightsMomentumOptimize(ML2_AttentionWeightsKey(attention), ML2_AttentionWeightsKey(attentionGradient), learningRate, decayRate, ML2_AttentionWeightsKey(average));
//     ML2_WeightsMomentumOptimize(ML2_AttentionWeightsValue(attention), ML2_AttentionWeightsValue(attentionGradient), learningRate, decayRate, ML2_AttentionWeightsValue(average));
// }

// ML2_FN void ML2_AttentionAdagradOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_LayerAttention squareSum) {
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, attentionGradient.info));
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, squareSum.info));
//     ML2_WeightsAdagradOptimize(ML2_AttentionWeightsQuery(attention), ML2_AttentionWeightsQuery(attentionGradient), learningRate, ML2_AttentionWeightsQuery(squareSum));
//     ML2_WeightsAdagradOptimize(ML2_AttentionWeightsKey(attention), ML2_AttentionWeightsKey(attentionGradient), learningRate, ML2_AttentionWeightsKey(squareSum));
//     ML2_WeightsAdagradOptimize(ML2_AttentionWeightsValue(attention), ML2_AttentionWeightsValue(attentionGradient), learningRate, ML2_AttentionWeightsValue(squareSum));
// }

// ML2_FN void ML2_AttentionRMSPropOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerAttention squareAverage) {
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, attentionGradient.info));
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, squareAverage.info));
//     ML2_WeightsRMSPropOptimize(ML2_AttentionWeightsQuery(attention), ML2_AttentionWeightsQuery(attentionGradient), learningRate, decayRate, ML2_AttentionWeightsQuery(squareAverage));
//     ML2_WeightsRMSPropOptimize(ML2_AttentionWeightsKey(attention), ML2_AttentionWeightsKey(attentionGradient), learningRate, decayRate, ML2_AttentionWeightsKey(squareAverage));
//     ML2_WeightsRMSPropOptimize(ML2_AttentionWeightsValue(attention), ML2_AttentionWeightsValue(attentionGradient), learningRate, decayRate, ML2_AttentionWeightsValue(squareAverage));
// }

// ML2_FN void ML2_AttentionAdamOptimize(ML2_LayerAttention attention, ML2_LayerAttention attentionGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerAttention average, ML2_LayerAttention squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, attentionGradient.info));
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, average.info));
//     ML2_HARD_ASSERT(ML2_AttentionInfoSame(attention.info, squareAverage.info));
//     ML2_WeightsAdamOptimize(ML2_AttentionWeightsQuery(attention), ML2_AttentionWeightsQuery(attentionGradient), learningRate, decayRate1, decayRate2, ML2_AttentionWeightsQuery(average), ML2_AttentionWeightsQuery(squareAverage), decayingWeight1, decayingWeight2);
//     ML2_WeightsAdamOptimize(ML2_AttentionWeightsKey(attention), ML2_AttentionWeightsKey(attentionGradient), learningRate, decayRate1, decayRate2, ML2_AttentionWeightsKey(average), ML2_AttentionWeightsKey(squareAverage), decayingWeight1, decayingWeight2);
//     ML2_WeightsAdamOptimize(ML2_AttentionWeightsValue(attention), ML2_AttentionWeightsValue(attentionGradient), learningRate, decayRate1, decayRate2, ML2_AttentionWeightsValue(average), ML2_AttentionWeightsValue(squareAverage), decayingWeight1, decayingWeight2);
// }

// ML2_FN ML2_Size ML2_AttentionSizeof(ML2_LayerAttention attention) {
//     return ML2_WeightsSizeof(ML2_AttentionWeightsQuery(attention)) + ML2_WeightsSizeof(ML2_AttentionWeightsKey(attention)) + ML2_WeightsSizeof(ML2_AttentionWeightsValue(attention));
// }

// ML2_LayerAttention ⬆️

ML2_FN ML2_Layer ML2_LayerNew(ML2_LayerInfo info) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return (ML2_Layer){info.type, .as.weights = ML2_WeightsNew(ML2_LayerInfoAsWeights(info))};
        case ML2_LayerTypeBiases: return (ML2_Layer){info.type, .as.biases = ML2_BiasesNew(ML2_LayerInfoAsBiases(info))};
        case ML2_LayerTypeActivation: return (ML2_Layer){info.type, .as.activation = ML2_LayerInfoAsActivation(info)};
        case ML2_LayerTypeLinear: return (ML2_Layer){info.type, .as.linear = ML2_LinearNew(ML2_LayerInfoAsLinear(info))};
        case ML2_LayerTypeFilters: return (ML2_Layer){info.type, .as.filters = ML2_FiltersNew(ML2_LayerInfoAsFilters(info))};
        case ML2_LayerTypeConv: return (ML2_Layer){info.type, .as.conv = ML2_ConvNew(ML2_LayerInfoAsConv(info))};
        case ML2_LayerTypeFlatten: return (ML2_Layer){info.type, {}};
        case ML2_LayerTypePool: return (ML2_Layer){info.type, .as.pool = ML2_LayerInfoAsPool(info)};
        case ML2_LayerTypeFusedWeights: return (ML2_Layer){info.type, .as.fusedWeights = ML2_FusedWeightsNew(ML2_LayerInfoAsFusedWeights(info))};
        case ML2_LayerTypeAttention: return (ML2_Layer){info.type, .as.attention = ML2_LayerInfoAsAttention(info)};
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerDestroy(ML2_Layer *layer) {
    switch (layer->type) {
        case ML2_LayerTypeWeights: ML2_WeightsDestroy(&layer->as.weights); break;
        case ML2_LayerTypeBiases: ML2_BiasesDestroy(&layer->as.biases); break;
        case ML2_LayerTypeActivation: *layer = (ML2_Layer){}; break;
        case ML2_LayerTypeLinear: ML2_LinearDestroy(&layer->as.linear); break;
        case ML2_LayerTypeFilters: ML2_FiltersDestroy(&layer->as.filters); break;
        case ML2_LayerTypeConv: ML2_ConvDestroy(&layer->as.conv); break;
        case ML2_LayerTypeFlatten: *layer = (ML2_Layer){}; break;
        case ML2_LayerTypePool: *layer = (ML2_Layer){}; break;
        case ML2_LayerTypeFusedWeights: ML2_FusedWeightsDestroy(&layer->as.fusedWeights); break;
        case ML2_LayerTypeAttention: *layer = (ML2_Layer){}; break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return (ML2_LayerInfo){layer.type, .as.weights = ML2_LayerAsWeights(layer).info};
        case ML2_LayerTypeBiases: return (ML2_LayerInfo){layer.type, .as.biases = ML2_LayerAsBiases(layer).info};
        case ML2_LayerTypeActivation: return (ML2_LayerInfo){layer.type, .as.activation = ML2_LayerAsActivation(layer)};
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

ML2_FN void ML2_LayerClear(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsClear(ML2_LayerAsWeights(layer)); break;
        case ML2_LayerTypeBiases: ML2_BiasesClear(ML2_LayerAsBiases(layer)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
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

ML2_FN void ML2_LayerSum(ML2_Layer dest, ML2_Layer src) {
    ML2_HARD_ASSERT(ML2_LayerInfoSame(ML2_LayerAsInfo(dest), ML2_LayerAsInfo(src)));
    switch (dest.type) {
        case ML2_LayerTypeWeights: ML2_WeightsSum(ML2_LayerAsWeights(dest), ML2_LayerAsWeights(src)); break;
        case ML2_LayerTypeBiases: ML2_BiasesSum(ML2_LayerAsBiases(dest), ML2_LayerAsBiases(src)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
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

ML2_FN void ML2_LayerInfoPrint(ML2_LayerInfo layer, ML2_Int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsInfoPrint(ML2_LayerInfoAsWeights(layer), indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesInfoPrint(ML2_LayerInfoAsBiases(layer), indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationInfoPrint(ML2_LayerInfoAsActivation(layer), indent); break;
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

ML2_FN void ML2_LayerPrint(ML2_Layer layer, ML2_Int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsPrint(ML2_LayerAsWeights(layer), indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesPrint(ML2_LayerAsBiases(layer), indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationInfoPrint(ML2_LayerAsActivation(layer), indent); break;
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

ML2_FN bool ML2_LayerInfoForwardCompatible(ML2_LayerInfo info, ML2_LayerCacheInfo input) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsInfoForwardCompatible(ML2_LayerInfoAsWeights(info), input);
        case ML2_LayerTypeBiases: return ML2_BiasesInfoForwardCompatible(ML2_LayerInfoAsBiases(info), input);
        case ML2_LayerTypeActivation: return ML2_ActivationInfoForwardCompatible(ML2_LayerInfoAsActivation(info), input);
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
        case ML2_LayerTypeActivation: return ML2_ActivationInfoForward(ML2_LayerInfoAsActivation(info), input);
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

ML2_FN bool ML2_LayerForwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsForwardCompatible(ML2_LayerInfoAsWeights(layer), input, output);
        case ML2_LayerTypeBiases: return ML2_BiasesForwardCompatible(ML2_LayerInfoAsBiases(layer), input, output);
        case ML2_LayerTypeActivation: return ML2_ActivationForwardCompatible(input, output);
        case ML2_LayerTypeLinear: return ML2_LinearForwardCompatible(ML2_LayerInfoAsLinear(layer), input, output);
        case ML2_LayerTypeFilters: return ML2_FiltersForwardCompatible(ML2_LayerInfoAsFilters(layer), input, output);
        case ML2_LayerTypeConv: return ML2_ConvForwardCompatible(ML2_LayerInfoAsConv(layer), input, output);
        case ML2_LayerTypeFlatten: return ML2_FlattenForwardCompatible(input, output);
        case ML2_LayerTypePool: return ML2_PoolForwardCompatible(ML2_LayerInfoAsPool(layer), input, output);
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsForwardCompatible(ML2_LayerInfoAsFusedWeights(layer), input, output);
        case ML2_LayerTypeAttention: return ML2_AttentionForwardCompatible(input, output);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_LayerForwardCompatible(ML2_LayerAsInfo(layer), ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsForward(ML2_LayerAsWeights(layer), input, output); break;
        case ML2_LayerTypeBiases: ML2_BiasesForward(ML2_LayerAsBiases(layer), input, output); break;
        case ML2_LayerTypeActivation: ML2_ActivationForward(ML2_LayerAsActivation(layer), input, output); break;
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
        case ML2_LayerTypeActivation: return ML2_ActivationCacheBackwardCompatible(ML2_LayerInfoAsActivation(layer), input, inputGradient, output, outputGradient);
        case ML2_LayerTypeLinear: return ML2_LinearCacheBackwardCompatible(ML2_LayerInfoAsLinear(layer), inputGradient, outputGradient);
        case ML2_LayerTypeFilters: return ML2_FiltersCacheBackwardCompatible(ML2_LayerInfoAsFilters(layer), inputGradient, outputGradient);
        case ML2_LayerTypeConv: return ML2_ConvCacheBackwardCompatible(ML2_LayerInfoAsConv(layer), inputGradient, outputGradient);
        case ML2_LayerTypeFlatten: return ML2_FlattenCacheBackwardCompatible(inputGradient, outputGradient);
        case ML2_LayerTypePool: return ML2_PoolCacheBackwardCompatible(ML2_LayerInfoAsPool(layer), input, inputGradient, output, outputGradient);
        case ML2_LayerTypeFusedWeights: return ML2_FusedWeightsCacheBackwardCompatible(ML2_LayerInfoAsFusedWeights(layer), inputGradient, outputGradient);
        case ML2_LayerTypeAttention: return ML2_AttentionCacheBackwardCompatible(input, inputGradient, outputGradient);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerCacheBackward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    ML2_HARD_ASSERT(ML2_LayerCacheBackwardCompatible(ML2_LayerAsInfo(layer), ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(outputGradient)));
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsCacheBackward(ML2_LayerAsWeights(layer), inputGradient, outputGradient); break;
        case ML2_LayerTypeBiases: ML2_BiasesCacheBackward(inputGradient, outputGradient); break;
        case ML2_LayerTypeActivation: ML2_ActivationCacheBackward(ML2_LayerAsActivation(layer), input, inputGradient, output, outputGradient); break;
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
        case ML2_LayerTypeWeights: return  ML2_WeightsSizeof(ML2_LayerAsWeights(layer));
        case ML2_LayerTypeBiases: return ML2_BiasesSizeof(ML2_LayerAsBiases(layer));
        case ML2_LayerTypeActivation: return 0;
        case ML2_LayerTypeLinear: return ML2_LinearSizeof(ML2_LayerAsLinear(layer));
        case ML2_LayerTypeFilters: return  ML2_FiltersSizeof(ML2_LayerAsFilters(layer));
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

ML2_FN ML2_LayerCacheScalars ML2_ScalarsNew(ML2_LayerCacheInfoScalars info) {
    ML2_LayerCacheScalars scalars = {
        info,
        {ML2_RELIABLE_CALLOC(info.samples * info.scalars, sizeof(*scalars.data.scalars))}
    };
    return scalars;
}

ML2_FN void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars) {
    ML2_FREE(scalars->data.scalars);
    *scalars = (ML2_LayerCacheScalars){};
}

ML2_FN ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, ML2_Int sample, ML2_Int scalar) {
    ML2_SOFT_ASSERT(0 <= sample && sample < scalars.info.samples && 0 <= scalar && scalar < scalars.info.scalars && "OUT OF BOUNDS INDICES");
    return &scalars.data.scalars[sample * scalars.info.scalars + scalar];
}

ML2_FN ML2_LayerCacheInfoScalars ML2_LayerCacheInfoAsScalars(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeScalars);
    return info.as.scalars;
}

ML2_FN ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeScalars);
    return cache.as.scalars;
}

ML2_FN void ML2_ScalarsClear(ML2_LayerCacheScalars scalars) {
    for (ML2_Int s = 0; s < scalars.info.samples; s++) {
        for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
            *ML2_ScalarsAt(scalars, s, c) = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_FN void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    ML2_Int samples = dest.info.samples;
    ML2_Int scalars = dest.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            *ML2_ScalarsAt(dest, s, c) = *ML2_ScalarsAt(src, s, c);
        }
    }
}

ML2_FN void ML2_ScalarsSum(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    ML2_Int samples = dest.info.samples;
    ML2_Int scalars = dest.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            *ML2_ScalarsAt(dest, s, c) += *ML2_ScalarsAt(src, s, c);
        }
    }
}

ML2_FN void ML2_ScalarsRand(ML2_LayerCacheScalars scalars, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int s = 0; s < scalars.info.samples; s++) {
        for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
            *ML2_ScalarsAt(scalars, s, c) = ML2_ScalarRand(low, high);
        }
    }
}

ML2_FN void ML2_ScalarsXavierInit(ML2_LayerCacheScalars scalars) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (scalars.info.samples + scalars.info.scalars));
    ML2_ScalarsRand(scalars, -limit, limit);
}

ML2_FN void ML2_ScalarsHeInit(ML2_LayerCacheScalars scalars) {
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / scalars.info.scalars);
    for (ML2_Int s = 0; s < scalars.info.samples; s++) {
        for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
            *ML2_ScalarsAt(scalars, s, c) = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
        }
    }
}

ML2_FN void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, ML2_Int indent) {
    printf(ML2_INDENT("Scalars(%dx%d)\n", indent), info.samples, info.scalars);
}

ML2_FN void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, ML2_Int indent) {
    ML2_ScalarsInfoPrint(scalars.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (ML2_Int s = 0; s < scalars.info.samples; s++) {
            printf(ML2_INDENT("", indent));
            for (ML2_Int c = 0; c < scalars.info.scalars; c++) {
                printf(ML2_SCALAR_FMT " ", *ML2_ScalarsAt(scalars, s, c));
            }
            printf("\n");
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b) {
    return a.samples == b.samples && a.scalars == b.scalars;
}

ML2_FN ML2_Size ML2_ScalarsSizeof(ML2_LayerCacheScalars scalars) {
    return scalars.info.samples * scalars.info.scalars * sizeof(*scalars.data.scalars);
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expected.info, predicted.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar diff = *ML2_ScalarsAt(predicted, s, c) - *ML2_ScalarsAt(expected, s, c);
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

ML2_FN void ML2_ScalarsLossBackwardSquareAverage(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar diff = *ML2_ScalarsAt(predicted, s, c) - *ML2_ScalarsAt(expected, s, c);
            *ML2_ScalarsAt(gradient, s, c) += ML2_SCALAR_LITERAL(2.0) * diff / (samples * scalars);
        }
    }
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMAX(*ML2_ScalarsAt(predicted, s, c), ML2_Epsilon);
            ML2_Scalar expectedScalar = *ML2_ScalarsAt(expected, s, c);
            loss += -expectedScalar * ML2_LN(predictedScalar);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMAX(*ML2_ScalarsAt(predicted, s, c), ML2_Epsilon);
            ML2_Scalar expectedScalar = *ML2_ScalarsAt(expected, s, c);
            *ML2_ScalarsAt(gradient, s, c) += -(expectedScalar / predictedScalar) / samples;
        }
    }
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar max = ML2_NegInf;
        for (ML2_Int c = 0; c < scalars; c++) {
            max = ML2_FMAX(max, *ML2_ScalarsAt(predicted, s, c));
        }
        ML2_Scalar sum = {};
        for (ML2_Int c = 0; c < scalars; c++) {
            sum += ML2_EXP(*ML2_ScalarsAt(predicted, s, c) - max);
        }
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMAX(ML2_EXP(*ML2_ScalarsAt(predicted, s, c) - max) / sum, ML2_Epsilon);
            ML2_Scalar expectedScalar = *ML2_ScalarsAt(expected, s, c);
            loss += -expectedScalar * ML2_LN(predictedScalar);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        ML2_Scalar max = ML2_NegInf;
        for (ML2_Int c = 0; c < scalars; c++) {
            max = ML2_FMAX(max, *ML2_ScalarsAt(predicted, s, c));
        }
        ML2_Scalar sum = {};
        for (ML2_Int c = 0; c < scalars; c++) {
            sum += ML2_EXP(*ML2_ScalarsAt(predicted, s, c) - max);
        }
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_EXP(*ML2_ScalarsAt(predicted, s, c) - max) / sum;
            ML2_Scalar expectedScalar = *ML2_ScalarsAt(expected, s, c);
            *ML2_ScalarsAt(gradient, s, c) += (predictedScalar - expectedScalar) / samples;
        }
    }
}

ML2_FN ML2_Scalar ML2_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(*ML2_ScalarsAt(predicted, s, c), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
            ML2_Scalar expectedScalar = *ML2_ScalarsAt(expected, s, c);
            loss += -expectedScalar * ML2_LN(predictedScalar) - (ML2_SCALAR_LITERAL(1.0) - expectedScalar) * ML2_LN(ML2_SCALAR_LITERAL(1.0) - predictedScalar);
        }
    }
    loss /= (samples * scalars);
    return loss;
}

ML2_FN void ML2_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheScalars predicted, ML2_LayerCacheScalars expected, ML2_LayerCacheScalars gradient) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int c = 0; c < scalars; c++) {
            ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(*ML2_ScalarsAt(predicted, s, c), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
            ML2_Scalar expectedScalar = *ML2_ScalarsAt(expected, s, c);
            *ML2_ScalarsAt(gradient, s, c) += (predictedScalar - expectedScalar) / (predictedScalar * (ML2_SCALAR_LITERAL(1.0) - predictedScalar)) / (samples * scalars);
        }
    }
}

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheVectors ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Vectors(ML2_Int samples, ML2_Int vectors, ML2_Int scalars) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeVectors, .as.vectors = {samples, vectors, scalars}};
}

ML2_FN ML2_LayerCacheVectors ML2_VectorsNew(ML2_LayerCacheInfoVectors info) {
    ML2_LayerCacheVectors vectors = {
        info,
        {ML2_RELIABLE_CALLOC(info.samples * info.vectors * info.scalars, sizeof(*vectors.data.scalars))}
    };
    return vectors;
}

ML2_FN void ML2_VectorsDestroy(ML2_LayerCacheVectors *vectors) {
    ML2_FREE(vectors->data.scalars);
    *vectors = (ML2_LayerCacheVectors){};
}

ML2_FN ML2_Scalar *ML2_VectorsAt(ML2_LayerCacheVectors vectors, ML2_Int sample, ML2_Int vector, ML2_Int scalar) {
    ML2_SOFT_ASSERT(0 <= sample && sample < vectors.info.samples && 0 <= vector && vector < vectors.info.vectors && 0 <= scalar && scalar < vectors.info.scalars && "OUT OF BOUNDS INDICES");
    return &vectors.data.scalars[(sample * vectors.info.vectors + vector) * vectors.info.scalars + scalar];
}

ML2_FN ML2_LayerCacheInfoVectors ML2_LayerCacheInfoAsVectors(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeVectors);
    return info.as.vectors;
}

ML2_FN ML2_LayerCacheVectors ML2_LayerCacheAsVectors(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeVectors);
    return cache.as.vectors;
}

ML2_FN void ML2_VectorsClear(ML2_LayerCacheVectors vectors) {
    for (ML2_Int s = 0; s < vectors.info.samples; s++) {
        for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
            for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                *ML2_VectorsAt(vectors, s, v, c) = ML2_SCALAR_LITERAL(0.0);
            }
        }
    }
}

ML2_FN void ML2_VectorsCopy(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(dest.info, src.info));
    ML2_Int samples = dest.info.samples;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                *ML2_VectorsAt(dest, s, v, c) = *ML2_VectorsAt(src, s, v, c);
            }
        }
    }
}

ML2_FN void ML2_VectorsSum(ML2_LayerCacheVectors dest, ML2_LayerCacheVectors src) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(dest.info, src.info));
    ML2_Int samples = dest.info.samples;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                *ML2_VectorsAt(dest, s, v, c) += *ML2_VectorsAt(src, s, v, c);
            }
        }
    }
}

ML2_FN void ML2_VectorsRand(ML2_LayerCacheVectors vectors, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int s = 0; s < vectors.info.samples; s++) {
        for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
            for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                *ML2_VectorsAt(vectors, s, v, c) = ML2_ScalarRand(low, high);
            }
        }
    }
}

ML2_FN void ML2_VectorsXavierInit(ML2_LayerCacheVectors vectors) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (vectors.info.samples + vectors.info.vectors + vectors.info.scalars));
    ML2_VectorsRand(vectors, -limit, limit);
}

ML2_FN void ML2_VectorsHeInit(ML2_LayerCacheVectors vectors) {
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (vectors.info.vectors * vectors.info.scalars));
    for (ML2_Int s = 0; s < vectors.info.samples; s++) {
        for (ML2_Int v = 0; v < vectors.info.vectors; v++) {
            for (ML2_Int c = 0; c < vectors.info.scalars; c++) {
                *ML2_VectorsAt(vectors, s, v, c) = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
            }
        }
    }
}

ML2_FN void ML2_VectorsInfoPrint(ML2_LayerCacheInfoVectors info, ML2_Int indent) {
    printf(ML2_INDENT("Vectors(%dx%dx%d)\n", indent), info.samples, info.vectors, info.scalars);
}

ML2_FN void ML2_VectorsPrint(ML2_LayerCacheVectors vectors, ML2_Int indent) {
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
                        printf(ML2_SCALAR_FMT " ", *ML2_VectorsAt(vectors, s, v, c));
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

ML2_FN bool ML2_VectorsInfoSame(ML2_LayerCacheInfoVectors a, ML2_LayerCacheInfoVectors b) {
    return a.samples == b.samples && a.vectors == b.vectors && a.scalars == b.scalars;
}

ML2_FN ML2_Size ML2_VectorsSizeof(ML2_LayerCacheVectors vectors) {
    return vectors.info.samples * vectors.info.vectors * vectors.info.scalars * sizeof(*vectors.data.scalars);
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar diff = *ML2_VectorsAt(predicted, s, v, c) - *ML2_VectorsAt(expected, s, v, c);
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

ML2_FN void ML2_VectorsLossBackwardSquareAverage(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar diff = *ML2_VectorsAt(predicted, s, v, c) - *ML2_VectorsAt(expected, s, v, c);
                *ML2_VectorsAt(gradient, s, v, c) += ML2_SCALAR_LITERAL(2.0) * diff / (samples * vectors * scalars);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMAX(*ML2_VectorsAt(predicted, s, v, c), ML2_Epsilon);
                ML2_Scalar expectedScalar = *ML2_VectorsAt(expected, s, v, c);
                loss += -expectedScalar * ML2_LN(predictedScalar);
            }
        }
    }
    loss /= (samples * vectors);
    return loss;
}

ML2_FN void ML2_VectorsLossBackwardCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMAX(*ML2_VectorsAt(predicted, s, v, c), ML2_Epsilon);
                ML2_Scalar expectedScalar = *ML2_VectorsAt(expected, s, v, c);
                *ML2_VectorsAt(gradient, s, v, c) += -(expectedScalar / predictedScalar) / (samples * vectors);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar max = ML2_NegInf;
            for (ML2_Int c = 0; c < scalars; c++) {
                max = ML2_FMAX(max, *ML2_VectorsAt(predicted, s, v, c));
            }
            ML2_Scalar sum = {};
            for (ML2_Int c = 0; c < scalars; c++) {
                sum += ML2_EXP(*ML2_VectorsAt(predicted, s, v, c) - max);
            }
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMAX(ML2_EXP(*ML2_VectorsAt(predicted, s, v, c) - max) / sum, ML2_Epsilon);
                ML2_Scalar expectedScalar = *ML2_VectorsAt(expected, s, v, c);
                loss += -expectedScalar * ML2_LN(predictedScalar);
            }
        }
    }
    loss /= (samples * vectors);
    return loss;
}

ML2_FN void ML2_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            ML2_Scalar max = ML2_NegInf;
            for (ML2_Int c = 0; c < scalars; c++) {
                max = ML2_FMAX(max, *ML2_VectorsAt(predicted, s, v, c));
            }
            ML2_Scalar sum = {};
            for (ML2_Int c = 0; c < scalars; c++) {
                sum += ML2_EXP(*ML2_VectorsAt(predicted, s, v, c) - max);
            }
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_EXP(*ML2_VectorsAt(predicted, s, v, c) - max) / sum;
                ML2_Scalar expectedScalar = *ML2_VectorsAt(expected, s, v, c);
                *ML2_VectorsAt(gradient, s, v, c) += (predictedScalar - expectedScalar) / (samples * vectors);
            }
        }
    }
}

ML2_FN ML2_Scalar ML2_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    ML2_Scalar loss = {};
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(*ML2_VectorsAt(predicted, s, v, c), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
                ML2_Scalar expectedScalar = *ML2_VectorsAt(expected, s, v, c);
                loss += -expectedScalar * ML2_LN(predictedScalar) - (ML2_SCALAR_LITERAL(1.0) - expectedScalar) * ML2_LN(ML2_SCALAR_LITERAL(1.0) - predictedScalar);
            }
        }
    }
    loss /= (samples * vectors * scalars);
    return loss;
}

ML2_FN void ML2_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheVectors predicted, ML2_LayerCacheVectors expected, ML2_LayerCacheVectors gradient) {
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, expected.info));
    ML2_HARD_ASSERT(ML2_VectorsInfoSame(predicted.info, gradient.info));
    ML2_Int samples = predicted.info.samples;
    ML2_Int vectors = predicted.info.vectors;
    ML2_Int scalars = predicted.info.scalars;

    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int v = 0; v < vectors; v++) {
            for (ML2_Int c = 0; c < scalars; c++) {
                ML2_Scalar predictedScalar = ML2_FMIN(ML2_FMAX(*ML2_VectorsAt(predicted, s, v, c), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
                ML2_Scalar expectedScalar = *ML2_VectorsAt(expected, s, v, c);
                *ML2_VectorsAt(gradient, s, v, c) += (predictedScalar - expectedScalar) / (predictedScalar * (ML2_SCALAR_LITERAL(1.0) - predictedScalar)) / (samples * vectors * scalars);
            }
        }
    }
}

// ML2_LayerCacheVectors ⬆️

// ML2_LayerCacheMatrices ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Matrices(ML2_Int samples, ML2_Int height, ML2_Int width, ML2_Int channels) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeMatrices, .as.matrices = {samples, height, width, channels}};
}

ML2_FN ML2_LayerCacheMatrices ML2_MatricesNew(ML2_LayerCacheInfoMatrices info) {
    ML2_LayerCacheMatrices matrices = {
        info,
        {ML2_RELIABLE_CALLOC(info.samples * info.matrices * info.vectors * info.scalars, sizeof(*matrices.data.matrices))}
    };
    return matrices;
}

ML2_FN void ML2_MatricesDestroy(ML2_LayerCacheMatrices *matrices) {
    ML2_FREE(matrices->data.matrices);
    *matrices = (ML2_LayerCacheMatrices){};
}

ML2_FN ML2_Scalar *ML2_MatricesAt(ML2_LayerCacheMatrices matrices, ML2_Int s, ML2_Int m, ML2_Int v, ML2_Int c) {
    ML2_SOFT_ASSERT(0 <= s && s < matrices.info.samples && 0 <= m && m < matrices.info.matrices && 0 <= v && v < matrices.info.vectors && 0 <= c && c < matrices.info.scalars && "OUT OF BOUNDS INDICES");
    return &matrices.data.matrices[((s * matrices.info.matrices + m) * matrices.info.vectors + v) * matrices.info.scalars + c];
}

ML2_FN ML2_LayerCacheInfoMatrices ML2_LayerCacheInfoAsMatrices(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeMatrices);
    return info.as.matrices;
}

ML2_FN ML2_LayerCacheMatrices ML2_LayerCacheAsMatrices(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeMatrices);
    return cache.as.matrices;
}

ML2_FN void ML2_MatricesClear(ML2_LayerCacheMatrices matrices) {
    for (ML2_Int s = 0; s < matrices.info.samples; s++) {
        for (ML2_Int y = 0; y < matrices.info.matrices; y++) {
            for (ML2_Int x = 0; x < matrices.info.vectors; x++) {
                    for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                    *ML2_MatricesAt(matrices, s, y, x, c) = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_FN void ML2_MatricesCopy(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(dest.info, src.info));
    ML2_Int samples = dest.info.samples;
    ML2_Int matrices = dest.info.matrices;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_MatricesAt(dest, s, m, v, c) = *ML2_MatricesAt(src, s, m, v, c);
                }
            }
        }
    }
}

ML2_FN void ML2_MatricesSum(ML2_LayerCacheMatrices dest, ML2_LayerCacheMatrices src) {
    ML2_HARD_ASSERT(ML2_MatricesInfoSame(dest.info, src.info));
    ML2_Int samples = dest.info.samples;
    ML2_Int matrices = dest.info.matrices;
    ML2_Int vectors = dest.info.vectors;
    ML2_Int scalars = dest.info.scalars;
    for (ML2_Int s = 0; s < samples; s++) {
        for (ML2_Int m = 0; m < matrices; m++) {
            for (ML2_Int v = 0; v < vectors; v++) {
                for (ML2_Int c = 0; c < scalars; c++) {
                    *ML2_MatricesAt(dest, s, m, v, c) += *ML2_MatricesAt(src, s, m, v, c);
                }
            }
        }
    }
}

ML2_FN void ML2_MatricesRand(ML2_LayerCacheMatrices matrices, ML2_Scalar low, ML2_Scalar high) {
    for (ML2_Int s = 0; s < matrices.info.samples; s++) {
        for (ML2_Int m = 0; m < matrices.info.matrices; m++) {
            for (ML2_Int v = 0; v < matrices.info.vectors; v++) {
                for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                    *ML2_MatricesAt(matrices, s, m, v, c) = ML2_ScalarRand(low, high);
                }
            }
        }
    }
}

ML2_FN void ML2_MatricesXavierInit(ML2_LayerCacheMatrices matrices) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (matrices.info.samples + matrices.info.matrices + matrices.info.vectors + matrices.info.scalars));
    ML2_MatricesRand(matrices, -limit, limit);
}

ML2_FN void ML2_MatricesHeInit(ML2_LayerCacheMatrices matrices) {
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (matrices.info.matrices * matrices.info.vectors * matrices.info.scalars));
    for (ML2_Int s = 0; s < matrices.info.samples; s++) {
        for (ML2_Int m = 0; m < matrices.info.matrices; m++) {
            for (ML2_Int v = 0; v < matrices.info.vectors; v++) {
                for (ML2_Int c = 0; c < matrices.info.scalars; c++) {
                    *ML2_MatricesAt(matrices, s, m, v, c) = ML2_ScalarRandNormal(ML2_SCALAR_LITERAL(0.0), scale);
                }
            }
        }
    }
}

ML2_FN void ML2_MatricesInfoPrint(ML2_LayerCacheInfoMatrices info, ML2_Int indent) {
    printf(ML2_INDENT("Matrices(%dx%dx%dx%d)\n", indent), info.samples, info.matrices, info.vectors, info.scalars);
}

ML2_FN void ML2_MatricesPrint(ML2_LayerCacheMatrices matrices, ML2_Int indent) {
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
                                printf(ML2_SCALAR_FMT " ", *ML2_MatricesAt(matrices, s, m, v, c));
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

ML2_FN bool ML2_MatricesInfoSame(ML2_LayerCacheInfoMatrices a, ML2_LayerCacheInfoMatrices b) {
    return a.samples == b.samples && a.matrices == b.matrices && a.vectors == b.vectors && a.scalars == b.scalars;
}

ML2_FN ML2_Size ML2_MatricesSizeof(ML2_LayerCacheMatrices matrices) {
    return matrices.info.samples * matrices.info.matrices * matrices.info.vectors * matrices.info.scalars * sizeof(*matrices.data.matrices);
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_TODO("ML2_MatricesLossForwardSquareAverage");
}

ML2_FN void ML2_MatricesLossBackwardSquareAverage(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_TODO("ML2_MatricesLossBackwardSquareAverage");
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_TODO("ML2_MatricesLossForwardCrossEntropy");
}

ML2_FN void ML2_MatricesLossBackwardCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_TODO("ML2_MatricesLossBackwardCrossEntropy");
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_TODO("ML2_MatricesLossForwardSoftmaxCrossEntropy");
}

ML2_FN void ML2_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_TODO("ML2_MatricesLossBackwardSoftmaxCrossEntropy");
}

ML2_FN ML2_Scalar ML2_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected) {
    ML2_TODO("ML2_MatricesLossForwardBinaryCrossEntropy");
}

ML2_FN void ML2_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheMatrices predicted, ML2_LayerCacheMatrices expected, ML2_LayerCacheMatrices gradient) {
    ML2_TODO("ML2_MatricesLossBackwardBinaryCrossEntropy");
}

// ML2_LayerCacheMatrices ⬆️

ML2_FN ML2_LayerCache ML2_LayerCacheNew(ML2_LayerCacheInfo info) {
    switch (info.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCache){info.type, .as.scalars = ML2_ScalarsNew(ML2_LayerCacheInfoAsScalars(info))};
        case ML2_LayerCacheTypeVectors: return (ML2_LayerCache){info.type, .as.vectors = ML2_VectorsNew(ML2_LayerCacheInfoAsVectors(info))};
        case ML2_LayerCacheTypeMatrices: return (ML2_LayerCache){info.type, .as.matrices = ML2_MatricesNew(ML2_LayerCacheInfoAsMatrices(info))};
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
}

ML2_FN ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCacheInfo){cache.type, .as.scalars = ML2_LayerCacheAsScalars(cache).info};
        case ML2_LayerCacheTypeVectors: return (ML2_LayerCacheInfo){cache.type, .as.vectors = ML2_LayerCacheAsVectors(cache).info};
        case ML2_LayerCacheTypeMatrices: return (ML2_LayerCacheInfo){cache.type, .as.matrices = ML2_LayerCacheAsMatrices(cache).info};
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

ML2_FN void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, ML2_Int indent) {
    switch (layer.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsInfoPrint(ML2_LayerCacheInfoAsScalars(layer), indent); break;
        case ML2_LayerCacheTypeVectors: ML2_VectorsInfoPrint(ML2_LayerCacheInfoAsVectors(layer), indent); break;
        case ML2_LayerCacheTypeMatrices: ML2_MatricesInfoPrint(ML2_LayerCacheInfoAsMatrices(layer), indent); break;
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

ML2_FN bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsInfoSame(ML2_LayerCacheInfoAsScalars(a), ML2_LayerCacheInfoAsScalars(b));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsInfoSame(ML2_LayerCacheInfoAsVectors(a), ML2_LayerCacheInfoAsVectors(b));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesInfoSame(ML2_LayerCacheInfoAsMatrices(a), ML2_LayerCacheInfoAsMatrices(b));
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
    ML2_Arch arch = {layers, ML2_RELIABLE_CALLOC(layers, sizeof(*arch.infos))};
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

ML2_FN ML2_Model ML2_ModelNew(ML2_Arch arch) {
    ML2_Model model = {
        .layerCount = arch.layerCount,
        .layers = ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*model.layers)),
    };
    for (ML2_Int i = 0; i < arch.layerCount; i++) {
        model.layers[i] = ML2_LayerNew(arch.infos[i]);
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

ML2_FN void ML2_ModelClear(ML2_Model model) {
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerClear(model.layers[i]);
    }
}

ML2_FN void ML2_ModelSum(ML2_Model dest, ML2_Model src) {
    ML2_HARD_ASSERT(ML2_ModelInfoSame(dest, src));
    for (ML2_Int i = 0; i < dest.layerCount; i++) {
        ML2_LayerSum(dest.layers[i], src.layers[i]);
    }
}

ML2_FN void ML2_ModelInfoPrint(ML2_Model model, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerInfoPrint(ML2_LayerAsInfo(model.layers[i]), indent + ML2_Indentation);
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_ModelPrint(ML2_Model model, ML2_Int indent) {
    printf(ML2_INDENT("{\n", indent));
    for (ML2_Int i = 0; i < model.layerCount; i++) {
        ML2_LayerPrint(model.layers[i], indent + ML2_Indentation);
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN bool ML2_ModelInfoSame(ML2_Model a, ML2_Model b) {
    return a.layerCount == b.layerCount;
}

ML2_FN bool ML2_ModelForwardCompatible(ML2_Model model, ML2_ModelCache modelCache) {
    return model.layerCount == modelCache.layerCount;
}

ML2_FN void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache, ML2_LayerCache input) {
    ML2_HARD_ASSERT(ML2_ModelForwardCompatible(model, modelCache));
    ML2_LayerForward(model.layers[0], input, modelCache.layers[0]);
    for (ML2_Int i = 1; i < model.layerCount; i++) {
        ML2_LayerForward(model.layers[i], modelCache.layers[i - 1], modelCache.layers[i]);
    }
}

ML2_FN bool ML2_ModelBackwardCompatible(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient) {
    return ML2_ModelInfoSame(model, modelGradient) && model.layerCount == modelCache.layerCount && ML2_ModelCacheInfoSame(modelCache, modelCacheGradient);
}

ML2_FN void ML2_ModelBackward(ML2_Model model, ML2_Model modelGradient, ML2_ModelCache modelCache, ML2_ModelCache modelCacheGradient, ML2_Batch batch, ML2_LossBackward lossBackward) {
    ML2_HARD_ASSERT(ML2_ModelBackwardCompatible(model,modelGradient, modelCache, modelCacheGradient));
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

ML2_FN ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, ML2_BatchInfo info) {
    ML2_ModelCache modelCache = {arch.layerCount, ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*modelCache.layers))};

    ML2_LayerCacheInfo prevInfo = info.input;
    for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
        ML2_LayerCacheInfo nextInfo = ML2_LayerInfoForward(arch.infos[i], prevInfo);

        modelCache.layers[i] = ML2_LayerCacheNew(nextInfo);

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

ML2_FN void ML2_ModelCacheClear(ML2_ModelCache modelCache) {
    for (ML2_Int i = 0; i < modelCache.layerCount; i++) {
        ML2_LayerCacheClear(modelCache.layers[i]);
    }
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

ML2_FN bool ML2_ModelCacheInfoSame(ML2_ModelCache a, ML2_ModelCache b) {
    return a.layerCount == b.layerCount;
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

ML2_FN ML2_Batch ML2_BatchNew(ML2_BatchInfo info) {
    return (ML2_Batch){ML2_LayerCacheNew(info.input), ML2_LayerCacheNew(info.output)};
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

ML2_FN ML2_OptimizerMomentum ML2_MomentumNew(ML2_OptimizerParametersMomentum parameters, ML2_Arch arch) {
    ML2_OptimizerMomentum momentum = {
        parameters,
        {arch.layerCount, ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*momentum.state.averageLayers))}
    };
    for (ML2_Int i = 0; i < momentum.state.layerCount; i++) {
        momentum.state.averageLayers[i] = ML2_LayerNew(arch.infos[i]);
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
    return momentum.state.layerCount * sizeof(*momentum.state.averageLayers);
}

// ML2_OptimizerMomentum ⬆️

// ML2_OptimizerAdagrad ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adagrad(ML2_Scalar learningRate) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeAdagrad, .as.adagrad = {learningRate}};
}

ML2_FN ML2_OptimizerAdagrad ML2_AdagradNew(ML2_OptimizerParametersAdagrad parameters, ML2_Arch arch) {
    ML2_OptimizerAdagrad adagrad = {
        parameters,
        {arch.layerCount, ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*adagrad.state.squareSumLayers))}
    };
    for (ML2_Int i = 0; i < adagrad.state.layerCount; i++) {
        adagrad.state.squareSumLayers[i] = ML2_LayerNew(arch.infos[i]);
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
    return adagrad.state.layerCount * sizeof(*adagrad.state.squareSumLayers);
}

// ML2_OptimizerAdagrad ⬆️

// ML2_OptimizerRMSProp ⬇️

ML2_FN ML2_OptimizerParameters ML2_RMSProp(ML2_Scalar learningRate, ML2_Scalar decayRate) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeRMSProp, .as.RMSProp = {learningRate, decayRate}};
}

ML2_FN ML2_OptimizerRMSProp ML2_RMSPropNew(ML2_OptimizerParametersRMSProp parameters, ML2_Arch arch) {
    ML2_OptimizerRMSProp RMSProp = {
        parameters,
        {arch.layerCount, ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*RMSProp.state.squareAverageLayers))}
    };
    for (ML2_Int i = 0; i < RMSProp.state.layerCount; i++) {
        RMSProp.state.squareAverageLayers[i] = ML2_LayerNew(arch.infos[i]);
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
    return RMSProp.state.layerCount * sizeof(*RMSProp.state.squareAverageLayers);
}

// ML2_OptimizerRMSProp ⬆️

// ML2_OptimizerAdam ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adam(ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2) {
    return (ML2_OptimizerParameters){ML2_OptimizerTypeAdam, .as.adam = {learningRate, decayRate1, decayRate2}};
}

ML2_FN ML2_OptimizerAdam ML2_AdamNew(ML2_OptimizerParametersAdam parameters, ML2_Arch arch) {
    ML2_OptimizerAdam adam = {
        parameters,
        {arch.layerCount, ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*adam.state.averageLayers)), ML2_RELIABLE_CALLOC(arch.layerCount, sizeof(*adam.state.squareAverageLayers)), parameters.decayRate1, parameters.decayRate2}
    };
    for (ML2_Int i = 0; i < adam.state.layerCount; i++) {
        adam.state.averageLayers[i] = ML2_LayerNew(arch.infos[i]);
        adam.state.squareAverageLayers[i] = ML2_LayerNew(arch.infos[i]);
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
    return adam.state.layerCount * (sizeof(*adam.state.averageLayers) + sizeof(*adam.state.squareAverageLayers));
}

// ML2_OptimizerAdam ⬆️

ML2_FN ML2_Optimizer ML2_OptimizerNew(ML2_OptimizerParameters parameters, ML2_Arch arch) {
    switch (parameters.type) {
        case ML2_OptimizerTypeGradientDescent: return (ML2_Optimizer){parameters.type, .as.gradientDescent = {ML2_OptimizerParametersAsGradientDescent(parameters)}};
        case ML2_OptimizerTypeMomentum: return (ML2_Optimizer){parameters.type, .as.momentum = ML2_MomentumNew(ML2_OptimizerParametersAsMomentum(parameters), arch)};
        case ML2_OptimizerTypeAdagrad: return (ML2_Optimizer){parameters.type, .as.adagrad = ML2_AdagradNew(ML2_OptimizerParametersAsAdagrad(parameters), arch)};
        case ML2_OptimizerTypeRMSProp: return (ML2_Optimizer){parameters.type, .as.RMSProp = ML2_RMSPropNew(ML2_OptimizerParametersAsRMSProp(parameters), arch)};
        case ML2_OptimizerTypeAdam: return (ML2_Optimizer){parameters.type, .as.adam = ML2_AdamNew(ML2_OptimizerParametersAsAdam(parameters), arch)};
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

ML2_FN void ML2_OptimizerDestroy(ML2_Optimizer *optimizer) {
    switch (optimizer->type) {
        case ML2_OptimizerTypeGradientDescent: *optimizer = (ML2_Optimizer){}; break;
        case ML2_OptimizerTypeMomentum: ML2_MomentumDestroy(&optimizer->as.momentum); break;
        case ML2_OptimizerTypeAdagrad: ML2_AdagradDestroy(&optimizer->as.adagrad); break;
        case ML2_OptimizerTypeRMSProp: ML2_RMSPropDestroy(&optimizer->as.RMSProp); break;
        case ML2_OptimizerTypeAdam: ML2_AdamDestroy(&optimizer->as.adam); break;
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
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
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardSquareAverage works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // ML2_Scalar loss = {};
    // for (ML2_Int s = 0; s < samples; s++) {
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar diff = *ML2_ScalarsAt(predictedScalars, s, c) - *ML2_ScalarsAt(expectedScalars, s, c);
    //         loss += diff * diff;
    //     }
    // }
    // // TODO(8/4/2026 10:15:17pm):
    // //      all other losses divide only by samples, this causes a inconsistency in code,
    // //      and the numbers users see for the loss changes significantly, this will also
    // //      cause the learning rate to have to be smaller,
    // //      all losses should use the same method, i dont know which though
    // loss /= (samples * scalars);
    // return loss;
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardSquareAverage(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardSquareAverage(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardSquareAverage(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardSquareAverage works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // for (ML2_Int s = 0; s < samples; s++) {
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar diff = *ML2_ScalarsAt(predictedScalars, s, c) - *ML2_ScalarsAt(expectedScalars, s, c);
    //         *ML2_ScalarsAt(gradientScalars, s, c) += ML2_SCALAR_LITERAL(2.0) * diff / (samples * scalars);
    //     }
    // }
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossBackwardSquareAverage(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossBackwardSquareAverage(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossBackwardSquareAverage(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Scalar ML2_LossForwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardCrossEntropy works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // ML2_Scalar loss = {};
    // for (ML2_Int s = 0; s < samples; s++) {
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar predicted = ML2_FMAX(*ML2_ScalarsAt(predictedScalars, s, c), ML2_Epsilon);
    //         ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, s, c);
    //         loss += -expected * ML2_LN(predicted);
    //     }
    // }
    // loss /= samples;
    // return loss;
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardCrossEntropy works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // for (ML2_Int s = 0; s < samples; s++) {
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar predicted = ML2_FMAX(*ML2_ScalarsAt(predictedScalars, s, c), ML2_Epsilon);
    //         ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, s, c);
    //         *ML2_ScalarsAt(gradientScalars, s, c) += -(expected / predicted) / samples;
    //     }
    // }
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossBackwardCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossBackwardCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossBackwardCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Scalar ML2_LossForwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardSoftmaxCrossEntropy works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // ML2_Scalar loss = {};
    // for (ML2_Int s = 0; s < samples; s++) {
    //     ML2_Scalar max = ML2_NegInf;
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         max = ML2_FMAX(max, *ML2_ScalarsAt(predictedScalars, s, c));
    //     }
    //     ML2_Scalar sum = {};
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         sum += ML2_EXP(*ML2_ScalarsAt(predictedScalars, s, c) - max);
    //     }
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar predicted = ML2_FMAX(ML2_EXP(*ML2_ScalarsAt(predictedScalars, s, c) - max) / sum, ML2_Epsilon);
    //         ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, s, c);
    //         loss += -expected * ML2_LN(predicted);
    //     }
    // }
    // loss /= samples;
    // return loss;
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardSoftmaxCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardSoftmaxCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardSoftmaxCrossEntropy works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // for (ML2_Int s = 0; s < samples; s++) {
    //     ML2_Scalar max = ML2_NegInf;
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         max = ML2_FMAX(max, *ML2_ScalarsAt(predictedScalars, s, c));
    //     }
    //     ML2_Scalar sum = {};
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         sum += ML2_EXP(*ML2_ScalarsAt(predictedScalars, s, c) - max);
    //     }
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar predicted = ML2_EXP(*ML2_ScalarsAt(predictedScalars, s, c) - max) / sum;
    //         ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, s, c);
    //         *ML2_ScalarsAt(gradientScalars, s, c) += (predicted - expected) / samples;
    //     }
    // }
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossBackwardSoftmaxCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_Scalar ML2_LossForwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardBinaryCrossEntropy works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // ML2_Scalar loss = {};
    // for (ML2_Int s = 0; s < samples; s++) {
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar predicted = ML2_FMIN(ML2_FMAX(*ML2_ScalarsAt(predictedScalars, s, c), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
    //         ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, s, c);
    //         loss += -expected * ML2_LN(predicted) - (ML2_SCALAR_LITERAL(1.0) - expected) * ML2_LN(ML2_SCALAR_LITERAL(1.0) - predicted);
    //     }
    // }
    // loss /= (samples * scalars);
    // return loss;
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossForwardBinaryCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossForwardBinaryCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossForwardBinaryCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LossBackwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    // ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardBinaryCrossEntropy works on scalars only");
    // ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    // ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    // ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    // ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    // ML2_Int samples = expectedScalars.info.samples;
    // ML2_Int scalars = expectedScalars.info.scalars;

    // for (ML2_Int s = 0; s < samples; s++) {
    //     for (ML2_Int c = 0; c < scalars; c++) {
    //         ML2_Scalar predicted = ML2_FMIN(ML2_FMAX(*ML2_ScalarsAt(predictedScalars, s, c), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
    //         ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, s, c);
    //         *ML2_ScalarsAt(gradientScalars, s, c) += (predicted - expected) / (predicted * (ML2_SCALAR_LITERAL(1.0) - predicted)) / (samples * scalars);
    //     }
    // }
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(expected)));
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(ML2_LayerCacheAsInfo(predicted), ML2_LayerCacheAsInfo(gradient)));
    switch (predicted.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsLossBackwardBinaryCrossEntropy(ML2_LayerCacheAsScalars(predicted), ML2_LayerCacheAsScalars(expected), ML2_LayerCacheAsScalars(gradient));
        case ML2_LayerCacheTypeVectors: return ML2_VectorsLossBackwardBinaryCrossEntropy(ML2_LayerCacheAsVectors(predicted), ML2_LayerCacheAsVectors(expected), ML2_LayerCacheAsVectors(gradient));
        case ML2_LayerCacheTypeMatrices: return ML2_MatricesLossBackwardBinaryCrossEntropy(ML2_LayerCacheAsMatrices(predicted), ML2_LayerCacheAsMatrices(expected), ML2_LayerCacheAsMatrices(gradient));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_Loss ⬆️

#endif // ML2_IMPLEMENTATION

#ifdef ML2_STRIP_PREFIX

#ifndef _ML2_H_STRIP_PREFIX_GUARD
#define _ML2_H_STRIP_PREFIX_GUARD

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
#define ActivationInfoForwardCompatible ML2_ActivationInfoForwardCompatible
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

#endif // _ML2_H_STRIP_PREFIX_GUARD

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