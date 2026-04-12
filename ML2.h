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

#include <stdio.h>
#include <string.h>
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

#ifndef NDEBUG
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
#endif // NDEBUG

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

#ifndef ML2_CALLOC
    #include <stdlib.h>
    #define ML2_CALLOC calloc
#endif // ML2_CALLOC

#ifndef ML2_FREE
    #include <stdlib.h>
    #define ML2_FREE free
#endif // ML2_FREE

#ifndef ML2_RELIABLE_CALLOC
    static inline void *ML2_ReliableCalloc(size_t count, size_t size) {
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

#ifndef ML2_INDENTATION
    #define ML2_INDENTATION 4
#endif // ML2_INDENTATION
ML2_VAR constexpr int ML2_Indentation = ML2_INDENTATION;

#define ML2_INDENT(fstr, i) "%*s" fstr, (i), ""

// ML2_Scalar ⬇️

// NOTE: Customizable, but make sure to define all of these
#ifndef ML2_SCALAR_TYPE
    #define ML2_SCALAR_TYPE float

    #define ML2_SCALAR_LITERAL(x) x##f
    #define ML2_EXP expf
    #define ML2_LOG logf
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

// TODO: figure out how to align the floats nicely
#define ML2_SCALAR_FMT "%.3f"

// ML2_Scalar ⬆️

// ML2_Layer ⬇️

typedef enum : int {
    ML2_LayerTypeWeights = 1,
    ML2_LayerTypeBiases,
    ML2_LayerTypeActivation,
    ML2_LayerTypeLinear,
    ML2_LayerTypeFilters,
    ML2_LayerTypeConv,
    ML2_LayerTypeFlatten,
} ML2_LayerType;

// ML2_LayerWeights ⬇️

typedef struct {
    int outputs;
    int inputs;
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
    int inputs;
} ML2_LayerInfoBiases;

typedef struct {
    ML2_LayerInfoBiases info;
    struct {
        ML2_Scalar *biases;
    } data;
} ML2_LayerBiases;

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

typedef enum : int {
    ML2_ActivationTypeReLU = 1,
    ML2_ActivationTypeSigmoid,
    ML2_ActivationTypeSin,
    ML2_ActivationTypeCos,
    ML2_ActivationTypeTanh,
    ML2_ActivationTypeSoftmax,
} ML2_ActivationType;

typedef struct {
    ML2_ActivationType type;
} ML2_LayerInfoActivation;

typedef struct {
    ML2_LayerInfoActivation info;
} ML2_LayerActivation;

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

typedef struct {
    int outputs;
    int inputs;
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
    int outputs;
    int inputs;
    int height;
    int width;
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
    int outputs;
    int inputs;
    int height;
    int width;
} ML2_LayerInfoConv;

typedef struct {
    ML2_LayerInfoConv info;
    struct {
        ML2_Scalar *weights;
        ML2_Scalar *biases;
    } data;
} ML2_LayerConv;

// ML2_LayerConv ⬆️

typedef struct {
    ML2_LayerType type;
    union {
        ML2_LayerInfoWeights weights;
        ML2_LayerInfoBiases biases;
        ML2_LayerInfoActivation activation;
        ML2_LayerInfoLinear linear;
        ML2_LayerInfoFilters filters;
        ML2_LayerInfoConv conv;
    } as;
} ML2_LayerInfo;

typedef struct {
    ML2_LayerType type;
    union {
        ML2_LayerWeights weights;
        ML2_LayerBiases biases;
        ML2_LayerActivation activation;
        ML2_LayerLinear linear;
        ML2_LayerFilters filters;
        ML2_LayerConv conv;
    } as;
} ML2_Layer;

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

typedef enum : int {
    ML2_LayerCacheTypeScalars = 1,
    ML2_LayerCacheTypeImages,
} ML2_LayerCacheType;

// ML2_LayerCacheScalars ⬇️

typedef struct {
    int samples;
    int scalars;
} ML2_LayerCacheInfoScalars;

typedef struct {
    ML2_LayerCacheInfoScalars info;
    struct {
        ML2_Scalar *scalars;
    } data;
} ML2_LayerCacheScalars;

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheImages ⬇️

typedef struct {
    int samples;
    int channels;
    int height;
    int width;
} ML2_LayerCacheInfoImages;

typedef struct {
    ML2_LayerCacheInfoImages info;
    struct {
        ML2_Scalar *images;
    } data;
} ML2_LayerCacheImages;

// ML2_LayerCacheImages ⬆️

typedef struct {
    ML2_LayerCacheType type;
    union {
        ML2_LayerCacheInfoScalars scalars;
        ML2_LayerCacheInfoImages images;
    } as;
} ML2_LayerCacheInfo;

typedef struct {
    ML2_LayerCacheType type;
    union {
        ML2_LayerCacheScalars scalars;
        ML2_LayerCacheImages images;
    } as;
} ML2_LayerCache;

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

typedef struct {
    int layerCount;
    ML2_LayerInfo *infos;
} ML2_Arch;

// ML2_Arch ⬆️

// ML2_Model ⬇️

typedef struct {
    int layerCount;
    ML2_Layer *layers;
} ML2_Model;

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

typedef struct {
    int layerCount;
    ML2_LayerCache *caches;
    ML2_LayerCache *cachesGradients;
    ML2_Layer *gradients;
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

typedef enum : int {
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
        int layerCount;
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
        int layerCount;
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
        int layerCount;
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
        int layerCount;
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

// ML2_Scalar ⬇️

ML2_FN ML2_Scalar ML2_ScalarRand(ML2_Scalar low, ML2_Scalar high);
ML2_FN ML2_Scalar ML2_ScalarRandNormal();
ML2_FN ML2_Scalar ML2_ScalarReLUForward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarReLUBackward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarSigmoidForward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarSigmoidBackward(ML2_Scalar output);
ML2_FN ML2_Scalar ML2_ScalarSinForward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarSinBackward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarCosForward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarCosBackward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarTanhForward(ML2_Scalar input);
ML2_FN ML2_Scalar ML2_ScalarTanhBackward(ML2_Scalar output);

// ML2_Scalar ⬆️

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_FN ML2_LayerInfo ML2_Weights(int outputs, int inputs);
ML2_FN ML2_LayerWeights ML2_WeightsNew(ML2_LayerInfoWeights info);
ML2_FN void ML2_WeightsDestroy(ML2_LayerWeights *weights);
ML2_FN ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, int i, int j);
ML2_FN ML2_LayerInfoWeights ML2_LayerInfoAsWeights(ML2_LayerInfo info);
ML2_FN ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer);
ML2_FN void ML2_WeightsClear(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_WeightsXavierInit(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsHeInit(ML2_LayerWeights weights);
ML2_FN void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, int indent);
ML2_FN void ML2_WeightsPrint(ML2_LayerWeights weights, int indent);
ML2_FN bool ML2_WeightsInfoForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_WeightsInfoForward(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_FN bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b);
ML2_FN bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_WeightsBackwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_WeightsBackward(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate);
ML2_FN void ML2_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average);
ML2_FN void ML2_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum);
ML2_FN void ML2_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage);
ML2_FN void ML2_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_FN ML2_LayerInfo ML2_Biases(int inputs);
ML2_FN ML2_LayerBiases ML2_BiasesNew(ML2_LayerInfoBiases info);
ML2_FN void ML2_BiasesDestroy(ML2_LayerBiases *biases);
ML2_FN ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, int i);
ML2_FN ML2_LayerInfoBiases ML2_LayerInfoAsBiases(ML2_LayerInfo info);
ML2_FN ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer);
ML2_FN void ML2_BiasesClear(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_BiasesXavierInit(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesHeInit(ML2_LayerBiases biases);
ML2_FN void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, int indent);
ML2_FN void ML2_BiasesPrint(ML2_LayerBiases biases, int indent);
ML2_FN bool ML2_BiasesInfoForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_BiasesInfoForward(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_FN bool ML2_BiasesInfoSame(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b);
ML2_FN bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_BiasesBackwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient, bool cacheBackward);
ML2_FN void ML2_BiasesBackward(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate);
ML2_FN void ML2_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average);
ML2_FN void ML2_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum);
ML2_FN void ML2_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage);
ML2_FN void ML2_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

// TODO(6/4/2026 2:32:30am): maybe delete this function its more pleasent to use the ones below it
ML2_FN ML2_LayerInfo ML2_Activation(ML2_ActivationType type);
ML2_FN ML2_LayerInfo ML2_ReLU();
ML2_FN ML2_LayerInfo ML2_Sigmoid();
ML2_FN ML2_LayerInfo ML2_Sin();
ML2_FN ML2_LayerInfo ML2_Cos();
ML2_FN ML2_LayerInfo ML2_Tanh();
ML2_FN ML2_LayerInfo ML2_Softmax();
ML2_FN ML2_LayerInfoActivation ML2_LayerInfoAsActivation(ML2_LayerInfo info);
ML2_FN ML2_LayerActivation ML2_LayerAsActivation(ML2_Layer layer);
ML2_FN const char *ML2_ActivationNameOf(ML2_ActivationType type);
ML2_FN void ML2_ActivationInfoPrint(ML2_LayerInfoActivation info, int indent);
ML2_FN void ML2_ActivationPrint(ML2_LayerActivation activation, int indent);
ML2_FN bool ML2_ActivationForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN bool ML2_ActivationBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ActivationReLUForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationReLUBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ActivationSigmoidForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationSigmoidBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ActivationSinForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationSinBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ActivationCosForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationCosBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ActivationTanhForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationTanhBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ActivationSoftmaxForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationSoftmaxBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ActivationForward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_ActivationBackward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

ML2_FN ML2_LayerInfo ML2_Linear(int outputs, int inputs);
ML2_FN ML2_LayerLinear ML2_LinearNew(ML2_LayerInfoLinear info);
ML2_FN void ML2_LinearDestroy(ML2_LayerLinear *linear);
ML2_FN ML2_LayerInfoLinear ML2_LayerInfoAsLinear(ML2_LayerInfo info);
ML2_FN ML2_LayerLinear ML2_LayerAsLinear(ML2_Layer layer);
ML2_FN ML2_LayerInfoWeights ML2_LinearInfoWeights(ML2_LayerInfoLinear linear);
ML2_FN ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear);
ML2_FN ML2_LayerInfoBiases ML2_LinearInfoBiases(ML2_LayerInfoLinear linear);
ML2_FN ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear);
ML2_FN void ML2_LinearClear(ML2_LayerLinear linear);
ML2_FN void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LinearXavierInit(ML2_LayerLinear linear);
ML2_FN void ML2_LinearHeInit(ML2_LayerLinear linear);
ML2_FN void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, int indent);
ML2_FN void ML2_LinearPrint(ML2_LayerLinear linear, int indent);
ML2_FN bool ML2_LinearInfoForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_LinearInfoForward(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_FN bool ML2_LinearInfoSame(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b);
ML2_FN bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_LinearBackwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_LinearBackward(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_LinearGradientDescentOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate);
ML2_FN void ML2_LinearMomentumOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear average);
ML2_FN void ML2_LinearAdagradOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_LayerLinear squareSum);
ML2_FN void ML2_LinearRMSPropOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerLinear squareAverage);
ML2_FN void ML2_LinearAdamOptimize(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerLinear average, ML2_LayerLinear squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_FN ML2_LayerInfo ML2_Filters(int outputs, int inputs, int height, int width);
ML2_FN ML2_LayerFilters ML2_FiltersNew(ML2_LayerInfoFilters info);
ML2_FN void ML2_FiltersDestroy(ML2_LayerFilters *filters);
ML2_FN ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, int i, int j, int k, int l);
ML2_FN ML2_LayerInfoFilters ML2_LayerInfoAsFilters(ML2_LayerInfo info);
ML2_FN ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer);
ML2_FN void ML2_FiltersClear(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_FiltersXavierInit(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersHeInit(ML2_LayerFilters filters);
ML2_FN void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, int indent);
ML2_FN void ML2_FiltersPrint(ML2_LayerFilters filters, int indent);
ML2_FN bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_FN bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b);
ML2_FN bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FiltersBackwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FiltersBackward(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_FiltersGradientDescentOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate);
ML2_FN void ML2_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average);
ML2_FN void ML2_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum);
ML2_FN void ML2_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage);
ML2_FN void ML2_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_FN ML2_LayerInfo ML2_Conv(int outputs, int inputs, int height, int width);
ML2_FN ML2_LayerConv ML2_ConvNew(ML2_LayerInfoConv info);
ML2_FN void ML2_ConvDestroy(ML2_LayerConv *conv);
ML2_FN ML2_LayerInfoConv ML2_LayerInfoAsConv(ML2_LayerInfo info);
ML2_FN ML2_LayerConv ML2_LayerAsConv(ML2_Layer layer);
ML2_FN ML2_LayerInfoFilters ML2_ConvInfoFilters(ML2_LayerInfoConv conv);
ML2_FN ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv);
ML2_FN ML2_LayerInfoBiases ML2_ConvInfoBiases(ML2_LayerInfoConv conv);
ML2_FN ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv);
ML2_FN void ML2_ConvClear(ML2_LayerConv conv);
ML2_FN void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ConvXavierInit(ML2_LayerConv conv);
ML2_FN void ML2_ConvHeInit(ML2_LayerConv conv);
ML2_FN void ML2_ConvInfoPrint(ML2_LayerInfoConv info, int indent);
ML2_FN void ML2_ConvPrint(ML2_LayerConv conv, int indent);
ML2_FN bool ML2_ConvInfoForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_ConvInfoForward(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_FN bool ML2_ConvInfoSame(ML2_LayerInfoConv a, ML2_LayerInfoConv b);
ML2_FN bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_ConvBackwardCompatible(ML2_LayerInfoConv conv, ML2_LayerInfoConv convGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_ConvBackward(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_ConvGradientDescentOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate);
ML2_FN void ML2_ConvMomentumOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv average);
ML2_FN void ML2_ConvAdagradOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_LayerConv squareSum);
ML2_FN void ML2_ConvRMSPropOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerConv squareAverage);
ML2_FN void ML2_ConvAdamOptimize(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerConv average, ML2_LayerConv squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

// ML2_LayerConv ⬆️

// ML2_LayerFlatten ⬇️

ML2_FN ML2_LayerInfo ML2_Flatten();
ML2_FN void ML2_FlattenInfoPrint(int indent);
ML2_FN void ML2_FlattenPrint(int indent);
ML2_FN bool ML2_FlattenInfoForwardCompatible(ML2_LayerCacheInfo input);
ML2_FN ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerCacheInfo input);
ML2_FN bool ML2_FlattenForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN void ML2_FlattenForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_FN bool ML2_FlattenBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient);
ML2_FN void ML2_FlattenBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward);

// ML2_LayerFlatten ⬆️

ML2_FN ML2_Layer ML2_LayerNew(ML2_LayerInfo info);
ML2_FN void ML2_LayerDestroy(ML2_Layer *layer);
ML2_FN ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer);
ML2_FN void ML2_LayerClear(ML2_Layer layer);
ML2_FN void ML2_LayerRand(ML2_Layer layer, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_LayerXavierInit(ML2_Layer layer);
ML2_FN void ML2_LayerHeInit(ML2_Layer layer);
ML2_FN void ML2_LayerInfoPrint(ML2_LayerInfo layer, int indent);
ML2_FN void ML2_LayerPrint(ML2_Layer layer, int indent);
ML2_FN ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input);
ML2_FN void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output);
ML2_FN void ML2_LayerBackward(ML2_Layer layer, ML2_Layer layerGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward);
ML2_FN void ML2_LayerGradientDescentOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate);
ML2_FN void ML2_LayerMomentumOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer average);
ML2_FN void ML2_LayerAdagradOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Layer squareSum);
ML2_FN void ML2_LayerRMSPropOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer squareAverage);
ML2_FN void ML2_LayerAdamOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_Layer average, ML2_Layer squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2);

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

// ML2_LayerCacheScalars ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Scalars(int samples, int scalars);
ML2_FN ML2_LayerCacheScalars ML2_ScalarsNew(ML2_LayerCacheInfoScalars info);
ML2_FN void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars);
ML2_FN ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, int sample, int scalar);
ML2_FN ML2_LayerCacheInfoScalars ML2_LayerCacheInfoAsScalars(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache);
ML2_FN void ML2_ScalarsClear(ML2_LayerCacheScalars scalars);
ML2_FN void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_FN void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, int indent);
ML2_FN void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, int indent);
ML2_FN bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b);

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheImages ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Images(int samples, int channels, int height, int width);
ML2_FN ML2_LayerCacheImages ML2_ImagesNew(ML2_LayerCacheInfoImages info);
ML2_FN void ML2_ImagesDestroy(ML2_LayerCacheImages *images);
ML2_FN ML2_Scalar *ML2_ImagesAt(ML2_LayerCacheImages images, int sample, int channel, int y, int x);
ML2_FN ML2_LayerCacheInfoImages ML2_LayerCacheInfoAsImages(ML2_LayerCacheInfo info);
ML2_FN ML2_LayerCacheImages ML2_LayerCacheAsImages(ML2_LayerCache cache);
ML2_FN void ML2_ImagesClear(ML2_LayerCacheImages images);
ML2_FN void ML2_ImagesCopy(ML2_LayerCacheImages dest, ML2_LayerCacheImages src);
ML2_FN void ML2_ImagesInfoPrint(ML2_LayerCacheInfoImages info, int indent);
ML2_FN void ML2_ImagesPrint(ML2_LayerCacheImages images, int indent);
ML2_FN bool ML2_ImagesInfoSame(ML2_LayerCacheInfoImages a, ML2_LayerCacheInfoImages b);

// ML2_LayerCacheImages ⬆️

ML2_FN ML2_LayerCache ML2_LayerCacheNew(ML2_LayerCacheInfo info);
ML2_FN void ML2_LayerCacheDestroy(ML2_LayerCache *cache);
ML2_FN ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheClear(ML2_LayerCache cache);
ML2_FN void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src);
ML2_FN void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, int indent);
ML2_FN void ML2_LayerCachePrint(ML2_LayerCache cache, int indent);
ML2_FN bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b);

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

#define ML2_ArchMake(...) ((ML2_Arch){sizeof((ML2_LayerInfo[]){__VA_ARGS__}) / sizeof(ML2_LayerInfo), (ML2_LayerInfo[]){__VA_ARGS__}})
ML2_FN ML2_Arch ML2_ArchNew(int layers, ML2_LayerInfo infos[layers]);
ML2_FN void ML2_ArchDestroy(ML2_Arch *arch);

// ML2_Arch ⬆️

// ML2_Model ⬇️

ML2_FN ML2_Model ML2_ModelNew(ML2_Arch arch);
ML2_FN void ML2_ModelDestroy(ML2_Model *model);
ML2_FN void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high);
ML2_FN void ML2_ModelXavierInit(ML2_Model model);
ML2_FN void ML2_ModelHeInit(ML2_Model model);
ML2_FN void ML2_ModelInfoPrint(ML2_Model model, int indent);
ML2_FN void ML2_ModelPrint(ML2_Model model, int indent);
ML2_FN void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache);
ML2_FN void ML2_ModelBackward(ML2_Model model, ML2_ModelCache modelCache);
ML2_FN bool ML2_ModelGradientDescentOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache);
ML2_FN void ML2_ModelGradientDescentOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerGradientDescent gradientDescent);
ML2_FN bool ML2_ModelMomentumOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerMomentum momentum);
ML2_FN void ML2_ModelMomentumOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerMomentum momentum);
ML2_FN bool ML2_ModelAdagradOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdagrad adagrad);
ML2_FN void ML2_ModelAdagradOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdagrad adagrad);
ML2_FN bool ML2_ModelRMSPropOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerRMSProp RMSProp);
ML2_FN void ML2_ModelRMSPropOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerRMSProp RMSProp);
ML2_FN bool ML2_ModelAdamOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdam adam);
ML2_FN void ML2_ModelAdamOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdam *adam);
ML2_FN void ML2_ModelOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_Optimizer *optimizer);

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_FN ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, ML2_BatchInfo info);
ML2_FN void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache);
ML2_FN void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, int indent);
ML2_FN void ML2_ModelCachePrint(ML2_ModelCache modelCache, int indent);
ML2_FN ML2_LayerCache ML2_ModelCacheInput(ML2_ModelCache modelCache);
ML2_FN ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache);
ML2_FN ML2_LayerCache ML2_ModelCacheOutputGradient(ML2_ModelCache modelCache);
ML2_FN void ML2_ModelCacheCopyBatchInput(ML2_ModelCache modelCache, ML2_Batch batch);
ML2_FN ML2_Scalar ML2_ModelCacheLossForward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossForward lossForward);
ML2_FN void ML2_ModelCacheLossBackward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossBackward lossBackward);

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_FN ML2_BatchInfo ML2_BatchInfoMake(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_FN ML2_Batch ML2_BatchNew(ML2_BatchInfo info);
ML2_FN void ML2_BatchDestroy(ML2_Batch *batch);
ML2_FN void ML2_BatchInfoPrint(ML2_Batch batch, int indent);
ML2_FN void ML2_BatchPrint(ML2_Batch batch, int indent);

// ML2_Batch ⬆️

// ML2_Optimizer ⬇️

// ML2_OptimizerGradientDescent ⬇️

ML2_FN ML2_OptimizerParameters ML2_GradientDescent(ML2_Scalar learningRate);
ML2_FN ML2_OptimizerParametersGradientDescent ML2_OptimizerParametersAsGradientDescent(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerGradientDescent ML2_OptimizerAsGradientDescent(ML2_Optimizer optimizer);

// ML2_OptimizerGradientDescent ⬆️

// ML2_OptimizerMomentum ⬇️

ML2_FN ML2_OptimizerParameters ML2_Momentum(ML2_Scalar learningRate, ML2_Scalar decayRate);
ML2_FN ML2_OptimizerMomentum ML2_MomentumNew(ML2_OptimizerParametersMomentum parameters, ML2_Arch arch);
ML2_FN void ML2_MomentumDestroy(ML2_OptimizerMomentum *momentum);
ML2_FN void ML2_MomentumReset(ML2_OptimizerMomentum momentum);
ML2_FN ML2_OptimizerParametersMomentum ML2_OptimizerParametersAsMomentum(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerMomentum ML2_OptimizerAsMomentum(ML2_Optimizer optimizer);

// ML2_OptimizerMomentum ⬆️

// ML2_OptimizerAdagrad ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adagrad(ML2_Scalar learningRate);
ML2_FN ML2_OptimizerAdagrad ML2_AdagradNew(ML2_OptimizerParametersAdagrad parameters, ML2_Arch arch);
ML2_FN void ML2_AdagradDestroy(ML2_OptimizerAdagrad *adagrad);
ML2_FN void ML2_AdagradReset(ML2_OptimizerAdagrad adagrad);
ML2_FN ML2_OptimizerParametersAdagrad ML2_OptimizerParametersAsAdagrad(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerAdagrad ML2_OptimizerAsAdagrad(ML2_Optimizer optimizer);

// ML2_OptimizerAdagrad ⬆️

// ML2_OptimizerRMSProp ⬇️

ML2_FN ML2_OptimizerParameters ML2_RMSProp(ML2_Scalar learningRate, ML2_Scalar decayRate);
ML2_FN ML2_OptimizerRMSProp ML2_RMSPropNew(ML2_OptimizerParametersRMSProp parameters, ML2_Arch arch);
ML2_FN void ML2_RMSPropDestroy(ML2_OptimizerRMSProp *RMSProp);
ML2_FN void ML2_RMSPropReset(ML2_OptimizerRMSProp RMSProp);
ML2_FN ML2_OptimizerParametersRMSProp ML2_OptimizerParametersAsRMSProp(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerRMSProp ML2_OptimizerAsRMSProp(ML2_Optimizer optimizer);

// ML2_OptimizerRMSProp ⬆️

// ML2_OptimizerAdam ⬇️

ML2_FN ML2_OptimizerParameters ML2_Adam(ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2);
ML2_FN ML2_OptimizerAdam ML2_AdamNew(ML2_OptimizerParametersAdam parameters, ML2_Arch arch);
ML2_FN void ML2_AdamDestroy(ML2_OptimizerAdam *adam);
ML2_FN void ML2_AdamReset(ML2_OptimizerAdam *adam);
ML2_FN ML2_OptimizerParametersAdam ML2_OptimizerParametersAsAdam(ML2_OptimizerParameters parameters);
ML2_FN ML2_OptimizerAdam ML2_OptimizerAsAdam(ML2_Optimizer optimizer);

// ML2_OptimizerAdam ⬆️

ML2_FN ML2_Optimizer ML2_OptimizerNew(ML2_OptimizerParameters parameters, ML2_Arch arch);
ML2_FN void ML2_OptimizerDestroy(ML2_Optimizer *optimizer);
ML2_FN void ML2_OptimizerReset(ML2_Optimizer *optimizer);

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

// ML2_Scalar ⬇️

ML2_FN ML2_Scalar ML2_ScalarRand(ML2_Scalar low, ML2_Scalar high) {
    return ((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX) * (high - low) + low;
}

// TODO: i just yoinked this from claude, i dont know if this is what i want it to look like
ML2_FN ML2_Scalar ML2_ScalarRandNormal() {
    ML2_Scalar u1 = ML2_FMAX((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX, ML2_Epsilon);
    ML2_Scalar u2 = (ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX;
    return ML2_SQRT(-ML2_SCALAR_LITERAL(2.0) * ML2_LOG(u1)) * ML2_COS(ML2_SCALAR_LITERAL(2.0) * ML2_Pi * u2);
}

ML2_FN ML2_Scalar ML2_ScalarReLUForward(ML2_Scalar input) {
    return input > ML2_SCALAR_LITERAL(0.0) ? input : ML2_SCALAR_LITERAL(0.0);
}

ML2_FN ML2_Scalar ML2_ScalarReLUBackward(ML2_Scalar input) {
    return input > ML2_SCALAR_LITERAL(0.0) ? ML2_SCALAR_LITERAL(1.0) : ML2_SCALAR_LITERAL(0.0);
}

ML2_FN ML2_Scalar ML2_ScalarSigmoidForward(ML2_Scalar input) {
    return ML2_SCALAR_LITERAL(1.0) / (ML2_SCALAR_LITERAL(1.0) + ML2_EXP(-input));
}

ML2_FN ML2_Scalar ML2_ScalarSigmoidBackward(ML2_Scalar output) {
    return output * (ML2_SCALAR_LITERAL(1.0) - output);
}

ML2_FN ML2_Scalar ML2_ScalarSinForward(ML2_Scalar input) {
    return ML2_SIN(input);
}

ML2_FN ML2_Scalar ML2_ScalarSinBackward(ML2_Scalar input) {
    return ML2_COS(input);
}

ML2_FN ML2_Scalar ML2_ScalarCosForward(ML2_Scalar input) {
    return ML2_COS(input);
}

ML2_FN ML2_Scalar ML2_ScalarCosBackward(ML2_Scalar input) {
    return -ML2_SIN(input);
}

ML2_FN ML2_Scalar ML2_ScalarTanhForward(ML2_Scalar input) {
    return ML2_TANH(input);
}

ML2_FN ML2_Scalar ML2_ScalarTanhBackward(ML2_Scalar output) {
    return ML2_SCALAR_LITERAL(1.0) - output * output;
}

// ML2_Scalar ⬆️

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_FN ML2_LayerInfo ML2_Weights(int outputs, int inputs) {
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

ML2_FN ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, int i, int j) {
    ML2_SOFT_ASSERT(0 <= i && i < weights.info.outputs && 0 <= j && j < weights.info.inputs && "OUT OF BOUNDS INDICES");
    return &weights.data.weights[i * weights.info.inputs + j];
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
    for (int i = 0; i < weights.info.outputs; i++) {
        for (int j = 0; j < weights.info.inputs; j++) {
            *ML2_WeightsAt(weights, i, j) = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_FN void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < weights.info.outputs; i++) {
        for (int j = 0; j < weights.info.inputs; j++) {
            *ML2_WeightsAt(weights, i, j) = ML2_ScalarRand(low, high);
        }
    }
}

ML2_FN void ML2_WeightsXavierInit(ML2_LayerWeights weights) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (weights.info.inputs + weights.info.outputs));
    ML2_WeightsRand(weights, -limit, limit);
}

ML2_FN void ML2_WeightsHeInit(ML2_LayerWeights weights) {
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / weights.info.inputs);
    for (int i = 0; i < weights.info.outputs; i++) {
        for (int j = 0; j < weights.info.inputs; j++) {
            *ML2_WeightsAt(weights, i, j) = scale * ML2_ScalarRandNormal();
        }
    }
}

ML2_FN void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, int indent) {
    printf(ML2_INDENT("Weights(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_FN void ML2_WeightsPrint(ML2_LayerWeights weights, int indent) {
    ML2_WeightsInfoPrint(weights.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < weights.info.outputs; i++) {
            printf(ML2_INDENT("", indent));
            for (int j = 0; j < weights.info.inputs; j++) {
                printf(ML2_SCALAR_FMT " ", *ML2_WeightsAt(weights, i, j));
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
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
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
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b) {
    return a.inputs == b.inputs && a.outputs == b.outputs;
}

ML2_FN bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoScalars inputScalars = ML2_LayerCacheInfoAsScalars(input);
            ML2_LayerCacheInfoScalars outputScalars = ML2_LayerCacheInfoAsScalars(output);
            return weights.inputs == inputScalars.scalars && weights.outputs == outputScalars.scalars && inputScalars.samples == outputScalars.samples;
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
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
            int samples = inputScalars.info.samples;
            int outputs = weights.info.outputs;
            int inputs = weights.info.inputs;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < outputs; j++) {
                    ML2_Scalar sum = {};
                    for (int k = 0; k < inputs; k++) {
                        sum += *ML2_ScalarsAt(inputScalars, i, k) * *ML2_WeightsAt(weights, j, k);
                    }
                    *ML2_ScalarsAt(outputScalars, i, j) = sum;
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_WeightsBackwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerInfoWeights weightsGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsInfoSame(weights, weightsGradient) && ML2_LayerCacheInfoSame(input, inputGradient) && ML2_WeightsForwardCompatible(weights, input, outputGradient);
}

ML2_FN void ML2_WeightsBackward(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_WeightsBackwardCompatible(weights.info, weightsGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputScalars.info.samples;
            int outputs = weights.info.outputs;
            int inputs = weights.info.inputs;

            ML2_WeightsClear(weightsGradient);
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < outputs; j++) {
                    for (int k = 0; k < inputs; k++) {
                        *ML2_WeightsAt(weightsGradient, j, k) += *ML2_ScalarsAt(inputScalars, i, k) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                }
            }

            if (cacheBackward) {
                ML2_ScalarsClear(inputGradientScalars);
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < outputs; j++) {
                        for (int k = 0; k < inputs; k++) {
                            *ML2_ScalarsAt(inputGradientScalars, i, k) += *ML2_WeightsAt(weights, j, k) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                        }
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_WeightsGradientDescentOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    int outputs = weights.info.outputs;
    int inputs = weights.info.inputs;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            *ML2_WeightsAt(weights, i, j) -= *ML2_WeightsAt(weightsGradient, i, j) * learningRate;
        }
    }
}

ML2_FN void ML2_WeightsMomentumOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights average) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, average.info));
    int outputs = weights.info.outputs;
    int inputs = weights.info.inputs;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            *ML2_WeightsAt(average, i, j) = decayRate * *ML2_WeightsAt(average, i, j) - learningRate * *ML2_WeightsAt(weightsGradient, i, j);
            *ML2_WeightsAt(weights, i, j) += *ML2_WeightsAt(average, i, j);
        }
    }
}

ML2_FN void ML2_WeightsAdagradOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_LayerWeights squareSum) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, squareSum.info));
    int outputs = weights.info.outputs;
    int inputs = weights.info.inputs;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            *ML2_WeightsAt(squareSum, i, j) += *ML2_WeightsAt(weightsGradient, i, j) * *ML2_WeightsAt(weightsGradient, i, j);
            *ML2_WeightsAt(weights, i, j) -= *ML2_WeightsAt(weightsGradient, i, j) * learningRate / ML2_SQRT(*ML2_WeightsAt(squareSum, i, j) + ML2_Epsilon);
        }
    }
}

ML2_FN void ML2_WeightsRMSPropOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerWeights squareAverage) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, squareAverage.info));
    int outputs = weights.info.outputs;
    int inputs = weights.info.inputs;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            *ML2_WeightsAt(squareAverage, i, j) = decayRate * *ML2_WeightsAt(squareAverage, i, j) + (ML2_SCALAR_LITERAL(1.0) - decayRate) * *ML2_WeightsAt(weightsGradient, i, j) * *ML2_WeightsAt(weightsGradient, i, j);
            *ML2_WeightsAt(weights, i, j) -= *ML2_WeightsAt(weightsGradient, i, j) * learningRate / ML2_SQRT(*ML2_WeightsAt(squareAverage, i, j) + ML2_Epsilon);
        }
    }
}

ML2_FN void ML2_WeightsAdamOptimize(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerWeights average, ML2_LayerWeights squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, weightsGradient.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, average.info));
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(weights.info, squareAverage.info));
    int outputs = weights.info.outputs;
    int inputs = weights.info.inputs;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            *ML2_WeightsAt(average, i, j) = decayRate1 * *ML2_WeightsAt(average, i, j) + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * *ML2_WeightsAt(weightsGradient, i, j);
            *ML2_WeightsAt(squareAverage, i, j) = decayRate2 * *ML2_WeightsAt(squareAverage, i, j) + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * *ML2_WeightsAt(weightsGradient, i, j) * *ML2_WeightsAt(weightsGradient, i, j);
            ML2_Scalar averageWeighted = *ML2_WeightsAt(average, i, j) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
            ML2_Scalar squareAverageWeighed = *ML2_WeightsAt(squareAverage, i, j) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
            *ML2_WeightsAt(weights, i, j) -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
        }
    }
}

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_FN ML2_LayerInfo ML2_Biases(int inputs) {
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

ML2_FN ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, int i) {
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
    for (int i = 0; i < biases.info.inputs; i++) {
        *ML2_BiasesAt(biases, i) = ML2_SCALAR_LITERAL(0.0);
    }
}

ML2_FN void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < biases.info.inputs; i++) {
        biases.data.biases[i] = ML2_ScalarRand(low, high);
    }
}

ML2_FN void ML2_BiasesXavierInit(ML2_LayerBiases biases) {
    ML2_BiasesClear(biases);
}

ML2_FN void ML2_BiasesHeInit(ML2_LayerBiases biases) {
    ML2_BiasesClear(biases);
}

ML2_FN void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, int indent) {
    printf(ML2_INDENT("Biases(%d)\n", indent), info.inputs);
}

ML2_FN void ML2_BiasesPrint(ML2_LayerBiases biases, int indent) {
    ML2_BiasesInfoPrint(biases.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_INDENT("", indent));
        for (int i = 0; i < biases.info.inputs; i++) {
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
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
            return biases.inputs == inputImages.channels;
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
        case ML2_LayerCacheTypeImages: {
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
        case ML2_LayerCacheTypeImages: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
            ML2_LayerCacheInfoImages outputImages = ML2_LayerCacheInfoAsImages(output);
            return biases.inputs == outputImages.channels && ML2_ImagesInfoSame(inputImages, outputImages);
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
            int samples = inputScalars.info.samples;
            int inputs = biases.info.inputs;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < inputs; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = *ML2_ScalarsAt(inputScalars, i, j) + *ML2_BiasesAt(biases, j);
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            *ML2_ImagesAt(outputImages, i, j, y, x) = *ML2_ImagesAt(inputImages, i, j, y, x) + *ML2_BiasesAt(biases, j);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_BiasesBackwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerInfoBiases biasesGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient, bool cacheBackward) {
    // // TODO(28/3/2026 17:51:52): when cacheBackward is false i only want to check the biases is compatible with outputGradient, this whole thing is a hack maybe theres a better way
    // return ML2_BiasesInfoSame(biases, biasesGradient) && ML2_LayerCacheInfoSame(input, inputGradient) && (cacheBackward ? ML2_BiasesForwardCompatible(biases, input, outputGradient) : true);
    if (!ML2_BiasesInfoSame(biases, biasesGradient)) return false;
    switch (outputGradient.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheInfoScalars outputGradientScalars = ML2_LayerCacheInfoAsScalars(outputGradient);
            if (biases.inputs != outputGradientScalars.scalars) return false;
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages outputGradientImages = ML2_LayerCacheInfoAsImages(outputGradient);
            if (biases.inputs != outputGradientImages.channels) return false;
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }

    if (cacheBackward) {
        if (!ML2_LayerCacheInfoSame(input, inputGradient)) return false;
        if (!ML2_BiasesForwardCompatible(biases, input, outputGradient)) return false;
    }
    return true;
}

ML2_FN void ML2_BiasesBackward(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_BiasesBackwardCompatible(biases.info, biasesGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient), cacheBackward));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = outputGradientScalars.info.samples;
            int inputs = outputGradientScalars.info.scalars;

            ML2_BiasesClear(biasesGradient);
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < inputs; j++) {
                    *ML2_BiasesAt(biasesGradient, j) += *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
            }

            if (cacheBackward) {
                ML2_ScalarsCopy(inputGradientScalars, outputGradientScalars);
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = outputGradientImages.info.samples;
            int channels = outputGradientImages.info.channels;
            int height = outputGradientImages.info.height;
            int width = outputGradientImages.info.width;

            ML2_BiasesClear(biasesGradient);
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    ML2_Scalar acc = {};
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            acc += *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                        }
                    }
                    *ML2_BiasesAt(biasesGradient, j) += acc;
                }
            }

            if (cacheBackward) {
                ML2_ImagesCopy(inputGradientImages, outputGradientImages);
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_BiasesGradientDescentOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    int inputs = biases.info.inputs;
    for (int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(biasesGradient, i) * learningRate;
    }
}

ML2_FN void ML2_BiasesMomentumOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases average) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, average.info));
    int inputs = biases.info.inputs;
    for (int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(average, i) = decayRate * *ML2_BiasesAt(average, i) - learningRate * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(biases, i) += *ML2_BiasesAt(average, i);
    }
}

ML2_FN void ML2_BiasesAdagradOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_LayerBiases squareSum) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, squareSum.info));
    int inputs = biases.info.inputs;
    for (int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(squareSum, i) += *ML2_BiasesAt(biasesGradient, i) * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(biasesGradient, i) * learningRate / ML2_SQRT(*ML2_BiasesAt(squareSum, i) + ML2_Epsilon);
    }
}

ML2_FN void ML2_BiasesRMSPropOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerBiases squareAverage) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, squareAverage.info));
    int inputs = biases.info.inputs;
    for (int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(squareAverage, i) = decayRate * *ML2_BiasesAt(squareAverage, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate) * *ML2_BiasesAt(biasesGradient, i) * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(biasesGradient, i) * learningRate / ML2_SQRT(*ML2_BiasesAt(squareAverage, i) + ML2_Epsilon);
    }
}

ML2_FN void ML2_BiasesAdamOptimize(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerBiases average, ML2_LayerBiases squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, biasesGradient.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, average.info));
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(biases.info, squareAverage.info));
    int inputs = biases.info.inputs;
    for (int i = 0; i < inputs; i++) {
        *ML2_BiasesAt(average, i) = decayRate1 * *ML2_BiasesAt(average, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * *ML2_BiasesAt(biasesGradient, i);
        *ML2_BiasesAt(squareAverage, i) = decayRate2 * *ML2_BiasesAt(squareAverage, i) + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * *ML2_BiasesAt(biasesGradient, i) * *ML2_BiasesAt(biasesGradient, i);
        ML2_Scalar averageWeighted = *ML2_BiasesAt(average, i) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
        ML2_Scalar squareAverageWeighed = *ML2_BiasesAt(squareAverage, i) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
        *ML2_BiasesAt(biases, i) -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
    }
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

ML2_FN ML2_LayerInfo ML2_Softmax() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.activation = {ML2_ActivationTypeSoftmax}};
}

ML2_FN ML2_LayerInfoActivation ML2_LayerInfoAsActivation(ML2_LayerInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerTypeActivation);
    return info.as.activation;
}

ML2_FN ML2_LayerActivation ML2_LayerAsActivation(ML2_Layer layer) {
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
        case ML2_ActivationTypeSoftmax: return "Softmax";
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_FN void ML2_ActivationInfoPrint(ML2_LayerInfoActivation info, int indent) {
    printf(ML2_INDENT("Activation(\"%s\")\n", indent), ML2_ActivationNameOf(info.type));
}

ML2_FN void ML2_ActivationPrint(ML2_LayerActivation activation, int indent) {
    ML2_ActivationInfoPrint(activation.info, indent);
}

ML2_FN bool ML2_ActivationForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return ML2_LayerCacheInfoSame(input, output);
}

ML2_FN bool ML2_ActivationBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, output) && ML2_LayerCacheInfoSame(input, inputGradient) && ML2_LayerCacheInfoSame(input, outputGradient);
}

ML2_FN void ML2_ActivationReLUForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = ML2_ScalarReLUForward(*ML2_ScalarsAt(inputScalars, i, j));
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            *ML2_ImagesAt(outputImages, i, j, y, x) = ML2_ScalarReLUForward(*ML2_ImagesAt(inputImages, i, j, y, x));
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationReLUBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ActivationBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < scalars; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) = ML2_ScalarReLUBackward(*ML2_ScalarsAt(inputScalars, i, j)) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < channels; j++) {
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                *ML2_ImagesAt(inputGradientImages, i, j, y, x) = ML2_ScalarReLUBackward(*ML2_ImagesAt(inputImages, i, j, y, x)) * *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationSigmoidForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            int samples = outputScalars.info.samples;
            int scalars = outputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = ML2_ScalarSigmoidForward(*ML2_ScalarsAt(inputScalars, i, j));
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = outputImages.info.samples;
            int channels = outputImages.info.channels;
            int height = outputImages.info.height;
            int width = outputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            *ML2_ImagesAt(outputImages, i, j, y, x) = ML2_ScalarSigmoidForward(*ML2_ImagesAt(inputImages, i, j, y, x));
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationSigmoidBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ActivationBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputGradientScalars.info.samples;
            int scalars = inputGradientScalars.info.scalars;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < scalars; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) = ML2_ScalarSigmoidBackward(*ML2_ScalarsAt(outputScalars, i, j)) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = inputGradientImages.info.samples;
            int channels = inputGradientImages.info.channels;
            int height = inputGradientImages.info.height;
            int width = inputGradientImages.info.width;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < channels; j++) {
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                *ML2_ImagesAt(inputGradientImages, i, j, y, x) = ML2_ScalarSigmoidBackward(*ML2_ImagesAt(outputImages, i, j, y, x)) * *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationSinForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            int samples = outputScalars.info.samples;
            int scalars = outputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = ML2_ScalarSinForward(*ML2_ScalarsAt(inputScalars, i, j));
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = outputImages.info.samples;
            int channels = outputImages.info.channels;
            int height = outputImages.info.height;
            int width = outputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            *ML2_ImagesAt(outputImages, i, j, y, x) = ML2_ScalarSinForward(*ML2_ImagesAt(inputImages, i, j, y, x));
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationSinBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ActivationBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < scalars; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) = ML2_ScalarSinBackward(*ML2_ScalarsAt(inputScalars, i, j)) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < channels; j++) {
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                *ML2_ImagesAt(inputGradientImages, i, j, y, x) = ML2_ScalarSinBackward(*ML2_ImagesAt(inputImages, i, j, y, x)) * *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationCosForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            int samples = outputScalars.info.samples;
            int scalars = outputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = ML2_ScalarCosForward(*ML2_ScalarsAt(inputScalars, i, j));
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = outputImages.info.samples;
            int channels = outputImages.info.channels;
            int height = outputImages.info.height;
            int width = outputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            *ML2_ImagesAt(outputImages, i, j, y, x) = ML2_ScalarCosForward(*ML2_ImagesAt(inputImages, i, j, y, x));
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationCosBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ActivationBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < scalars; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) = ML2_ScalarCosBackward(*ML2_ScalarsAt(inputScalars, i, j)) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < channels; j++) {
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                *ML2_ImagesAt(inputGradientImages, i, j, y, x) = ML2_ScalarCosBackward(*ML2_ImagesAt(inputImages, i, j, y, x)) * *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationTanhForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            int samples = outputScalars.info.samples;
            int scalars = outputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = ML2_ScalarTanhForward(*ML2_ScalarsAt(inputScalars, i, j));
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = outputImages.info.samples;
            int channels = outputImages.info.channels;
            int height = outputImages.info.height;
            int width = outputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            *ML2_ImagesAt(outputImages, i, j, y, x) = ML2_ScalarTanhForward(*ML2_ImagesAt(inputImages, i, j, y, x));
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationTanhBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ActivationBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputGradientScalars.info.samples;
            int scalars = inputGradientScalars.info.scalars;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < scalars; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) = ML2_ScalarTanhBackward(*ML2_ScalarsAt(outputScalars, i, j)) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = inputGradientImages.info.samples;
            int channels = inputGradientImages.info.channels;
            int height = inputGradientImages.info.height;
            int width = inputGradientImages.info.width;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < channels; j++) {
                        for (int y = 0; y < height; y++) {
                            for (int x = 0; x < width; x++) {
                                *ML2_ImagesAt(inputGradientImages, i, j, y, x) = ML2_ScalarTanhBackward(*ML2_ImagesAt(outputImages, i, j, y, x)) * *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationSoftmaxForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = ML2_LayerCacheAsScalars(input);
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            int samples = outputScalars.info.samples;
            int scalars = outputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                ML2_Scalar max = *ML2_ScalarsAt(inputScalars, i, 0);
                for (int j = 1; j < scalars; j++) {
                    ML2_Scalar s = *ML2_ScalarsAt(inputScalars, i, j);
                    if (s > max) max = s;
                }
                ML2_Scalar sum = {};
                for (int j = 0; j < scalars; j++) {
                    ML2_Scalar e = ML2_EXP(*ML2_ScalarsAt(inputScalars, i, j) - max);
                    *ML2_ScalarsAt(outputScalars, i, j) = e;
                    sum += e;
                }
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) /= sum;
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Softmax may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationSoftmaxBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ActivationBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            int samples = inputGradientScalars.info.samples;
            int scalars = inputGradientScalars.info.scalars;

            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    ML2_Scalar sum = {};
                    for (int j = 0; j < scalars; j++) {
                        sum += *ML2_ScalarsAt(outputScalars, i, j) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                    }
                    for (int j = 0; j < scalars; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) = *ML2_ScalarsAt(outputScalars, i, j) * (*ML2_ScalarsAt(outputGradientScalars, i, j) - sum);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Softmax may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_ActivationForward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache output) {
    switch (activation.info.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationReLUForward(input, output); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationSigmoidForward(input, output); break;
        case ML2_ActivationTypeSin: ML2_ActivationSinForward(input, output); break;
        case ML2_ActivationTypeCos: ML2_ActivationCosForward(input, output); break;
        case ML2_ActivationTypeTanh: ML2_ActivationTanhForward(input, output); break;
        case ML2_ActivationTypeSoftmax: ML2_ActivationSoftmaxForward(input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_FN void ML2_ActivationBackward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    switch (activation.info.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationReLUBackward(input, inputGradient, output, outputGradient, cacheBackward); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationSigmoidBackward(input, inputGradient, output, outputGradient, cacheBackward); break;
        case ML2_ActivationTypeSin: ML2_ActivationSinBackward(input, inputGradient, output, outputGradient, cacheBackward); break;
        case ML2_ActivationTypeCos: ML2_ActivationCosBackward(input, inputGradient, output, outputGradient, cacheBackward); break;
        case ML2_ActivationTypeTanh: ML2_ActivationTanhBackward(input, inputGradient, output, outputGradient, cacheBackward); break;
        case ML2_ActivationTypeSoftmax: ML2_ActivationSoftmaxBackward(input, inputGradient, output, outputGradient, cacheBackward); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

ML2_FN ML2_LayerInfo ML2_Linear(int outputs, int inputs) {
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
    return (ML2_LayerInfoWeights){linear.outputs, linear.inputs};
}

ML2_FN ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear) {
    return (ML2_LayerWeights){ML2_LinearInfoWeights(linear.info), {linear.data.weights}};
}

ML2_FN ML2_LayerInfoBiases ML2_LinearInfoBiases(ML2_LayerInfoLinear linear) {
    return (ML2_LayerInfoBiases){linear.outputs};
}

ML2_FN ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear) {
    return (ML2_LayerBiases){ML2_LinearInfoBiases(linear.info), {linear.data.biases}};
}

ML2_FN void ML2_LinearClear(ML2_LayerLinear linear) {
    ML2_WeightsClear(ML2_LinearWeights(linear));
    ML2_BiasesClear(ML2_LinearBiases(linear));
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

ML2_FN void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, int indent) {
    printf(ML2_INDENT("Linear(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_FN void ML2_LinearPrint(ML2_LayerLinear linear, int indent) {
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
            ML2_BiasesForward(ML2_LinearBiases(linear), output, output);
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LinearBackwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerInfoLinear linearGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_WeightsBackwardCompatible(ML2_LinearInfoWeights(linear), ML2_LinearInfoWeights(linearGradient), input, inputGradient, outputGradient) && ML2_BiasesBackwardCompatible(ML2_LinearInfoBiases(linear), ML2_LinearInfoBiases(linearGradient), input, inputGradient, outputGradient, false);
}

ML2_FN void ML2_LinearBackward(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_LinearBackwardCompatible(linear.info, linearGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_WeightsBackward(ML2_LinearWeights(linear), ML2_LinearWeights(linearGradient), input, inputGradient, outputGradient, cacheBackward);
            ML2_BiasesBackward(ML2_LinearBiases(linear), ML2_LinearBiases(linearGradient), input, inputGradient, outputGradient, false);
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
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

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_FN ML2_LayerInfo ML2_Filters(int outputs, int inputs, int height, int width) {
    return (ML2_LayerInfo){ML2_LayerTypeFilters, .as.filters = {outputs, inputs, height, width}};
}

ML2_FN ML2_LayerFilters ML2_FiltersNew(ML2_LayerInfoFilters info) {
    ML2_LayerFilters filters = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs * info.height * info.width, sizeof(*filters.data.weights))}
    };
    return filters;
}

ML2_FN void ML2_FiltersDestroy(ML2_LayerFilters *filters) {
    ML2_FREE(filters->data.weights);
    *filters = (ML2_LayerFilters){};
}

ML2_FN ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, int i, int j, int k, int l) {
    ML2_SOFT_ASSERT(0 <= i && i < filters.info.outputs && 0 <= j && j < filters.info.inputs && 0 <= k && k < filters.info.height && 0 <= l && l < filters.info.width && "OUT OF BOUNDS INDICES");
    return &filters.data.weights[((i * filters.info.inputs + j) * filters.info.height + k) * filters.info.width + l];
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
    for (int i = 0; i < filters.info.outputs; i++) {
        for (int j = 0; j < filters.info.inputs; j++) {
            for (int y = 0; y < filters.info.height; y++) {
                for (int x = 0; x < filters.info.width; x++) {
                    *ML2_FiltersAt(filters, i, j, y, x) = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < filters.info.outputs; i++) {
        for (int j = 0; j < filters.info.inputs; j++) {
            for (int y = 0; y < filters.info.height; y++) {
                for (int x = 0; x < filters.info.width; x++) {
                    *ML2_FiltersAt(filters, i, j, y, x) = ML2_ScalarRand(low, high);
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
    ML2_Scalar scale = ML2_SQRT(ML2_SCALAR_LITERAL(2.0) / (filters.info.inputs * filters.info.height * filters.info.width));
    for (int i = 0; i < filters.info.outputs; i++) {
        for (int j = 0; j < filters.info.inputs; j++) {
            for (int y = 0; y < filters.info.height; y++) {
                for (int x = 0; x < filters.info.width; x++) {
                    *ML2_FiltersAt(filters, i, j, y, x) = scale * ML2_ScalarRandNormal();
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, int indent) {
    printf(ML2_INDENT("Filters(%dx%dx%dx%d)\n", indent), info.outputs, info.inputs, info.height, info.width);
}

ML2_FN void ML2_FiltersPrint(ML2_LayerFilters filters, int indent) {
    ML2_FiltersInfoPrint(filters.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < filters.info.outputs; i++) {
            for (int j = 0; j < filters.info.inputs; j++) {
                printf(ML2_INDENT("Weights[%d,%d]:\n", indent), i, j);
                printf(ML2_INDENT("{\n", indent));
                {
                    indent += ML2_Indentation;
                    for (int y = 0; y < filters.info.height; y++) {
                        printf(ML2_INDENT("", indent));
                        for (int x = 0; x < filters.info.width; x++) {
                            printf(ML2_SCALAR_FMT " ", *ML2_FiltersAt(filters, i, j, y, x));
                        }
                        printf("\n");
                    }
                    indent -= ML2_Indentation;
                }
                printf(ML2_INDENT("}\n", indent));
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
            int height = inputImages.height - filters.height + 1;
            int width = inputImages.width - filters.width + 1;
            return filters.inputs == inputImages.channels && height > 0 && width > 0;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FiltersInfoForwardCompatible(filters, input));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
            int height = inputImages.height - filters.height + 1;
            int width = inputImages.width - filters.width + 1;
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeImages, .as.images = {inputImages.samples, filters.outputs, height, width}};
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b) {
    return a.outputs == b.outputs && a.inputs == b.inputs && a.height == b.height && a.width == b.width;
}

ML2_FN bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
            ML2_LayerCacheInfoImages outputImages = ML2_LayerCacheInfoAsImages(output);
            int height = inputImages.height - filters.height + 1;
            int width = inputImages.width - filters.width + 1;
            return inputImages.samples == outputImages.samples && inputImages.channels == filters.inputs && outputImages.channels == filters.outputs && outputImages.height == height && outputImages.width == width;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FiltersForwardCompatible(filters.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            int samples = outputImages.info.samples;
            int outputs = outputImages.info.channels;
            int inputs = inputImages.info.channels;

            int outputHeight = outputImages.info.height;
            int outputWidth = outputImages.info.width;
            int filtersHeight = filters.info.height;
            int filtersWidth = filters.info.width;

            // TODO: find a way to not clear it
            ML2_ImagesClear(outputImages);
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < outputs; j++) {
                    for (int k = 0; k < inputs; k++) {
                        for (int yk = 0; yk < filtersHeight; yk++) {
                            for (int xk = 0; xk < filtersWidth; xk++) {
                                for (int y = 0; y < outputHeight; y++) {
                                    for (int x = 0; x < outputWidth; x++) {
                                        *ML2_ImagesAt(outputImages, i, j, y, x) += *ML2_ImagesAt(inputImages, i, k, y + yk, x + xk) * *ML2_FiltersAt(filters, j, k, yk, xk);
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

ML2_FN bool ML2_FiltersBackwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerInfoFilters filtersGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersInfoSame(filters, filtersGradient) && ML2_LayerCacheInfoSame(input, inputGradient) && ML2_FiltersForwardCompatible(filters, input, outputGradient);
}

ML2_FN void ML2_FiltersBackward(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_FiltersBackwardCompatible(filters.info, filtersGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            int samples = inputImages.info.samples;
            int outputs = filters.info.outputs;
            int inputs = filters.info.inputs;

            int inputHeight = inputGradientImages.info.height;
            int inputWidth = inputGradientImages.info.width;
            int outputHeight = outputGradientImages.info.height;
            int outputWidth = outputGradientImages.info.width;
            int filtersHeight = filters.info.height;
            int filtersWidth = filters.info.width;

            ML2_FiltersClear(filtersGradient);
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < outputs; j++) {
                    for (int k = 0; k < inputs; k++) {
                        for (int yk = 0; yk < filtersHeight; yk++) {
                            for (int xk = 0; xk < filtersWidth; xk++) {
                                ML2_Scalar acc = {};
                                for (int y = 0; y < outputHeight; y++) {
                                    for (int x = 0; x < outputWidth; x++) {
                                        acc += *ML2_ImagesAt(inputImages, i, k, y + yk, x + xk) * *ML2_ImagesAt(outputGradientImages, i, j, y, x);
                                    }
                                }
                                *ML2_FiltersAt(filtersGradient, j, k, yk, xk) += acc;
                            }
                        }
                    }
                }
            }

            // TODO(11/4/2026 8:47:40pm):
            //      i made this optimization that moved the j loop down near near yk and xk,
            //      after a bunch of testing it seems to be faster,
            //      even though the filters iteration order is "inefficient"
            //      this is probably because most filters are small, so they fit in cache easily
            //      and also because now there is no ImagesClear, overall this might make it faster
            //      but if im writing this todo then obviously im still unsure

            // if (cacheBackward) {
            //     ML2_ImagesClear(inputGradientImages);
            //     for (int i = 0; i < samples; i++) {
            //         for (int j = 0; j < outputs; j++) {
            //             for (int k = 0; k < inputs; k++) {
            //                 for (int y = 0; y < inputHeight; y++) {
            //                     for (int x = 0; x < inputWidth; x++) {
            //                         int yk_min = (y >= outputHeight) ? y - outputHeight + 1 : 0;
            //                         int yk_max = (y < filtersHeight) ? y : filtersHeight - 1;
            //                         int xk_min = (x >= outputWidth) ? x - outputWidth + 1 : 0;
            //                         int xk_max = (x < filtersWidth) ? x : filtersWidth - 1;
            //                         ML2_Scalar acc = {};
            //                         for (int yk = yk_min; yk <= yk_max; yk++) {
            //                             for (int xk = xk_min; xk <= xk_max; xk++) {
            //                                 acc += *ML2_FiltersAt(filters, j, k, yk, xk) * *ML2_ImagesAt(outputGradientImages, i, j, y - yk, x - xk);
            //                             }
            //                         }
            //                         *ML2_ImagesAt(inputGradientImages, i, k, y, x) += acc;
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }
            if (cacheBackward) {
                for (int i = 0; i < samples; i++) {
                    for (int k = 0; k < inputs; k++) {
                        for (int y = 0; y < inputHeight; y++) {
                            for (int x = 0; x < inputWidth; x++) {
                                int yk_min = (y >= outputHeight) ? y - outputHeight + 1 : 0;
                                int yk_max = (y < filtersHeight) ? y : filtersHeight - 1;
                                int xk_min = (x >= outputWidth) ? x - outputWidth + 1 : 0;
                                int xk_max = (x < filtersWidth) ? x : filtersWidth - 1;
                                ML2_Scalar sum = {};
                                for (int j = 0; j < outputs; j++) {
                                    for (int yk = yk_min; yk <= yk_max; yk++) {
                                        for (int xk = xk_min; xk <= xk_max; xk++) {
                                            sum += *ML2_FiltersAt(filters, j, k, yk, xk) * *ML2_ImagesAt(outputGradientImages, i, j, y - yk, x - xk);
                                        }
                                    }
                                }
                                *ML2_ImagesAt(inputGradientImages, i, k, y, x) = sum;
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
    int outputs = filters.info.outputs;
    int inputs = filters.info.inputs;
    int height = filters.info.height;
    int width = filters.info.width;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_FiltersAt(filters, i, j, y, x) -= *ML2_FiltersAt(filtersGradient, i, j, y, x) * learningRate;
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersMomentumOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters average) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, average.info));
    int outputs = filters.info.outputs;
    int inputs = filters.info.inputs;
    int height = filters.info.height;
    int width = filters.info.width;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_FiltersAt(average, i, j, y, x) = decayRate * *ML2_FiltersAt(average, i, j, y, x) - learningRate * *ML2_FiltersAt(filtersGradient, i, j, y, x);
                    *ML2_FiltersAt(filters, i, j, y, x) += *ML2_FiltersAt(average, i, j, y, x);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersAdagradOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_LayerFilters squareSum) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, squareSum.info));
    int outputs = filters.info.outputs;
    int inputs = filters.info.inputs;
    int height = filters.info.height;
    int width = filters.info.width;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_FiltersAt(squareSum, i, j, y, x) += *ML2_FiltersAt(filtersGradient, i, j, y, x) * *ML2_FiltersAt(filtersGradient, i, j, y, x);
                    *ML2_FiltersAt(filters, i, j, y, x) -= *ML2_FiltersAt(filtersGradient, i, j, y, x) * learningRate / ML2_SQRT(*ML2_FiltersAt(squareSum, i, j, y, x) + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersRMSPropOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_LayerFilters squareAverage) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, squareAverage.info));
    int outputs = filters.info.outputs;
    int inputs = filters.info.inputs;
    int height = filters.info.height;
    int width = filters.info.width;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_FiltersAt(squareAverage, i, j, y, x) = decayRate * *ML2_FiltersAt(squareAverage, i, j, y, x) + (ML2_SCALAR_LITERAL(1.0) - decayRate) * *ML2_FiltersAt(filtersGradient, i, j, y, x) * *ML2_FiltersAt(filtersGradient, i, j, y, x);
                    *ML2_FiltersAt(filters, i, j, y, x) -= *ML2_FiltersAt(filtersGradient, i, j, y, x) * learningRate / ML2_SQRT(*ML2_FiltersAt(squareAverage, i, j, y, x) + ML2_Epsilon);
                }
            }
        }
    }
}

ML2_FN void ML2_FiltersAdamOptimize(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_LayerFilters average, ML2_LayerFilters squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, filtersGradient.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, average.info));
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(filters.info, squareAverage.info));
    int outputs = filters.info.outputs;
    int inputs = filters.info.inputs;
    int height = filters.info.height;
    int width = filters.info.width;
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_FiltersAt(average, i, j, y, x) = decayRate1 * *ML2_FiltersAt(average, i, j, y, x) + (ML2_SCALAR_LITERAL(1.0) - decayRate1) * *ML2_FiltersAt(filtersGradient, i, j, y, x);
                    *ML2_FiltersAt(squareAverage, i, j, y, x) = decayRate2 * *ML2_FiltersAt(squareAverage, i, j, y, x) + (ML2_SCALAR_LITERAL(1.0) - decayRate2) * *ML2_FiltersAt(filtersGradient, i, j, y, x) * *ML2_FiltersAt(filtersGradient, i, j, y, x);
                    ML2_Scalar averageWeighted = *ML2_FiltersAt(average, i, j, y, x) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight1);
                    ML2_Scalar squareAverageWeighed = *ML2_FiltersAt(squareAverage, i, j, y, x) / (ML2_SCALAR_LITERAL(1.0) - decayingWeight2);
                    *ML2_FiltersAt(filters, i, j, y, x) -= averageWeighted * learningRate / (ML2_SQRT(squareAverageWeighed) + ML2_Epsilon);
                }
            }
        }
    }
}

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_FN ML2_LayerInfo ML2_Conv(int outputs, int inputs, int height, int width) {
    return (ML2_LayerInfo){ML2_LayerTypeConv, .as.conv = {outputs, inputs, height, width}};
}

ML2_FN ML2_LayerConv ML2_ConvNew(ML2_LayerInfoConv info) {
    ML2_LayerConv conv = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs * info.height * info.width, sizeof(*conv.data.weights)),
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
    return (ML2_LayerInfoFilters){conv.outputs, conv.inputs, conv.height, conv.width};
}

ML2_FN ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv) {
    return (ML2_LayerFilters){ML2_ConvInfoFilters(conv.info), {conv.data.weights}};
}

ML2_FN ML2_LayerInfoBiases ML2_ConvInfoBiases(ML2_LayerInfoConv conv) {
    return (ML2_LayerInfoBiases){conv.outputs};
}

ML2_FN ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv) {
    return (ML2_LayerBiases){ML2_ConvInfoBiases(conv.info), {conv.data.biases}};
}

ML2_FN void ML2_ConvClear(ML2_LayerConv conv) {
    ML2_FiltersClear(ML2_ConvFilters(conv));
    ML2_BiasesClear(ML2_ConvBiases(conv));
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

ML2_FN void ML2_ConvInfoPrint(ML2_LayerInfoConv info, int indent) {
    printf(ML2_INDENT("Conv(%dx%dx%dx%d)\n", indent), info.outputs, info.inputs, info.height, info.width);
}

ML2_FN void ML2_ConvPrint(ML2_LayerConv conv, int indent) {
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
        }
        case ML2_LayerCacheTypeImages: {
            ML2_FiltersForward(ML2_ConvFilters(conv), input, output);
            ML2_BiasesForward(ML2_ConvBiases(conv), output, output);
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_ConvBackwardCompatible(ML2_LayerInfoConv conv, ML2_LayerInfoConv convGradient, ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_FiltersBackwardCompatible(ML2_ConvInfoFilters(conv), ML2_ConvInfoFilters(convGradient), input, inputGradient, outputGradient) && ML2_BiasesBackwardCompatible(ML2_ConvInfoBiases(conv), ML2_ConvInfoBiases(convGradient), input, inputGradient, outputGradient, false);
}

ML2_FN void ML2_ConvBackward(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_ConvBackwardCompatible(conv.info, convGradient.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_FiltersBackward(ML2_ConvFilters(conv), ML2_ConvFilters(convGradient), input, inputGradient, outputGradient, cacheBackward);
            ML2_BiasesBackward(ML2_ConvBiases(conv), ML2_ConvBiases(convGradient), input, inputGradient, outputGradient, false);
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

// ML2_LayerConv ⬆️

// ML2_LayerFlatten ⬇️

ML2_FN ML2_LayerInfo ML2_Flatten() {
    return (ML2_LayerInfo){ML2_LayerTypeFlatten, {}};
}

ML2_FN void ML2_FlattenInfoPrint(int indent) {
    printf(ML2_INDENT("Flatten()\n", indent));
}

ML2_FN void ML2_FlattenPrint(int indent) {
    ML2_FlattenInfoPrint(indent);
}

ML2_FN bool ML2_FlattenInfoForwardCompatible(ML2_LayerCacheInfo input) {
    return input.type == ML2_LayerCacheTypeImages;
}

ML2_FN ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FlattenInfoForwardCompatible(input));
    ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeScalars, .as.scalars = {inputImages.samples, inputImages.channels * inputImages.height * inputImages.width}};
}

ML2_FN bool ML2_FlattenForwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != ML2_LayerCacheTypeImages || output.type != ML2_LayerCacheTypeScalars) return false;
    ML2_LayerCacheInfoImages inputImages = ML2_LayerCacheInfoAsImages(input);
    ML2_LayerCacheInfoScalars outputScalars = ML2_LayerCacheInfoAsScalars(output);
    return inputImages.samples == outputScalars.samples && inputImages.channels * inputImages.height * inputImages.width == outputScalars.scalars;
}

ML2_FN void ML2_FlattenForward(ML2_LayerCache input, ML2_LayerCache output) {
    ML2_HARD_ASSERT(ML2_FlattenForwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output)));
    ML2_LayerCacheImages inputImages = ML2_LayerCacheAsImages(input);
    ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
    int samples = inputImages.info.samples;
    int channels = inputImages.info.channels;
    int height = inputImages.info.height;
    int width = inputImages.info.width;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < channels; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_ScalarsAt(outputScalars, i, (j * height + y) * width + x) = *ML2_ImagesAt(inputImages, i, j, y, x);
                }
            }
        }
    }
}

ML2_FN bool ML2_FlattenBackwardCompatible(ML2_LayerCacheInfo input, ML2_LayerCacheInfo inputGradient, ML2_LayerCacheInfo outputGradient) {
    return ML2_LayerCacheInfoSame(input, inputGradient) && ML2_FlattenForwardCompatible(input, outputGradient);
}

ML2_FN void ML2_FlattenBackward(ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient, bool cacheBackward) {
    ML2_HARD_ASSERT(ML2_FlattenBackwardCompatible(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient), ML2_LayerCacheAsInfo(outputGradient)));
    ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
    ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
    int samples = inputGradientImages.info.samples;
    int channels = inputGradientImages.info.channels;
    int height = inputGradientImages.info.height;
    int width = inputGradientImages.info.width;

    if (cacheBackward) {
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < channels; j++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        *ML2_ImagesAt(inputGradientImages, i, j, y, x) = *ML2_ScalarsAt(outputGradientScalars, i, (j * height + y) * width + x);
                    }
                }
            }
        }
    }
}

// ML2_LayerFlatten ⬆️

ML2_FN ML2_Layer ML2_LayerNew(ML2_LayerInfo info) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return (ML2_Layer){info.type, .as.weights = ML2_WeightsNew(ML2_LayerInfoAsWeights(info))};
        case ML2_LayerTypeBiases: return (ML2_Layer){info.type, .as.biases = ML2_BiasesNew(ML2_LayerInfoAsBiases(info))};
        case ML2_LayerTypeActivation: return (ML2_Layer){info.type, .as.activation = {ML2_LayerInfoAsActivation(info)}};
        case ML2_LayerTypeLinear: return (ML2_Layer){info.type, .as.linear = ML2_LinearNew(ML2_LayerInfoAsLinear(info))};
        case ML2_LayerTypeFilters: return (ML2_Layer){info.type, .as.filters = ML2_FiltersNew(ML2_LayerInfoAsFilters(info))};
        case ML2_LayerTypeConv: return (ML2_Layer){info.type, .as.conv = ML2_ConvNew(ML2_LayerInfoAsConv(info))};
        case ML2_LayerTypeFlatten: return (ML2_Layer){info.type, {}};
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
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return (ML2_LayerInfo){layer.type, .as.weights = ML2_LayerAsWeights(layer).info};
        case ML2_LayerTypeBiases: return (ML2_LayerInfo){layer.type, .as.biases = ML2_LayerAsBiases(layer).info};
        case ML2_LayerTypeActivation: return (ML2_LayerInfo){layer.type, .as.activation = ML2_LayerAsActivation(layer).info};
        case ML2_LayerTypeLinear: return (ML2_LayerInfo){layer.type, .as.linear = ML2_LayerAsLinear(layer).info};
        case ML2_LayerTypeFilters: return (ML2_LayerInfo){layer.type, .as.filters = ML2_LayerAsFilters(layer).info};
        case ML2_LayerTypeConv: return (ML2_LayerInfo){layer.type, .as.conv = ML2_LayerAsConv(layer).info};
        case ML2_LayerTypeFlatten: return (ML2_LayerInfo){layer.type, {}};
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
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerInfoPrint(ML2_LayerInfo layer, int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsInfoPrint(ML2_LayerInfoAsWeights(layer), indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesInfoPrint(ML2_LayerInfoAsBiases(layer), indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationInfoPrint(ML2_LayerInfoAsActivation(layer), indent); break;
        case ML2_LayerTypeLinear: ML2_LinearInfoPrint(ML2_LayerInfoAsLinear(layer), indent); break;
        case ML2_LayerTypeFilters: ML2_FiltersInfoPrint(ML2_LayerInfoAsFilters(layer), indent); break;
        case ML2_LayerTypeConv: ML2_ConvInfoPrint(ML2_LayerInfoAsConv(layer), indent); break;
        case ML2_LayerTypeFlatten: ML2_FlattenInfoPrint(indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerPrint(ML2_Layer layer, int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsPrint(ML2_LayerAsWeights(layer), indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesPrint(ML2_LayerAsBiases(layer), indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationPrint(ML2_LayerAsActivation(layer), indent); break;
        case ML2_LayerTypeLinear: ML2_LinearPrint(ML2_LayerAsLinear(layer), indent); break;
        case ML2_LayerTypeFilters: ML2_FiltersPrint(ML2_LayerAsFilters(layer), indent); break;
        case ML2_LayerTypeConv: ML2_ConvPrint(ML2_LayerAsConv(layer), indent); break;
        case ML2_LayerTypeFlatten: ML2_FlattenPrint(indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsInfoForward(ML2_LayerInfoAsWeights(info), input);
        case ML2_LayerTypeBiases: return ML2_BiasesInfoForward(ML2_LayerInfoAsBiases(info), input);
        case ML2_LayerTypeActivation: return input;
        case ML2_LayerTypeLinear: return ML2_LinearInfoForward(ML2_LayerInfoAsLinear(info), input);
        case ML2_LayerTypeFilters: return ML2_FiltersInfoForward(ML2_LayerInfoAsFilters(info), input);
        case ML2_LayerTypeConv: return ML2_ConvInfoForward(ML2_LayerInfoAsConv(info), input);
        case ML2_LayerTypeFlatten: return ML2_FlattenInfoForward(input);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsForward(ML2_LayerAsWeights(layer), input, output); break;
        case ML2_LayerTypeBiases: ML2_BiasesForward(ML2_LayerAsBiases(layer), input, output); break;
        case ML2_LayerTypeActivation: ML2_ActivationForward(ML2_LayerAsActivation(layer), input, output); break;
        case ML2_LayerTypeLinear: ML2_LinearForward(ML2_LayerAsLinear(layer), input, output); break;
        case ML2_LayerTypeFilters: ML2_FiltersForward(ML2_LayerAsFilters(layer), input, output); break;
        case ML2_LayerTypeConv: ML2_ConvForward(ML2_LayerAsConv(layer), input, output); break;
        case ML2_LayerTypeFlatten: ML2_FlattenForward(input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerBackward(ML2_Layer layer, ML2_Layer layerGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient, bool cacheBackward) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsBackward(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(layerGradient), input, inputGradient, outputGradient, cacheBackward); break;
        case ML2_LayerTypeBiases: ML2_BiasesBackward(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(layerGradient), input, inputGradient, outputGradient, cacheBackward); break;
        case ML2_LayerTypeActivation: ML2_ActivationBackward(ML2_LayerAsActivation(layer), input, inputGradient, output, outputGradient, cacheBackward); break;
        case ML2_LayerTypeLinear: ML2_LinearBackward(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(layerGradient), input, inputGradient, outputGradient, cacheBackward); break;
        case ML2_LayerTypeFilters: ML2_FiltersBackward(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(layerGradient), input, inputGradient, outputGradient, cacheBackward); break;
        case ML2_LayerTypeConv: ML2_ConvBackward(ML2_LayerAsConv(layer), ML2_LayerAsConv(layerGradient), input, inputGradient, outputGradient, cacheBackward); break;
        case ML2_LayerTypeFlatten: ML2_FlattenBackward(input, inputGradient, outputGradient, cacheBackward); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerGradientDescentOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsGradientDescentOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate); break;
        case ML2_LayerTypeBiases: ML2_BiasesGradientDescentOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearGradientDescentOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate); break;
        case ML2_LayerTypeFilters: ML2_FiltersGradientDescentOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate); break;
        case ML2_LayerTypeConv: ML2_ConvGradientDescentOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerMomentumOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer average) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsMomentumOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, decayRate, ML2_LayerAsWeights(average)); break;
        case ML2_LayerTypeBiases: ML2_BiasesMomentumOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, decayRate, ML2_LayerAsBiases(average)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearMomentumOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, decayRate, ML2_LayerAsLinear(average)); break;
        case ML2_LayerTypeFilters: ML2_FiltersMomentumOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, decayRate, ML2_LayerAsFilters(average)); break;
        case ML2_LayerTypeConv: ML2_ConvMomentumOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, decayRate, ML2_LayerAsConv(average)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerAdagradOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Layer squareSum) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsAdagradOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, ML2_LayerAsWeights(squareSum)); break;
        case ML2_LayerTypeBiases: ML2_BiasesAdagradOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, ML2_LayerAsBiases(squareSum)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearAdagradOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, ML2_LayerAsLinear(squareSum)); break;
        case ML2_LayerTypeFilters: ML2_FiltersAdagradOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, ML2_LayerAsFilters(squareSum)); break;
        case ML2_LayerTypeConv: ML2_ConvAdagradOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, ML2_LayerAsConv(squareSum)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerRMSPropOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate, ML2_Layer squareAverage) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsRMSPropOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, decayRate, ML2_LayerAsWeights(squareAverage)); break;
        case ML2_LayerTypeBiases: ML2_BiasesRMSPropOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, decayRate, ML2_LayerAsBiases(squareAverage)); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearRMSPropOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, decayRate, ML2_LayerAsLinear(squareAverage)); break;
        case ML2_LayerTypeFilters: ML2_FiltersRMSPropOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, decayRate, ML2_LayerAsFilters(squareAverage)); break;
        case ML2_LayerTypeConv: ML2_ConvRMSPropOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, decayRate, ML2_LayerAsConv(squareAverage)); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_FN void ML2_LayerAdamOptimize(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate, ML2_Scalar decayRate1, ML2_Scalar decayRate2, ML2_Layer average, ML2_Layer squareAverage, ML2_Scalar decayingWeight1, ML2_Scalar decayingWeight2) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsAdamOptimize(ML2_LayerAsWeights(layer), ML2_LayerAsWeights(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsWeights(average), ML2_LayerAsWeights(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeBiases: ML2_BiasesAdamOptimize(ML2_LayerAsBiases(layer), ML2_LayerAsBiases(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsBiases(average), ML2_LayerAsBiases(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearAdamOptimize(ML2_LayerAsLinear(layer), ML2_LayerAsLinear(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsLinear(average), ML2_LayerAsLinear(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeFilters: ML2_FiltersAdamOptimize(ML2_LayerAsFilters(layer), ML2_LayerAsFilters(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsFilters(average), ML2_LayerAsFilters(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeConv: ML2_ConvAdamOptimize(ML2_LayerAsConv(layer), ML2_LayerAsConv(gradient), learningRate, decayRate1, decayRate2, ML2_LayerAsConv(average), ML2_LayerAsConv(squareAverage), decayingWeight1, decayingWeight2); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

// ML2_LayerCacheScalars ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Scalars(int samples, int scalars) {
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

ML2_FN ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, int sample, int scalar) {
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
    for (int i = 0; i < scalars.info.samples; i++) {
        for (int j = 0; j < scalars.info.scalars; j++) {
            *ML2_ScalarsAt(scalars, i, j) = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_FN void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(dest.info, src.info));
    int samples = dest.info.samples;
    int scalars = dest.info.scalars;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            *ML2_ScalarsAt(dest, i, j) = *ML2_ScalarsAt(src, i, j);
        }
    }
}

ML2_FN void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, int indent) {
    printf(ML2_INDENT("Scalars(%dx%d)\n", indent), info.samples, info.scalars);
}

ML2_FN void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, int indent) {
    ML2_ScalarsInfoPrint(scalars.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < scalars.info.samples; i++) {
            printf(ML2_INDENT("", indent));
            for (int j = 0; j < scalars.info.scalars; j++) {
                printf(ML2_SCALAR_FMT " ", *ML2_ScalarsAt(scalars, i, j));
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

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheImages ⬇️

ML2_FN ML2_LayerCacheInfo ML2_Images(int samples, int channels, int height, int width) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeImages, .as.images = {samples, channels, height, width}};
}

ML2_FN ML2_LayerCacheImages ML2_ImagesNew(ML2_LayerCacheInfoImages info) {
    ML2_LayerCacheImages images = {
        info,
        {ML2_RELIABLE_CALLOC(info.samples * info.channels * info.height * info.width, sizeof(*images.data.images))}
    };
    return images;
}

ML2_FN void ML2_ImagesDestroy(ML2_LayerCacheImages *images) {
    ML2_FREE(images->data.images);
    *images = (ML2_LayerCacheImages){};
}

ML2_FN ML2_Scalar *ML2_ImagesAt(ML2_LayerCacheImages images, int sample, int channel, int y, int x) {
    ML2_SOFT_ASSERT(0 <= sample && sample < images.info.samples && 0 <= channel && channel < images.info.channels && 0 <= y && y < images.info.height && 0 <= x && x < images.info.width && "OUT OF BOUNDS INDICES");
    return &images.data.images[((sample * images.info.channels + channel) * images.info.height + y) * images.info.width + x];
}

ML2_FN ML2_LayerCacheInfoImages ML2_LayerCacheInfoAsImages(ML2_LayerCacheInfo info) {
    ML2_HARD_ASSERT(info.type == ML2_LayerCacheTypeImages);
    return info.as.images;
}

ML2_FN ML2_LayerCacheImages ML2_LayerCacheAsImages(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeImages);
    return cache.as.images;
}

ML2_FN void ML2_ImagesClear(ML2_LayerCacheImages images) {
    for (int i = 0; i < images.info.samples; i++) {
        for (int j = 0; j < images.info.channels; j++) {
            for (int y = 0; y < images.info.height; y++) {
                for (int x = 0; x < images.info.width; x++) {
                    *ML2_ImagesAt(images, i, j, y, x) = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_FN void ML2_ImagesCopy(ML2_LayerCacheImages dest, ML2_LayerCacheImages src) {
    ML2_HARD_ASSERT(ML2_ImagesInfoSame(dest.info, src.info));
    int samples = dest.info.samples;
    int channels = dest.info.channels;
    int height = dest.info.height;
    int width = dest.info.width;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < channels; j++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *ML2_ImagesAt(dest, i, j, y, x) = *ML2_ImagesAt(src, i, j, y, x);
                }
            }
        }
    }
}

ML2_FN void ML2_ImagesInfoPrint(ML2_LayerCacheInfoImages info, int indent) {
    printf(ML2_INDENT("Images(%dx%dx%dx%d)\n", indent), info.samples, info.channels, info.height, info.width);
}

ML2_FN void ML2_ImagesPrint(ML2_LayerCacheImages images, int indent) {
    ML2_ImagesInfoPrint(images.info, indent);
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < images.info.samples; i++) {
            printf(ML2_INDENT("Samples[%d]:\n", indent), i);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                for (int j = 0; j < images.info.channels; j++) {
                    printf(ML2_INDENT("Channels[%d]:\n", indent), j);
                    printf(ML2_INDENT("{\n", indent));
                    {
                        indent += ML2_Indentation;
                        for (int y = 0; y < images.info.height; y++) {
                            printf(ML2_INDENT("", indent));
                            for (int x = 0; x < images.info.width; x++) {
                                printf(ML2_SCALAR_FMT " ", *ML2_ImagesAt(images, i, j, y, x));
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
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN bool ML2_ImagesInfoSame(ML2_LayerCacheInfoImages a, ML2_LayerCacheInfoImages b) {
    return a.samples == b.samples && a.channels == b.channels && a.height == b.height && a.width == b.width;
}

// ML2_LayerCacheImages ⬆️

ML2_FN ML2_LayerCache ML2_LayerCacheNew(ML2_LayerCacheInfo info) {
    switch (info.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCache){info.type, .as.scalars = ML2_ScalarsNew(ML2_LayerCacheInfoAsScalars(info))};
        case ML2_LayerCacheTypeImages: return (ML2_LayerCache){info.type, .as.images = ML2_ImagesNew(ML2_LayerCacheInfoAsImages(info))};
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheDestroy(ML2_LayerCache *cache) {
    switch (cache->type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsDestroy(&cache->as.scalars); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesDestroy(&cache->as.images); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCacheInfo){cache.type, .as.scalars = ML2_LayerCacheAsScalars(cache).info};
        case ML2_LayerCacheTypeImages: return (ML2_LayerCacheInfo){cache.type, .as.images = ML2_LayerCacheAsImages(cache).info};
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheClear(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsClear(ML2_LayerCacheAsScalars(cache)); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesClear(ML2_LayerCacheAsImages(cache)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src) {
    switch (dest.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsCopy(ML2_LayerCacheAsScalars(dest), ML2_LayerCacheAsScalars(src)); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesCopy(ML2_LayerCacheAsImages(dest), ML2_LayerCacheAsImages(src)); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, int indent) {
    switch (layer.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsInfoPrint(ML2_LayerCacheInfoAsScalars(layer), indent); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesInfoPrint(ML2_LayerCacheInfoAsImages(layer), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN void ML2_LayerCachePrint(ML2_LayerCache cache, int indent) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsPrint(ML2_LayerCacheAsScalars(cache), indent); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesPrint(ML2_LayerCacheAsImages(cache), indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_FN bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsInfoSame(ML2_LayerCacheInfoAsScalars(a), ML2_LayerCacheInfoAsScalars(b));
        case ML2_LayerCacheTypeImages: return ML2_ImagesInfoSame(ML2_LayerCacheInfoAsImages(a), ML2_LayerCacheInfoAsImages(b));
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

ML2_FN ML2_Arch ML2_ArchNew(int layers, ML2_LayerInfo infos[static layers]) {
    ML2_Arch arch = {layers, ML2_RELIABLE_CALLOC(layers, sizeof(*arch.infos))};
    memcpy(arch.infos, infos, layers * sizeof(*arch.infos));
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
    for (int i = 0; i < arch.layerCount; i++) {
        model.layers[i] = ML2_LayerNew(arch.infos[i]);
    }
    return model;
}

ML2_FN void ML2_ModelDestroy(ML2_Model *model) {
    for (int i = 0; i < model->layerCount; i++) {
        ML2_LayerDestroy(&model->layers[i]);
    }
    ML2_FREE(model->layers);
    *model = (ML2_Model){};
}

ML2_FN void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerRand(model.layers[i], low, high);
    }
}

ML2_FN void ML2_ModelXavierInit(ML2_Model model) {
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerXavierInit(model.layers[i]);
    }
}

ML2_FN void ML2_ModelHeInit(ML2_Model model) {
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerHeInit(model.layers[i]);
    }
}

ML2_FN void ML2_ModelInfoPrint(ML2_Model model, int indent) {
    printf(ML2_INDENT("{\n", indent));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerInfoPrint(ML2_LayerAsInfo(model.layers[i]), indent + ML2_Indentation);
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_ModelPrint(ML2_Model model, int indent) {
    printf(ML2_INDENT("{\n", indent));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerPrint(model.layers[i], indent + ML2_Indentation);
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache) {
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerForward(model.layers[i], modelCache.caches[i], modelCache.caches[i + 1]);
    }
}

ML2_FN void ML2_ModelBackward(ML2_Model model, ML2_ModelCache modelCache) {
    for (int i = modelCache.layerCount - 2; i >= 0; i--) {
        bool cacheBackward = (i > 0) ? true : false;
        ML2_LayerBackward(model.layers[i], modelCache.gradients[i], modelCache.caches[i], modelCache.cachesGradients[i], modelCache.caches[i + 1], modelCache.cachesGradients[i + 1], cacheBackward);
    }
}

ML2_FN bool ML2_ModelGradientDescentOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache) {
    return model.layerCount == modelCache.layerCount - 1;
}

ML2_FN void ML2_ModelGradientDescentOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerGradientDescent gradientDescent) {
    ML2_HARD_ASSERT(ML2_ModelGradientDescentOptimizeCompatible(model, modelCache));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerGradientDescentOptimize(model.layers[i], modelCache.gradients[i], gradientDescent.parameters.learningRate);
    }
}

ML2_FN bool ML2_ModelMomentumOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerMomentum momentum) {
    return model.layerCount == modelCache.layerCount - 1 && model.layerCount == momentum.state.layerCount;
}

ML2_FN void ML2_ModelMomentumOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerMomentum momentum) {
    ML2_HARD_ASSERT(ML2_ModelMomentumOptimizeCompatible(model, modelCache, momentum));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerMomentumOptimize(model.layers[i], modelCache.gradients[i], momentum.parameters.learningRate, momentum.parameters.decayRate, momentum.state.averageLayers[i]);
    }
}

ML2_FN bool ML2_ModelAdagradOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdagrad adagrad) {
    return model.layerCount == modelCache.layerCount - 1 && model.layerCount == adagrad.state.layerCount;
}

ML2_FN void ML2_ModelAdagradOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdagrad adagrad) {
    ML2_HARD_ASSERT(ML2_ModelAdagradOptimizeCompatible(model, modelCache, adagrad));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerAdagradOptimize(model.layers[i], modelCache.gradients[i], adagrad.parameters.learningRate, adagrad.state.squareSumLayers[i]);
    }
}

ML2_FN bool ML2_ModelRMSPropOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerRMSProp RMSProp) {
    return model.layerCount == modelCache.layerCount - 1 && model.layerCount == RMSProp.state.layerCount;
}

ML2_FN void ML2_ModelRMSPropOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerRMSProp RMSProp) {
    ML2_HARD_ASSERT(ML2_ModelRMSPropOptimizeCompatible(model, modelCache, RMSProp));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerRMSPropOptimize(model.layers[i], modelCache.gradients[i], RMSProp.parameters.learningRate, RMSProp.parameters.decayRate, RMSProp.state.squareAverageLayers[i]);
    }
}

ML2_FN bool ML2_ModelAdamOptimizeCompatible(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdam adam) {
    return model.layerCount == modelCache.layerCount - 1 && model.layerCount == adam.state.layerCount;
}

ML2_FN void ML2_ModelAdamOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_OptimizerAdam *adam) {
    ML2_HARD_ASSERT(ML2_ModelAdamOptimizeCompatible(model, modelCache, *adam));
    for (int i = 0; i < model.layerCount; i++) {
        ML2_LayerAdamOptimize(model.layers[i], modelCache.gradients[i], adam->parameters.learningRate, adam->parameters.decayRate1, adam->parameters.decayRate2, adam->state.averageLayers[i], adam->state.squareAverageLayers[i], adam->state.decayingWeight1, adam->state.decayingWeight2);
    }
    adam->state.decayingWeight1 *= adam->parameters.decayRate1;
    adam->state.decayingWeight2 *= adam->parameters.decayRate2;
}

ML2_FN void ML2_ModelOptimize(ML2_Model model, ML2_ModelCache modelCache, ML2_Optimizer *optimizer) {
    switch (optimizer->type) {
        case ML2_OptimizerTypeGradientDescent: ML2_ModelGradientDescentOptimize(model, modelCache, optimizer->as.gradientDescent); break;
        case ML2_OptimizerTypeMomentum: ML2_ModelMomentumOptimize(model, modelCache, optimizer->as.momentum); break;
        case ML2_OptimizerTypeAdagrad: ML2_ModelAdagradOptimize(model, modelCache, optimizer->as.adagrad); break;
        case ML2_OptimizerTypeRMSProp: ML2_ModelRMSPropOptimize(model, modelCache, optimizer->as.RMSProp); break;
        case ML2_OptimizerTypeAdam: ML2_ModelAdamOptimize(model, modelCache, &optimizer->as.adam); break;
        default: ML2_UNREACHABLE("Unknown ML2_OptimizerType");
    }
}

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_FN ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, ML2_BatchInfo info) {
    int modelCacheLayers = arch.layerCount + 1;
    ML2_ModelCache modelCache = {
        .layerCount = modelCacheLayers,
        .caches = ML2_RELIABLE_CALLOC(modelCacheLayers, sizeof(*modelCache.caches)),
        .cachesGradients = ML2_RELIABLE_CALLOC(modelCacheLayers, sizeof(*modelCache.cachesGradients)),
        .gradients = ML2_RELIABLE_CALLOC(modelCacheLayers - 1, sizeof(*modelCache.gradients)),
    };

    for (int i = 0; i < modelCacheLayers - 1; i++) {
        // NOTE: gradients are the same as the model's layers
        modelCache.gradients[i] = ML2_LayerNew(arch.infos[i]);
    }

    modelCache.caches[0] = ML2_LayerCacheNew(info.input);
    modelCache.cachesGradients[0] = ML2_LayerCacheNew(info.input);
    ML2_LayerCacheInfo prevCacheInfo = info.input;
    for (int i = 0; i < modelCacheLayers - 1; i++) {
        ML2_LayerCacheInfo nextCacheInfo = ML2_LayerInfoForward(arch.infos[i], prevCacheInfo);
        modelCache.caches[i + 1] = ML2_LayerCacheNew(nextCacheInfo);
        modelCache.cachesGradients[i + 1] = ML2_LayerCacheNew(nextCacheInfo);
        prevCacheInfo = nextCacheInfo;
    }
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(info.output, prevCacheInfo));
    return modelCache;
}

ML2_FN void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache) {
    for (int i = 0; i < modelCache->layerCount - 1; i++) {
        ML2_LayerDestroy(&modelCache->gradients[i]);
    }

    for (int i = 0; i < modelCache->layerCount; i++) {
        ML2_LayerCacheDestroy(&modelCache->caches[i]);
        ML2_LayerCacheDestroy(&modelCache->cachesGradients[i]);
    }

    ML2_FREE(modelCache->caches);
    ML2_FREE(modelCache->cachesGradients);
    ML2_FREE(modelCache->gradients);

    *modelCache = (ML2_ModelCache){};
}

ML2_FN void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, int indent) {
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        const char cacheStr[] = "Cache";
        const char gradientStr[] = "Gradient";
        const char cacheGradientStr[] = "CacheGradient";
        size_t alignIndent = 0;
        if (sizeof(cacheStr) > alignIndent) alignIndent = sizeof(cacheStr);
        if (sizeof(gradientStr) > alignIndent) alignIndent = sizeof(gradientStr);
        if (sizeof(cacheGradientStr) > alignIndent) alignIndent = sizeof(cacheGradientStr);

        for (int i = 0; i < modelCache.layerCount; i++) {
            printf(ML2_INDENT("%s[%d]: ", indent), cacheStr, i);
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(modelCache.caches[i]), alignIndent - sizeof(cacheStr));
            printf(ML2_INDENT("%s[%d]: ", indent), cacheGradientStr, i);
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(modelCache.cachesGradients[i]), alignIndent - sizeof(cacheGradientStr));
            if (i < modelCache.layerCount - 1) {
                printf(ML2_INDENT("%s[%d]: ", indent), gradientStr, i);
                ML2_LayerInfoPrint(ML2_LayerAsInfo(modelCache.gradients[i]), alignIndent - sizeof(gradientStr));
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN void ML2_ModelCachePrint(ML2_ModelCache modelCache, int indent) {
    printf(ML2_INDENT("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < modelCache.layerCount; i++) {
            printf(ML2_INDENT("Cache[%d]\n", indent), i);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                ML2_LayerCachePrint(modelCache.caches[i], indent);
                indent -= ML2_Indentation;
            }
            printf(ML2_INDENT("}\n", indent));
            printf(ML2_INDENT("CacheGradient[%d]\n", indent), i);
            printf(ML2_INDENT("{\n", indent));
            {
                indent += ML2_Indentation;
                ML2_LayerCachePrint(modelCache.cachesGradients[i], indent);
                indent -= ML2_Indentation;
            }
            printf(ML2_INDENT("}\n", indent));
            if (i < modelCache.layerCount - 1) {
                printf(ML2_INDENT("Gradient[%d]\n", indent), i);
                printf(ML2_INDENT("{\n", indent));
                {
                    indent += ML2_Indentation;
                    ML2_LayerPrint(modelCache.gradients[i], indent);
                    indent -= ML2_Indentation;
                }
                printf(ML2_INDENT("}\n", indent));
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_INDENT("}\n", indent));
}

ML2_FN ML2_LayerCache ML2_ModelCacheInput(ML2_ModelCache modelCache) {
    return modelCache.caches[0];
}

ML2_FN ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache) {
    return modelCache.caches[modelCache.layerCount - 1];
}

ML2_FN ML2_LayerCache ML2_ModelCacheOutputGradient(ML2_ModelCache modelCache) {
    return modelCache.cachesGradients[modelCache.layerCount - 1];
}

ML2_FN void ML2_ModelCacheCopyBatchInput(ML2_ModelCache modelCache, ML2_Batch batch) {
    ML2_LayerCacheCopy(ML2_ModelCacheInput(modelCache), batch.input);
}

ML2_FN ML2_Scalar ML2_ModelCacheLossForward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossForward lossForward) {
    return lossForward(ML2_ModelCacheOutput(modelCache), batch.output);
}

ML2_FN void ML2_ModelCacheLossBackward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossBackward lossBackward) {
    lossBackward(ML2_ModelCacheOutput(modelCache), batch.output, ML2_ModelCacheOutputGradient(modelCache));
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

ML2_FN void ML2_BatchInfoPrint(ML2_Batch batch, int indent) {
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

ML2_FN void ML2_BatchPrint(ML2_Batch batch, int indent) {
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
    for (int i = 0; i < momentum.state.layerCount; i++) {
        momentum.state.averageLayers[i] = ML2_LayerNew(arch.infos[i]);
    }
    return momentum;
}

ML2_FN void ML2_MomentumDestroy(ML2_OptimizerMomentum *momentum) {
    for (int i = 0; i < momentum->state.layerCount; i++) {
        ML2_LayerDestroy(&momentum->state.averageLayers[i]);
    }
    ML2_FREE(momentum->state.averageLayers);
    *momentum = (ML2_OptimizerMomentum){};
}

ML2_FN void ML2_MomentumReset(ML2_OptimizerMomentum momentum) {
    for (int i = 0; i < momentum.state.layerCount; i++) {
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
    for (int i = 0; i < adagrad.state.layerCount; i++) {
        adagrad.state.squareSumLayers[i] = ML2_LayerNew(arch.infos[i]);
    }
    return adagrad;
}

ML2_FN void ML2_AdagradDestroy(ML2_OptimizerAdagrad *adagrad) {
    for (int i = 0; i < adagrad->state.layerCount; i++) {
        ML2_LayerDestroy(&adagrad->state.squareSumLayers[i]);
    }
    ML2_FREE(adagrad->state.squareSumLayers);
    *adagrad = (ML2_OptimizerAdagrad){};
}

ML2_FN void ML2_AdagradReset(ML2_OptimizerAdagrad adagrad) {
    for (int i = 0; i < adagrad.state.layerCount; i++) {
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
    for (int i = 0; i < RMSProp.state.layerCount; i++) {
        RMSProp.state.squareAverageLayers[i] = ML2_LayerNew(arch.infos[i]);
    }
    return RMSProp;
}

ML2_FN void ML2_RMSPropDestroy(ML2_OptimizerRMSProp *RMSProp) {
    for (int i = 0; i < RMSProp->state.layerCount; i++) {
        ML2_LayerDestroy(&RMSProp->state.squareAverageLayers[i]);
    }
    ML2_FREE(RMSProp->state.squareAverageLayers);
    *RMSProp = (ML2_OptimizerRMSProp){};
}

ML2_FN void ML2_RMSPropReset(ML2_OptimizerRMSProp RMSProp) {
    for (int i = 0; i < RMSProp.state.layerCount; i++) {
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
    for (int i = 0; i < adam.state.layerCount; i++) {
        adam.state.averageLayers[i] = ML2_LayerNew(arch.infos[i]);
        adam.state.squareAverageLayers[i] = ML2_LayerNew(arch.infos[i]);
    }
    return adam;
}

ML2_FN void ML2_AdamDestroy(ML2_OptimizerAdam *adam) {
    for (int i = 0; i < adam->state.layerCount; i++) {
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
    for (int i = 0; i < adam->state.layerCount; i++) {
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

// ML2_Optimizer ⬆️

// ML2_Loss ⬇️

ML2_FN ML2_Scalar ML2_LossForwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardSquareAverage works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    ML2_Scalar loss = {};
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar diff = *ML2_ScalarsAt(predictedScalars, i, j) - *ML2_ScalarsAt(expectedScalars, i, j);
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

ML2_FN void ML2_LossBackwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardSquareAverage works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar diff = *ML2_ScalarsAt(predictedScalars, i, j) - *ML2_ScalarsAt(expectedScalars, i, j);
            *ML2_ScalarsAt(gradientScalars, i, j) = ML2_SCALAR_LITERAL(2.0) * diff / (samples * scalars);
        }
    }
}

ML2_FN ML2_Scalar ML2_LossForwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    ML2_Scalar loss = {};
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar predicted = ML2_FMAX(*ML2_ScalarsAt(predictedScalars, i, j), ML2_Epsilon);
            ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, i, j);
            loss -= expected * ML2_LOG(predicted);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_LossBackwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar predicted = ML2_FMAX(*ML2_ScalarsAt(predictedScalars, i, j), ML2_Epsilon);
            ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, i, j);
            *ML2_ScalarsAt(gradientScalars, i, j) = -(expected / predicted) / samples;
        }
    }
}

ML2_FN ML2_Scalar ML2_LossForwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardSoftmaxCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    ML2_Scalar loss = {};
    for (int i = 0; i < samples; i++) {
        ML2_Scalar max = *ML2_ScalarsAt(predictedScalars, i, 0);
        for (int j = 1; j < scalars; j++) {
            ML2_Scalar scalar = *ML2_ScalarsAt(predictedScalars, i, j);
            if (scalar > max) max = scalar;
        }
        ML2_Scalar sum = {};
        for (int j = 0; j < scalars; j++) {
            sum += ML2_EXP(*ML2_ScalarsAt(predictedScalars, i, j) - max);
        }
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar predicted = ML2_FMAX(ML2_EXP(*ML2_ScalarsAt(predictedScalars, i, j) - max) / sum, ML2_Epsilon);
            ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, i, j);
            loss -= expected * ML2_LOG(predicted);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_LossBackwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardSoftmaxCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    for (int i = 0; i < samples; i++) {
        ML2_Scalar max = *ML2_ScalarsAt(predictedScalars, i, 0);
        for (int j = 1; j < scalars; j++) {
            ML2_Scalar scalar = *ML2_ScalarsAt(predictedScalars, i, j);
            if (scalar > max) max = scalar;
        }
        ML2_Scalar sum = {};
        for (int j = 0; j < scalars; j++) {
            sum += ML2_EXP(*ML2_ScalarsAt(predictedScalars, i, j) - max);
        }
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar predicted = ML2_EXP(*ML2_ScalarsAt(predictedScalars, i, j) - max) / sum;
            ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, i, j);
            *ML2_ScalarsAt(gradientScalars, i, j) = (predicted - expected) / samples;
        }
    }
}

ML2_FN ML2_Scalar ML2_LossForwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardBinaryCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    ML2_Scalar loss = {};
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar predicted = ML2_FMIN(ML2_FMAX(*ML2_ScalarsAt(predictedScalars, i, j), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
            ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, i, j);
            loss -= expected * ML2_LOG(predicted) + (ML2_SCALAR_LITERAL(1.0) - expected) * ML2_LOG(ML2_SCALAR_LITERAL(1.0) - predicted);
        }
    }
    loss /= samples;
    return loss;
}

ML2_FN void ML2_LossBackwardBinaryCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardBinaryCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = ML2_LayerCacheAsScalars(predicted);
    ML2_LayerCacheScalars expectedScalars = ML2_LayerCacheAsScalars(expected);
    ML2_LayerCacheScalars gradientScalars = ML2_LayerCacheAsScalars(gradient);
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, predictedScalars.info));
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(expectedScalars.info, gradientScalars.info));
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar predicted = ML2_FMIN(ML2_FMAX(*ML2_ScalarsAt(predictedScalars, i, j), ML2_Epsilon), ML2_SCALAR_LITERAL(1.0) - ML2_Epsilon);
            ML2_Scalar expected = *ML2_ScalarsAt(expectedScalars, i, j);
            *ML2_ScalarsAt(gradientScalars, i, j) = (predicted - expected) / (predicted * (ML2_SCALAR_LITERAL(1.0) - predicted)) / samples;
        }
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

    [ ] Pooling
        MaxPool and AvgPool layers
        CNNs are basically unusable without them

    [ ] Padding
        Conv currently only does valid convolution (output shrinks)
        need same/full padding so output can match input size

    [ ] BatchNorm
        normalizes each layer's output during training
        makes training significantly more stable on deep networks

    [ ] Dropout
        randomly zeros activations during training
        prevents overfitting, ~10 lines to implement

    [ ] Leaky ReLU
        ReLU kills neurons that go negative permanently
        Leaky ReLU lets a small gradient through, fixes dying ReLU problem

    [ ] Model save/load
        currently a trained model lives only in memory
        need to write/read weights to disk

    [ ] Metrics
        accuracy for classification
        nothing else needed, loss alone is not enough to evaluate a model
*/