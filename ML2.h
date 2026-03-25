#ifndef _ML2_H
#define _ML2_H

/*
    this library aims for minimal memory allocations

    *New calls allocate memory, so use *Destroy to free the memory
*/

#ifndef ML2_DEF
    #define ML2_DEF static inline
#endif // ML2_DEF

#include <stdio.h>
#include <string.h>
#include <math.h>

#define ML2_UNREACHABLE(msg, ...)                                        \
    do {                                                                 \
        fprintf(stderr, "UNREACHABLE: \"" msg "\":\n"                    \
                        "    file:     |%s|\n"                           \
                        "    function: |%s|\n"                           \
                        "    line:     |%d|\n",                          \
                __VA_OPT__(__VA_ARGS__,) __FILE__, __func__, __LINE__);  \
        abort();                                                         \
    } while (0)

#define ML2_TODO(msg, ...)                                               \
    do {                                                                 \
        fprintf(stderr, "TODO: \"" msg "\":\n"                           \
                        "    file:     |%s|\n"                           \
                        "    function: |%s|\n"                           \
                        "    line:     |%d|\n",                          \
                __VA_OPT__(__VA_ARGS__,) __FILE__, __func__, __LINE__);  \
        abort();                                                         \
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
            // fputs does not have internal allocations
            fputs("ML2_ReliableCalloc: Failed to allocate memory\n", stderr);
            abort();
        }
        return ptr;
    }
    #define ML2_RELIABLE_CALLOC ML2_ReliableCalloc
#endif // ML2_RELIABLE_CALLOC

#ifndef ML2_Indentation
    #define ML2_Indentation 4
#endif // ML2_Indentation

#define ML2_Indent(fstr, i) "%*s" fstr, (i), ""

// ML2_Scalar ⬇️

// NOTE: Customizable, but make sure to define all of these
#ifndef ML2_SCALAR_TYPE
    #define ML2_SCALAR_TYPE float

    #define ML2_SCALAR_LITERAL(x) x##f
    #define ML2_SQRT sqrtf
    #define ML2_LOG logf
    #define ML2_EXP expf
#endif // ML2_SCALAR_TYPE
typedef ML2_SCALAR_TYPE ML2_Scalar;

// TODO: figure out how to align the floats nicely
#define ML2_ScalarFmt "%.3f"

// ML2_Scalar ⬆️

// ML2_Layer ⬇️

typedef enum {
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

typedef enum {
    ML2_ActivationTypeReLU = 1,
    ML2_ActivationTypeSigmoid,
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

// ML2_LayerFlatten ⬇️

// TODO: Flatten has no state, maybe it shouldnt even have an .as but more context and experimentation is required that i dont have right now
typedef struct {
    int _;
} ML2_LayerInfoFlatten;

typedef struct {
    ML2_LayerInfoFlatten info;
} ML2_LayerFlatten;

// ML2_LayerFlatten ⬆️

typedef struct {
    ML2_LayerType type;
    union {
        ML2_LayerInfoWeights Weights;
        ML2_LayerInfoBiases Biases;
        ML2_LayerInfoActivation Activation;
        ML2_LayerInfoLinear Linear;
        ML2_LayerInfoFilters Filters;
        ML2_LayerInfoConv Conv;
        ML2_LayerInfoFlatten Flatten;
    } as;
} ML2_LayerInfo;

typedef struct {
    ML2_LayerType type;
    union {
        ML2_LayerWeights Weights;
        ML2_LayerBiases Biases;
        ML2_LayerActivation Activation;
        ML2_LayerLinear Linear;
        ML2_LayerFilters Filters;
        ML2_LayerConv Conv;
        ML2_LayerFlatten Flatten;
    } as;
} ML2_Layer;

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

typedef enum {
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
        ML2_LayerCacheInfoScalars Scalars;
        ML2_LayerCacheInfoImages Images;
    } as;
} ML2_LayerCacheInfo;

typedef struct {
    ML2_LayerCacheType type;
    union {
        ML2_LayerCacheScalars Scalars;
        ML2_LayerCacheImages Images;
    } as;
} ML2_LayerCache;

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

typedef struct {
    int layers;
    ML2_LayerInfo *infos;
} ML2_Arch;

// ML2_Arch ⬆️

// ML2_Model ⬇️

typedef struct {
    int count;
    ML2_Layer *layers;
} ML2_Model;

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

typedef struct {
    int layers;
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

// ML2_Loss ⬇️

typedef ML2_Scalar (*ML2_LossForward)(ML2_LayerCache predicted, ML2_LayerCache expected);
typedef void (*ML2_LossBackward)(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

// ML2_Loss ⬆️

// ML2_Scalar ⬇️

ML2_DEF ML2_Scalar ML2_RandScalar(ML2_Scalar low, ML2_Scalar high);

// ML2_Scalar ⬆️

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_DEF ML2_LayerInfo ML2_Weights(int outputs, int inputs);
ML2_DEF ML2_LayerWeights ML2_WeightsNew(ML2_LayerInfoWeights info);
ML2_DEF void ML2_WeightsDestroy(ML2_LayerWeights *weights);
ML2_DEF ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, int i, int j);
ML2_DEF ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer);
ML2_DEF void ML2_WeightsClear(ML2_LayerWeights weights);
ML2_DEF void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_WeightsXavierRand(ML2_LayerWeights weights);
ML2_DEF void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, int indent);
ML2_DEF void ML2_WeightsPrint(ML2_LayerWeights weights, int indent);
ML2_DEF bool ML2_WeightsInfoForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_DEF void ML2_WeightsInfoForwardCompatibleAssert(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_DEF ML2_LayerCacheInfo ML2_WeightsInfoForward(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input);
ML2_DEF bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b);
ML2_DEF void ML2_WeightsInfoSameAssert(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b);
ML2_DEF bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_WeightsForwardCompatibleAssert(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_WeightsBackward(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_WeightsGradientDescent(ML2_LayerWeights weights, ML2_LayerWeights gradientWeights, ML2_Scalar learningRate);

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_DEF ML2_LayerInfo ML2_Biases(int inputs);
ML2_DEF ML2_LayerBiases ML2_BiasesNew(ML2_LayerInfoBiases info);
ML2_DEF void ML2_BiasesDestroy(ML2_LayerBiases *biases);
ML2_DEF ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, int i);
ML2_DEF ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer);
ML2_DEF void ML2_BiasesClear(ML2_LayerBiases biases);
ML2_DEF void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_BiasesXavierRand(ML2_LayerBiases biases);
ML2_DEF void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, int indent);
ML2_DEF void ML2_BiasesPrint(ML2_LayerBiases biases, int indent);
ML2_DEF bool ML2_BiasesInfoForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_DEF void ML2_BiasesInfoForwardCompatibleAssert(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_DEF ML2_LayerCacheInfo ML2_BiasesInfoForward(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input);
ML2_DEF bool ML2_BiasesInfoSame(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b);
ML2_DEF void ML2_BiasesInfoSameAssert(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b);
ML2_DEF bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_BiasesForwardCompatibleAssert(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_BiasesBackward(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_BiasesGradientDescent(ML2_LayerBiases biases, ML2_LayerBiases gradientBiases, ML2_Scalar learningRate);

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

ML2_DEF ML2_LayerInfo ML2_Activation(ML2_ActivationType type);
ML2_DEF ML2_LayerInfo ML2_ActivationReLU();
ML2_DEF ML2_LayerInfo ML2_ActivationSigmoid();
ML2_DEF ML2_LayerInfo ML2_ActivationSoftmax();
ML2_DEF ML2_LayerActivation ML2_ActivationNew(ML2_LayerInfoActivation info);
ML2_DEF void ML2_ActivationDestroy(ML2_LayerActivation *activation);
ML2_DEF ML2_LayerActivation ML2_LayerAsActivation(ML2_Layer layer);
ML2_DEF const char *ML2_ActivationNameOf(ML2_ActivationType type);
ML2_DEF void ML2_ActivationInfoPrint(ML2_LayerInfoActivation info, int indent);
ML2_DEF void ML2_ActivationPrint(ML2_LayerActivation activation, int indent);
// ML2_DEF ML2_LayerCacheInfo ML2_ActivationInfoForward(ML2_LayerInfoActivation activation [[maybe_unused]], ML2_LayerCacheInfo input);
// ML2_DEF bool ML2_ActivationInfoSame(ML2_LayerInfoActivation a, ML2_LayerInfoActivation b);
// ML2_DEF void ML2_ActivationInfoSameAssert(ML2_LayerInfoActivation a, ML2_LayerInfoActivation b);
// ML2_DEF bool ML2_ActivationForwardCompatible(ML2_LayerInfoActivation activation, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
// ML2_DEF void ML2_ActivationForwardCompatibleAssert(ML2_LayerInfoActivation activation, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_ActivationReLUForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_ActivationReLUBackward(ML2_LayerCache input, ML2_LayerCache output, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_ActivationSigmoidForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_ActivationSigmoidBackward(ML2_LayerCache input, ML2_LayerCache output, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_ActivationSoftmaxForward(ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_ActivationSoftmaxBackward(ML2_LayerCache input, ML2_LayerCache output, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_ActivationForward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_ActivationBackward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

ML2_DEF ML2_LayerInfo ML2_Linear(int outputs, int inputs);
ML2_DEF ML2_LayerLinear ML2_LinearNew(ML2_LayerInfoLinear info);
ML2_DEF void ML2_LinearDestroy(ML2_LayerLinear *linear);
ML2_DEF ML2_LayerLinear ML2_LayerAsLinear(ML2_Layer layer);
ML2_DEF ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear);
ML2_DEF ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear);
ML2_DEF void ML2_LinearClear(ML2_LayerLinear linear);
ML2_DEF void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_LinearXavierRand(ML2_LayerLinear linear);
ML2_DEF void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, int indent);
ML2_DEF void ML2_LinearPrint(ML2_LayerLinear linear, int indent);
ML2_DEF bool ML2_LinearInfoForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_DEF void ML2_LinearInfoForwardCompatibleAssert(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_DEF ML2_LayerCacheInfo ML2_LinearInfoForward(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input);
ML2_DEF bool ML2_LinearInfoSame(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b);
ML2_DEF void ML2_LinearInfoSameAssert(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b);
ML2_DEF bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_LinearForwardCompatibleAssert(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_LinearBackward(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_LinearGradientDescent(ML2_LayerLinear linear, ML2_LayerLinear gradientLinear, ML2_Scalar learningRate);

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_DEF ML2_LayerInfo ML2_Filters(int outputs, int inputs, int height, int width);
ML2_DEF ML2_LayerFilters ML2_FiltersNew(ML2_LayerInfoFilters info);
ML2_DEF void ML2_FiltersDestroy(ML2_LayerFilters *filters);
ML2_DEF ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, int i, int j, int k, int l);
ML2_DEF ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer);
ML2_DEF void ML2_FiltersClear(ML2_LayerFilters filters);
ML2_DEF void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_FiltersXavierRand(ML2_LayerFilters filters);
ML2_DEF void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, int indent);
ML2_DEF void ML2_FiltersPrint(ML2_LayerFilters filters, int indent);
ML2_DEF bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_DEF void ML2_FiltersInfoForwardCompatibleAssert(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_DEF ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input);
ML2_DEF bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b);
ML2_DEF void ML2_FiltersInfoSameAssert(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b);
ML2_DEF bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_FiltersForwardCompatibleAssert(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_FiltersBackward(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_FiltersGradientDescent(ML2_LayerFilters filters, ML2_LayerFilters gradientFilters, ML2_Scalar learningRate);

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_DEF ML2_LayerInfo ML2_Conv(int inputs, int outputs, int height, int width);
ML2_DEF ML2_LayerConv ML2_ConvNew(ML2_LayerInfoConv info);
ML2_DEF void ML2_ConvDestroy(ML2_LayerConv *conv);
ML2_DEF ML2_LayerConv ML2_LayerAsConv(ML2_Layer layer);
ML2_DEF ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv);
ML2_DEF ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv);
ML2_DEF void ML2_ConvClear(ML2_LayerConv conv);
ML2_DEF void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_ConvXavierRand(ML2_LayerConv conv);
ML2_DEF void ML2_ConvInfoPrint(ML2_LayerInfoConv info, int indent);
ML2_DEF void ML2_ConvPrint(ML2_LayerConv conv, int indent);
ML2_DEF bool ML2_ConvInfoForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_DEF void ML2_ConvInfoForwardCompatibleAssert(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_DEF ML2_LayerCacheInfo ML2_ConvInfoForward(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input);
ML2_DEF bool ML2_ConvInfoSame(ML2_LayerInfoConv a, ML2_LayerInfoConv b);
ML2_DEF void ML2_ConvInfoSameAssert(ML2_LayerInfoConv a, ML2_LayerInfoConv b);
ML2_DEF bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_ConvForwardCompatibleAssert(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_ConvBackward(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);
ML2_DEF void ML2_ConvGradientDescent(ML2_LayerConv conv, ML2_LayerConv gradientLinear, ML2_Scalar learningRate);

// ML2_LayerConv ⬆️

// ML2_LayerFlatten ⬇️

ML2_DEF ML2_LayerInfo ML2_Flatten();
ML2_DEF ML2_LayerFlatten ML2_FlattenNew(ML2_LayerInfoFlatten info);
ML2_DEF void ML2_FlattenDestroy(ML2_LayerFlatten *flatten);
ML2_DEF void ML2_FlattenInfoPrint(ML2_LayerInfoFlatten flatten, int indent);
ML2_DEF void ML2_FlattenPrint(ML2_LayerFlatten flatten, int indent);
ML2_DEF bool ML2_FlattenInfoForwardCompatible(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input);
ML2_DEF void ML2_FlattenInfoForwardCompatibleAssert(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input);
ML2_DEF ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input);
ML2_DEF bool ML2_FlattenForwardCompatible(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_FlattenForwardCompatibleAssert(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_FlattenForward(ML2_LayerFlatten flatten, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_FlattenBackward(ML2_LayerFlatten flatten, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient);

// ML2_LayerFlatten ⬆️

ML2_DEF ML2_Layer ML2_LayerNew(ML2_LayerInfo info);
ML2_DEF void ML2_LayerDestroy(ML2_Layer *layer);
ML2_DEF ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer);
ML2_DEF void ML2_LayerClear(ML2_Layer layer);
ML2_DEF void ML2_LayerRand(ML2_Layer layer, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_LayerXavierRand(ML2_Layer layer);
ML2_DEF void ML2_LayerInfoPrint(ML2_LayerInfo layer, int indent);
ML2_DEF void ML2_LayerPrint(ML2_Layer layer, int indent);
ML2_DEF ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input);
// TODO(4/3/2026 6:52:54): these 3 functions below are never used, the their subversions are
// ML2_DEF bool ML2_LayerInfoSame(ML2_LayerInfo a, ML2_LayerInfo b);
// ML2_DEF void ML2_LayerInfoSameAssert(ML2_LayerInfo a, ML2_LayerInfo b);
// ML2_DEF bool ML2_LayerInfoForwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output);
ML2_DEF void ML2_LayerBackward(ML2_Layer layer, ML2_Layer layerGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient);
ML2_DEF void ML2_LayerGradientDescent(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate);

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

// ML2_LayerCacheScalars ⬇️

ML2_DEF ML2_LayerCacheInfo ML2_Scalars(int samples, int scalars);
ML2_DEF ML2_LayerCacheScalars ML2_ScalarsNew(ML2_LayerCacheInfoScalars info);
ML2_DEF void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars);
ML2_DEF ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, int sample, int scalar);
ML2_DEF ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache);
ML2_DEF void ML2_ScalarsClear(ML2_LayerCacheScalars scalars);
ML2_DEF void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src);
ML2_DEF void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, int indent);
ML2_DEF void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, int indent);
ML2_DEF bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b);
ML2_DEF void ML2_ScalarsInfoSameAssert(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b);

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheImages ⬇️

ML2_DEF ML2_LayerCacheInfo ML2_Images(int samples, int channels, int height, int width);
ML2_DEF ML2_LayerCacheImages ML2_ImagesNew(ML2_LayerCacheInfoImages info);
ML2_DEF void ML2_ImagesDestroy(ML2_LayerCacheImages *images);
ML2_DEF ML2_Scalar *ML2_ImagesAt(ML2_LayerCacheImages images, int sample, int channel, int y, int x);
ML2_DEF ML2_LayerCacheImages ML2_LayerCacheAsImages(ML2_LayerCache cache);
ML2_DEF void ML2_ImagesClear(ML2_LayerCacheImages images);
ML2_DEF void ML2_ImagesCopy(ML2_LayerCacheImages dest, ML2_LayerCacheImages src);
ML2_DEF void ML2_ImagesInfoPrint(ML2_LayerCacheInfoImages info, int indent);
ML2_DEF void ML2_ImagesPrint(ML2_LayerCacheImages images, int indent);
ML2_DEF bool ML2_ImagesInfoSame(ML2_LayerCacheInfoImages a, ML2_LayerCacheInfoImages b);
ML2_DEF void ML2_ImagesInfoSameAssert(ML2_LayerCacheInfoImages a, ML2_LayerCacheInfoImages b);

// ML2_LayerCacheImages ⬆️

ML2_DEF ML2_LayerCache ML2_LayerCacheNew(ML2_LayerCacheInfo info);
ML2_DEF void ML2_LayerCacheDestroy(ML2_LayerCache *cache);
ML2_DEF ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache);
ML2_DEF void ML2_LayerCacheClear(ML2_LayerCache cache);
ML2_DEF void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src);
ML2_DEF void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, int indent);
ML2_DEF void ML2_LayerCachePrint(ML2_LayerCache cache, int indent);
ML2_DEF bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b);
ML2_DEF void ML2_LayerCacheInfoSameAssert(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b);

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

ML2_DEF ML2_Arch ML2_ArchNew(int layers, ML2_LayerInfo infos[layers]);
ML2_DEF void ML2_ArchDestroy(ML2_Arch *arch);

// ML2_Arch ⬆️

// ML2_Model ⬇️

ML2_DEF ML2_Model ML2_ModelNew(ML2_Arch arch);
ML2_DEF void ML2_ModelDestroy(ML2_Model *model);
ML2_DEF void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_ModelXavierRand(ML2_Model model);
ML2_DEF void ML2_ModelInfoPrint(ML2_Model model, int indent);
ML2_DEF void ML2_ModelPrint(ML2_Model model, int indent);
ML2_DEF void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache);
ML2_DEF void ML2_ModelBackward(ML2_Model model, ML2_ModelCache modelCache);
ML2_DEF void ML2_ModelGradientDescent(ML2_Model model, ML2_ModelCache modelCache, ML2_Scalar learningRate);

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_DEF ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, ML2_BatchInfo info);
ML2_DEF void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache);
ML2_DEF void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, int indent);
ML2_DEF void ML2_ModelCachePrint(ML2_ModelCache modelCache, int indent);
ML2_DEF ML2_LayerCache ML2_ModelCacheInput(ML2_ModelCache modelCache);
ML2_DEF ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache);
ML2_DEF ML2_LayerCache ML2_ModelCacheOutputGradient(ML2_ModelCache modelCache);
ML2_DEF void ML2_ModelCacheCopyBatchInput(ML2_ModelCache modelCache, ML2_Batch batch);
ML2_DEF ML2_Scalar ML2_ModelCacheLossForward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossForward lossForward);
ML2_DEF void ML2_ModelCacheLossBackward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossBackward lossBackward);
ML2_DEF void ML2_ModelCacheClearGradients(ML2_ModelCache modelCache);

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_DEF ML2_BatchInfo ML2_BatchInfoMake(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output);
ML2_DEF ML2_Batch ML2_BatchNew(ML2_BatchInfo info);
ML2_DEF void ML2_BatchDestroy(ML2_Batch *batch);
ML2_DEF void ML2_BatchInfoPrint(ML2_Batch batch, int indent);
ML2_DEF void ML2_BatchPrint(ML2_Batch batch, int indent);

// ML2_Batch ⬆️

// ML2_Loss ⬇️

ML2_DEF ML2_Scalar ML2_LossForwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_DEF void ML2_LossBackwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);
ML2_DEF ML2_Scalar ML2_LossForwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_DEF void ML2_LossBackwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);
ML2_DEF ML2_Scalar ML2_LossForwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected);
ML2_DEF void ML2_LossBackwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient);

// ML2_Loss ⬆️

#endif // _ML2_H

#ifdef ML2_IMPLEMENTATION

// ML2_ ⬇️
// ML2_ ⬆️

// ML2_Scalar ⬇️

ML2_DEF ML2_Scalar ML2_RandScalar(ML2_Scalar low, ML2_Scalar high) {
    return ((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX) * (high - low) + low;
}

// ML2_Scalar ⬆️

// ML2_Layer ⬇️

// ML2_LayerWeights ⬇️

ML2_DEF ML2_LayerInfo ML2_Weights(int outputs, int inputs) {
    return (ML2_LayerInfo){ML2_LayerTypeWeights, .as.Weights = {outputs, inputs}};
}

ML2_DEF ML2_LayerWeights ML2_WeightsNew(ML2_LayerInfoWeights info) {
    ML2_LayerWeights weights = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs, sizeof(*weights.data.weights))}
    };
    return weights;
}

ML2_DEF void ML2_WeightsDestroy(ML2_LayerWeights *weights) {
    ML2_FREE(weights->data.weights);
    *weights = (ML2_LayerWeights){};
}

ML2_DEF ML2_Scalar *ML2_WeightsAt(ML2_LayerWeights weights, int i, int j) {
    ML2_SOFT_ASSERT(0 <= i && i < weights.info.outputs && 0 <= j && j < weights.info.inputs && "OUT OF BOUNDS INDICES");
    return &weights.data.weights[i * weights.info.inputs + j];
}

ML2_DEF ML2_LayerWeights ML2_LayerAsWeights(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeWeights);
    return layer.as.Weights;
}

ML2_DEF void ML2_WeightsClear(ML2_LayerWeights weights) {
    for (int i = 0; i < weights.info.outputs; i++) {
        for (int j = 0; j < weights.info.inputs; j++) {
            *ML2_WeightsAt(weights, i, j) = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_DEF void ML2_WeightsRand(ML2_LayerWeights weights, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < weights.info.outputs; i++) {
        for (int j = 0; j < weights.info.inputs; j++) {
            *ML2_WeightsAt(weights, i, j) = ML2_RandScalar(low, high);
        }
    }
}

ML2_DEF void ML2_WeightsXavierRand(ML2_LayerWeights weights) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / (weights.info.inputs + weights.info.outputs));
    ML2_WeightsRand(weights, -limit, limit);
}

ML2_DEF void ML2_WeightsInfoPrint(ML2_LayerInfoWeights info, int indent) {
    printf(ML2_Indent("Weights(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_DEF void ML2_WeightsPrint(ML2_LayerWeights weights, int indent) {
    ML2_WeightsInfoPrint(weights.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int j = 0; j < weights.info.inputs; j++) {
            printf(ML2_Indent("", indent));
            for (int i = 0; i < weights.info.outputs; i++) {
                printf(ML2_ScalarFmt " ", *ML2_WeightsAt(weights, i, j));
            }
            printf("\n");
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF bool ML2_WeightsInfoForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return weights.inputs == input.as.Scalars.scalars;
        }
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_WeightsInfoForwardCompatibleAssert(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_WeightsInfoForwardCompatible(weights, input) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH WEIGHTS");
}

ML2_DEF ML2_LayerCacheInfo ML2_WeightsInfoForward(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input) {
    ML2_WeightsInfoForwardCompatibleAssert(weights, input);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return (ML2_LayerCacheInfo){input.type, .as.Scalars = {input.as.Scalars.samples, weights.outputs}};
        }
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF bool ML2_WeightsInfoSame(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b) {
    return a.inputs == b.inputs && a.outputs == b.outputs;
}

ML2_DEF void ML2_WeightsInfoSameAssert(ML2_LayerInfoWeights a, ML2_LayerInfoWeights b) {
    ML2_HARD_ASSERT(ML2_WeightsInfoSame(a, b) && "WEIGHTS MUST HAVE THE SAME INFO");
}

ML2_DEF bool ML2_WeightsForwardCompatible(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheInfoScalars outputScalars = output.as.Scalars;
            return weights.inputs == inputScalars.scalars && weights.outputs == outputScalars.scalars && inputScalars.samples == outputScalars.samples;
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Weights may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_WeightsForwardCompatibleAssert(ML2_LayerInfoWeights weights, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    ML2_HARD_ASSERT(ML2_WeightsForwardCompatible(weights, input, output) && "LAYERS MUST BE FORWARD COMPATIBLE WITH WEIGHTS");
}

ML2_DEF void ML2_WeightsForward(ML2_LayerWeights weights, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_WeightsForwardCompatibleAssert(weights.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = output.as.Scalars;
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

ML2_DEF void ML2_WeightsBackward(ML2_LayerWeights weights, ML2_LayerWeights weightsGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_WeightsInfoSameAssert(weights.info, weightsGradient.info);
    ML2_WeightsForwardCompatibleAssert(weights.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsInfoSameAssert(inputScalars.info, inputGradientScalars.info);
            int samples = inputScalars.info.samples;
            int outputs = weights.info.outputs;
            int inputs = weights.info.inputs;
            // input: samples x inputs
            // inputGradient: samples x inputs
            // weights: outputs x inputs
            // weightsGradient: outputs x inputs
            // outputGradient: samples x outputs

            for (int i = 0; i < samples; i++) {
                for (int k = 0; k < outputs; k++) {
                    for (int j = 0; j < inputs; j++) {
                        *ML2_WeightsAt(weightsGradient, k, j) += *ML2_ScalarsAt(outputGradientScalars, i, k) * *ML2_ScalarsAt(inputScalars, i, j);
                    }
                }
            }

            ML2_ScalarsClear(inputGradientScalars);
            for (int i = 0; i < samples; i++) {
                for (int k = 0; k < outputs; k++) {
                    for (int j = 0; j < inputs; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) += *ML2_ScalarsAt(outputGradientScalars, i, k) * *ML2_WeightsAt(weights, k, j);
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

ML2_DEF void ML2_WeightsGradientDescent(ML2_LayerWeights weights, ML2_LayerWeights gradientWeights, ML2_Scalar learningRate) {
    ML2_WeightsInfoSameAssert(weights.info, gradientWeights.info);
    for (int i = 0; i < weights.info.outputs; i++) {
        for (int j = 0; j < weights.info.inputs; j++) {
            *ML2_WeightsAt(weights, i, j) -= *ML2_WeightsAt(gradientWeights, i, j) * learningRate;
        }
    }
}

// ML2_LayerWeights ⬆️

// ML2_LayerBiases ⬇️

ML2_DEF ML2_LayerInfo ML2_Biases(int inputs) {
    return (ML2_LayerInfo){ML2_LayerTypeBiases, .as.Biases = {inputs}};
}

ML2_DEF ML2_LayerBiases ML2_BiasesNew(ML2_LayerInfoBiases info) {
    ML2_LayerBiases biases = {
        info,
        {ML2_RELIABLE_CALLOC(info.inputs, sizeof(*biases.data.biases))}
    };
    return biases;
}

ML2_DEF void ML2_BiasesDestroy(ML2_LayerBiases *biases) {
    ML2_FREE(biases->data.biases);
    *biases = (ML2_LayerBiases){};
}

ML2_DEF ML2_Scalar *ML2_BiasesAt(ML2_LayerBiases biases, int i) {
    ML2_SOFT_ASSERT(0 <= i && i < biases.info.inputs && "OUT OF BOUNDS INDICES");
    return &biases.data.biases[i];
}

ML2_DEF ML2_LayerBiases ML2_LayerAsBiases(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeBiases);
    return layer.as.Biases;
}

ML2_DEF void ML2_BiasesClear(ML2_LayerBiases biases) {
    for (int i = 0; i < biases.info.inputs; i++) {
        *ML2_BiasesAt(biases, i) = ML2_SCALAR_LITERAL(0.0);
    }
}

ML2_DEF void ML2_BiasesRand(ML2_LayerBiases biases, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < biases.info.inputs; i++) {
        biases.data.biases[i] = ML2_RandScalar(low, high);
    }
}

ML2_DEF void ML2_BiasesXavierRand(ML2_LayerBiases biases) {
    ML2_BiasesClear(biases);
}

ML2_DEF void ML2_BiasesInfoPrint(ML2_LayerInfoBiases info, int indent) {
    printf(ML2_Indent("Biases(%d)\n", indent), info.inputs);
}

ML2_DEF void ML2_BiasesPrint(ML2_LayerBiases biases, int indent) {
    ML2_BiasesInfoPrint(biases.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_Indent("", indent));
        for (int i = 0; i < biases.info.inputs; i++) {
            printf(ML2_ScalarFmt " ", *ML2_BiasesAt(biases, i));
        }
        printf("\n");
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF bool ML2_BiasesInfoForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return biases.inputs == input.as.Scalars.scalars;
        }
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Biases may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_BiasesInfoForwardCompatibleAssert(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_BiasesInfoForwardCompatible(biases, input) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH BIASES");
}

ML2_DEF ML2_LayerCacheInfo ML2_BiasesInfoForward(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input) {
    ML2_BiasesInfoForwardCompatibleAssert(biases, input);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return input;
        }
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Biases may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF bool ML2_BiasesInfoSame(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b) {
    return a.inputs == b.inputs;
}

ML2_DEF void ML2_BiasesInfoSameAssert(ML2_LayerInfoBiases a, ML2_LayerInfoBiases b) {
    ML2_HARD_ASSERT(ML2_BiasesInfoSame(a, b) && "BIASES MUST HAVE THE SAME INFO");
}

ML2_DEF bool ML2_BiasesForwardCompatible(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheInfoScalars outputScalars = output.as.Scalars;
            return biases.inputs == outputScalars.scalars && ML2_ScalarsInfoSame(inputScalars, outputScalars);
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Biases may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_BiasesForwardCompatibleAssert(ML2_LayerInfoBiases biases, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    ML2_HARD_ASSERT(ML2_BiasesForwardCompatible(biases, input, output) && "LAYERS MUST BE FORWARD COMPATIBLE WITH BIASES");
}

ML2_DEF void ML2_BiasesForward(ML2_LayerBiases biases, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_BiasesForwardCompatibleAssert(biases.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = output.as.Scalars;
            int samples = inputScalars.info.samples;
            int inputs = biases.info.inputs;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < inputs; j++) {
                    *ML2_ScalarsAt(outputScalars, i, j) = *ML2_ScalarsAt(inputScalars, i, j) + *ML2_BiasesAt(biases, j);
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Biases may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_BiasesBackward(ML2_LayerBiases biases, ML2_LayerBiases biasesGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_BiasesInfoSameAssert(biases.info, biasesGradient.info);
    ML2_BiasesForwardCompatibleAssert(biases.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsInfoSameAssert(inputScalars.info, inputGradientScalars.info);
            int samples = inputScalars.info.samples;
            int inputs = biases.info.inputs;
            // input: samples x inputs
            // inputGradient: samples x inputs
            // biases: inputs
            // biasesGradient: inputs
            // outputGradient: samples x inputs

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < inputs; j++) {
                    *ML2_BiasesAt(biasesGradient, j) += *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
            }

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < inputs; j++) {
                    *ML2_ScalarsAt(inputGradientScalars, i, j) = *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Biases may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_BiasesGradientDescent(ML2_LayerBiases biases, ML2_LayerBiases gradientBiases, ML2_Scalar learningRate) {
    ML2_BiasesInfoSameAssert(biases.info, gradientBiases.info);
    for (int i = 0; i < biases.info.inputs; i++) {
        *ML2_BiasesAt(biases, i) -= *ML2_BiasesAt(gradientBiases, i) * learningRate;
    }
}

// ML2_LayerBiases ⬆️

// ML2_LayerActivation ⬇️

ML2_DEF ML2_LayerInfo ML2_Activation(ML2_ActivationType type) {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.Activation = {type}};
}

ML2_DEF ML2_LayerInfo ML2_ActivationReLU() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.Activation = {ML2_ActivationTypeReLU}};
}

ML2_DEF ML2_LayerInfo ML2_ActivationSigmoid() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.Activation = {ML2_ActivationTypeSigmoid}};
}

ML2_DEF ML2_LayerInfo ML2_ActivationSoftmax() {
    return (ML2_LayerInfo){ML2_LayerTypeActivation, .as.Activation = {ML2_ActivationTypeSoftmax}};
}

ML2_DEF ML2_LayerActivation ML2_ActivationNew(ML2_LayerInfoActivation info) {
    return (ML2_LayerActivation){info};
}

ML2_DEF void ML2_ActivationDestroy(ML2_LayerActivation *activation) {
    *activation = (ML2_LayerActivation){};
}

ML2_DEF ML2_LayerActivation ML2_LayerAsActivation(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeActivation);
    return layer.as.Activation;
}

ML2_DEF const char *ML2_ActivationNameOf(ML2_ActivationType type) {
    switch (type) {
        case ML2_ActivationTypeReLU: return "ReLU";
        case ML2_ActivationTypeSigmoid: return "Sigmoid";
        case ML2_ActivationTypeSoftmax: return "Softmax";
        default: ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_DEF void ML2_ActivationInfoPrint(ML2_LayerInfoActivation info, int indent) {
    printf(ML2_Indent("Activation(\"%s\")\n", indent), ML2_ActivationNameOf(info.type));
}

ML2_DEF void ML2_ActivationPrint(ML2_LayerActivation activation, int indent) {
    ML2_ActivationInfoPrint(activation.info, indent);
}

// NOTE: this only verifies that input has a valid type
// ML2_DEF ML2_LayerCacheInfo ML2_ActivationInfoForward(ML2_LayerInfoActivation activation [[maybe_unused]], ML2_LayerCacheInfo input) {
//     switch (input.type) {
//         case ML2_LayerCacheTypeScalars:
//         case ML2_LayerCacheTypeImages: return input;
//         default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
//     }
// }

// ML2_DEF bool ML2_ActivationInfoSame(ML2_LayerInfoActivation a, ML2_LayerInfoActivation b) {
//     return a.type == b.type;
// }

// ML2_DEF void ML2_ActivationInfoSameAssert(ML2_LayerInfoActivation a, ML2_LayerInfoActivation b) {
//     ML2_HARD_ASSERT(ML2_ActivationInfoSame(a, b) && "ACTIVATIONS MUST HAVE THE SAME INFO");
// }

// ML2_DEF bool ML2_ActivationForwardCompatible(ML2_LayerInfoActivation activation [[maybe_unused]], ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
//     return ML2_LayerCacheInfoSame(input, output);
// }

// ML2_DEF void ML2_ActivationForwardCompatibleAssert(ML2_LayerInfoActivation activation, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
//     ML2_HARD_ASSERT(ML2_ActivationForwardCompatible(activation, input, output) && "LAYERS MUST BE FORWARD COMPATIBLE WITH ACTIVATION");
// }

ML2_DEF void ML2_ActivationReLUForward(ML2_LayerCache input, ML2_LayerCache output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_ScalarsInfoSameAssert(inputScalars.info, outputScalars.info);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    ML2_Scalar relu = *ML2_ScalarsAt(inputScalars, i, j) > ML2_SCALAR_LITERAL(0.0) ? *ML2_ScalarsAt(inputScalars, i, j) : ML2_SCALAR_LITERAL(0.0);
                    *ML2_ScalarsAt(outputScalars, i, j) = relu;
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = input.as.Images;
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            ML2_ImagesInfoSameAssert(inputImages.info, outputImages.info);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < height; k++) {
                        for (int l = 0; l < width; l++) {
                            ML2_Scalar relu = *ML2_ImagesAt(inputImages, i, j, k, l) > ML2_SCALAR_LITERAL(0.0) ? *ML2_ImagesAt(inputImages, i, j, k, l) : ML2_SCALAR_LITERAL(0.0);
                            *ML2_ImagesAt(outputImages, i, j, k, l) = relu;
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ActivationReLUBackward(ML2_LayerCache input, ML2_LayerCache output, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsInfoSameAssert(inputScalars.info, outputScalars.info);
            ML2_ScalarsInfoSameAssert(inputGradientScalars.info, outputGradientScalars.info);
            ML2_ScalarsInfoSameAssert(inputScalars.info, inputGradientScalars.info);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    ML2_Scalar reluDerivative = *ML2_ScalarsAt(inputScalars, i, j) > ML2_SCALAR_LITERAL(0.0) ? ML2_SCALAR_LITERAL(1.0) : ML2_SCALAR_LITERAL(0.0);
                    *ML2_ScalarsAt(inputGradientScalars, i, j) = reluDerivative * *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = input.as.Images;
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            ML2_ImagesInfoSameAssert(inputImages.info, outputImages.info);
            ML2_ImagesInfoSameAssert(inputGradientImages.info, outputGradientImages.info);
            ML2_ImagesInfoSameAssert(inputImages.info, inputGradientImages.info);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < height; k++) {
                        for (int l = 0; l < width; l++) {
                            ML2_Scalar reluDerivative = *ML2_ImagesAt(inputImages, i, j, k, l) > ML2_SCALAR_LITERAL(0.0) ? ML2_SCALAR_LITERAL(1.0) : ML2_SCALAR_LITERAL(0.0);
                            *ML2_ImagesAt(inputGradientImages, i, j, k, l) = reluDerivative * *ML2_ImagesAt(outputGradientImages, i, j, k, l);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ActivationSigmoidForward(ML2_LayerCache input, ML2_LayerCache output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_ScalarsInfoSameAssert(inputScalars.info, outputScalars.info);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    ML2_Scalar sigmoid = ML2_SCALAR_LITERAL(1.0) / (ML2_SCALAR_LITERAL(1.0) + ML2_EXP(-*ML2_ScalarsAt(inputScalars, i, j)));
                    *ML2_ScalarsAt(outputScalars, i, j) = sigmoid;
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = input.as.Images;
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            ML2_ImagesInfoSameAssert(inputImages.info, outputImages.info);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < height; k++) {
                        for (int l = 0; l < width; l++) {
                            ML2_Scalar sigmoid = ML2_SCALAR_LITERAL(1.0) / (ML2_SCALAR_LITERAL(1.0) + ML2_EXP(-*ML2_ImagesAt(inputImages, i, j, k, l)));
                            *ML2_ImagesAt(outputImages, i, j, k, l) = sigmoid;
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ActivationSigmoidBackward(ML2_LayerCache input, ML2_LayerCache output, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsInfoSameAssert(inputScalars.info, outputScalars.info);
            ML2_ScalarsInfoSameAssert(inputGradientScalars.info, outputGradientScalars.info);
            ML2_ScalarsInfoSameAssert(inputScalars.info, inputGradientScalars.info);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < scalars; j++) {
                    ML2_Scalar sigmoidDerivative = *ML2_ScalarsAt(outputScalars, i, j) * (ML2_SCALAR_LITERAL(1.0) - *ML2_ScalarsAt(outputScalars, i, j));
                    *ML2_ScalarsAt(inputGradientScalars, i, j) = sigmoidDerivative * *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = input.as.Images;
            ML2_LayerCacheImages outputImages = ML2_LayerCacheAsImages(output);
            ML2_LayerCacheImages inputGradientImages = ML2_LayerCacheAsImages(inputGradient);
            ML2_LayerCacheImages outputGradientImages = ML2_LayerCacheAsImages(outputGradient);
            ML2_ImagesInfoSameAssert(inputImages.info, outputImages.info);
            ML2_ImagesInfoSameAssert(inputGradientImages.info, outputGradientImages.info);
            ML2_ImagesInfoSameAssert(inputImages.info, inputGradientImages.info);
            int samples = inputImages.info.samples;
            int channels = inputImages.info.channels;
            int height = inputImages.info.height;
            int width = inputImages.info.width;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < height; k++) {
                        for (int l = 0; l < width; l++) {
                            ML2_Scalar sigmoidDerivative = *ML2_ImagesAt(outputImages, i, j, k, l) * (ML2_SCALAR_LITERAL(1.0) - *ML2_ImagesAt(outputImages, i, j, k, l));
                            *ML2_ImagesAt(inputGradientImages, i, j, k, l) = sigmoidDerivative * *ML2_ImagesAt(outputGradientImages, i, j, k, l);
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ActivationSoftmaxForward(ML2_LayerCache input, ML2_LayerCache output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_ScalarsInfoSameAssert(inputScalars.info, outputScalars.info);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

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

ML2_DEF void ML2_ActivationSoftmaxBackward(ML2_LayerCache input, ML2_LayerCache output, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = ML2_LayerCacheAsScalars(output);
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsInfoSameAssert(inputScalars.info, outputScalars.info);
            ML2_ScalarsInfoSameAssert(inputGradientScalars.info, outputGradientScalars.info);
            ML2_ScalarsInfoSameAssert(inputScalars.info, inputGradientScalars.info);
            int samples = inputScalars.info.samples;
            int scalars = inputScalars.info.scalars;

            for (int i = 0; i < samples; i++) {
                ML2_Scalar dot = {};
                for (int j = 0; j < scalars; j++) {
                    dot += *ML2_ScalarsAt(outputScalars, i, j) * *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
                for (int j = 0; j < scalars; j++) {
                    *ML2_ScalarsAt(inputGradientScalars, i, j) = *ML2_ScalarsAt(outputScalars, i, j) * (*ML2_ScalarsAt(outputGradientScalars, i, j) - dot);
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Softmax may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ActivationForward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache output) {
    switch (activation.info.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationReLUForward(input, output); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationSigmoidForward(input, output); break;
        case ML2_ActivationTypeSoftmax: ML2_ActivationSoftmaxForward(input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

ML2_DEF void ML2_ActivationBackward(ML2_LayerActivation activation, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    switch (activation.info.type) {
        case ML2_ActivationTypeReLU: ML2_ActivationReLUBackward(input, output, inputGradient, outputGradient); break;
        case ML2_ActivationTypeSigmoid: ML2_ActivationSigmoidBackward(input, output, inputGradient, outputGradient); break;
        case ML2_ActivationTypeSoftmax: ML2_ActivationSoftmaxBackward(input, output, inputGradient, outputGradient); break;
        default: ML2_UNREACHABLE("Unknown ML2_ActivationType");
    }
}

// ML2_LayerActivation ⬆️

// ML2_LayerLinear ⬇️

ML2_DEF ML2_LayerInfo ML2_Linear(int outputs, int inputs) {
    return (ML2_LayerInfo){ML2_LayerTypeLinear, .as.Linear = {outputs, inputs}};
}

ML2_DEF ML2_LayerLinear ML2_LinearNew(ML2_LayerInfoLinear info) {
    ML2_LayerLinear linear = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs, sizeof(*linear.data.weights)),
         ML2_RELIABLE_CALLOC(info.outputs, sizeof(*linear.data.biases))}
    };
    return linear;
}

ML2_DEF void ML2_LinearDestroy(ML2_LayerLinear *linear) {
    ML2_FREE(linear->data.weights);
    ML2_FREE(linear->data.biases);
    *linear = (ML2_LayerLinear){};
}

ML2_DEF ML2_LayerLinear ML2_LayerAsLinear(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeLinear);
    return layer.as.Linear;
}

ML2_DEF ML2_LayerWeights ML2_LinearWeights(ML2_LayerLinear linear) {
    return (ML2_LayerWeights){{linear.info.outputs, linear.info.inputs}, {linear.data.weights}};
}

ML2_DEF ML2_LayerBiases ML2_LinearBiases(ML2_LayerLinear linear) {
    return (ML2_LayerBiases){{linear.info.outputs}, {linear.data.biases}};
}

ML2_DEF void ML2_LinearClear(ML2_LayerLinear linear) {
    ML2_WeightsClear(ML2_LinearWeights(linear));
    ML2_BiasesClear(ML2_LinearBiases(linear));
}

ML2_DEF void ML2_LinearRand(ML2_LayerLinear linear, ML2_Scalar low, ML2_Scalar high) {
    ML2_WeightsRand(ML2_LinearWeights(linear), low, high);
    ML2_BiasesRand(ML2_LinearBiases(linear), low, high);
}

ML2_DEF void ML2_LinearXavierRand(ML2_LayerLinear linear) {
    ML2_WeightsXavierRand(ML2_LinearWeights(linear));
    ML2_BiasesXavierRand(ML2_LinearBiases(linear));
}

ML2_DEF void ML2_LinearInfoPrint(ML2_LayerInfoLinear info, int indent) {
    printf(ML2_Indent("Linear(%dx%d)\n", indent), info.outputs, info.inputs);
}

ML2_DEF void ML2_LinearPrint(ML2_LayerLinear linear, int indent) {
    ML2_LinearInfoPrint(linear.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        ML2_WeightsPrint(ML2_LinearWeights(linear), indent);
        ML2_BiasesPrint(ML2_LinearBiases(linear), indent);
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF bool ML2_LinearInfoForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return linear.inputs == input.as.Scalars.scalars;
        }
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LinearInfoForwardCompatibleAssert(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_LinearInfoForwardCompatible(linear, input) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH LINEAR");
}

ML2_DEF ML2_LayerCacheInfo ML2_LinearInfoForward(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input) {
    ML2_LinearInfoForwardCompatibleAssert(linear, input);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            return (ML2_LayerCacheInfo){input.type, .as.Scalars = {input.as.Scalars.samples, linear.outputs}};
        }
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF bool ML2_LinearInfoSame(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b) {
    return a.inputs == b.inputs && a.outputs == b.outputs;
}

ML2_DEF void ML2_LinearInfoSameAssert(ML2_LayerInfoLinear a, ML2_LayerInfoLinear b) {
    ML2_HARD_ASSERT(ML2_LinearInfoSame(a, b) && "LINEARS MUST HAVE THE SAME INFO");
}

ML2_DEF bool ML2_LinearForwardCompatible(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheInfoScalars outputScalars = output.as.Scalars;
            return linear.inputs == inputScalars.scalars && linear.outputs == outputScalars.scalars && inputScalars.samples == outputScalars.samples;
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LinearForwardCompatibleAssert(ML2_LayerInfoLinear linear, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    ML2_HARD_ASSERT(ML2_LinearForwardCompatible(linear, input, output) && "LAYERS MUST BE FORWARD COMPATIBLE WITH LINEAR");
}

ML2_DEF void ML2_LinearForward(ML2_LayerLinear linear, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_LinearForwardCompatibleAssert(linear.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars outputScalars = output.as.Scalars;
            ML2_LayerWeights weights = ML2_LinearWeights(linear);
            ML2_LayerBiases biases = ML2_LinearBiases(linear);
            int samples = inputScalars.info.samples;
            int outputs = linear.info.outputs;
            int inputs = linear.info.inputs;

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < outputs; j++) {
                    ML2_Scalar sum = *ML2_BiasesAt(biases, j);
                    for (int k = 0; k < inputs; k++) {
                        sum += *ML2_ScalarsAt(inputScalars, i, k) * *ML2_WeightsAt(weights, j, k);
                    }
                    *ML2_ScalarsAt(outputScalars, i, j) = sum;
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LinearBackward(ML2_LayerLinear linear, ML2_LayerLinear linearGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_LinearInfoSameAssert(linear.info, linearGradient.info);
    ML2_LinearForwardCompatibleAssert(linear.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_LayerCacheScalars inputScalars = input.as.Scalars;
            ML2_LayerCacheScalars inputGradientScalars = ML2_LayerCacheAsScalars(inputGradient);
            ML2_LayerCacheScalars outputGradientScalars = ML2_LayerCacheAsScalars(outputGradient);
            ML2_ScalarsInfoSameAssert(inputScalars.info, inputGradientScalars.info);
            ML2_LayerWeights weights = ML2_LinearWeights(linear);
            ML2_LayerWeights weightsGradient = ML2_LinearWeights(linearGradient);
            ML2_LayerBiases biasesGradient = ML2_LinearBiases(linearGradient);
            int samples = inputScalars.info.samples;
            int outputs = linear.info.outputs;
            int inputs = linear.info.inputs;
            // input: samples x inputs
            // inputGradient: samples x inputs
            // linear: outputs x inputs
            // linearGradient: outputs x inputs
            // weights: outputs x inputs
            // weightsGradient: outputs x inputs
            // biases: outputs (not modified)
            // biasesGradient: outputs
            // outputGradient: samples x outputs

            for (int i = 0; i < samples; i++) {
                for (int k = 0; k < outputs; k++) {
                    for (int j = 0; j < inputs; j++) {
                        *ML2_WeightsAt(weightsGradient, k, j) += *ML2_ScalarsAt(outputGradientScalars, i, k) * *ML2_ScalarsAt(inputScalars, i, j);
                    }
                }
            }

            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < outputs; j++) {
                    *ML2_BiasesAt(biasesGradient, j) += *ML2_ScalarsAt(outputGradientScalars, i, j);
                }
            }

            ML2_ScalarsClear(inputGradientScalars);
            for (int i = 0; i < samples; i++) {
                for (int k = 0; k < outputs; k++) {
                    for (int j = 0; j < inputs; j++) {
                        *ML2_ScalarsAt(inputGradientScalars, i, j) += *ML2_ScalarsAt(outputGradientScalars, i, k) * *ML2_WeightsAt(weights, k, j);
                    }
                }
            }
        } break;
        case ML2_LayerCacheTypeImages: {
            ML2_TODO("Images -> Linear may or may not be a feature in the future");
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LinearGradientDescent(ML2_LayerLinear linear, ML2_LayerLinear gradientLinear, ML2_Scalar learningRate) {
    ML2_LinearInfoSameAssert(linear.info, gradientLinear.info);
    ML2_WeightsGradientDescent(ML2_LinearWeights(linear), ML2_LinearWeights(gradientLinear), learningRate);
    ML2_BiasesGradientDescent(ML2_LinearBiases(linear), ML2_LinearBiases(gradientLinear), learningRate);
}

// ML2_LayerLinear ⬆️

// ML2_LayerFilters ⬇️

ML2_DEF ML2_LayerInfo ML2_Filters(int outputs, int inputs, int height, int width) {
    return (ML2_LayerInfo){ML2_LayerTypeFilters, .as.Filters = {outputs, inputs, height, width}};
}

ML2_DEF ML2_LayerFilters ML2_FiltersNew(ML2_LayerInfoFilters info) {
    ML2_LayerFilters filters = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs * info.height * info.width, sizeof(*filters.data.weights))}
    };
    return filters;
}

ML2_DEF void ML2_FiltersDestroy(ML2_LayerFilters *filters) {
    ML2_FREE(filters->data.weights);
    *filters = (ML2_LayerFilters){};
}

ML2_DEF ML2_Scalar *ML2_FiltersAt(ML2_LayerFilters filters, int i, int j, int k, int l) {
    return &filters.data.weights[
        i * (filters.info.inputs * filters.info.height * filters.info.width) +
        j * (filters.info.height * filters.info.width) +
        k * (filters.info.width) +
        l
    ];
}

ML2_DEF ML2_LayerFilters ML2_LayerAsFilters(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeFilters);
    return layer.as.Filters;
}

ML2_DEF void ML2_FiltersClear(ML2_LayerFilters filters) {
    for (int i = 0; i < filters.info.outputs; i++) {
        for (int j = 0; j < filters.info.inputs; j++) {
            for (int k = 0; k < filters.info.height; k++) {
                for (int l = 0; l < filters.info.width; l++) {
                    *ML2_FiltersAt(filters, i, j, k, l) = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_DEF void ML2_FiltersRand(ML2_LayerFilters filters, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < filters.info.outputs; i++) {
        for (int j = 0; j < filters.info.inputs; j++) {
            for (int k = 0; k < filters.info.height; k++) {
                for (int l = 0; l < filters.info.width; l++) {
                    *ML2_FiltersAt(filters, i, j, k, l) = ML2_RandScalar(low, high);
                }
            }
        }
    }
}

ML2_DEF void ML2_FiltersXavierRand(ML2_LayerFilters filters) {
    ML2_Scalar limit = ML2_SQRT(ML2_SCALAR_LITERAL(6.0) / ((filters.info.inputs + filters.info.outputs) * filters.info.height * filters.info.width));
    ML2_FiltersRand(filters, -limit, limit);
}

ML2_DEF void ML2_FiltersInfoPrint(ML2_LayerInfoFilters info, int indent) {
    printf(ML2_Indent("Filters(%dx%dx%dx%d)\n", indent), info.outputs, info.inputs, info.height, info.width);
}

ML2_DEF void ML2_FiltersPrint(ML2_LayerFilters filters, int indent) {
    ML2_FiltersInfoPrint(filters.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < filters.info.outputs; i++) {
            for (int j = 0; j < filters.info.inputs; j++) {
                printf(ML2_Indent("Weights[%d,%d]:\n", indent), i, j);
                printf(ML2_Indent("{\n", indent));
                {
                    indent += ML2_Indentation;
                    for (int k = 0; k < filters.info.height; k++) {
                        printf(ML2_Indent("", indent));
                        for (int l = 0; l < filters.info.width; l++) {
                            printf(ML2_ScalarFmt " ", *ML2_FiltersAt(filters, i, j, k, l));
                        }
                        printf("\n");
                    }
                    indent -= ML2_Indentation;
                }
                printf(ML2_Indent("}\n", indent));
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF bool ML2_FiltersInfoForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = input.as.Images;
            int height = inputImages.height - filters.height + 1;
            int width = inputImages.width - filters.width + 1;
            return filters.inputs == inputImages.channels && height > 0 && inputImages.width - width > 0;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_FiltersInfoForwardCompatibleAssert(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FiltersInfoForwardCompatible(filters, input) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH FILTERS");
}

ML2_DEF ML2_LayerCacheInfo ML2_FiltersInfoForward(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input) {
    ML2_FiltersInfoForwardCompatibleAssert(filters, input);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = input.as.Images;
            int height = inputImages.height - filters.height + 1;
            int width = inputImages.width - filters.width + 1;
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeImages, .as.Images = {inputImages.samples, filters.outputs, height, width}};
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF bool ML2_FiltersInfoSame(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b) {
    return a.outputs == b.outputs && a.inputs == b.inputs && a.height == b.height && a.width == b.width;
}

ML2_DEF void ML2_FiltersInfoSameAssert(ML2_LayerInfoFilters a, ML2_LayerInfoFilters b) {
    ML2_HARD_ASSERT(ML2_FiltersInfoSame(a, b) && "FILTERS SHAPES MUST MATCH");
}

ML2_DEF bool ML2_FiltersForwardCompatible(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoImages inputImages = input.as.Images;
            ML2_LayerCacheInfoImages outputImages = output.as.Images;
            int height = inputImages.height - filters.height + 1;
            int width = inputImages.width - filters.width + 1;
            return inputImages.samples == outputImages.samples && inputImages.channels == filters.inputs && outputImages.channels == filters.outputs && outputImages.height == height && outputImages.width == width;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_FiltersForwardCompatibleAssert(ML2_LayerInfoFilters filters, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    ML2_HARD_ASSERT(ML2_FiltersForwardCompatible(filters, input, output) && "LAYERS MUST BE FORWARD COMPATIBLE WITH FILTERS");
}

ML2_DEF void ML2_FiltersForward(ML2_LayerFilters filters, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_FiltersForwardCompatibleAssert(filters.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Filters may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = input.as.Images;
            ML2_LayerCacheImages outputImages = output.as.Images;
            int samples = inputImages.info.samples;
            int outputs = filters.info.outputs;
            int inputs = filters.info.inputs;

            // TODO: benchmark which is better
            // TODO: chatgpt said swapping (y, x, yk, xk) to (yk, xk, y, x) is better so check that
            if (1) {
                ML2_ImagesClear(outputImages);
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < outputs; j++) {
                        for (int k = 0; k < inputs; k++) {
                            for (int y = 0; y < outputImages.info.height; y++) {
                                for (int x = 0; x < outputImages.info.width; x++) {
                                    for (int yk = 0; yk < filters.info.height; yk++) {
                                        for (int xk = 0; xk < filters.info.width; xk++) {
                                            *ML2_ImagesAt(outputImages, i, j, y, x) += *ML2_ImagesAt(inputImages, i, k, y + yk, x + xk) * *ML2_FiltersAt(filters, j, k, yk, xk);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < outputs; j++) {
                        for (int y = 0; y < outputImages.info.height; y++) {
                            for (int x = 0; x < outputImages.info.width; x++) {
                                ML2_Scalar sum = {};
                                for (int k = 0; k < inputs; k++) {
                                    for (int yk = 0; yk < filters.info.height; yk++) {
                                        for (int xk = 0; xk < filters.info.width; xk++) {
                                            sum += *ML2_ImagesAt(inputImages, i, k, y + yk, x + xk) * *ML2_FiltersAt(filters, j, k, yk, xk);
                                        }
                                    }
                                }
                                *ML2_ImagesAt(outputImages, i, j, y, x) = sum;
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_FiltersBackward(ML2_LayerFilters filters, ML2_LayerFilters filtersGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_TODO("ML2_FiltersBackward");
}

ML2_DEF void ML2_FiltersGradientDescent(ML2_LayerFilters filters, ML2_LayerFilters gradientFilters, ML2_Scalar learningRate) {
    ML2_TODO("ML2_FiltersGradientDescent");
}

// ML2_LayerFilters ⬆️

// ML2_LayerConv ⬇️

ML2_DEF ML2_LayerInfo ML2_Conv(int outputs, int inputs, int height, int width) {
    return (ML2_LayerInfo){ML2_LayerTypeConv, .as.Conv = {outputs, inputs, height, width}};
}

ML2_DEF ML2_LayerConv ML2_ConvNew(ML2_LayerInfoConv info) {
    ML2_LayerConv conv = {
        info,
        {ML2_RELIABLE_CALLOC(info.outputs * info.inputs * info.height * info.width, sizeof(*conv.data.weights)),
         ML2_RELIABLE_CALLOC(info.outputs, sizeof(*conv.data.biases))}
    };
    return conv;
}

ML2_DEF void ML2_ConvDestroy(ML2_LayerConv *conv) {
    ML2_FREE(conv->data.weights);
    ML2_FREE(conv->data.biases);
    *conv = (ML2_LayerConv){};
}

ML2_DEF ML2_LayerConv ML2_LayerAsConv(ML2_Layer layer) {
    ML2_HARD_ASSERT(layer.type == ML2_LayerTypeConv);
    return layer.as.Conv;
}

ML2_DEF ML2_LayerFilters ML2_ConvFilters(ML2_LayerConv conv) {
    return (ML2_LayerFilters){{conv.info.outputs, conv.info.inputs, conv.info.height, conv.info.width}, {conv.data.weights}};
}

ML2_DEF ML2_LayerBiases ML2_ConvBiases(ML2_LayerConv conv) {
    return (ML2_LayerBiases){{conv.info.outputs}, {conv.data.biases}};
}

ML2_DEF void ML2_ConvClear(ML2_LayerConv conv) {
    ML2_FiltersClear(ML2_ConvFilters(conv));
    ML2_BiasesClear(ML2_ConvBiases(conv));
}

ML2_DEF void ML2_ConvRand(ML2_LayerConv conv, ML2_Scalar low, ML2_Scalar high) {
    ML2_FiltersRand(ML2_ConvFilters(conv), low, high);
    ML2_BiasesRand(ML2_ConvBiases(conv), low, high);
}

ML2_DEF void ML2_ConvXavierRand(ML2_LayerConv conv) {
    ML2_FiltersXavierRand(ML2_ConvFilters(conv));
    ML2_BiasesXavierRand(ML2_ConvBiases(conv));
}

ML2_DEF void ML2_ConvInfoPrint(ML2_LayerInfoConv info, int indent) {
    printf(ML2_Indent("Conv(%dx%dx%dx%d)\n", indent), info.outputs, info.inputs, info.height, info.width);
}

ML2_DEF void ML2_ConvPrint(ML2_LayerConv conv, int indent) {
    ML2_ConvInfoPrint(conv.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        ML2_FiltersPrint(ML2_ConvFilters(conv), indent);
        ML2_BiasesPrint(ML2_ConvBiases(conv), indent);
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

// TODO: this logic is a duplicate from the Filters functions

ML2_DEF bool ML2_ConvInfoForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = input.as.Images;
            int height = inputImages.height - conv.height + 1;
            int width = inputImages.width - conv.width + 1;
            return conv.inputs == inputImages.channels && height > 0 && inputImages.width - width > 0;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ConvInfoForwardCompatibleAssert(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_ConvInfoForwardCompatible(conv, input) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH CONV");
}

ML2_DEF ML2_LayerCacheInfo ML2_ConvInfoForward(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input) {
    ML2_ConvInfoForwardCompatibleAssert(conv, input);
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheInfoImages inputImages = input.as.Images;
            int height = inputImages.height - conv.height + 1;
            int width = inputImages.width - conv.width + 1;
            return (ML2_LayerCacheInfo){ML2_LayerCacheTypeImages, .as.Images = {inputImages.samples, conv.outputs, height, width}};
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF bool ML2_ConvInfoSame(ML2_LayerInfoConv a, ML2_LayerInfoConv b) {
    return a.outputs == b.outputs && a.inputs == b.inputs && a.height == b.height && a.width == b.width;
}

ML2_DEF void ML2_ConvInfoSameAssert(ML2_LayerInfoConv a, ML2_LayerInfoConv b) {
    ML2_HARD_ASSERT(ML2_ConvInfoSame(a, b) && "CONV SHAPES MUST MATCH");
}

ML2_DEF bool ML2_ConvForwardCompatible(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            if (input.type != output.type) return false;
            ML2_LayerCacheInfoImages inputImages = input.as.Images;
            ML2_LayerCacheInfoImages outputImages = output.as.Images;
            int height = inputImages.height - conv.height + 1;
            int width = inputImages.width - conv.width + 1;
            return inputImages.samples == outputImages.samples && inputImages.channels == conv.inputs && outputImages.channels == conv.outputs && outputImages.height == height && outputImages.width == width;
        }
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ConvForwardCompatibleAssert(ML2_LayerInfoConv conv, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    ML2_HARD_ASSERT(ML2_ConvForwardCompatible(conv, input, output) && "LAYERS MUST BE FORWARD COMPATIBLE WITH CONV");
}

ML2_DEF void ML2_ConvForward(ML2_LayerConv conv, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_ConvForwardCompatibleAssert(conv.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output));
    switch (input.type) {
        case ML2_LayerCacheTypeScalars: {
            ML2_TODO("Scalars -> Conv may or may not be a feature in the future");
        }
        case ML2_LayerCacheTypeImages: {
            ML2_LayerCacheImages inputImages = input.as.Images;
            ML2_LayerCacheImages outputImages = output.as.Images;
            ML2_LayerFilters filters = ML2_ConvFilters(conv);
            ML2_LayerBiases biases = ML2_ConvBiases(conv);
            int samples = inputImages.info.samples;
            int outputs = conv.info.outputs;
            int inputs = conv.info.inputs;

            // TODO: benchmark which is better
            // TODO: chatgpt said swapping (y, x, yk, xk) to (yk, xk, y, x) is better so check that
            if (1) {
                // ML2_ImagesClear(outputImages);
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < outputs; j++) {
                        for (int y = 0; y < outputImages.info.height; y++) {
                            for (int x = 0; x < outputImages.info.width; x++) {
                                *ML2_ImagesAt(outputImages, i, j, y, x) = *ML2_BiasesAt(biases, j);
                            }
                        }
                    }
                }
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < outputs; j++) {
                        for (int k = 0; k < inputs; k++) {
                            for (int y = 0; y < outputImages.info.height; y++) {
                                for (int x = 0; x < outputImages.info.width; x++) {
                                    for (int yk = 0; yk < conv.info.height; yk++) {
                                        for (int xk = 0; xk < conv.info.width; xk++) {
                                            *ML2_ImagesAt(outputImages, i, j, y, x) += *ML2_ImagesAt(inputImages, i, k, y + yk, x + xk) * *ML2_FiltersAt(filters, j, k, yk, xk);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < outputs; j++) {
                        for (int y = 0; y < outputImages.info.height; y++) {
                            for (int x = 0; x < outputImages.info.width; x++) {
                                ML2_Scalar sum = *ML2_BiasesAt(biases, j);
                                for (int k = 0; k < inputs; k++) {
                                    for (int yk = 0; yk < conv.info.height; yk++) {
                                        for (int xk = 0; xk < conv.info.width; xk++) {
                                            sum += *ML2_ImagesAt(inputImages, i, k, y + yk, x + xk) * *ML2_FiltersAt(filters, j, k, yk, xk);
                                        }
                                    }
                                }
                                *ML2_ImagesAt(outputImages, i, j, y, x) = sum;
                            }
                        }
                    }
                }
            }
        } break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_ConvBackward(ML2_LayerConv conv, ML2_LayerConv convGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_TODO("ML2_ConvBackward");
}

ML2_DEF void ML2_ConvGradientDescent(ML2_LayerConv conv, ML2_LayerConv gradientLinear, ML2_Scalar learningRate) {
    ML2_TODO("ML2_ConvGradientDescent");
}

// ML2_LayerConv ⬆️

// ML2_LayerFlatten ⬇️

ML2_DEF ML2_LayerInfo ML2_Flatten() {
    return (ML2_LayerInfo){ML2_LayerTypeFlatten, .as.Flatten = {}};
}

ML2_DEF ML2_LayerFlatten ML2_FlattenNew(ML2_LayerInfoFlatten info) {
    return (ML2_LayerFlatten){info};
}

ML2_DEF void ML2_FlattenDestroy(ML2_LayerFlatten *flatten) {
    *flatten = (ML2_LayerFlatten){};
}

ML2_DEF void ML2_FlattenInfoPrint(ML2_LayerInfoFlatten flatten, int indent) {
    printf(ML2_Indent("Flatten()\n", indent));
}

ML2_DEF void ML2_FlattenPrint(ML2_LayerFlatten flatten, int indent) {
    ML2_FlattenInfoPrint(flatten.info, indent);
}

ML2_DEF bool ML2_FlattenInfoForwardCompatible(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input) {
    return input.type == ML2_LayerCacheTypeImages;
}

ML2_DEF void ML2_FlattenInfoForwardCompatibleAssert(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input) {
    ML2_HARD_ASSERT(ML2_FlattenInfoForwardCompatible(flatten, input) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH FLATTEN");
}

ML2_DEF ML2_LayerCacheInfo ML2_FlattenInfoForward(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input) {
    ML2_FlattenInfoForwardCompatibleAssert(flatten, input);
    ML2_LayerCacheInfoImages inputImages = input.as.Images;
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeScalars, .as.Scalars = {inputImages.samples, inputImages.channels * inputImages.height * inputImages.width}};
}

ML2_DEF bool ML2_FlattenForwardCompatible(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    if (input.type != ML2_LayerCacheTypeImages || output.type != ML2_LayerCacheTypeScalars) return false;
    ML2_LayerCacheInfoImages inputImages = input.as.Images;
    ML2_LayerCacheInfoScalars outputScalars = output.as.Scalars;
    return inputImages.samples == outputScalars.samples && inputImages.channels * inputImages.height * inputImages.width == outputScalars.scalars;
}

ML2_DEF void ML2_FlattenForwardCompatibleAssert(ML2_LayerInfoFlatten flatten, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    ML2_HARD_ASSERT(ML2_FlattenForwardCompatible(flatten, input, output) && "INPUT MUST BE INFO FORWARD COMPATIBLE WITH FLATTEN");
}

ML2_DEF void ML2_FlattenForward(ML2_LayerFlatten flatten, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_FlattenForwardCompatibleAssert(flatten.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(output));
    ML2_LayerCacheImages inputImages = input.as.Images;
    ML2_LayerCacheScalars outputScalars = output.as.Scalars;
    int samples = inputImages.info.samples;
    int channels = inputImages.info.channels;
    int height = inputImages.info.height;
    int width = inputImages.info.width;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    *ML2_ScalarsAt(outputScalars, i, j * (height + width) + k * (width) + l) = *ML2_ImagesAt(inputImages, i, j, k, l);
                }
            }
        }
    }
}

ML2_DEF void ML2_FlattenBackward(ML2_LayerFlatten flatten, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache outputGradient) {
    ML2_LayerCacheInfoSameAssert(ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(inputGradient));
    ML2_FlattenForwardCompatibleAssert(flatten.info, ML2_LayerCacheAsInfo(input), ML2_LayerCacheAsInfo(outputGradient));
    ML2_LayerCacheImages inputImages = input.as.Images;
    ML2_LayerCacheImages inputGradientImages = inputGradient.as.Images;
    ML2_LayerCacheScalars outputGradientScalars = outputGradient.as.Scalars;
    int samples = inputImages.info.samples;
    int channels = inputImages.info.channels;
    int height = inputImages.info.height;
    int width = inputImages.info.width;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    *ML2_ImagesAt(inputGradientImages, i, j, k, l) = *ML2_ScalarsAt(outputGradientScalars, i, j * (height + width) + k * (width) + l);
                }
            }
        }
    }
}

// ML2_LayerFlatten ⬆️

ML2_DEF ML2_Layer ML2_LayerNew(ML2_LayerInfo info) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return (ML2_Layer){info.type, .as.Weights = ML2_WeightsNew(info.as.Weights)};
        case ML2_LayerTypeBiases: return (ML2_Layer){info.type, .as.Biases = ML2_BiasesNew(info.as.Biases)};
        case ML2_LayerTypeActivation: return (ML2_Layer){info.type, .as.Activation = ML2_ActivationNew(info.as.Activation)};
        case ML2_LayerTypeLinear: return (ML2_Layer){info.type, .as.Linear = ML2_LinearNew(info.as.Linear)};
        case ML2_LayerTypeFilters: return (ML2_Layer){info.type, .as.Filters = ML2_FiltersNew(info.as.Filters)};
        case ML2_LayerTypeConv: return (ML2_Layer){info.type, .as.Conv = ML2_ConvNew(info.as.Conv)};
        case ML2_LayerTypeFlatten: return (ML2_Layer){info.type, .as.Flatten = ML2_FlattenNew(info.as.Flatten)};
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerDestroy(ML2_Layer *layer) {
    switch (layer->type) {
        case ML2_LayerTypeWeights: ML2_WeightsDestroy(&layer->as.Weights); break;
        case ML2_LayerTypeBiases: ML2_BiasesDestroy(&layer->as.Biases); break;
        case ML2_LayerTypeActivation: ML2_ActivationDestroy(&layer->as.Activation); break;
        case ML2_LayerTypeLinear: ML2_LinearDestroy(&layer->as.Linear); break;
        case ML2_LayerTypeFilters: ML2_FiltersDestroy(&layer->as.Filters); break;
        case ML2_LayerTypeConv: ML2_ConvDestroy(&layer->as.Conv); break;
        case ML2_LayerTypeFlatten: ML2_FlattenDestroy(&layer->as.Flatten); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF ML2_LayerInfo ML2_LayerAsInfo(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: return (ML2_LayerInfo){layer.type, .as.Weights = layer.as.Weights.info};
        case ML2_LayerTypeBiases: return (ML2_LayerInfo){layer.type, .as.Biases = layer.as.Biases.info};
        case ML2_LayerTypeActivation: return (ML2_LayerInfo){layer.type, .as.Activation = layer.as.Activation.info};
        case ML2_LayerTypeLinear: return (ML2_LayerInfo){layer.type, .as.Linear = layer.as.Linear.info};
        case ML2_LayerTypeFilters: return (ML2_LayerInfo){layer.type, .as.Filters = layer.as.Filters.info};
        case ML2_LayerTypeConv: return (ML2_LayerInfo){layer.type, .as.Conv = layer.as.Conv.info};
        case ML2_LayerTypeFlatten: return (ML2_LayerInfo){layer.type, .as.Flatten = layer.as.Flatten.info};
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerClear(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsClear(layer.as.Weights); break;
        case ML2_LayerTypeBiases: ML2_BiasesClear(layer.as.Biases); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearClear(layer.as.Linear); break;
        case ML2_LayerTypeFilters: ML2_FiltersClear(layer.as.Filters); break;
        case ML2_LayerTypeConv: ML2_ConvClear(layer.as.Conv); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerRand(ML2_Layer layer, ML2_Scalar low, ML2_Scalar high) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsRand(layer.as.Weights, low, high); break;
        case ML2_LayerTypeBiases: ML2_BiasesRand(layer.as.Biases, low, high); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearRand(layer.as.Linear, low, high); break;
        case ML2_LayerTypeFilters: ML2_FiltersRand(layer.as.Filters, low, high); break;
        case ML2_LayerTypeConv: ML2_ConvRand(layer.as.Conv, low, high); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerXavierRand(ML2_Layer layer) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsXavierRand(layer.as.Weights); break;
        case ML2_LayerTypeBiases: ML2_BiasesXavierRand(layer.as.Biases); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearXavierRand(layer.as.Linear); break;
        case ML2_LayerTypeFilters: ML2_FiltersXavierRand(layer.as.Filters); break;
        case ML2_LayerTypeConv: ML2_ConvXavierRand(layer.as.Conv); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerInfoPrint(ML2_LayerInfo layer, int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsInfoPrint(layer.as.Weights, indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesInfoPrint(layer.as.Biases, indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationInfoPrint(layer.as.Activation, indent); break;
        case ML2_LayerTypeLinear: ML2_LinearInfoPrint(layer.as.Linear, indent); break;
        case ML2_LayerTypeFilters: ML2_FiltersInfoPrint(layer.as.Filters, indent); break;
        case ML2_LayerTypeConv: ML2_ConvInfoPrint(layer.as.Conv, indent); break;
        case ML2_LayerTypeFlatten: ML2_FlattenInfoPrint(layer.as.Flatten, indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerPrint(ML2_Layer layer, int indent) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsPrint(layer.as.Weights, indent); break;
        case ML2_LayerTypeBiases: ML2_BiasesPrint(layer.as.Biases, indent); break;
        case ML2_LayerTypeActivation: ML2_ActivationPrint(layer.as.Activation, indent); break;
        case ML2_LayerTypeLinear: ML2_LinearPrint(layer.as.Linear, indent); break;
        case ML2_LayerTypeFilters: ML2_FiltersPrint(layer.as.Filters, indent); break;
        case ML2_LayerTypeConv: ML2_ConvPrint(layer.as.Conv, indent); break;
        case ML2_LayerTypeFlatten: ML2_FlattenPrint(layer.as.Flatten, indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF ML2_LayerCacheInfo ML2_LayerInfoForward(ML2_LayerInfo info, ML2_LayerCacheInfo input) {
    switch (info.type) {
        case ML2_LayerTypeWeights: return ML2_WeightsInfoForward(info.as.Weights, input);
        case ML2_LayerTypeBiases: return ML2_BiasesInfoForward(info.as.Biases, input);
        case ML2_LayerTypeActivation: return input;
        case ML2_LayerTypeLinear: return ML2_LinearInfoForward(info.as.Linear, input);
        case ML2_LayerTypeFilters: return ML2_FiltersInfoForward(info.as.Filters, input);
        case ML2_LayerTypeConv: return ML2_ConvInfoForward(info.as.Conv, input);
        case ML2_LayerTypeFlatten: return ML2_FlattenInfoForward(info.as.Flatten, input);
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

// ML2_DEF bool ML2_LayerInfoSame(ML2_LayerInfo a, ML2_LayerInfo b) {
//     if (a.type != b.type) return false;
//     switch (a.type) {
//         case ML2_LayerTypeWeights: return ML2_WeightsInfoSame(a.as.Weights, b.as.Weights);
//         case ML2_LayerTypeBiases: return ML2_BiasesInfoSame(a.as.Biases, b.as.Biases);
//         case ML2_LayerTypeActivation: return ML2_ActivationInfoSame(a.as.Activation, b.as.Activation);
//         case ML2_LayerTypeLinear: return ML2_LinearInfoSame(a.as.Linear, b.as.Linear);
//         case ML2_LayerTypeConv: return ML2_ConvInfoSame(a.as.Conv, b.as.Conv);
//         default: ML2_UNREACHABLE("Unknown ML2_LayerType");
//     }
//     return true;
// }

// ML2_DEF void ML2_LayerInfoSameAssert(ML2_LayerInfo a, ML2_LayerInfo b) {
//     ML2_HARD_ASSERT(ML2_LayerInfoSame(a, b) && "LAYERS MUST HAVE THE SAME INFO");
// }

// ML2_DEF bool ML2_LayerInfoForwardCompatible(ML2_LayerInfo layer, ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
//     switch (layer.type) {
//         case ML2_LayerTypeWeights: return ML2_WeightsForwardCompatible(layer.as.Weights, input, output);
//         case ML2_LayerTypeBiases: return ML2_BiasesForwardCompatible(layer.as.Biases, input, output);
//         case ML2_LayerTypeActivation: return ML2_ActivationForwardCompatible(layer.as.Activation, input, output);
//         case ML2_LayerTypeLinear: return ML2_LinearInfoForwardCompatible(layer.as.Linear, input, output);
//         case ML2_LayerTypeConv: return ML2_ConvInfoForwardCompatible(layer.as.Conv, input, output);
//         default: ML2_UNREACHABLE("Unknown ML2_LayerType");
//     }
// }

ML2_DEF void ML2_LayerForward(ML2_Layer layer, ML2_LayerCache input, ML2_LayerCache output) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsForward(layer.as.Weights, input, output); break;
        case ML2_LayerTypeBiases: ML2_BiasesForward(layer.as.Biases, input, output); break;
        case ML2_LayerTypeActivation: ML2_ActivationForward(layer.as.Activation, input, output); break;
        case ML2_LayerTypeLinear: ML2_LinearForward(layer.as.Linear, input, output); break;
        case ML2_LayerTypeFilters: ML2_FiltersForward(layer.as.Filters, input, output); break;
        case ML2_LayerTypeConv: ML2_ConvForward(layer.as.Conv, input, output); break;
        case ML2_LayerTypeFlatten: ML2_FlattenForward(layer.as.Flatten, input, output); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerBackward(ML2_Layer layer, ML2_Layer layerGradient, ML2_LayerCache input, ML2_LayerCache inputGradient, ML2_LayerCache output, ML2_LayerCache outputGradient) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsBackward(layer.as.Weights, ML2_LayerAsWeights(layerGradient), input, inputGradient, outputGradient); break;
        case ML2_LayerTypeBiases: ML2_BiasesBackward(layer.as.Biases, ML2_LayerAsBiases(layerGradient), input, inputGradient, outputGradient); break;
        case ML2_LayerTypeActivation: ML2_ActivationBackward(layer.as.Activation, input, inputGradient, output, outputGradient); break;
        case ML2_LayerTypeLinear: ML2_LinearBackward(layer.as.Linear, ML2_LayerAsLinear(layerGradient), input, inputGradient, outputGradient); break;
        case ML2_LayerTypeFilters: ML2_FiltersBackward(layer.as.Filters, ML2_LayerAsFilters(layerGradient), input, inputGradient, outputGradient); break;
        case ML2_LayerTypeConv: ML2_ConvBackward(layer.as.Conv, ML2_LayerAsConv(layerGradient), input, inputGradient, outputGradient); break;
        case ML2_LayerTypeFlatten: ML2_FlattenBackward(layer.as.Flatten, input, inputGradient, outputGradient); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

ML2_DEF void ML2_LayerGradientDescent(ML2_Layer layer, ML2_Layer gradient, ML2_Scalar learningRate) {
    switch (layer.type) {
        case ML2_LayerTypeWeights: ML2_WeightsGradientDescent(layer.as.Weights, ML2_LayerAsWeights(gradient), learningRate); break;
        case ML2_LayerTypeBiases: ML2_BiasesGradientDescent(layer.as.Biases, ML2_LayerAsBiases(gradient), learningRate); break;
        case ML2_LayerTypeActivation: /* No action needed */ break;
        case ML2_LayerTypeLinear: ML2_LinearGradientDescent(layer.as.Linear, ML2_LayerAsLinear(gradient), learningRate); break;
        case ML2_LayerTypeFilters: ML2_FiltersGradientDescent(layer.as.Filters, ML2_LayerAsFilters(gradient), learningRate); break;
        case ML2_LayerTypeConv: ML2_ConvGradientDescent(layer.as.Conv, ML2_LayerAsConv(gradient), learningRate); break;
        case ML2_LayerTypeFlatten: /* No action needed */ break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerType");
    }
}

// ML2_Layer ⬆️

// ML2_LayerCache ⬇️

// ML2_LayerCacheScalars ⬇️

ML2_DEF ML2_LayerCacheInfo ML2_Scalars(int samples, int scalars) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeScalars, .as.Scalars = {samples, scalars}};
}

ML2_DEF ML2_LayerCacheScalars ML2_ScalarsNew(ML2_LayerCacheInfoScalars info) {
    ML2_LayerCacheScalars scalars = {
        info,
        {ML2_RELIABLE_CALLOC(info.samples * info.scalars, sizeof(*scalars.data.scalars))}
    };
    return scalars;
}

ML2_DEF void ML2_ScalarsDestroy(ML2_LayerCacheScalars *scalars) {
    ML2_FREE(scalars->data.scalars);
    *scalars = (ML2_LayerCacheScalars){};
}

ML2_DEF ML2_Scalar *ML2_ScalarsAt(ML2_LayerCacheScalars scalars, int sample, int scalar) {
    ML2_SOFT_ASSERT(0 <= sample && sample < scalars.info.samples && 0 <= scalar && scalar < scalars.info.scalars && "OUT OF BOUNDS INDICES");
    return &scalars.data.scalars[sample * scalars.info.scalars + scalar];
}

ML2_DEF ML2_LayerCacheScalars ML2_LayerCacheAsScalars(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeScalars);
    return cache.as.Scalars;
}

ML2_DEF void ML2_ScalarsClear(ML2_LayerCacheScalars scalars) {
    for (int i = 0; i < scalars.info.samples; i++) {
        for (int j = 0; j < scalars.info.scalars; j++) {
            *ML2_ScalarsAt(scalars, i, j) = ML2_SCALAR_LITERAL(0.0);
        }
    }
}

ML2_DEF void ML2_ScalarsCopy(ML2_LayerCacheScalars dest, ML2_LayerCacheScalars src) {
    ML2_ScalarsInfoSameAssert(dest.info, src.info);
    int samples = dest.info.samples;
    int scalars = dest.info.scalars;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            *ML2_ScalarsAt(dest, i, j) = *ML2_ScalarsAt(src, i, j);
        }
    }
}

ML2_DEF void ML2_ScalarsInfoPrint(ML2_LayerCacheInfoScalars info, int indent) {
    printf(ML2_Indent("Scalars(%dx%d)\n", indent), info.samples, info.scalars);
}

ML2_DEF void ML2_ScalarsPrint(ML2_LayerCacheScalars scalars, int indent) {
    ML2_ScalarsInfoPrint(scalars.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < scalars.info.samples; i++) {
            printf(ML2_Indent("", indent));
            for (int j = 0; j < scalars.info.scalars; j++) {
                printf(ML2_ScalarFmt " ", *ML2_ScalarsAt(scalars, i, j));
            }
            printf("\n");
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF bool ML2_ScalarsInfoSame(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b) {
    return a.samples == b.samples && a.scalars == b.scalars;
}

ML2_DEF void ML2_ScalarsInfoSameAssert(ML2_LayerCacheInfoScalars a, ML2_LayerCacheInfoScalars b) {
    ML2_HARD_ASSERT(ML2_ScalarsInfoSame(a, b) && "SCALARS SHAPES MUST MATCH");
}

// ML2_LayerCacheScalars ⬆️

// ML2_LayerCacheImages ⬇️

ML2_DEF ML2_LayerCacheInfo ML2_Images(int samples, int channels, int height, int width) {
    return (ML2_LayerCacheInfo){ML2_LayerCacheTypeImages, .as.Images = {samples, channels, height, width}};
}

ML2_DEF ML2_LayerCacheImages ML2_ImagesNew(ML2_LayerCacheInfoImages info) {
    ML2_LayerCacheImages images = {
        info,
        {ML2_RELIABLE_CALLOC(info.samples * info.channels * info.height * info.width, sizeof(*images.data.images))}
    };
    return images;
}

ML2_DEF void ML2_ImagesDestroy(ML2_LayerCacheImages *images) {
    ML2_FREE(images->data.images);
    *images = (ML2_LayerCacheImages){};
}

ML2_DEF ML2_Scalar *ML2_ImagesAt(ML2_LayerCacheImages images, int sample, int channel, int y, int x) {
    return &images.data.images[
        sample  * (images.info.channels + images.info.height + images.info.width) +
        channel * (images.info.height + images.info.width) +
        y       * (images.info.width) +
        x
    ];
}

ML2_DEF ML2_LayerCacheImages ML2_LayerCacheAsImages(ML2_LayerCache cache) {
    ML2_HARD_ASSERT(cache.type == ML2_LayerCacheTypeImages);
    return cache.as.Images;
}

ML2_DEF void ML2_ImagesClear(ML2_LayerCacheImages images) {
    for (int i = 0; i < images.info.samples; i++) {
        for (int j = 0; j < images.info.channels; j++) {
            for (int k = 0; k < images.info.height; k++) {
                for (int l = 0; l < images.info.width; l++) {
                    *ML2_ImagesAt(images, i, j, k, l) = ML2_SCALAR_LITERAL(0.0);
                }
            }
        }
    }
}

ML2_DEF void ML2_ImagesCopy(ML2_LayerCacheImages dest, ML2_LayerCacheImages src) {
    ML2_ImagesInfoSameAssert(dest.info, src.info);
    int samples = dest.info.samples;
    int channels = dest.info.channels;
    int height = dest.info.height;
    int width = dest.info.width;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    *ML2_ImagesAt(dest, i, j, k, l) = *ML2_ImagesAt(src, i, j, k, l);
                }
            }
        }
    }
}

ML2_DEF void ML2_ImagesInfoPrint(ML2_LayerCacheInfoImages info, int indent) {
    printf(ML2_Indent("Images(%dx%dx%dx%d)\n", indent), info.samples, info.channels, info.height, info.width);
}

ML2_DEF void ML2_ImagesPrint(ML2_LayerCacheImages images, int indent) {
    ML2_ImagesInfoPrint(images.info, indent);
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < images.info.samples; i++) {
            printf(ML2_Indent("Samples[%d]:\n", indent), i);
            printf(ML2_Indent("{\n", indent));
            {
                indent += ML2_Indentation;
                for (int j = 0; j < images.info.channels; j++) {
                    printf(ML2_Indent("Channels[%d]:\n", indent), j);
                    printf(ML2_Indent("{\n", indent));
                    {
                        indent += ML2_Indentation;
                        for (int k = 0; k < images.info.height; k++) {
                            printf(ML2_Indent("", indent));
                            for (int l = 0; l < images.info.width; l++) {
                                printf(ML2_ScalarFmt " ", *ML2_ImagesAt(images, i, j, k, l));
                            }
                            printf("\n");
                        }
                        indent -= ML2_Indentation;
                    }
                    printf(ML2_Indent("}\n", indent));
                }
                indent -= ML2_Indentation;
            }
            printf(ML2_Indent("}\n", indent));
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF bool ML2_ImagesInfoSame(ML2_LayerCacheInfoImages a, ML2_LayerCacheInfoImages b) {
    return a.samples == b.samples && a.channels == b.channels && a.height == b.height && a.width == b.width;
}

ML2_DEF void ML2_ImagesInfoSameAssert(ML2_LayerCacheInfoImages a, ML2_LayerCacheInfoImages b){
    ML2_HARD_ASSERT(ML2_ImagesInfoSame(a, b) && "IMAGES SHAPES MUST MATCH");
}

// ML2_LayerCacheImages ⬆️

ML2_DEF ML2_LayerCache ML2_LayerCacheNew(ML2_LayerCacheInfo info) {
    switch (info.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCache){info.type, .as.Scalars = ML2_ScalarsNew(info.as.Scalars)};
        case ML2_LayerCacheTypeImages: return (ML2_LayerCache){info.type, .as.Images = ML2_ImagesNew(info.as.Images)};
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LayerCacheDestroy(ML2_LayerCache *cache) {
    switch (cache->type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsDestroy(&cache->as.Scalars); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesDestroy(&cache->as.Images); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF ML2_LayerCacheInfo ML2_LayerCacheAsInfo(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: return (ML2_LayerCacheInfo){cache.type, .as.Scalars = cache.as.Scalars.info};
        case ML2_LayerCacheTypeImages: return (ML2_LayerCacheInfo){cache.type, .as.Images = cache.as.Images.info};
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LayerCacheClear(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsClear(cache.as.Scalars); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesClear(cache.as.Images); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LayerCacheCopy(ML2_LayerCache dest, ML2_LayerCache src) {
    switch (dest.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsCopy(dest.as.Scalars, src.as.Scalars); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesCopy(dest.as.Images, src.as.Images); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LayerCacheInfoPrint(ML2_LayerCacheInfo layer, int indent) {
    switch (layer.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsInfoPrint(layer.as.Scalars, indent); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesInfoPrint(layer.as.Images, indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LayerCachePrint(ML2_LayerCache cache, int indent) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: ML2_ScalarsPrint(cache.as.Scalars, indent); break;
        case ML2_LayerCacheTypeImages: ML2_ImagesPrint(cache.as.Images, indent); break;
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF bool ML2_LayerCacheInfoSame(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b) {
    if (a.type != b.type) return false;
    switch (a.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsInfoSame(a.as.Scalars, b.as.Scalars);
        case ML2_LayerCacheTypeImages: return ML2_ImagesInfoSame(a.as.Images, b.as.Images);
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

ML2_DEF void ML2_LayerCacheInfoSameAssert(ML2_LayerCacheInfo a, ML2_LayerCacheInfo b) {
    ML2_HARD_ASSERT(ML2_LayerCacheInfoSame(a, b) && "LAYER CACHES MUST HAVE THE SAME INFO");
}

// ML2_LayerCache ⬆️

// ML2_Arch ⬇️

#define ML2_ArchMake(...) ((ML2_Arch){sizeof((ML2_LayerInfo[]){__VA_ARGS__}) / sizeof(ML2_LayerInfo), (ML2_LayerInfo[]){__VA_ARGS__}})

ML2_DEF ML2_Arch ML2_ArchNew(int layers, ML2_LayerInfo infos[static layers]) {
    ML2_Arch arch = {layers, ML2_RELIABLE_CALLOC(layers, sizeof(*arch.infos))};
    memcpy(arch.infos, infos, layers * sizeof(*arch.infos));
    return arch;
}

ML2_DEF void ML2_ArchDestroy(ML2_Arch *arch) {
    ML2_FREE(arch->infos);
    *arch = (ML2_Arch){};
}

// ML2_Arch ⬆️

// ML2_Model ⬇️

ML2_DEF ML2_Model ML2_ModelNew(ML2_Arch arch) {
    ML2_Model model = {
        .count = arch.layers,
        .layers = ML2_RELIABLE_CALLOC(arch.layers, sizeof(*model.layers)),
    };
    for (int i = 0; i < arch.layers; i++) {
        model.layers[i] = ML2_LayerNew(arch.infos[i]);
    }
    return model;
}

ML2_DEF void ML2_ModelDestroy(ML2_Model *model) {
    for (int i = 0; i < model->count; i++) {
        ML2_LayerDestroy(&model->layers[i]);
    }
    ML2_FREE(model->layers);
    *model = (ML2_Model){};
}

ML2_DEF void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < model.count; i++) {
        ML2_LayerRand(model.layers[i], low, high);
    }
}

ML2_DEF void ML2_ModelXavierRand(ML2_Model model) {
    for (int i = 0; i < model.count; i++) {
        ML2_LayerXavierRand(model.layers[i]);
    }
}

ML2_DEF void ML2_ModelInfoPrint(ML2_Model model, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < model.count; i++) {
        ML2_LayerInfoPrint(ML2_LayerAsInfo(model.layers[i]), indent + ML2_Indentation);
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF void ML2_ModelPrint(ML2_Model model, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < model.count; i++) {
        ML2_LayerPrint(model.layers[i], indent + ML2_Indentation);
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache) {
    for (int i = 0; i < model.count; i++) {
        ML2_LayerForward(model.layers[i], modelCache.caches[i], modelCache.caches[i + 1]);
    }
}

ML2_DEF void ML2_ModelBackward(ML2_Model model, ML2_ModelCache modelCache) {
    // TODO: the gradient of the first layer does not need to be calculated, but it does get calculated
    for (int i = modelCache.layers - 2; i >= 0; i--) {
        ML2_LayerBackward(model.layers[i], modelCache.gradients[i], modelCache.caches[i], modelCache.cachesGradients[i], modelCache.caches[i + 1], modelCache.cachesGradients[i + 1]);
    }
}

ML2_DEF void ML2_ModelGradientDescent(ML2_Model model, ML2_ModelCache modelCache, ML2_Scalar learningRate) {
    for (int i = 0; i < model.count; i++) {
        ML2_LayerGradientDescent(model.layers[i], modelCache.gradients[i], learningRate);
    }
}

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_DEF ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, ML2_BatchInfo info) {
    int modelCacheLayers = arch.layers + 1;
    ML2_ModelCache modelCache = {
        .layers = modelCacheLayers,
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
    ML2_LayerCacheInfoSameAssert(info.output, prevCacheInfo);
    return modelCache;
}

ML2_DEF void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache) {
    for (int i = 0; i < modelCache->layers - 1; i++) {
        ML2_LayerDestroy(&modelCache->gradients[i]);
    }

    for (int i = 0; i < modelCache->layers; i++) {
        ML2_LayerCacheDestroy(&modelCache->caches[i]);
        ML2_LayerCacheDestroy(&modelCache->cachesGradients[i]);
    }

    ML2_FREE(modelCache->caches);
    ML2_FREE(modelCache->cachesGradients);
    ML2_FREE(modelCache->gradients);

    *modelCache = (ML2_ModelCache){};
}

ML2_DEF void ML2_ModelCacheInfoPrint(ML2_ModelCache modelCache, int indent) {
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        const char cacheStr[] = "Cache";
        const char gradientStr[] = "Gradient";
        const char cacheGradientStr[] = "CacheGradient";
        size_t alignIndent = 0;
        if (sizeof(cacheStr) > alignIndent) alignIndent = sizeof(cacheStr);
        if (sizeof(gradientStr) > alignIndent) alignIndent = sizeof(gradientStr);
        if (sizeof(cacheGradientStr) > alignIndent) alignIndent = sizeof(cacheGradientStr);

        for (int i = 0; i < modelCache.layers; i++) {
            printf(ML2_Indent("%s[%d]: ", indent), cacheStr, i);
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(modelCache.caches[i]), alignIndent - sizeof(cacheStr));
            printf(ML2_Indent("%s[%d]: ", indent), cacheGradientStr, i);
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(modelCache.cachesGradients[i]), alignIndent - sizeof(cacheGradientStr));
            if (i < modelCache.layers - 1) {
                printf(ML2_Indent("%s[%d]: ", indent), gradientStr, i);
                ML2_LayerInfoPrint(ML2_LayerAsInfo(modelCache.gradients[i]), alignIndent - sizeof(gradientStr));
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF void ML2_ModelCachePrint(ML2_ModelCache modelCache, int indent) {
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        for (int i = 0; i < modelCache.layers; i++) {
            printf(ML2_Indent("Cache[%d]\n", indent), i);
            printf(ML2_Indent("{\n", indent));
            {
                indent += ML2_Indentation;
                ML2_LayerCachePrint(modelCache.caches[i], indent);
                indent -= ML2_Indentation;
            }
            printf(ML2_Indent("}\n", indent));
            printf(ML2_Indent("CacheGradient[%d]\n", indent), i);
            printf(ML2_Indent("{\n", indent));
            {
                indent += ML2_Indentation;
                ML2_LayerCachePrint(modelCache.cachesGradients[i], indent);
                indent -= ML2_Indentation;
            }
            printf(ML2_Indent("}\n", indent));
            if (i < modelCache.layers - 1) {
                printf(ML2_Indent("Gradient[%d]\n", indent), i);
                printf(ML2_Indent("{\n", indent));
                {
                    indent += ML2_Indentation;
                    ML2_LayerPrint(modelCache.gradients[i], indent);
                    indent -= ML2_Indentation;
                }
                printf(ML2_Indent("}\n", indent));
            }
        }
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF ML2_LayerCache ML2_ModelCacheInput(ML2_ModelCache modelCache) {
    return modelCache.caches[0];
}

ML2_DEF ML2_LayerCache ML2_ModelCacheOutput(ML2_ModelCache modelCache) {
    return modelCache.caches[modelCache.layers - 1];
}

ML2_DEF ML2_LayerCache ML2_ModelCacheOutputGradient(ML2_ModelCache modelCache) {
    return modelCache.cachesGradients[modelCache.layers - 1];
}

ML2_DEF void ML2_ModelCacheCopyBatchInput(ML2_ModelCache modelCache, ML2_Batch batch) {
    ML2_LayerCacheCopy(ML2_ModelCacheInput(modelCache), batch.input);
}

ML2_DEF ML2_Scalar ML2_ModelCacheLossForward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossForward lossForward) {
    return lossForward(ML2_ModelCacheOutput(modelCache), batch.output);
}

ML2_DEF void ML2_ModelCacheLossBackward(ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossBackward lossBackward) {
    lossBackward(ML2_ModelCacheOutput(modelCache), batch.output, ML2_ModelCacheOutputGradient(modelCache));
}

ML2_DEF void ML2_ModelCacheClearGradients(ML2_ModelCache modelCache) {
    // NOTE: clearing out cachesGradients is unnecessary, as they are temporary for backpropagation
    for (int i = 0; i < modelCache.layers - 1; i++) {
        ML2_LayerClear(modelCache.gradients[i]);
    }
}

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_DEF ML2_BatchInfo ML2_BatchInfoMake(ML2_LayerCacheInfo input, ML2_LayerCacheInfo output) {
    return (ML2_BatchInfo){input, output};
}

ML2_DEF ML2_Batch ML2_BatchNew(ML2_BatchInfo info) {
    return (ML2_Batch){ML2_LayerCacheNew(info.input), ML2_LayerCacheNew(info.output)};
}

ML2_DEF void ML2_BatchDestroy(ML2_Batch *batch) {
    ML2_LayerCacheDestroy(&batch->input);
    ML2_LayerCacheDestroy(&batch->output);
    // NOTE: this is not required because LayerCacheDestroy will zero it out, but whatever
    *batch = (ML2_Batch){};
}

ML2_DEF void ML2_BatchInfoPrint(ML2_Batch batch, int indent) {
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_Indent("Input:\n", indent));
        printf(ML2_Indent("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(batch.input), indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_Indent("}\n", indent));

        printf(ML2_Indent("Output:\n", indent));
        printf(ML2_Indent("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCacheInfoPrint(ML2_LayerCacheAsInfo(batch.output), indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_Indent("}\n", indent));
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF void ML2_BatchPrint(ML2_Batch batch, int indent) {
    printf(ML2_Indent("{\n", indent));
    {
        indent += ML2_Indentation;
        printf(ML2_Indent("Input:\n", indent));
        printf(ML2_Indent("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCachePrint(batch.input, indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_Indent("}\n", indent));

        printf(ML2_Indent("Output:\n", indent));
        printf(ML2_Indent("{\n", indent));
        {
            indent += ML2_Indentation;
            ML2_LayerCachePrint(batch.output, indent);
            indent -= ML2_Indentation;
        }
        printf(ML2_Indent("}\n", indent));
        indent -= ML2_Indentation;
    }
    printf(ML2_Indent("}\n", indent));
}

// ML2_Batch ⬆️

// ML2_Loss ⬇️

ML2_DEF ML2_Scalar ML2_LossForwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardSquareAverage works on scalars only");
    ML2_LayerCacheScalars predictedScalars = predicted.as.Scalars;
    ML2_LayerCacheScalars expectedScalars = expected.as.Scalars;
    ML2_ScalarsInfoSameAssert(expectedScalars.info, predictedScalars.info);
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    ML2_Scalar loss = {};
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar diff = *ML2_ScalarsAt(predictedScalars, i, j) - *ML2_ScalarsAt(expectedScalars, i, j);
            loss += diff * diff;
        }
    }
    loss /= (samples * scalars);
    return loss;
}

ML2_DEF void ML2_LossBackwardSquareAverage(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardSquareAverage works on scalars only");
    ML2_LayerCacheScalars predictedScalars = predicted.as.Scalars;
    ML2_LayerCacheScalars expectedScalars = expected.as.Scalars;
    ML2_LayerCacheScalars gradientScalars = gradient.as.Scalars;
    ML2_ScalarsInfoSameAssert(expectedScalars.info, predictedScalars.info);
    ML2_ScalarsInfoSameAssert(expectedScalars.info, gradientScalars.info);
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            ML2_Scalar diff = *ML2_ScalarsAt(predictedScalars, i, j) - *ML2_ScalarsAt(expectedScalars, i, j);
            *ML2_ScalarsAt(gradientScalars, i, j) = ML2_SCALAR_LITERAL(2.0) * diff / (samples * scalars);
        }
    }
}

ML2_DEF ML2_Scalar ML2_LossForwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = predicted.as.Scalars;
    ML2_LayerCacheScalars expectedScalars = expected.as.Scalars;
    ML2_ScalarsInfoSameAssert(expectedScalars.info, predictedScalars.info);
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    ML2_Scalar loss = {};
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            // TODO(12/3/26 22:40:30): 1e-7 is a magic value chosen by chatGPT, it should be checked if it is really the best value
            ML2_Scalar p = fmaxf(*ML2_ScalarsAt(predictedScalars, i, j), 1e-7f);
            loss -= *ML2_ScalarsAt(expectedScalars, i, j) * ML2_LOG(p);
        }
    }
    loss /= samples;
    return loss;
}

ML2_DEF void ML2_LossBackwardCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = predicted.as.Scalars;
    ML2_LayerCacheScalars expectedScalars = expected.as.Scalars;
    ML2_LayerCacheScalars gradientScalars = gradient.as.Scalars;
    ML2_ScalarsInfoSameAssert(expectedScalars.info, predictedScalars.info);
    ML2_ScalarsInfoSameAssert(expectedScalars.info, gradientScalars.info);
    int samples = expectedScalars.info.samples;
    int scalars = expectedScalars.info.scalars;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < scalars; j++) {
            // TODO(12/3/26 22:40:30): 1e-7 is a magic value chosen by chatGPT, it should be checked if it is really the best value
            ML2_Scalar p = fmaxf(*ML2_ScalarsAt(predictedScalars, i, j), 1e-7f);
            *ML2_ScalarsAt(gradientScalars, i, j) = -(*ML2_ScalarsAt(expectedScalars, i, j) / p) / samples;
        }
    }
}

ML2_DEF ML2_Scalar ML2_LossForwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && "ML2_LossForwardSoftmaxCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = predicted.as.Scalars;
    ML2_LayerCacheScalars expectedScalars = expected.as.Scalars;
    ML2_ScalarsInfoSameAssert(expectedScalars.info, predictedScalars.info);
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
            ML2_Scalar softmax = ML2_EXP(*ML2_ScalarsAt(predictedScalars, i, j) - max) / sum;
            ML2_Scalar p = fmaxf(softmax, 1e-7f);
            loss -= *ML2_ScalarsAt(expectedScalars, i, j) * ML2_LOG(p);
        }
    }
    loss /= samples;
    return loss;
}

ML2_DEF void ML2_LossBackwardSoftmaxCrossEntropy(ML2_LayerCache predicted, ML2_LayerCache expected, ML2_LayerCache gradient) {
    ML2_HARD_ASSERT(predicted.type == ML2_LayerCacheTypeScalars && expected.type == ML2_LayerCacheTypeScalars && gradient.type == ML2_LayerCacheTypeScalars && "ML2_LossBackwardSoftmaxCrossEntropy works on scalars only");
    ML2_LayerCacheScalars predictedScalars = predicted.as.Scalars;
    ML2_LayerCacheScalars expectedScalars = expected.as.Scalars;
    ML2_LayerCacheScalars gradientScalars = gradient.as.Scalars;
    ML2_ScalarsInfoSameAssert(expectedScalars.info, predictedScalars.info);
    ML2_ScalarsInfoSameAssert(expectedScalars.info, gradientScalars.info);
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
            ML2_Scalar softmax = ML2_EXP(*ML2_ScalarsAt(predictedScalars, i, j) - max) / sum;
            *ML2_ScalarsAt(gradientScalars, i, j) = (softmax - *ML2_ScalarsAt(expectedScalars, i, j)) / samples;
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
#define WeightsInfoSameAssert ML2_WeightsInfoSameAssert
#define WeightsInfoForwardCompatible ML2_WeightsInfoForwardCompatible
#define WeightsInfoForwardCompatibleAssert ML2_WeightsInfoForwardCompatibleAssert
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
#define BiasesInfoSameAssert ML2_BiasesInfoSameAssert
#define BiasesInfoForwardCompatible ML2_BiasesInfoForwardCompatible
#define BiasesInfoForwardCompatibleAssert ML2_BiasesInfoForwardCompatibleAssert
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
#define ActivationReLU ML2_ActivationReLU
#define ActivationSigmoid ML2_ActivationSigmoid
#define ActivationSoftmax ML2_ActivationSoftmax
#define ActivationNew ML2_ActivationNew
#define ActivationDestroy ML2_ActivationDestroy
#define LayerAsActivation ML2_LayerAsActivation
#define ActivationNameOf ML2_ActivationNameOf
#define ActivationInfoPrint ML2_ActivationInfoPrint
#define ActivationPrint ML2_ActivationPrint
#define ActivationInfoForward ML2_ActivationInfoForward
#define ActivationInfoSame ML2_ActivationInfoSame
#define ActivationInfoSameAssert ML2_ActivationInfoSameAssert
#define ActivationInfoForwardCompatible ML2_ActivationInfoForwardCompatible
#define ActivationInfoForwardCompatibleAssert ML2_ActivationInfoForwardCompatibleAssert
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
#define LinearInfoSameAssert ML2_LinearInfoSameAssert
#define LinearInfoForwardCompatible ML2_LinearInfoForwardCompatible
#define LinearInfoForwardCompatibleAssert ML2_LinearInfoForwardCompatibleAssert
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
#define LayerInfoSameAssert ML2_LayerInfoSameAssert
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
#define ScalarsInfoSameAssert ML2_ScalarsInfoSameAssert
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
#define LayerCacheInfoSameAssert ML2_LayerCacheInfoSameAssert
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
#define ModelCacheClearGradients ML2_ModelCacheClearGradients

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
    BUG(5/3/2026 20:35:53):
        Issue with stripping prefixes:
            The name "ML2_Scalars" becomes "Scalars" when stripped,
            however that is the same name that a "ML2_LayerCache" uses
            as a variable in the union, therefore you cannot do
            X.as.Scalars
            because it expands to
            X.as.ML2_Scalars
            same exact issue also applies to "ML2_Weights" "ML2_Biases" etc
*/