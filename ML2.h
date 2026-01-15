#ifndef _ML2_H
#    define _ML2_H

/*
    this library aims for minimal memory allocations

    *New calls allocate memory, so use *Destroy to free the memory
*/

#    ifndef ML2_DEF
#        define ML2_DEF static inline
#    endif // ML2_DEF

#    include <stdio.h>
#    include <stdbool.h>
#    include <string.h>
#    include <math.h>

#    define ML2_UNREACHABLE(msg, ...)                     \
        do {                                              \
            fprintf(stderr, "UNREACHABLE: \"" msg "\":\n" \
                            "    file:     |%s|\n"        \
                            "    function: |%s|\n"        \
                            "    line:     |%d|\n",       \
                ##__VA_ARGS__,                            \
                __FILE__,                                 \
                __func__,                                 \
                __LINE__);                                \
            abort();                                      \
        } while (0)

#    define ML2_TODO(msg, ...)                      \
        do {                                        \
            fprintf(stderr, "TODO: \"" msg "\":\n"  \
                            "    file:     |%s|\n"  \
                            "    function: |%s|\n"  \
                            "    line:     |%d|\n", \
                ##__VA_ARGS__,                      \
                __FILE__,                           \
                __func__,                           \
                __LINE__);                          \
            abort();                                \
        } while (0)

#    ifndef ML2_ASSERT
#        include <assert.h>
#        define ML2_ASSERT assert
#    endif // ML2_ASSERT

#    ifndef ML2_CALLOC
#        include <stdlib.h>
#        define ML2_CALLOC calloc
#    endif // ML2_CALLOC

#    ifndef ML2_FREE
#        include <stdlib.h>
#        define ML2_FREE free
#    endif // ML2_FREE

#    ifndef ML2_RELIABLE_CALLOC
static inline void *ML2_ReliableCalloc(size_t count, size_t size) {
    void *ptr = ML2_CALLOC(count, size);
    ML2_ASSERT(ptr != NULL && "BUY MORE RAM");
    return ptr;
}
#        define ML2_RELIABLE_CALLOC ML2_ReliableCalloc
#    endif // ML2_RELIABLE_CALLOC

#    ifndef ML2_Indentation
#        define ML2_Indentation 4
#    endif // ML2_Indentation

#    define ML2_Indent(fstr, i) "%*s" fstr, (i), ""

// Customizable but make sure to define all of these
#    ifndef ML2_SCALAR_TYPE
#        define ML2_SCALAR_TYPE float

#        define ML2_SCALAR_LITERAL(x) x##f
#        define ML2_EXP expf
#    endif // ML2_SCALAR_TYPE
typedef ML2_SCALAR_TYPE ML2_Scalar;

// TODO: figure out how to align the floats nicely
#    define ML2_ScalarFmt "%.3f"

typedef enum {
    ML2_ActNone = 0,
    ML2_ActReLU,
    ML2_ActSigmoid,
    ML2_ActCount,
} ML2_ActType;
static_assert(ML2_ActCount == 3, "ACTIVATIONS"); // for development

typedef ML2_Scalar (*ML2_Act)(ML2_Scalar);

// TODO: check if i can put values on the stack
typedef struct {
    int rows, cols, stride;
    // all matrix operations expect non-overlapping values
    ML2_Scalar *restrict values;
} ML2_Matrix;

// TODO: this struct is pretty useless, it should have more stuff
typedef struct {
    int nodes;
    ML2_ActType actType;
} ML2_Layer;

typedef struct {
    int count;
    ML2_Layer *layers;
} ML2_Arch;

#    define ML2_ArchMake(...)                                                \
        {                                                                    \
            .count = (sizeof((ML2_Layer[])__VA_ARGS__) / sizeof(ML2_Layer)), \
            .layers = (ML2_Layer[])__VA_ARGS__,                              \
        }

typedef struct {
    int count;
    ML2_Matrix *weights;
    ML2_Matrix *biases;
    // TODO: use the last activation for stuff like softmax
    ML2_ActType *activations;
} ML2_Model;

typedef struct {
    int count;
    ML2_Matrix *values;
    ML2_Matrix *valuesGradient;
    ML2_Matrix *weightsGradient;
    ML2_Matrix *biasesGradient;
} ML2_ModelCache;

typedef struct {
    int samples, inputs, outputs;
    ML2_Scalar *values;
} ML2_Batch;

typedef ML2_Scalar (*ML2_LossForward)(ML2_Matrix target, ML2_Matrix prediction);
typedef void (*ML2_LossBackward)(ML2_Matrix gradient, ML2_Matrix target, ML2_Matrix prediction);

ML2_DEF ML2_Scalar ML2_ScalarRand(ML2_Scalar low, ML2_Scalar high);

ML2_DEF ML2_Scalar ML2_ReLU(ML2_Scalar x);
ML2_DEF ML2_Scalar ML2_ReLUDerivative(ML2_Scalar x);
ML2_DEF ML2_Scalar ML2_Sigmoid(ML2_Scalar x);
ML2_DEF ML2_Scalar ML2_SigmoidDerivative(ML2_Scalar y);
ML2_DEF ML2_Act ML2_ActOf(ML2_ActType t);
ML2_DEF ML2_Act ML2_ActDerivativeOf(ML2_ActType t);
ML2_DEF const char *ML2_ActName(ML2_ActType t);

ML2_DEF ML2_Scalar *ML2_MatrixAt(ML2_Matrix matrix, int row, int col);
ML2_DEF ML2_Matrix ML2_MatrixNew(int rows, int cols);
ML2_DEF void ML2_MatrixDestroy(ML2_Matrix *matrix);
ML2_DEF void ML2_MatrixRand(ML2_Matrix matrix, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_MatrixPrint(ML2_Matrix matrix, int indent);
ML2_DEF ML2_Matrix ML2_MatrixRow(ML2_Matrix matrix, int row);
ML2_DEF ML2_Matrix ML2_MatrixCol(ML2_Matrix matrix, int col);
ML2_DEF bool ML2_MatrixEqual(ML2_Matrix a, ML2_Matrix b);
ML2_DEF void ML2_MatrixAssertEqual(ML2_Matrix a, ML2_Matrix b);
// Dest[i,j] = 0
ML2_DEF void ML2_MatrixClear(ML2_Matrix dest);
// Dest[i,j] *= Scale
ML2_DEF void ML2_MatrixScale(ML2_Matrix dest, ML2_Scalar scale);
// Dest[i,j] = Src[i,j]
ML2_DEF void ML2_MatrixCopy(ML2_Matrix dest, ML2_Matrix src);
// Dest[i,j] = Σ(A[i,k] * B[k,j])
ML2_DEF void ML2_MatrixDot(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b);
ML2_DEF bool ML2_MatrixDotCompatible(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b);
ML2_DEF void ML2_MatrixDotAssertCompatible(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b);
// Dest[i,j] += A[i,j]
ML2_DEF void ML2_MatrixSum(ML2_Matrix dest, ML2_Matrix a);
ML2_DEF bool ML2_MatrixSumCompatible(ML2_Matrix dest, ML2_Matrix a);
ML2_DEF void ML2_MatrixSumAssertCompatible(ML2_Matrix dest, ML2_Matrix a);
// Dest[i,j] = f(Dest[i,j]);
ML2_DEF void ML2_MatrixActivate(ML2_Matrix dest, ML2_Act f);

ML2_DEF ML2_Model ML2_ModelNew(ML2_Arch arch);
ML2_DEF void ML2_ModelDestroy(ML2_Model *model);
ML2_DEF void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_ModelPrint(ML2_Model model, int indent);
// Z[i+1] = Act(W*X[i]+B);
ML2_DEF void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch);
ML2_DEF ML2_Scalar ML2_ModelLoss(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossForward lossForward);
ML2_DEF void ML2_ModelGradientFiniteDiff(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch, ML2_Scalar epsilon, ML2_LossForward lossForward);
ML2_DEF void ML2_ModelGradientBackprop(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossBackward lossBackward);
ML2_DEF void ML2_ModelGradientDescent(ML2_Model model, ML2_ModelCache modelCache, ML2_Scalar learningRate);
ML2_DEF bool ML2_ModelCompatibleCache(ML2_Model model, ML2_ModelCache modelCache);
ML2_DEF void ML2_ModelAssertCompatibleCache(ML2_Model model, ML2_ModelCache modelCache);

ML2_DEF ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, int samples);
ML2_DEF void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache);
ML2_DEF void ML2_ModelCachePrint(ML2_ModelCache modelCache, int indent);
ML2_DEF void ML2_ModelCacheClear(ML2_ModelCache modelCache);
ML2_DEF ML2_Matrix ML2_ModelCacheInput(ML2_ModelCache modelCache);
ML2_DEF ML2_Matrix ML2_ModelCacheOutput(ML2_ModelCache modelCache);
ML2_DEF ML2_Matrix ML2_ModelCacheInputGradient(ML2_ModelCache modelCache);
ML2_DEF ML2_Matrix ML2_ModelCacheOutputGradient(ML2_ModelCache modelCache);
ML2_DEF bool ML2_ModelCacheCompatibleBatch(ML2_ModelCache modelCache, ML2_Batch batch);
ML2_DEF void ML2_ModelCacheAssertCompatibleBatch(ML2_ModelCache modelCache, ML2_Batch batch);

ML2_DEF ML2_Batch ML2_BatchNew(int samples, int inputs, int outputs);
ML2_DEF void ML2_BatchDestroy(ML2_Batch *batch);
ML2_DEF void ML2_BatchPrint(ML2_Batch batch, int indent);
ML2_DEF ML2_Batch ML2_BatchSlice(ML2_Batch batch, int start, int samples);
ML2_DEF ML2_Matrix ML2_BatchInput(ML2_Batch batch);
ML2_DEF ML2_Matrix ML2_BatchOutput(ML2_Batch batch);
ML2_DEF ML2_Matrix ML2_BatchSampleInput(ML2_Batch batch, int sample);
ML2_DEF ML2_Matrix ML2_BatchSampleOutput(ML2_Batch batch, int sample);

ML2_DEF ML2_Scalar ML2_LossForwardSquareError(ML2_Matrix target, ML2_Matrix prediction);
ML2_DEF void ML2_LossBackwardSquareError(ML2_Matrix gradient, ML2_Matrix target, ML2_Matrix prediction);

#endif // _ML2_H

#ifdef ML2_IMPLEMENTATION

// ML2_ ⬇️
// ML2_ ⬆️

// ML2_Scalar ⬇️

ML2_DEF ML2_Scalar ML2_ScalarRand(ML2_Scalar low, ML2_Scalar high) {
    return ((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX) * (high - low) + low;
}

// ML2_Scalar ⬆️

// ML2_Act ⬇️

ML2_DEF ML2_Scalar ML2_ReLU(ML2_Scalar x) {
    if (x > ML2_SCALAR_LITERAL(0.0)) return x;
    return ML2_SCALAR_LITERAL(0.0);
}

ML2_DEF ML2_Scalar ML2_ReLUDerivative(ML2_Scalar x) {
    if (x > ML2_SCALAR_LITERAL(0.0)) return ML2_SCALAR_LITERAL(1.0);
    return ML2_SCALAR_LITERAL(0.0);
}

ML2_DEF ML2_Scalar ML2_Sigmoid(ML2_Scalar x) {
    return ML2_SCALAR_LITERAL(1.0) / (ML2_SCALAR_LITERAL(1.0) + ML2_EXP(-x));
}

ML2_DEF ML2_Scalar ML2_SigmoidDerivative(ML2_Scalar y) {
    return y * (ML2_SCALAR_LITERAL(1.0) - y);
}

// TODO: ActNone can be a linear function that just returns x, so that you dont need to if when you use Acts
ML2_DEF ML2_Act ML2_ActOf(ML2_ActType t) {
    switch (t) {
        case ML2_ActNone:
            return NULL;
        case ML2_ActReLU:
            return ML2_ReLU;
        case ML2_ActSigmoid:
            return ML2_Sigmoid;
        default:
            ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_DEF ML2_Act ML2_ActDerivativeOf(ML2_ActType t) {
    switch (t) {
        case ML2_ActNone:
            return NULL;
        case ML2_ActReLU:
            return ML2_ReLUDerivative;
        case ML2_ActSigmoid:
            return ML2_SigmoidDerivative;
        default:
            ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_DEF const char *ML2_ActName(ML2_ActType t) {
    switch (t) {
        case ML2_ActNone:
            return "None";
        case ML2_ActReLU:
            return "ReLU";
        case ML2_ActSigmoid:
            return "Sigmoid";
        default:
            ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

// ML2_Act ⬆️

// ML2_Matrix ⬇️

ML2_DEF ML2_Scalar *ML2_MatrixAt(ML2_Matrix matrix, int row, int col) {
    ML2_ASSERT(0 <= row && row < matrix.rows && 0 <= col && col < matrix.cols && "INVALID MATRIX INDICES");
    return &matrix.values[row * matrix.stride + col];
}

ML2_DEF ML2_Matrix ML2_MatrixNew(int rows, int cols) {
    ML2_Matrix matrix = {
        .rows = rows,
        .cols = cols,
        .stride = cols,
        .values = ML2_RELIABLE_CALLOC(rows * cols, sizeof(*matrix.values)),
    };
    return matrix;
}

ML2_DEF void ML2_MatrixDestroy(ML2_Matrix *matrix) {
    ML2_FREE(matrix->values);
    memset(matrix, 0, sizeof(*matrix));
}

ML2_DEF void ML2_MatrixRand(ML2_Matrix matrix, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            *ML2_MatrixAt(matrix, i, j) = ML2_ScalarRand(low, high);
        }
    }
}

ML2_DEF void ML2_MatrixPrint(ML2_Matrix matrix, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int row = 0; row < matrix.rows; row++) {
        printf(ML2_Indent("", indent + ML2_Indentation));
        for (int col = 0; col < matrix.cols; col++) {
            printf(ML2_ScalarFmt " ", *ML2_MatrixAt(matrix, row, col));
        }
        printf("\n");
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF ML2_Matrix ML2_MatrixRow(ML2_Matrix matrix, int row) {
    ML2_Matrix r = {
        .rows = 1,
        .cols = matrix.cols,
        .stride = matrix.stride,
        .values = ML2_MatrixAt(matrix, row, 0),
    };
    return r;
}

ML2_DEF ML2_Matrix ML2_MatrixCol(ML2_Matrix matrix, int col) {
    ML2_Matrix c = {
        .rows = matrix.rows,
        .cols = 1,
        .stride = matrix.stride,
        .values = ML2_MatrixAt(matrix, 0, col),
    };
    return c;
}

ML2_DEF bool ML2_MatrixEqual(ML2_Matrix a, ML2_Matrix b) {
    return a.rows == b.rows &&
           a.cols == b.cols;
}

ML2_DEF void ML2_MatrixAssertEqual(ML2_Matrix a, ML2_Matrix b) {
    ML2_ASSERT(ML2_MatrixEqual(a, b) && "MATRIX SIZES MUST MATCH");
}

// Dest[i,j] = 0
ML2_DEF void ML2_MatrixClear(ML2_Matrix dest) {
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = 0;
        }
    }
}

// Dest[i,j] *= Scale
ML2_DEF void ML2_MatrixScale(ML2_Matrix dest, ML2_Scalar scale) {
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) *= scale;
        }
    }
}

// Dest[i,j] = Src[i,j]
ML2_DEF void ML2_MatrixCopy(ML2_Matrix dest, ML2_Matrix src) {
    ML2_MatrixAssertEqual(dest, src);
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = *ML2_MatrixAt(src, i, j);
        }
    }
}

// Dest[i,j] = Σ(A[i,k] * B[k,j])
ML2_DEF void ML2_MatrixDot(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b) {
    ML2_MatrixDotAssertCompatible(dest, a, b);
    int inner = a.cols;

    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            ML2_Scalar sum = ML2_SCALAR_LITERAL(0.0);
            for (int k = 0; k < inner; k++) {
                sum += *ML2_MatrixAt(a, i, k) * *ML2_MatrixAt(b, k, j);
            }
            *ML2_MatrixAt(dest, i, j) = sum;
        }
    }
}

ML2_DEF bool ML2_MatrixDotCompatible(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b) {
    if (a.cols != b.rows) return false;
    if (dest.rows != a.rows) return false;
    if (dest.cols != b.cols) return false;
    if (dest.values == a.values) return false;
    if (dest.values == b.values) return false;
    return true;
}

ML2_DEF void ML2_MatrixDotAssertCompatible(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b) {
    ML2_ASSERT(ML2_MatrixDotCompatible(dest, a, b) && "MATRICES MUST BE DOT COMPATIBLE");
}

// Dest[i,j] += A[i,j]
ML2_DEF void ML2_MatrixSum(ML2_Matrix dest, ML2_Matrix a) {
    ML2_MatrixSumAssertCompatible(dest, a);
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) += *ML2_MatrixAt(a, i, j);
        }
    }
}

ML2_DEF bool ML2_MatrixSumCompatible(ML2_Matrix dest, ML2_Matrix a) {
    if (!ML2_MatrixEqual(dest, a)) return false;
    if (dest.values == a.values) return false;
    return true;
}

ML2_DEF void ML2_MatrixSumAssertCompatible(ML2_Matrix dest, ML2_Matrix a) {
    ML2_ASSERT(ML2_MatrixSumCompatible(dest, a) && "MATRICES MUST BE SUM COMPATIBLE");
}

// Dest[i,j] = f(Dest[i,j])
ML2_DEF void ML2_MatrixActivate(ML2_Matrix dest, ML2_Act f) {
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = f(*ML2_MatrixAt(dest, i, j));
        }
    }
}

// ML2_Matrix ⬆️

// ML2_Model ⬇️

ML2_DEF ML2_Model ML2_ModelNew(ML2_Arch arch) {
    int layerCount = arch.count;
    ML2_Model model = {.count = layerCount - 1};
    model.weights = ML2_RELIABLE_CALLOC(layerCount - 1, sizeof(*model.weights));
    model.biases = ML2_RELIABLE_CALLOC(layerCount - 1, sizeof(*model.biases));
    model.activations = ML2_RELIABLE_CALLOC(layerCount - 1, sizeof(*model.activations));
    for (int i = 0; i < layerCount - 1; i++) {
        int nodes = arch.layers[i].nodes;
        ML2_ASSERT(nodes > 0 && "LAYERS MUST HAVE MORE THAN 0 NODES");
        int nextNodes = arch.layers[i + 1].nodes;
        ML2_ASSERT(nextNodes > 0 && "LAYERS MUST HAVE MORE THAN 0 NODES"); // TODO: this assert runs twice on the same values, problematic but not really
        model.weights[i] = ML2_MatrixNew(nodes, nextNodes);
        model.biases[i] = ML2_MatrixNew(1, nextNodes);
        model.activations[i] = arch.layers[i].actType;
    }
    return model;
}

ML2_DEF void ML2_ModelDestroy(ML2_Model *model) {
    for (int i = 0; i < model->count; i++) {
        ML2_MatrixDestroy(&model->weights[i]);
        ML2_MatrixDestroy(&model->biases[i]);
    }
    ML2_FREE(model->weights);
    ML2_FREE(model->biases);
    ML2_FREE(model->activations);
    memset(model, 0, sizeof(*model));
}

ML2_DEF void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < model.count; i++) {
        ML2_MatrixRand(model.weights[i], low, high);
        ML2_MatrixRand(model.biases[i], low, high);
    }
}

ML2_DEF void ML2_ModelPrint(ML2_Model model, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < model.count; i++) {
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix biases = model.biases[i];
        // printf("%*sWeights[%d](%dx%d):\n", indent + ML2_Indentation, "", i, weights.rows, weights.cols);
        printf(ML2_Indent("Weights[%d](%dx%d):\n", indent + ML2_Indentation), i, weights.rows, weights.cols);
        ML2_MatrixPrint(weights, indent + ML2_Indentation);
        // printf("%*sBiases[%d](%dx%d):\n", indent + ML2_Indentation, "", i, biases.rows, biases.cols);
        printf(ML2_Indent("Biases[%d](%dx%d):\n", indent + ML2_Indentation), i, biases.rows, biases.cols);
        ML2_MatrixPrint(biases, indent + ML2_Indentation);
        // printf("%*sActivation[%d]: \"%s\"\n", indent + ML2_Indentation, "", i, ML2_ActName(model.activations[i]));
        printf(ML2_Indent("Activation[%d]: \"%s\"\n", indent + ML2_Indentation), i, ML2_ActName(model.activations[i]));
    }
    printf(ML2_Indent("}\n", indent));
}

// Z[i+1] = Act(X[i]*W+B)
ML2_DEF void ML2_ModelForward(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch) {
    ML2_ModelAssertCompatibleCache(model, modelCache);
    ML2_ModelCacheAssertCompatibleBatch(modelCache, batch);

    ML2_MatrixCopy(ML2_ModelCacheInput(modelCache), ML2_BatchInput(batch));
    for (int i = 0; i < model.count; i++) {
        ML2_Matrix nextValues = modelCache.values[i + 1];
        ML2_MatrixDot(nextValues, modelCache.values[i], model.weights[i]);
        for (int j = 0; j < batch.samples; j++) {
            ML2_Matrix valuesRow = ML2_MatrixRow(nextValues, j);
            ML2_MatrixSum(valuesRow, model.biases[i]);
        }
        if (model.activations[i] != ML2_ActNone) {
            ML2_MatrixActivate(nextValues, ML2_ActOf(model.activations[i]));
        }
    }
}

ML2_DEF ML2_Scalar ML2_ModelLoss(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossForward lossForward) {
    ML2_ModelAssertCompatibleCache(model, modelCache);
    ML2_ModelCacheAssertCompatibleBatch(modelCache, batch);

    ML2_ModelForward(model, modelCache, batch);
    // it is asserted that
    // modelOutput.rows == batch.samples
    // modelOutput.cols == batch.outputs
    ML2_Matrix modelOutput = ML2_ModelCacheOutput(modelCache);
    ML2_Matrix batchOutput = ML2_BatchOutput(batch);
    return lossForward(batchOutput, modelOutput);
}

ML2_DEF void ML2_ModelGradientFiniteDiff(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch, ML2_Scalar epsilon, ML2_LossForward lossForward) {
    ML2_ASSERT(epsilon > 0 && "EPSILON MUST BE POSITIVE");
    ML2_ModelAssertCompatibleCache(model, modelCache);
    ML2_ModelCacheAssertCompatibleBatch(modelCache, batch);
    // f'(x) = lim(e -> 0): (f(x + e) - f(x)) / e
    ML2_Scalar prev = ML2_SCALAR_LITERAL(0.0);
    ML2_Scalar curLoss = ML2_ModelLoss(model, modelCache, batch, lossForward);
    for (int i = 0; i < model.count; i++) {
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix weightsGradient = modelCache.weightsGradient[i];
        for (int j = 0; j < weights.rows; j++) {
            for (int k = 0; k < weights.cols; k++) {
                ML2_Scalar *cur = ML2_MatrixAt(weights, j, k);
                prev = *cur;
                *cur += epsilon;
                ML2_Scalar newLoss = ML2_ModelLoss(model, modelCache, batch, lossForward);
                *ML2_MatrixAt(weightsGradient, j, k) = (newLoss - curLoss) / epsilon;
                *cur = prev;
            }
        }

        ML2_Matrix biases = model.biases[i];
        ML2_Matrix biasesGradient = modelCache.biasesGradient[i];
        for (int j = 0; j < biases.rows; j++) {
            ML2_Scalar *cur = ML2_MatrixAt(biases, 0, j);
            prev = *cur;
            *cur += epsilon;
            ML2_Scalar newLoss = ML2_ModelLoss(model, modelCache, batch, lossForward);
            *ML2_MatrixAt(biasesGradient, 0, j) = (newLoss - curLoss) / epsilon;
            *cur = prev;
        }
    }
}

ML2_DEF void ML2_ModelGradientBackprop(ML2_Model model, ML2_ModelCache modelCache, ML2_Batch batch, ML2_LossBackward lossBackward) {
    ML2_ModelAssertCompatibleCache(model, modelCache);
    ML2_ModelCacheAssertCompatibleBatch(modelCache, batch);

    ML2_ModelCacheClear(modelCache);

    ML2_ModelForward(model, modelCache, batch);
    // these 3 matrices are asserted to be compatible sizes from the 2 asserts above
    ML2_Matrix modelOutput = ML2_ModelCacheOutput(modelCache);
    ML2_Matrix modelOutputGradient = ML2_ModelCacheOutputGradient(modelCache);
    ML2_Matrix batchOutput = ML2_BatchOutput(batch);

    lossBackward(modelOutputGradient, batchOutput, modelOutput);

    int samples = batch.samples;
    for (int l = model.count - 1; l >= 0; l--) {
        ML2_Matrix values = modelCache.values[l];
        ML2_Matrix weights = model.weights[l];
        ML2_ActType actType = model.activations[l];
        ML2_Matrix valuesGradient = modelCache.valuesGradient[l];
        ML2_Matrix weightsGradient = modelCache.weightsGradient[l];
        ML2_Matrix biasesGradient = modelCache.biasesGradient[l];
        ML2_Matrix nextValues = modelCache.values[l + 1];
        ML2_Matrix nextValuesGradient = modelCache.valuesGradient[l + 1];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < nextValues.cols; j++) {
                ML2_Scalar chainDerivative = *ML2_MatrixAt(nextValuesGradient, i, j);
                if (actType != ML2_ActNone) {
                    chainDerivative *= ML2_ActDerivativeOf(actType)(*ML2_MatrixAt(nextValues, i, j));
                }
                for (int k = 0; k < weights.rows; k++) {
                    *ML2_MatrixAt(valuesGradient, i, k) += chainDerivative * *ML2_MatrixAt(weights, k, j);
                    *ML2_MatrixAt(weightsGradient, k, j) += chainDerivative * *ML2_MatrixAt(values, i, k);
                }
                *ML2_MatrixAt(biasesGradient, 0, j) += chainDerivative;
            }
        }
    }
}

ML2_DEF void ML2_ModelGradientDescent(ML2_Model model, ML2_ModelCache modelCache, ML2_Scalar learningRate) {
    ML2_ModelAssertCompatibleCache(model, modelCache);
    for (int i = 0; i < model.count; i++) {
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix weightsGradient = modelCache.weightsGradient[i];
        for (int j = 0; j < weights.rows; j++) {
            for (int k = 0; k < weights.cols; k++) {
                *ML2_MatrixAt(weights, j, k) -= learningRate * *ML2_MatrixAt(weightsGradient, j, k);
            }
        }

        ML2_Matrix biases = model.biases[i];
        ML2_Matrix biasesGradient = modelCache.biasesGradient[i];
        for (int j = 0; j < biases.cols; j++) {
            *ML2_MatrixAt(biases, 0, j) -= learningRate * *ML2_MatrixAt(biasesGradient, 0, j);
        }
    }
}

ML2_DEF bool ML2_ModelCompatibleCache(ML2_Model model, ML2_ModelCache modelCache) {
    if (model.count + 1 != modelCache.count) return false;
    for (int i = 0; i < model.count; i++) {
        ML2_Matrix nextValues = modelCache.values[i + 1];
        if (!ML2_MatrixDotCompatible(nextValues, modelCache.values[i], model.weights[i])) return false;
        ML2_Matrix nextValuesRow = ML2_MatrixRow(nextValues, 0);
        if (!ML2_MatrixSumCompatible(nextValuesRow, model.biases[i])) return false;
        if (!ML2_MatrixEqual(model.weights[i], modelCache.weightsGradient[i])) return false;
        if (!ML2_MatrixEqual(model.biases[i], modelCache.biasesGradient[i])) return false;
    }
    return true;
}

ML2_DEF void ML2_ModelAssertCompatibleCache(ML2_Model model, ML2_ModelCache modelCache) {
    ML2_ASSERT(ML2_ModelCompatibleCache(model, modelCache) && "MODEL MUST BE COMPATIBLE WITH CACHE");
}

// ML2_Model ⬆️

// ML2_ModelCache ⬇️

ML2_DEF ML2_ModelCache ML2_ModelCacheNew(ML2_Arch arch, int samples) {
    int layerCount = arch.count;
    ML2_ModelCache modelCache = {.count = layerCount};
    modelCache.values = ML2_RELIABLE_CALLOC(layerCount, sizeof(*modelCache.values));
    modelCache.valuesGradient = ML2_RELIABLE_CALLOC(layerCount, sizeof(*modelCache.valuesGradient));
    modelCache.weightsGradient = ML2_RELIABLE_CALLOC(layerCount - 1, sizeof(*modelCache.weightsGradient));
    modelCache.biasesGradient = ML2_RELIABLE_CALLOC(layerCount - 1, sizeof(*modelCache.biasesGradient));
    for (int i = 0; i < layerCount; i++) {
        int nodes = arch.layers[i].nodes;
        ML2_ASSERT(nodes > 0 && "LAYERS MUST HAVE MORE THAN 0 NODES");
        modelCache.values[i] = ML2_MatrixNew(samples, nodes);
        modelCache.valuesGradient[i] = ML2_MatrixNew(samples, nodes);
        if (i < layerCount - 1) {
            int nextNodes = arch.layers[i + 1].nodes;
            ML2_ASSERT(nextNodes > 0 && "LAYERS MUST HAVE MORE THAN 0 NODES"); // TODO: this assert runs twice on the same values, problematic but not really
            modelCache.weightsGradient[i] = ML2_MatrixNew(nodes, nextNodes);
            modelCache.biasesGradient[i] = ML2_MatrixNew(1, nextNodes);
        }
    }
    return modelCache;
}

ML2_DEF void ML2_ModelCacheDestroy(ML2_ModelCache *modelCache) {
    for (int i = 0; i < modelCache->count; i++) {
        ML2_MatrixDestroy(&modelCache->values[i]);
        ML2_MatrixDestroy(&modelCache->valuesGradient[i]);
        if (i < modelCache->count - 1) {
            ML2_MatrixDestroy(&modelCache->weightsGradient[i]);
            ML2_MatrixDestroy(&modelCache->biasesGradient[i]);
        }
    }
    ML2_FREE(modelCache->values);
    ML2_FREE(modelCache->valuesGradient);
    ML2_FREE(modelCache->weightsGradient);
    ML2_FREE(modelCache->biasesGradient);
    memset(modelCache, 0, sizeof(*modelCache));
}

ML2_DEF void ML2_ModelCachePrint(ML2_ModelCache modelCache, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < modelCache.count; i++) {
        ML2_Matrix values = modelCache.values[i];
        ML2_Matrix valuesGradient = modelCache.valuesGradient[i];
        ML2_Matrix weightsGradient = modelCache.weightsGradient[i];
        ML2_Matrix biasesGradient = modelCache.biasesGradient[i];
        printf(ML2_Indent("Values[%d](%dx%d):\n", indent + ML2_Indentation), i, values.rows, values.cols);
        ML2_MatrixPrint(values, indent + ML2_Indentation);
        printf(ML2_Indent("Values Gradient[%d](%dx%d):\n", indent + ML2_Indentation), i, valuesGradient.rows, valuesGradient.cols);
        ML2_MatrixPrint(valuesGradient, indent + ML2_Indentation);
        if (i < modelCache.count - 1) {
            printf(ML2_Indent("Weights Gradient[%d](%dx%d):\n", indent + ML2_Indentation), i, weightsGradient.rows, weightsGradient.cols);
            ML2_MatrixPrint(weightsGradient, indent + ML2_Indentation);
            printf(ML2_Indent("Biases Gradient[%d](%dx%d):\n", indent + ML2_Indentation), i, biasesGradient.rows, biasesGradient.cols);
            ML2_MatrixPrint(biasesGradient, indent + ML2_Indentation);
        }
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF void ML2_ModelCacheClear(ML2_ModelCache modelCache) {
    for (int i = 0; i < modelCache.count; i++) {
        ML2_MatrixClear(modelCache.values[i]);
        ML2_MatrixClear(modelCache.valuesGradient[i]);
        if (i < modelCache.count - 1) {
            ML2_MatrixClear(modelCache.weightsGradient[i]);
            ML2_MatrixClear(modelCache.biasesGradient[i]);
        }
    }
}

ML2_DEF ML2_Matrix ML2_ModelCacheInput(ML2_ModelCache modelCache) {
    return modelCache.values[0];
}

ML2_DEF ML2_Matrix ML2_ModelCacheOutput(ML2_ModelCache modelCache) {
    return modelCache.values[modelCache.count - 1];
}

ML2_DEF ML2_Matrix ML2_ModelCacheInputGradient(ML2_ModelCache modelCache) {
    return modelCache.valuesGradient[0];
}

ML2_DEF ML2_Matrix ML2_ModelCacheOutputGradient(ML2_ModelCache modelCache) {
    return modelCache.valuesGradient[modelCache.count - 1];
}

ML2_DEF bool ML2_ModelCacheCompatibleBatch(ML2_ModelCache modelCache, ML2_Batch batch) {
    ML2_Matrix modelInput = ML2_ModelCacheInput(modelCache);
    ML2_Matrix modelOutput = ML2_ModelCacheOutput(modelCache);
    ML2_Matrix modelInputGradient = ML2_ModelCacheInputGradient(modelCache);
    ML2_Matrix modelOutputGradient = ML2_ModelCacheOutputGradient(modelCache);
    ML2_Matrix batchInput = ML2_BatchInput(batch);
    ML2_Matrix batchOutput = ML2_BatchOutput(batch);
    // TODO: the batch may have less samples than the ModelCache, in case of batching, so modelInput.cols >= samples
    if (!ML2_MatrixEqual(modelInput, batchInput)) return false;
    if (!ML2_MatrixEqual(modelOutput, batchOutput)) return false;

    // TODO: this assertion shouldn't trigger IF AND ONLY IF the modelCache was created through ModelCacheNew properly
    if (!ML2_MatrixEqual(modelInputGradient, batchInput)) return false;
    if (!ML2_MatrixEqual(modelOutputGradient, batchOutput)) return false;
    return true;
}

ML2_DEF void ML2_ModelCacheAssertCompatibleBatch(ML2_ModelCache modelCache, ML2_Batch batch) {
    ML2_ASSERT(ML2_ModelCacheCompatibleBatch(modelCache, batch) && "MODEL CACHE MUST BE COMPATIBLE WITH BATCH");
}

// ML2_ModelCache ⬆️

// ML2_Batch ⬇️

ML2_DEF ML2_Batch ML2_BatchNew(int samples, int inputs, int outputs) {
    ML2_Batch batch = {
        .samples = samples,
        .inputs = inputs,
        .outputs = outputs,
        .values = ML2_CALLOC(samples * (inputs + outputs), sizeof(*batch.values)),
    };
    return batch;
}

ML2_DEF void ML2_BatchDestroy(ML2_Batch *batch) {
    ML2_FREE(batch->values);
    memset(batch, 0, sizeof(*batch));
}

ML2_DEF void ML2_BatchPrint(ML2_Batch batch, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < batch.samples; i++) {
        printf(ML2_Indent("", indent + ML2_Indentation));
        for (int j = 0; j < batch.inputs; j++) {
            printf(ML2_ScalarFmt, batch.values[i * (batch.inputs + batch.outputs) + j]);
            if (j < batch.inputs - 1) {
                printf(", ");
            }
        }
        printf("    ");
        for (int j = 0; j < batch.outputs; j++) {
            printf(ML2_ScalarFmt, batch.values[i * (batch.inputs + batch.outputs) + batch.inputs + j]);
            if (j < batch.outputs - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF ML2_Batch ML2_BatchSlice(ML2_Batch batch, int start, int samples) {
    ML2_ASSERT(0 <= start && samples > 0 && start + samples <= batch.samples && "INVALID BATCH SLICE");

    ML2_Batch batchSlice = {
        .inputs = batch.inputs,
        .outputs = batch.outputs,
        .samples = samples,
        .values = &batch.values[start * (batch.inputs + batch.outputs)],
    };
    return batchSlice;
}

ML2_DEF ML2_Matrix ML2_BatchInput(ML2_Batch batch) {
    ML2_Matrix input = {
        .rows = batch.samples,
        .cols = batch.inputs,
        .stride = batch.inputs + batch.outputs,
        .values = &batch.values[0],
    };
    return input;
}

ML2_DEF ML2_Matrix ML2_BatchOutput(ML2_Batch batch) {
    ML2_Matrix output = {
        .rows = batch.samples,
        .cols = batch.outputs,
        .stride = batch.inputs + batch.outputs,
        .values = &batch.values[batch.inputs],
    };
    return output;
}

ML2_DEF ML2_Matrix ML2_BatchSampleInput(ML2_Batch batch, int sample) {
    ML2_ASSERT(0 <= sample && sample < batch.samples && "INVALID SAMPLE INDEX");

    ML2_Matrix input = {
        .rows = 1,
        .cols = batch.inputs,
        .stride = batch.inputs + batch.outputs,
        .values = &batch.values[input.stride * sample],
    };
    return input;
}

ML2_DEF ML2_Matrix ML2_BatchSampleOutput(ML2_Batch batch, int sample) {
    ML2_ASSERT(0 <= sample && sample < batch.samples && "INVALID SAMPLE INDEX");

    ML2_Matrix output = {
        .rows = 1,
        .cols = batch.outputs,
        .stride = batch.inputs + batch.outputs,
        .values = &batch.values[output.stride * sample + batch.inputs],
    };
    return output;
}

// ML2_Batch ⬆️

// ML2_Loss ⬇️

ML2_DEF ML2_Scalar ML2_LossForwardSquareError(ML2_Matrix target, ML2_Matrix prediction) {
    ML2_MatrixAssertEqual(target, prediction);
    // target.rows == prediction.rows == samples
    // target.cols == prediction.cols == outputs
    int samples = target.rows;
    int outputs = target.cols;

    ML2_Scalar loss = ML2_SCALAR_LITERAL(0.0);
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < outputs; j++) {
            ML2_Scalar diff = *ML2_MatrixAt(target, i, j) - *ML2_MatrixAt(prediction, i, j);
            loss += diff * diff;
        }
    }
    loss /= (samples * outputs);
    return loss;
}

ML2_DEF void ML2_LossBackwardSquareError(ML2_Matrix gradient, ML2_Matrix target, ML2_Matrix prediction) {
    ML2_MatrixAssertEqual(gradient, target);
    ML2_MatrixAssertEqual(target, prediction);
    // gradient.rows == target.rows == prediction.rows == samples
    // gradient.cols == target.cols == prediction.cols == outputs
    int samples = target.rows;
    int outputs = target.cols;

    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < outputs; j++) {
            ML2_Scalar diff = *ML2_MatrixAt(target, i, j) - *ML2_MatrixAt(prediction, i, j);
            *ML2_MatrixAt(gradient, i, j) = ML2_SCALAR_LITERAL(-2.0) * diff / (samples * outputs);
        }
    }
}

// ML2_Loss ⬆️

#endif // ML2_IMPLEMENTATION

#ifdef ML2_STRIP_PREFIX

#    ifndef _ML2_H_STRIP_PREFIX_GUARD
#        define _ML2_H_STRIP_PREFIX_GUARD

#        define Scalar ML2_Scalar
#        define ActNone ML2_ActNone
#        define ActSigmoid ML2_ActSigmoid
#        define ActType ML2_ActType
#        define Act ML2_Act
#        define Matrix ML2_Matrix
#        define Arch ML2_Arch
#        define Layer ML2_Layer
#        define ArchMake ML2_ArchMake
#        define Model ML2_Model
#        define ModelCache ML2_ModelCache
#        define Batch ML2_Batch

#        define ScalarRand ML2_ScalarRand

#        define ReLU ML2_ReLU
#        define ReLUDerivative ML2_ReLUDerivative
#        define Sigmoid ML2_Sigmoid
#        define SigmoidDerivative ML2_SigmoidDerivative
#        define ActOf ML2_ActOf
#        define ActDerivativeOf ML2_ActDerivativeOf
#        define ActName ML2_ActName

#        define MatrixAt ML2_MatrixAt
#        define MatrixNew ML2_MatrixNew
#        define MatrixDestroy ML2_MatrixDestroy
#        define MatrixRand ML2_MatrixRand
#        define MatrixPrint ML2_MatrixPrint
#        define MatrixRow ML2_MatrixRow
#        define MatrixCol ML2_MatrixCol
#        define MatrixEqual ML2_MatrixEqual
#        define MatrixAssertEqual ML2_MatrixAssertEqual
#        define MatrixClear ML2_MatrixClear
#        define MatrixScale ML2_MatrixScale
#        define MatrixCopy ML2_MatrixCopy
#        define MatrixDot ML2_MatrixDot
#        define MatrixDotCompatible ML2_MatrixDotCompatible
#        define MatrixDotAssertCompatible ML2_MatrixDotAssertCompatible
#        define MatrixSum ML2_MatrixSum
#        define MatrixSumCompatible ML2_MatrixSumCompatible
#        define MatrixSumAssertCompatible ML2_MatrixSumAssertCompatible
#        define MatrixActivate ML2_MatrixActivate

#        define ModelNew ML2_ModelNew
#        define ModelDestroy ML2_ModelDestroy
#        define ModelRand ML2_ModelRand
#        define ModelPrint ML2_ModelPrint
#        define ModelForward ML2_ModelForward
#        define ModelLoss ML2_ModelLoss
#        define ModelGradientFiniteDiff ML2_ModelGradientFiniteDiff
#        define ModelGradientBackprop ML2_ModelGradientBackprop
#        define ModelGradientDescent ML2_ModelGradientDescent
#        define ModelCacheCompatible ML2_ModelCompatibleCache
#        define ModelCacheAssertCompatible ML2_ModelAssertCompatibleCache

#        define ModelCacheNew ML2_ModelCacheNew
#        define ModelCacheDestroy ML2_ModelCacheDestroy
#        define ModelCachePrint ML2_ModelCachePrint
#        define ModelCacheClear ML2_ModelCacheClear
#        define ModelCacheInput ML2_ModelCacheInput
#        define ModelCacheOutput ML2_ModelCacheOutput
#        define ModelCacheInputGradient ML2_ModelCacheInputGradient
#        define ModelCacheOutputGradient ML2_ModelCacheOutputGradient
#        define ModelCacheCompatibleBatch ML2_ModelCacheCompatibleBatch
#        define ModelCacheAssertCompatibleBatch ML2_ModelCacheAssertCompatibleBatch

#        define BatchNew ML2_BatchNew
#        define BatchDestroy ML2_BatchDestroy
#        define BatchPrint ML2_BatchPrint
#        define BatchSlice ML2_BatchSlice
#        define BatchInput ML2_BatchInput
#        define BatchOutput ML2_BatchOutput
#        define BatchSampleInput ML2_BatchSampleInput
#        define BatchSampleOutput ML2_BatchSampleOutput

#        define LossForward ML2_LossForward
#        define LossBackward ML2_LossBackward
#        define LossForwardSquareError ML2_LossForwardSquareError
#        define LossBackwardSquareError ML2_LossBackwardSquareError

#    endif // _ML2_H_STRIP_PREFIX_GUARD

#endif // ML2_STRIP_PREFIX

/* ChatGPTODO:

# ✅ **2. Proper Activation Functions**

Implement these:

### **2.1 ReLU**

```
f(x) = max(0, x)
f'(x) = x > 0 ? 1 : 0
```

Essential to avoid vanishing gradients.

### **2.2 LeakyReLU**

```
0.01*x for x<0
```

Helps dead ReLU problems.

### **2.3 Tanh**

Better than sigmoid for continuous outputs.

### **2.4 Sigmoid**

Keep it — but only use it at the output for binary tasks.

### **2.5 Softmax**

Absolutely needed for classification.

### **2.6 Linear (ActNone)**

Your current ActNone is correct for regression outputs.

---

# ✅ **3. Proper Initialization (no more collapsing slices)**

### **3.1 Xavier (Glorot) init**

For sigmoid/tanh layers:

```
scale = sqrt(6.0 / (fan_in + fan_out))
w = uniform(-scale, scale)
```

### **3.2 He initialization**

For ReLU layers:

```
scale = sqrt(2.0 / fan_in)
w = normal(0, scale)
```

Add:

```
InitXavier
InitHe
InitUniform
InitCustom
```

---

# ✅ **4. Real Loss Functions**

At least add these:

### **4.1 MSE (you have it already)**

Keep for regression only.

### **4.2 Binary Cross Entropy**

```
loss = -[y log(p) + (1-y) log(1-p)]
```

### **4.3 Categorical Cross Entropy**

For softmax output.

---

# ✅ **5. Optimizers (super important)**

Start simple:

### **5.1 SGD**

(You already have)

### **5.2 Momentum**

```
v = beta*v + grad
w -= lr * v
```

### **5.3 Adam**

This will make your library "real".

---

# ✅ **6. Dataset Utilities (you mostly have this)**

Add helpers:

* shuffle samples
* batch loader
* split train/test
* normalize data (very important)

---

# ✅ **7. Model API Improvements**

### Suggested naming for cached values:

```
values[i]        → activations[i]
z[i]             → preactivations[i] (optional)
weightsGradient  → dW
biasesGradient   → dB
```

Better names means fewer mistakes.

---

# 🔥 **PRIORITY ORDER (What to implement first)**

If you want ML2 to *actually learn*, do it in this order:

### **1. Backprop**

Your library is dead without this.

### **2. Xavier/He initialization**

This alone will fix your vanishing outputs.

### **3. ReLU + Tanh**

Critical for stable gradients.

### **4. Softmax + Cross-Entropy**

Needed for classification.

### **5. Adam optimizer**

Huge improvement to learning quality.

### **6. Batching**

Stops overfitting and makes gradients less noisy.

---

# ✔️ Want a code template?

I can give you:

* A clean backprop implementation for your architecture
* Xavier initialization code
* ReLU/Tanh/Softmax function + derivatives
* A standard neural-network training loop
* A stable gradient descent update pipeline

Just tell me:

**“Give me code templates for X and Y”**

And I’ll generate them in your exact ML2 API style.

*/