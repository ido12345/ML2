#ifndef _ML2_H
#define _ML2_H

#ifndef ML2_DEF
#    define ML2_DEF static inline
#endif // ML2_DEF

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define ML2_UNREACHABLE(msg, ...)                     \
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

#define ML2_TODO(msg, ...)                      \
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

#ifndef ML2_ASSERT
#    include <assert.h>
#    define ML2_ASSERT assert
#endif // ML2_ASSERT

#ifndef ML2_CALLOC
#    include <stdlib.h>
#    define ML2_CALLOC calloc
#endif // ML2_CALLOC

// #ifndef ML2_RELIABLE_CALLOC
// #    define ML2_RELIABLE_CALLOC(...) (ML2_ASSERT(ML2_CALLOC(__VA_ARGS__) != NULL))
// #endif // ML2_RELIABLE_CALLOC

#ifndef ML2_FREE
#    include <stdlib.h>
#    define ML2_FREE free
#endif // ML2_FREE

#ifndef ML2_RELIABLE_CALLOC
static inline void *ML2_ReliableCalloc(size_t count, size_t size) {
    void *ptr = ML2_CALLOC(count, size);
    ML2_ASSERT(ptr != NULL && "BUY MORE RAM");
    return ptr;
}
#    define ML2_RELIABLE_CALLOC ML2_ReliableCalloc
#endif // ML2_RELIABLE_CALLOC

#ifndef ML2_Indentation
#    define ML2_Indentation 4
#endif // ML2_Indentation

#define ML2_Indent(fstr, i) "%*s" fstr, (i), ""

// Customizable
typedef float ML2_Scalar;
// TODO: figure out how to align the floats nicely
#define ML2_ScalarFmt "%.3f"

typedef enum {
    ML2_ActNone = 0,
    ML2_ActSigmoid,
} ML2_ActType;

typedef ML2_Scalar (*ML2_Act)(ML2_Scalar);

// TODO: check if i can put values on the stack
typedef struct {
    int rows, cols, stride;
    ML2_Scalar *values;
} ML2_Matrix;

typedef int ML2_Arch[][2];

#define ML2_ArchNew(...)                                         \
    {                                                            \
        {(sizeof((ML2_Arch){__VA_ARGS__}) / sizeof(int[2])), 0}, \
        __VA_ARGS__}
/*
    Example:
        ML2_Arch arch = ML2_ArchNew(
            { 2, ML2_ActSigmoid },
            { 1, ML2_ActNone },
        );
    Try to understand it as a fun exercise
*/

// ML2_Arch ML2_ArchNew_()

typedef struct {
    int layers;
    // [firstIndex -> lastIndex]

    ML2_Matrix *values;  // [0 -> layers-1]
    ML2_Matrix *weights; // [0 -> layers-2]
    ML2_Matrix *biases;  // [0 -> layers-2]
    // TODO: use the last activation for stuff like softmax
    ML2_ActType *activations; // [0 -> layers-2]

    ML2_Matrix *valuesGradient;  // [0 -> layers-1]
    ML2_Matrix *weightsGradient; // [0 -> layers-2]
    ML2_Matrix *biasesGradient;  // [0 -> layers-2]
} ML2_Model;

typedef struct {
    int samples, inputs, outputs;
    ML2_Scalar *values;
} ML2_Data;

ML2_DEF ML2_Scalar ML2_RandScalar(ML2_Scalar low, ML2_Scalar high);

ML2_DEF ML2_Scalar ML2_Sigmoid(ML2_Scalar x);
ML2_DEF const char *ML2_ActName(ML2_ActType t);

ML2_DEF ML2_Scalar *ML2_MatrixAt(ML2_Matrix matrix, int row, int col);
ML2_DEF ML2_Matrix ML2_MatrixNew(int rows, int cols);
ML2_DEF void ML2_MatrixDestroy(ML2_Matrix *matrix);
ML2_DEF void ML2_MatrixRand(ML2_Matrix matrix, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_MatrixPrint(ML2_Matrix matrix, int indent);
ML2_DEF bool ML2_MatrixSame(ML2_Matrix a, ML2_Matrix b);
ML2_DEF bool ML2_MatrixSameFlipped(ML2_Matrix a, ML2_Matrix b);
ML2_DEF void ML2_MatrixAssertSame(ML2_Matrix a, ML2_Matrix b);
ML2_DEF void ML2_MatrixAssertSameFlipped(ML2_Matrix a, ML2_Matrix b);
// Dest[i,j] = Src[i,j]
ML2_DEF void ML2_MatrixCopy(ML2_Matrix dest, ML2_Matrix src);
// Dest[i,j] = Src[j,i]
ML2_DEF void ML2_MatrixCopyFlipped(ML2_Matrix dest, ML2_Matrix src);
// Dest[i,j] = Σ(A[i,k] * B[k,j]);
ML2_DEF void ML2_MatrixDot(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b);
// Dest[i,j] = A[i,j] + B[i,j]
ML2_DEF void ML2_MatrixSum(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b);
// Dest[i,j] = f(Dest[i,j]);
ML2_DEF void ML2_MatrixActivate(ML2_Matrix dest, ML2_Act f);

ML2_DEF ML2_Model ML2_ModelNew(ML2_Arch arch);
ML2_DEF void ML2_ModelDestroy(ML2_Model *model);
ML2_DEF void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high);
ML2_DEF void ML2_ModelPrint(ML2_Model model, int indent);
ML2_DEF void ML2_ModelPrintParameters(ML2_Model model, int indent);
ML2_DEF void ML2_ModelPrintGradient(ML2_Model model, int indent);
// Z[i+1] = Act(W*X[i]+B);
ML2_DEF void ML2_ModelForward(ML2_Model model);
ML2_DEF ML2_Matrix ML2_ModelInput(ML2_Model model);
ML2_DEF ML2_Matrix ML2_ModelOutput(ML2_Model model);
ML2_DEF void ML2_ModelSetInput(ML2_Model model, ML2_Matrix input);
ML2_DEF ML2_Scalar ML2_ModelSquareLoss(ML2_Model model, ML2_Data data);
ML2_DEF void ML2_ModelGradientFiniteDiff(ML2_Model model, ML2_Data data, ML2_Scalar epsilon);
ML2_DEF void ML2_ModelGradientDescent(ML2_Model model, ML2_Scalar learningRate);

ML2_DEF ML2_Data ML2_DataNew(int samples, int inputs, int outputs);
ML2_DEF void ML2_DataDestroy(ML2_Data *data);
ML2_DEF void ML2_DataPrint(ML2_Data data, int indent);
ML2_DEF ML2_Matrix ML2_DataInput(ML2_Data data);
ML2_DEF ML2_Matrix ML2_DataOutput(ML2_Data data);
ML2_DEF ML2_Matrix ML2_DataSampleInput(ML2_Data data, int sample);
ML2_DEF ML2_Matrix ML2_DataSampleOutput(ML2_Data data, int sample);

#endif // _ML2_H

#ifdef ML2_IMPLEMENTATION

ML2_DEF ML2_Scalar ML2_RandScalar(ML2_Scalar low, ML2_Scalar high) {
    return ((ML2_Scalar)rand() / (ML2_Scalar)RAND_MAX) * (high - low) + low;
}

// ML2_ ⬇️
// ML2_ ⬆️

// ML2_Act ⬇️

ML2_DEF ML2_Scalar ML2_Sigmoid(ML2_Scalar x) {
    return 1.0 / (1.0 + exp(-x));
}

ML2_DEF ML2_Act ML2_ActOf(ML2_ActType t) {
    switch (t) {
        case ML2_ActNone:
            return NULL;
        case ML2_ActSigmoid:
            return ML2_Sigmoid;
        default:
            ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

ML2_DEF const char *ML2_ActName(ML2_ActType t) {
    switch (t) {
        case ML2_ActNone:
            return "None";
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
            *ML2_MatrixAt(matrix, i, j) = ML2_RandScalar(low, high);
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

ML2_DEF bool ML2_MatrixSame(ML2_Matrix a, ML2_Matrix b) {
    return a.rows == b.rows &&
           a.cols == b.cols;
}

ML2_DEF bool ML2_MatrixSameFlipped(ML2_Matrix a, ML2_Matrix b) {
    return a.rows == b.cols &&
           a.cols == b.rows;
}

ML2_DEF void ML2_MatrixAssertSame(ML2_Matrix a, ML2_Matrix b) {
    ML2_ASSERT(ML2_MatrixSame(a, b) && "MATRIX SIZES MUST MATCH");
}

ML2_DEF void ML2_MatrixAssertSameFlipped(ML2_Matrix a, ML2_Matrix b) {
    ML2_ASSERT(ML2_MatrixSameFlipped(a, b) && "MATRIX SIZES MUST MATCH");
}

// void ML2_MatrixClear(ML2_Matrix dest) {
//     for (int i = 0; i < dest.rows; i++) {
//         for (int j = 0; j < dest.cols; j++) {
//             *ML2_MatrixAt(dest, i, j) = 0;
//         }
//     }
// }

// Dest[i,j] = Src[i,j]
ML2_DEF void ML2_MatrixCopy(ML2_Matrix dest, ML2_Matrix src) {
    ML2_MatrixAssertSame(dest, src);
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = *ML2_MatrixAt(src, i, j);
        }
    }
}

// Dest[i,j] = Src[j,i]
ML2_DEF void ML2_MatrixCopyFlipped(ML2_Matrix dest, ML2_Matrix src) {
    ML2_MatrixAssertSameFlipped(dest, src);
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = *ML2_MatrixAt(src, j, i);
        }
    }
}

// Dest[i,j] = Σ(A[i,k] * B[k,j])
ML2_DEF void ML2_MatrixDot(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b) {
    ML2_ASSERT(a.cols == b.rows && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(dest.rows == a.rows && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(dest.cols == b.cols && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(dest.values != a.values && "MATRIX VALUES MUST NOT OVERLAP");
    ML2_ASSERT(dest.values != b.values && "MATRIX VALUES MUST NOT OVERLAP");
    int inner = a.cols;

    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            ML2_Scalar sum = 0;
            for (int k = 0; k < inner; k++) {
                sum += *ML2_MatrixAt(a, i, k) * *ML2_MatrixAt(b, k, j);
            }
            *ML2_MatrixAt(dest, i, j) = sum;
        }
    }
}

// Dest[i,j] = A[i,j] + B[i,j]
ML2_DEF void ML2_MatrixSum(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b) {
    ML2_MatrixAssertSame(dest, a);
    ML2_MatrixAssertSame(dest, b);
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = *ML2_MatrixAt(a, i, j) + *ML2_MatrixAt(b, i, j);
        }
    }
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
    int layers = arch[0][0];
    ML2_Model model = {.layers = layers};
    model.values = ML2_RELIABLE_CALLOC(layers, sizeof(*model.values));
    model.weights = ML2_RELIABLE_CALLOC(layers - 1, sizeof(*model.weights));
    model.biases = ML2_RELIABLE_CALLOC(layers - 1, sizeof(*model.biases));
    model.activations = ML2_RELIABLE_CALLOC(layers - 1, sizeof(*model.activations));

    model.valuesGradient = ML2_RELIABLE_CALLOC(layers, sizeof(*model.valuesGradient));
    model.weightsGradient = ML2_RELIABLE_CALLOC(layers - 1, sizeof(*model.weightsGradient));
    model.biasesGradient = ML2_RELIABLE_CALLOC(layers - 1, sizeof(*model.biasesGradient));
    for (int i = 0; i < layers; i++) {
        model.values[i] = ML2_MatrixNew(arch[i + 1][0], 1);
        model.valuesGradient[i] = ML2_MatrixNew(arch[i + 1][0], 1);
        if (i < layers - 1) {
            model.weights[i] = ML2_MatrixNew(arch[i + 2][0], arch[i + 1][0]);
            model.biases[i] = ML2_MatrixNew(arch[i + 2][0], 1);
            model.activations[i] = (ML2_ActType)(arch[i + 1][1]);

            model.weightsGradient[i] = ML2_MatrixNew(arch[i + 2][0], arch[i + 1][0]);
            model.biasesGradient[i] = ML2_MatrixNew(arch[i + 2][0], 1);
        }
    }
    return model;
}

ML2_DEF void ML2_ModelDestroy(ML2_Model *model) {
    for (int i = 0; i < model->layers; i++) {
        ML2_MatrixDestroy(&model->values[i]);
        ML2_MatrixDestroy(&model->valuesGradient[i]);
        if (i < model->layers - 1) {
            ML2_MatrixDestroy(&model->weights[i]);
            ML2_MatrixDestroy(&model->biases[i]);

            ML2_MatrixDestroy(&model->weightsGradient[i]);
            ML2_MatrixDestroy(&model->biasesGradient[i]);
        }
    }
    ML2_FREE(model->values);
    ML2_FREE(model->weights);
    ML2_FREE(model->biases);
    ML2_FREE(model->activations);
    ML2_FREE(model->valuesGradient);
    ML2_FREE(model->weightsGradient);
    ML2_FREE(model->biasesGradient);
    memset(model, 0, sizeof(*model));
}

ML2_DEF void ML2_ModelRand(ML2_Model model, ML2_Scalar low, ML2_Scalar high) {
    for (int i = 0; i < model.layers - 1; i++) {
        ML2_MatrixRand(model.weights[i], low, high);
        ML2_MatrixRand(model.biases[i], low, high);
    }
}

ML2_DEF void ML2_ModelPrint(ML2_Model model, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < model.layers; i++) {
        ML2_Matrix values = model.values[i];
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix biases = model.biases[i];
        // printf("%*sValues[%d](%dx%d):\n", indent + ML2_Indentation, "", i, values.rows, values.cols);
        printf(ML2_Indent("Values[%d](%dx%d):\n", indent + ML2_Indentation), i, values.rows, values.cols);
        ML2_MatrixPrint(values, indent + ML2_Indentation);
        if (i < model.layers - 1) {
            // printf("%*sWeights[%d](%dx%d):\n", indent + ML2_Indentation, "", i, weights.rows, weights.cols);
            printf(ML2_Indent("Weights[%d](%dx%d):\n", indent + ML2_Indentation), i, weights.rows, weights.cols);
            ML2_MatrixPrint(weights, indent + ML2_Indentation);
            // printf("%*sBiases[%d](%dx%d):\n", indent + ML2_Indentation, "", i, biases.rows, biases.cols);
            printf(ML2_Indent("Biases[%d](%dx%d):\n", indent + ML2_Indentation), i, biases.rows, biases.cols);
            ML2_MatrixPrint(biases, indent + ML2_Indentation);
            // printf("%*sActivation[%d]: \"%s\"\n", indent + ML2_Indentation, "", i, ML2_ActName(model.activations[i]));
            printf(ML2_Indent("Activation[%d]: \"%s\"\n", indent + ML2_Indentation), i, ML2_ActName(model.activations[i]));
        }
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF void ML2_ModelPrintParameters(ML2_Model model, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < model.layers - 1; i++) {
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

ML2_DEF void ML2_ModelPrintGradient(ML2_Model model, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < model.layers; i++) {
        ML2_Matrix valuesGradient = model.valuesGradient[i];
        ML2_Matrix weightsGradient = model.weightsGradient[i];
        ML2_Matrix biasesGradient = model.biasesGradient[i];
        // printf("%*sValues[%d](%dx%d):\n", indent + ML2_Indentation, "", i, valuesGradient.rows, valuesGradient.cols);
        printf(ML2_Indent("Values[%d](%dx%d):\n", indent + ML2_Indentation), i, valuesGradient.rows, valuesGradient.cols);
        ML2_MatrixPrint(valuesGradient, indent + ML2_Indentation);
        if (i < model.layers - 1) {
            // printf("%*sWeights[%d](%dx%d):\n", indent + ML2_Indentation, "", i, weightsGradient.rows, weightsGradient.cols);
            printf(ML2_Indent("Weights[%d](%dx%d):\n", indent + ML2_Indentation), i, weightsGradient.rows, weightsGradient.cols);
            ML2_MatrixPrint(weightsGradient, indent + ML2_Indentation);
            // printf("%*sBiases[%d](%dx%d):\n", indent + ML2_Indentation, "", i, biasesGradient.rows, biasesGradient.cols);
            printf(ML2_Indent("Biases[%d](%dx%d):\n", indent + ML2_Indentation), i, biasesGradient.rows, biasesGradient.cols);
            ML2_MatrixPrint(biasesGradient, indent + ML2_Indentation);
            // printf("%*sActivation[%d]: \"%s\"\n", indent + ML2_Indentation, "", i, ML2_ActName(model.activations[i]));
            printf(ML2_Indent("Activation[%d]: \"%s\"\n", indent + ML2_Indentation), i, ML2_ActName(model.activations[i]));
        }
    }
    printf(ML2_Indent("}\n", indent));
}

// Z[i+1] = Act(W*X[i]+B)
ML2_DEF void ML2_ModelForward(ML2_Model model) {
    for (int i = 0; i < model.layers - 1; i++) {
        ML2_MatrixDot(model.values[i + 1], model.weights[i], model.values[i]);
        ML2_MatrixSum(model.values[i + 1], model.values[i + 1], model.biases[i]);
        if (model.activations[i]) {
            ML2_MatrixActivate(model.values[i + 1], ML2_ActOf(model.activations[i]));
        }
    }
}

ML2_DEF ML2_Matrix ML2_ModelInput(ML2_Model model) {
    return model.values[0];
}

ML2_DEF ML2_Matrix ML2_ModelOutput(ML2_Model model) {
    return model.values[model.layers - 1];
}

ML2_DEF void ML2_ModelSetInput(ML2_Model model, ML2_Matrix input) {
    ML2_MatrixCopyFlipped(ML2_ModelInput(model), input);
}

ML2_DEF ML2_Scalar ML2_ModelSquareLoss(ML2_Model model, ML2_Data data) {
    ML2_Scalar loss = 0;
    ML2_Matrix modelOutput = ML2_ModelOutput(model);
    ML2_ASSERT(modelOutput.rows == data.outputs && "MODEL AND DATA OUTPUT SIZES MUST MATCH");
    int outputs = modelOutput.rows;
    for (int i = 0; i < data.samples; i++) {
        ML2_Matrix sampleInput = ML2_DataSampleInput(data, i);
        ML2_Matrix sampleOutput = ML2_DataSampleOutput(data, i);
        ML2_ModelSetInput(model, sampleInput);
        ML2_ModelForward(model);
        for (int j = 0; j < outputs; j++) {
            ML2_Scalar diff = *ML2_MatrixAt(modelOutput, j, 0) - *ML2_MatrixAt(sampleOutput, 0, j);
            loss += diff * diff;
        }
    }
    loss /= (data.samples * outputs);
    return loss;
}

ML2_DEF void ML2_ModelGradientFiniteDiff(ML2_Model model, ML2_Data data, ML2_Scalar epsilon) {
    ML2_ASSERT(epsilon > 0 && "EPSILON MUST BE POSITIVE");
    // f'(x) = lim(e -> 0): (f(x + e) - f(x)) / e
    ML2_Scalar prev = 0;
    ML2_Scalar loss = ML2_ModelSquareLoss(model, data);
    for (int i = 0; i < model.layers - 1; i++) {
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix weightsGradient = model.weightsGradient[i];
        for (int j = 0; j < weights.rows; j++) {
            for (int k = 0; k < weights.cols; k++) {
                ML2_Scalar *cur = ML2_MatrixAt(weights, j, k);
                prev = *cur;
                *cur += epsilon;
                ML2_Scalar newLoss = ML2_ModelSquareLoss(model, data);
                *ML2_MatrixAt(weightsGradient, j, k) = (newLoss - loss) / epsilon;
                *cur = prev;
            }
        }

        ML2_Matrix biases = model.biases[i];
        ML2_Matrix biasesGradient = model.biasesGradient[i];
        for (int j = 0; j < biases.rows; j++) {
            ML2_Scalar *cur = ML2_MatrixAt(biases, j, 0);
            prev = *cur;
            *cur += epsilon;
            ML2_Scalar newLoss = ML2_ModelSquareLoss(model, data);
            *ML2_MatrixAt(biasesGradient, j, 0) = (newLoss - loss) / epsilon;
            *cur = prev;
        }
    }
}

ML2_DEF void ML2_ModelGradientDescent(ML2_Model model, ML2_Scalar learningRate) {
    for (int i = 0; i < model.layers - 1; i++) {
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix weightsGradient = model.weightsGradient[i];
        for (int j = 0; j < weights.rows; j++) {
            for (int k = 0; k < weights.cols; k++) {
                *ML2_MatrixAt(weights, j, k) -= learningRate * *ML2_MatrixAt(weightsGradient, j, k);
            }
        }

        ML2_Matrix biases = model.biases[i];
        ML2_Matrix biasesGradient = model.biasesGradient[i];
        for (int j = 0; j < biases.rows; j++) {
            *ML2_MatrixAt(biases, j, 0) -= learningRate * *ML2_MatrixAt(biasesGradient, j, 0);
        }
    }
}

// ML2_Model ⬆️

// ML2_Data ⬇️

ML2_DEF ML2_Data ML2_DataNew(int samples, int inputs, int outputs) {
    ML2_Data data = {
        .samples = samples,
        .inputs = inputs,
        .outputs = outputs,
        .values = ML2_CALLOC(samples * (inputs + outputs), sizeof(*data.values)),
    };
    return data;
}

ML2_DEF void ML2_DataDestroy(ML2_Data *data) {
    ML2_FREE(data->values);
    memset(data, 0, sizeof(*data));
}

ML2_DEF void ML2_DataPrint(ML2_Data data, int indent) {
    printf(ML2_Indent("{\n", indent));
    for (int i = 0; i < data.samples; i++) {
        printf(ML2_Indent("", indent + ML2_Indentation));
        for (int j = 0; j < data.inputs; j++) {
            printf(ML2_ScalarFmt, data.values[i * (data.inputs + data.outputs) + j]);
            if (j < data.inputs - 1) {
                printf(", ");
            }
        }
        printf("    ");
        for (int j = 0; j < data.outputs; j++) {
            printf(ML2_ScalarFmt, data.values[i * (data.inputs + data.outputs) + data.inputs + j]);
            if (j < data.outputs - 1) {
                printf(", ");
            }
        }
        printf("\n");
    }
    printf(ML2_Indent("}\n", indent));
}

ML2_DEF ML2_Matrix ML2_DataInput(ML2_Data data) {
    ML2_Matrix input = {
        .rows = data.samples,
        .cols = data.inputs,
        .stride = data.inputs + data.outputs,
        .values = &data.values[0],
    };
    return input;
}

ML2_DEF ML2_Matrix ML2_DataOutput(ML2_Data data) {
    ML2_Matrix output = {
        .rows = data.samples,
        .cols = data.outputs,
        .stride = data.inputs + data.outputs,
        .values = &data.values[data.inputs],
    };
    return output;
}

ML2_DEF ML2_Matrix ML2_DataSampleInput(ML2_Data data, int sample) {
    ML2_ASSERT(0 <= sample && sample < data.samples && "INVALID SAMPLE INDEX");

    ML2_Matrix input = {
        .rows = 1,
        .cols = data.inputs,
        .stride = data.inputs + data.outputs,
        .values = &data.values[input.stride * sample],
    };
    return input;
}

ML2_DEF ML2_Matrix ML2_DataSampleOutput(ML2_Data data, int sample) {
    ML2_ASSERT(0 <= sample && sample < data.samples && "INVALID SAMPLE INDEX");

    ML2_Matrix output = {
        .rows = 1,
        .cols = data.outputs,
        .stride = data.inputs + data.outputs,
        .values = &data.values[output.stride * sample + data.inputs],
    };
    return output;
}

// ML2_Data ⬆️

#ifdef ML2_STRIP_PREFIX

#    define Scalar ML2_Scalar
#    define ActNone ML2_ActNone
#    define ActSigmoid ML2_ActSigmoid
#    define Act ML2_Act
#    define Matrix ML2_Matrix
#    define Arch ML2_Arch
#    define ArchNew ML2_ArchNew
#    define Model ML2_Model
#    define Data ML2_Data

#    define Sigmoid ML2_Sigmoid
#    define ActName ML2_ActName
#    define MatrixAt ML2_MatrixAt
#    define MatrixNew ML2_MatrixNew
#    define MatrixDestroy ML2_MatrixDestroy
#    define MatrixRand ML2_MatrixRand
#    define MatrixPrint ML2_MatrixPrint
#    define MatrixSame ML2_MatrixSame
#    define MatrixSameFlipped ML2_MatrixSameFlipped
#    define MatrixAssertSame ML2_MatrixAssertSame
#    define MatrixAssertSameFlipped ML2_MatrixAssertSameFlipped
#    define MatrixCopy ML2_MatrixCopy
#    define MatrixCopyFlipped ML2_MatrixCopyFlipped
#    define MatrixDot ML2_MatrixDot
#    define MatrixSum ML2_MatrixSum
#    define MatrixActivate ML2_MatrixActivate
#    define ModelNew ML2_ModelNew
#    define ModelDestroy ML2_ModelDestroy
#    define ModelRand ML2_ModelRand
#    define ModelPrint ML2_ModelPrint
#    define ModelPrintParameters ML2_ModelPrintParameters
#    define ModelPrintGradient ML2_ModelPrintGradient
#    define ModelForward ML2_ModelForward
#    define ModelInput ML2_ModelInput
#    define ModelOutput ML2_ModelOutput
#    define ModelSetInput ML2_ModelSetInput
#    define ModelSquareLoss ML2_ModelSquareLoss
#    define ModelGradientFiniteDiff ML2_ModelGradientFiniteDiff
#    define ModelGradientDescent ML2_ModelGradientDescent
#    define DataNew ML2_DataNew
#    define DataDestroy ML2_DataDestroy
#    define DataPrint ML2_DataPrint
#    define DataInput ML2_DataInput
#    define DataOutput ML2_DataOutput
#    define DataSampleInput ML2_DataSampleInput
#    define DataSampleOutput ML2_DataSampleOutput

#endif // ML2_STRIP_PREFIX

#endif // ML2_IMPLEMENTATION