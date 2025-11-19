#ifndef _ML2_H
#define _ML2_H

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
                ##__VA_ARGS__,                        \
                __FILE__,                             \
                __func__,                             \
                __LINE__);                            \
        abort();                                      \
    } while (0)

#define ML2_TODO(msg, ...)                      \
    do {                                        \
        fprintf(stderr, "TODO: \"" msg "\":\n"  \
                        "    file:     |%s|\n"  \
                        "    function: |%s|\n"  \
                        "    line:     |%d|\n", \
                ##__VA_ARGS__,                  \
                __FILE__,                       \
                __func__,                       \
                __LINE__);                      \
        abort();                                \
    } while (0)

#ifndef ML2_CALLOC
#    include <stdlib.h>
#    define ML2_CALLOC calloc
#endif // ML2_CALLOC

#ifndef ML2_FREE
#    include <stdlib.h>
#    define ML2_FREE free
#endif // ML2_FREE

#ifndef ML2_ASSERT
#    include <assert.h>
#    define ML2_ASSERT assert
#endif // ML2_ASSERT

typedef float ML2_Scalar;

typedef enum {
    ML2_ActNone = 0,
    ML2_ActSigmoid,
    ML2_ActCount,
} ML2_ActType;

typedef ML2_Scalar (*ML2_Act)(ML2_Scalar);

// TODO: check if i can put values on the stack
typedef struct {
    int rows, cols, stride;
    ML2_Scalar *values;
} ML2_Matrix;

typedef int ML2_Arch[][2];

#define ML2_ArchNewStatic(...)                                       \
    {                                                                \
        { (sizeof((ML2_Arch){ __VA_ARGS__ }) / sizeof(int[2])), 0 }, \
        __VA_ARGS__                                                  \
    }
/*
    Example:
        ML2_Arch arch = ML2_ArchNewStatic(
            { 2, ML2_ActSigmoid },
            { 1, ML2_ActNone },
        );
    Try to understand it as a fun exercise
*/

typedef struct {
    ML2_Matrix *values;
    ML2_Matrix *weights;
    ML2_Matrix *biases;
    ML2_ActType *activations;
    int layers;
} ML2_Model;

typedef struct {
    int samples, inputs, outputs;
    ML2_Scalar *values;
} ML2_Data;

#endif // _ML2_H

#ifdef ML2_IMPLEMENTATION

// ML2_ ⬇️
// ML2_ ⬆️

// ML2_Act ⬇️

ML2_Scalar ML2_Sigmoid(ML2_Scalar x) {
    return 1.0 / (1.0 + exp(-x));
}

ML2_Act ML2_ActOf(ML2_ActType t) {
    switch (t) {
        case ML2_ActNone:
            return NULL;
        case ML2_ActSigmoid:
            return ML2_Sigmoid;
        default:
            ML2_UNREACHABLE("Unknown ML2_ActType");
    }
}

const char *ML2_ActName(ML2_ActType t) {
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

ML2_Scalar *ML2_MatrixAt(ML2_Matrix matrix, int row, int col) {
    if (0 <= row && row < matrix.rows &&
        0 <= col && col < matrix.cols) {
        return &matrix.values[row * matrix.stride + col];
    }
    return NULL;
}

ML2_Matrix ML2_MatrixNew(int rows, int cols) {
    ML2_Matrix matrix = {
        .rows = rows,
        .cols = cols,
        .stride = cols,
        .values = calloc(rows * cols, sizeof(matrix.values)),
    };
    return matrix;
}

void ML2_MatrixDestroy(ML2_Matrix *matrix) {
    free(matrix->values);
    memset(matrix, 0, sizeof(*matrix));
}

void ML2_MatrixPrint(ML2_Matrix matrix, int indent) {
    printf("%*s{\n", indent, "");
    for (int row = 0; row < matrix.rows; row++) {
        printf("%*s", indent + 4, "");
        for (int col = 0; col < matrix.cols; col++) {
            printf("%f ", *ML2_MatrixAt(matrix, row, col));
        }
        printf("\n");
    }
    printf("%*s}\n", indent, "");
}

// bool ML2_MatrixSame(ML2_Matrix a, ML2_Matrix b) {
//     return a.rows == b.rows &&
//            a.cols == b.cols;
// }

// void ML2_MatrixClear(ML2_Matrix dest) {
//     for (int row = 0; row < dest.rows; row++) {
//         for (int col = 0; col < dest.cols; col++) {
//             *ML2_MatrixAt(dest, row, col) = 0;
//         }
//     }
// }

void ML2_MatrixCopy(ML2_Matrix dest, ML2_Matrix src) {
    ML2_ASSERT(dest.rows == src.rows && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(dest.cols == src.cols && "MATRIX SIZES MUST MATCH");

    for (int row = 0; row < dest.rows; row++) {
        for (int col = 0; col < dest.cols; col++) {
            *ML2_MatrixAt(dest, row, col) = *ML2_MatrixAt(src, row, col);
        }
    }
}

// Dest[i,j] = Σ(A[i,k] * B[k,j])
void ML2_MatrixDot(ML2_Matrix dest, ML2_Matrix a, ML2_Matrix b) {
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

// Dest[i,j] += A[i,j]
void ML2_MatrixSum(ML2_Matrix dest, ML2_Matrix a) {
    ML2_ASSERT(dest.rows == a.rows && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(dest.cols == a.cols && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(dest.values != a.values && "MATRIX VALUES MUST NOT OVERLAP");

    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) += *ML2_MatrixAt(a, i, j);
        }
    }
}

// Dest[i,j] = f(Dest[i,j])
void ML2_MatrixActivate(ML2_Matrix dest, ML2_Act f) {
    for (int i = 0; i < dest.rows; i++) {
        for (int j = 0; j < dest.cols; j++) {
            *ML2_MatrixAt(dest, i, j) = f(*ML2_MatrixAt(dest, i, j));
        }
    }
}

// ML2_Matrix ⬆️

// ML2_Model ⬇️

ML2_Model ML2_ModelNew(ML2_Arch arch) {
    int layers = arch[0][0];
    ML2_Model model = { .layers = layers };
    model.values = calloc(layers, sizeof(*model.values));
    model.weights = calloc(layers - 1, sizeof(*model.weights));
    model.biases = calloc(layers - 1, sizeof(*model.biases));
    model.activations = calloc(layers - 1, sizeof(*model.activations));
    for (int i = 0; i < layers; i++) {
        model.values[i] = ML2_MatrixNew(arch[i + 1][0], 1);
        if (i < layers - 1) {
            model.weights[i] = ML2_MatrixNew(arch[i + 2][0], arch[i + 1][0]);
            model.biases[i] = ML2_MatrixNew(arch[i + 2][0], 1);
            model.activations[i] = (ML2_ActType)(arch[i + 1][1]);
        }
    }
    return model;
}

void ML2_ModelDestroy(ML2_Model *model) {
    for (int i = 0; i < model->layers; i++) {
        ML2_MatrixDestroy(&model->values[i]);
        if (i < model->layers - 1) {
            ML2_MatrixDestroy(&model->weights[i]);
            ML2_MatrixDestroy(&model->biases[i]);
        }
    }
    free(model->values);
    free(model->weights);
    free(model->biases);
    free(model->activations);
    memset(model, 0, sizeof(*model));
}

void ML2_ModelPrint(ML2_Model model, int indent) {
    printf("%*s{\n", indent, "");
    for (int i = 0; i < model.layers; i++) {
        ML2_Matrix values = model.values[i];
        ML2_Matrix weights = model.weights[i];
        ML2_Matrix biases = model.biases[i];
        printf("%*sValues[%d](%dx%d):\n", indent + 4, "", i, values.rows, values.cols);
        ML2_MatrixPrint(values, indent + 4);
        if (i < model.layers - 1) {
            printf("%*sWeights[%d](%dx%d):\n", indent + 4, "", i, weights.rows, weights.cols);
            ML2_MatrixPrint(weights, indent + 4);
            printf("%*sBiases[%d](%dx%d):\n", indent + 4, "", i, biases.rows, biases.cols);
            ML2_MatrixPrint(biases, indent + 4);
            printf("%*sActivation[%d]: \"%s\"\n", indent + 4, "", i, ML2_ActName(model.activations[i]));
        }
    }
    printf("%*s}\n", indent, "");
}

void ML2_ModelForward(ML2_Model model) {
    for (int i = 0; i < model.layers - 1; i++) {
        // z[i+1] = act(w*x[i] + b)
        ML2_MatrixDot(model.values[i + 1], model.weights[i], model.values[i]);
        ML2_MatrixSum(model.values[i + 1], model.biases[i]);
        if (model.activations[i]) {
            ML2_MatrixActivate(model.values[i + 1], ML2_ActOf(model.activations[i]));
        }
    }
}

ML2_Matrix ML2_ModelInput(ML2_Model model) {
    return model.values[0];
}

ML2_Matrix ML2_ModelOutput(ML2_Model model) {
    return model.values[model.layers - 1];
}

void ML2_ModelSetInput(ML2_Model model, ML2_Matrix input) {
    ML2_Matrix modelInput = ML2_ModelInput(model);
    ML2_ASSERT(modelInput.rows == input.cols && "MATRIX SIZES MUST MATCH");
    ML2_ASSERT(modelInput.cols == input.rows && "MATRIX SIZES MUST MATCH");

    for (int i = 0; i < modelInput.rows; i++) {
        *ML2_MatrixAt(modelInput, i, 0) = *ML2_MatrixAt(input, 0, i);
    }
}

void ML2_ModelGetOutput(ML2_Matrix output, ML2_Model model) {
    ML2_Matrix modelOutput = ML2_ModelOutput(model);
    ML2_ASSERT(output.cols == modelOutput.rows && "MATRIX SIZES MUST MATCH");

    for (int i = 0; i < modelOutput.cols; i++) {
        *ML2_MatrixAt(output, 0, i) = *ML2_MatrixAt(modelOutput, i, 0);
    }
}

// ML2_Model ⬆️

// ML2_Data ⬇️

ML2_Data ML2_DataNew(int samples, int inputs, int outputs) {
    ML2_Data data = {
        .samples = samples,
        .inputs = inputs,
        .outputs = outputs,
        .values = ML2_CALLOC(samples * (inputs + outputs), sizeof(*data.values)),
    };
    return data;
}

ML2_Matrix ML2_DataInput(ML2_Data data, int sample) {
    ML2_ASSERT(0 <= sample && sample < data.samples && "INVALID SAMPLE INDEX");

    ML2_Matrix input = {
        .rows = 1,
        .cols = data.inputs,
        .stride = data.inputs + data.outputs,
        .values = &data.values[input.stride * sample],
    };
    return input;
}

// ML2_Data ⬆️

#endif // ML2_IMPLEMENTATION