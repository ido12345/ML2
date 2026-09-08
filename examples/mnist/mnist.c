#define ML2_IMPLEMENTATION
#include "ML2.h"
#include <time.h>

static constexpr int trainingSamples = 60'000;
static constexpr int testingSamples = 10'000;
static constexpr int inputs = 784;
static constexpr int outputs = 10;
static constexpr int imageChannels = 1;
static constexpr int imageHeight = 28;
static constexpr int imageWidth = 28;
static_assert(inputs == imageHeight * imageWidth);

static constexpr ML2_Scalar learningRate = 0.035f;
// the number of threads that will be created
static constexpr int batches = 20;

static ML2_Model model = {};
static ML2_Model *modelGradient = {};
static ML2_ModelCache *modelCacheTraining = {};
static ML2_ModelCache *modelCacheGradientTraining = {};
static ML2_Batch *batchTraining = {};
static ML2_ModelCache modelCacheTesting = {};
static ML2_Batch batchTesting = {};
static ML2_LossForward lossForward = ML2_LossForwardSoftmaxCrossEntropy;
static ML2_LossBackward lossBackward = ML2_LossBackwardSoftmaxCrossEntropy;
static ML2_Optimizer optimizer = {};

// these 2 are built in for mnist_gui
static ML2_ModelCache modelCacheSample = {};
static ML2_LayerCache inputSampleSingle = {};

static void ModelRand() {
    srand(time(0));
    ML2_ModelXavierInit(model);
}

static void ModelInit() {
    constexpr int hiddenChannels1 = 4;
    constexpr int hiddenChannels2 = 4;
    ML2_Arch arch = ML2_ArchMake(
        ML2_Conv(
            imageChannels, hiddenChannels1,
            5, 5,
            0, 0,
            1, 1
        ),
        ML2_ReLU(),
        ML2_Conv(
            hiddenChannels1, hiddenChannels2,
            3, 3,
            0, 0,
            1, 1
        ),
        ML2_ReLU(),
        ML2_Flatten(),
        ML2_Linear(hiddenChannels2 * 22 * 22, outputs),
        // ML2_Softmax(),
    );

    model = ML2_ModelNew(arch);

    {
        FILE *mnist_train_input = fopen("data/mnist_train_input.bin", "rb");
        FILE *mnist_train_output = fopen("data/mnist_train_output.bin", "rb");
        modelGradient = ML2_RELIABLE_CALLOC(batches, sizeof(*modelGradient));
        modelCacheTraining = ML2_RELIABLE_CALLOC(batches, sizeof(*modelCacheTraining));
        modelCacheGradientTraining = ML2_RELIABLE_CALLOC(batches, sizeof(*modelCacheGradientTraining));
        batchTraining = ML2_RELIABLE_CALLOC(batches, sizeof(*batchTraining));
        int base = trainingSamples / batches;
        int remainder = trainingSamples % batches;
        for (int i = 0; i < batches; i++) {
            ML2_Size batchSize = base + (i < remainder ? 1 : 0);
            ML2_BatchInfo batchTrainingInfo = ML2_BatchInfoMake(
                ML2_Matrices(batchSize, imageHeight, imageWidth, imageChannels),
                ML2_Scalars(batchSize, outputs)
            );
            modelGradient[i] = ML2_ModelNew(arch);
            modelCacheTraining[i] = ML2_ModelCacheNew(arch, batchTrainingInfo);
            modelCacheGradientTraining[i] = ML2_ModelCacheNew(arch, batchTrainingInfo);
            batchTraining[i] = ML2_BatchNew(batchTrainingInfo);
            ML2_LayerCacheMatrices batchTrainingInput = ML2_LayerCacheAsMatrices(batchTraining[i].input);
            ML2_LayerCacheScalars batchTrainingOutput = ML2_LayerCacheAsScalars(batchTraining[i].output);
            ML2_HARD_ASSERT(fread(batchTrainingInput.data.matrices, sizeof(*batchTrainingInput.data.matrices), batchSize * inputs, mnist_train_input) == batchSize * inputs);
            ML2_HARD_ASSERT(fread(batchTrainingOutput.data.scalars, sizeof(*batchTrainingOutput.data.scalars), batchSize * outputs, mnist_train_output) == batchSize * outputs);
        }
        fclose(mnist_train_input);
        fclose(mnist_train_output);
    }
    ML2_BatchInfo batchTestingInfo = ML2_BatchInfoMake(
        ML2_Matrices(testingSamples, imageHeight, imageWidth, imageChannels),
        ML2_Scalars(testingSamples, outputs)
    );
    modelCacheTesting = ML2_ModelCacheNew(arch, batchTestingInfo);
    batchTesting = ML2_BatchNew(batchTestingInfo);
    
    ML2_LayerCacheMatrices batchTestingInput = ML2_LayerCacheAsMatrices(batchTesting.input);
    ML2_LayerCacheScalars batchTestingOutput = ML2_LayerCacheAsScalars(batchTesting.output);
    FILE *mnist_test_input = fopen("data/mnist_test_input.bin", "rb");
    FILE *mnist_test_output = fopen("data/mnist_test_output.bin", "rb");
    ML2_HARD_ASSERT(fread(batchTestingInput.data.matrices, sizeof(*batchTestingInput.data.matrices), testingSamples * inputs, mnist_test_input) == testingSamples * inputs);
    ML2_HARD_ASSERT(fread(batchTestingOutput.data.scalars, sizeof(*batchTestingOutput.data.scalars), testingSamples * outputs, mnist_test_output) == testingSamples * outputs);
    fclose(mnist_test_input);
    fclose(mnist_test_output);

    // optimizer = ML2_OptimizerNew(ML2_GradientDescent(learningRate), arch);
    optimizer = ML2_OptimizerNew(ML2_Adam(learningRate, 0.9, 0.999), arch);

    ML2_BatchInfo batchSampleInfo = ML2_BatchInfoMake(
        ML2_Matrices(1, imageHeight, imageWidth, imageChannels),
        ML2_Scalars(1, outputs)
    );
    modelCacheSample = ML2_ModelCacheNew(arch, batchSampleInfo);
    inputSampleSingle = ML2_LayerCacheNew(batchSampleInfo.input);

    ModelRand();
}

static void ModelDeinit() {
    ML2_ModelDestroy(&model);
    for (int i = 0; i < batches; i++) {
        ML2_ModelDestroy(&modelGradient[i]);
        ML2_ModelCacheDestroy(&modelCacheTraining[i]);
        ML2_ModelCacheDestroy(&modelCacheGradientTraining[i]);
        ML2_BatchDestroy(&batchTraining[i]);
    }
    ML2_ModelCacheDestroy(&modelCacheTesting);
    ML2_BatchDestroy(&batchTesting);
    ML2_OptimizerDestroy(&optimizer);

    ML2_ModelCacheDestroy(&modelCacheSample);
    ML2_LayerCacheDestroy(&inputSampleSingle);
}

static ML2_Scalar ModelLossTraining() {
    ML2_Scalar loss = {};
    #pragma omp parallel for
    for (int i = 0; i < batches; i++) {
        ML2_ModelCacheClear(modelCacheTraining[i]);
        ML2_ModelForward(model, modelCacheTraining[i], batchTraining[i].input);
        ML2_Scalar l = ML2_ModelCacheLoss(modelCacheTraining[i], batchTraining[i].output, lossForward);
        #pragma omp critical
        {
            loss += l;
        }
    }
    return loss / batches;
}

static ML2_Scalar ModelLossTesting() {
    ML2_ModelCacheClear(modelCacheTesting);
    ML2_ModelForward(model, modelCacheTesting, batchTesting.input);
    return ML2_ModelCacheLoss(modelCacheTesting, batchTesting.output, lossForward);
}

static void ModelTrain() {
    #pragma omp parallel for
    for (int i = 0; i < batches; i++) {
        ML2_ModelClear(modelGradient[i]);
        ML2_ModelCacheClear(modelCacheTraining[i]);
        ML2_ModelCacheClear(modelCacheGradientTraining[i]);
        ML2_ModelForward(model, modelCacheTraining[i], batchTraining[i].input);
        ML2_ModelBackward(model, modelGradient[i], modelCacheTraining[i], modelCacheGradientTraining[i], batchTraining[i], lossBackward);
    }
    for (int i = 1; i < batches; i++) {
        ML2_ModelSum(modelGradient[0], modelGradient[i]);
    }
    ML2_ModelOptimize(model, modelGradient[0], &optimizer);
}

[[maybe_unused]] static ML2_LayerCache ModelBatchTrainingInputSample(int sample) {
    for (int i = 0; i < batches; i++) {
        ML2_LayerCacheMatrices images = ML2_LayerCacheAsMatrices(batchTraining[i].input);
        if (sample < images.info.samples) {
            return (ML2_LayerCache){ML2_LayerCacheTypeMatrices, .as.matrices = {{1, images.info.matrices, images.info.vectors, images.info.scalars}, {ML2_MatricesAt(images, sample, 0, 0, 0)}}};
        }
        sample -= images.info.samples;
    }
    ML2_UNREACHABLE("Invalid Sample: %d", sample);
}

[[maybe_unused]] static ML2_LayerCacheScalars ModelBatchTrainingOutputSample(int sample) {
    for (int i = 0; i < batches; i++) {
        ML2_LayerCacheScalars scalars = ML2_LayerCacheAsScalars(batchTraining[i].output);
        if (sample < scalars.info.samples) {
            return (ML2_LayerCacheScalars){{1, scalars.info.scalars}, {ML2_ScalarsAt(scalars, sample, 0)}};
        }
        sample -= scalars.info.samples;
    }
    ML2_UNREACHABLE("Invalid Sample: %d", sample);
}

[[maybe_unused]] static void ModelInfoPrint() {
    printf("Model:\n");
    ML2_ModelInfoPrint(model, 0);
    for (int i = 0; i < batches; i++) {
        printf("BatchTraining[%d]:\n", i);
        ML2_BatchInfoPrint(batchTraining[i], 0);
        printf("Model[%d]:\n", i);
        ML2_ModelInfoPrint(modelGradient[i], 0);
        printf("ModelCacheTraining[%d]:\n", i);
        ML2_ModelCacheInfoPrint(modelCacheTraining[i], 0);
        printf("ModelCacheGradientTraining[%d]:\n", i);
        ML2_ModelCacheInfoPrint(modelCacheGradientTraining[i], 0);
    }
    printf("ModelCacheTesting:\n");
    ML2_ModelCacheInfoPrint(modelCacheTesting, 0);
    printf("BatchTesting:\n");
    ML2_BatchInfoPrint(batchTesting, 0);
    printf("Optimizer:\n");
    ML2_OptimizerInfoPrint(ML2_OptimizerAsParameters(optimizer), 0);
}

[[maybe_unused]] static ML2_Size ModelSizeof() {
    ML2_Size size = {};

    size += ML2_ModelSizeof(model);
    for (int i = 0; i < batches; i++) {
        size += ML2_BatchSizeof(batchTraining[i]);
        size += ML2_ModelSizeof(modelGradient[i]);
        size += ML2_ModelCacheSizeof(modelCacheTraining[i]);
        size += ML2_ModelCacheSizeof(modelCacheGradientTraining[i]);
    }
    size += ML2_ModelCacheSizeof(modelCacheTesting);
    size += ML2_BatchSizeof(batchTesting);
    size += ML2_OptimizerSizeof(optimizer);

    return size;
}

static void ModelEval(ML2_ModelCache modelCache, ML2_LayerCache input, ML2_LayerCache output) {
    ML2_ModelCacheClear(modelCache);
    ML2_ModelForward(model, modelCache, input);
    ML2_ActivationSoftmaxForward(ML2_ModelCacheOutput(modelCache), output);
}

#ifndef NO_MAIN
int main() {
    ModelInit();
    ModelInfoPrint();
    printf("Size: %zu\n", ModelSizeof());

    constexpr int iterations = 100;
    printf("Loss[0]: Training: %f, Testing: %f\n", ModelLossTraining(), ModelLossTesting());
    for (int i = 1; i <= iterations; i++) {
        ModelTrain();
        printf("Loss[%d]: Training: %f, Testing: %f\n", i, ModelLossTraining(), ModelLossTesting());
    }

    ML2_LayerCache output = ML2_LayerCacheNew(ML2_LayerCacheAsInfo(ML2_ModelCacheOutput(modelCacheTesting)));
    ModelEval(modelCacheTesting, batchTesting.input, output);

    ML2_LayerCacheMatrices input = ML2_LayerCacheAsMatrices(batchTesting.input);
    ML2_LayerCacheScalars expectedOutput = ML2_LayerCacheAsScalars(batchTesting.output);
    ML2_LayerCacheScalars modelOutput = ML2_LayerCacheAsScalars(output);

    int correct = 0;
    for (int s = 0; s < testingSamples; s++) {
        ML2_Scalar max = *ML2_ScalarsAt(modelOutput, s, 0);
        int maxIdx = 0;
        for (int i = 1; i < outputs; i++) {
            ML2_Scalar scalar = *ML2_ScalarsAt(modelOutput, s, i);
            if (scalar > max) {
                max = scalar;
                maxIdx = i;
            }
        }
        for (int i = 0; i < outputs; i++) {
            if (*ML2_ScalarsAt(expectedOutput, s, i) == 1.0f) {
                if (i == maxIdx) {
                    correct++;
                }
                break;
            }
        }
        if (1) {
            for (int i = 0; i < outputs; i++) {
                if (*ML2_ScalarsAt(expectedOutput, s, i) == 1.0f) {
                    if (i == maxIdx) {
                        printf("\e[0;32m"); // green
                    } else {
                        printf("\e[0;31m"); // red
                    }
                }
                printf("%d: %.0f (%5.3f)", i, *ML2_ScalarsAt(expectedOutput, s, i), *ML2_ScalarsAt(modelOutput, s, i));
                printf("\e[0m"); // reset color
                printf("\n");
            }

            for (int y = 0; y < imageHeight; y++) {
                for (int x = 0; x < imageWidth; x++) {
                    constexpr char brightness[] = ".:;-=+*#%@";
                    printf("%c ", brightness[(int)floorf(*ML2_MatricesAt(input, s, y, x, 0) * (sizeof(brightness) - 2))]);
                }
                printf("\n");
            }
            getchar();
        }
    }
    printf("\n");

    printf("Testing Accuracy: %.9g%%\n", (double)correct / (double)testingSamples * 100.0);

    ML2_LayerCacheDestroy(&output);
    ModelDeinit();
    return 0;
}
#endif // NO_MAIN