/*
    I asked chatGPT to give me a model that would be hard to make to see if i can utilize DAGs,
    i can and i realised important things about the design of the library,
    but the model he gave me is ridiculous,
    im leaving this and moving on to implementing other models by other REAL people

    below is the ridiculously large model chatGPT gave me 
*/

/*
Treat the picture as four blocks glued together by a DAG.

Input shape is `N x 1 x 28 x 28`, where `N` is batch size. The network keeps a tensor cache for every intermediate output, and the residual blocks reuse the same input tensor in two places: the main path and the skip path. That is the part that makes it a DAG instead of a plain chain.

The full flow is:

`Input -> Conv(3x3, 32, stride=1, pad=1) -> BN -> ReLU -> Residual Block -> Residual Block -> MaxPool(2x2) -> Downsample Residual Block -> MaxPool(2x2) -> Flatten -> Dense(128) -> ReLU -> Dense(10) -> Softmax`

The hidden detail is that each “block” is really a small subgraph.

The first convolution changes `1 x 28 x 28` into `32 x 28 x 28`. A `3x3` conv with `pad=1` and `stride=1` keeps width and height unchanged. Without padding, the image would shrink every time you apply a `3x3` filter. With `pad=1`, the output size is:

`out = (in + 2*pad - kernel) / stride + 1 = (28 + 2 - 3) + 1 = 28`

So the first hidden tensor becomes `32 x 28 x 28`.

BatchNorm comes next. For every channel separately, it computes mean and variance over the batch and spatial positions, normalizes the values, then applies learnable scale and shift. It has parameters like `gamma` and `beta`, but it also has running averages used only at inference time. That is an important hidden part: training and inference behave differently.

ReLU is just `max(0, x)`. It adds nonlinearity after normalization.

The first residual block is:

Main path:
`Conv(3x3, 32, stride=1, pad=1) -> BN -> ReLU -> Conv(3x3, 32, stride=1, pad=1) -> BN`

Skip path:
`identity`

Merge:
`Add(main, skip) -> ReLU`

This block works because both paths produce the same shape: `32 x 28 x 28`. The add is elementwise, so shapes must match exactly.

The reason this block helps is that the network does not need to learn an entirely new representation from scratch. It can learn a small correction to the input of the block. That usually makes training deeper CNNs much easier.

The second residual block changes channels from `32` to `64`.

Main path:
`Conv(3x3, 64, stride=1, pad=1) -> BN -> ReLU -> Conv(3x3, 64, stride=1, pad=1) -> BN`

Skip path:
`Conv(1x1, 64, stride=1, pad=0)`

Merge:
`Add(main, skip) -> ReLU`

The hidden reason for the `1x1` convolution is shape matching. The main path now outputs `64` channels, but the input still has `32`. You cannot add tensors with different channel counts, so the skip path uses a `1x1` projection to convert `32` channels into `64`.

A `1x1` convolution does not mix neighboring pixels spatially. It only mixes channels at the same pixel location. Think of it as a learned per-pixel linear transform across channels.

After that comes max pooling:

`MaxPool(2x2, stride=2)`

This halves the width and height. So `64 x 28 x 28` becomes `64 x 14 x 14`. MaxPool keeps only the maximum value in each `2x2` window. The hidden part here is that during backpropagation, the gradient flows only to the position that was the maximum in each window. So you usually need to store the max locations or recompute them.

The downsample residual block is where stride enters the model.

Main path:
`Conv(3x3, 64, stride=2, pad=1) -> BN -> ReLU -> Conv(3x3, 64, stride=1, pad=1) -> BN`

Skip path:
`Conv(1x1, 64, stride=2, pad=0)`

Merge:
`Add(main, skip) -> ReLU`

This block reduces spatial size from `14 x 14` to `7 x 7` on both paths. Again, the skip path exists only so the shapes match for the add. The stride-2 convolution is the learned downsampling operation.

Then another max pool:

`MaxPool(2x2, stride=2)`

From `64 x 7 x 7` this becomes `64 x 3 x 3`. The exact size is `floor(7 / 2) = 3`.

Then flatten:

`64 x 3 x 3 = 576`

Flatten is not a learned layer. It is only a reshape. It turns the final feature map into a vector so dense layers can consume it.

Then:

`Dense(128) -> ReLU -> Dense(10) -> Softmax`

The last dense layer outputs 10 values, one for each MNIST digit. Softmax turns them into probabilities. In practice, it is often better to train with raw logits and a fused cross-entropy loss for numerical stability. Internally, that means the final layer can just output the 10 logits, and the loss function handles softmax behavior.

If you are implementing this in your own engine, the important hidden execution detail is this:

For every residual block, the input tensor must be saved until both the main path and the skip path are finished. Then you add them. In graph terms:

`x -> main_path -> m`
`x -> skip_path -> s`
`m + s -> y`

So your executor cannot be purely “one layer after another” unless you model the block itself as one node that internally runs a subgraph.

Backward pass through the add is simple but important: the output gradient is copied to both branches. If `y = a + b`, then `dy/da = 1` and `dy/db = 1`, so both branches receive the same upstream gradient.

A practical implementation order for each residual block is:

1. Read the block input from cache.
2. Compute the main branch.
3. Compute the skip branch.
4. Add the two outputs elementwise.
5. Apply ReLU.
6. Store the result in the next cache slot.

For the shapes, the whole network is:

Input: `1 x 28 x 28`

After first conv: `32 x 28 x 28`

After residual block 1: `32 x 28 x 28`

After residual block 2: `64 x 28 x 28`

After maxpool: `64 x 14 x 14`

After downsample block: `64 x 7 x 7`

After maxpool: `64 x 3 x 3`

After flatten: `576`

After dense 128: `128`

After dense 10: `10`

The model is strong enough for MNIST, but the main reason to build it is not raw accuracy. It is that it forces you to implement the pieces that matter in a real DAG engine: branching, merging, shape matching with `1x1` projections, pooling, stride, and cache management.

A good way to implement it is to make each residual block a single higher-level module in your code, even if internally it uses multiple layers. That keeps the outer model simple while still letting the execution graph branch inside the block.

*/
#define ML2_IMPLEMENTATION
#include "ML2.h"

static ML2_Batch batchTraining = {};
static ML2_Batch batchTesting = {};

static constexpr int trainingSamples = 60'000;
static constexpr int testingSamples = 10'000;
static constexpr int imageChannels = 1;
static constexpr int imageHeight = 28;
static constexpr int imageWidth = 28;
static constexpr int inputs = 784;
static_assert(inputs == imageHeight * imageWidth);
static constexpr int outputs = 10;

static ML2_Model m1 = {}, m1Grad = {};
static ML2_ModelCache mc1 = {}, mc1Grad = {};
static ML2_Optimizer o1 = {};
static ML2_Model m2 = {}, m2Grad = {};
static ML2_Optimizer o2 = {};

static const ML2_LossForward lossForward = ML2_LossForwardCrossEntropy;
static const ML2_LossBackward lossBackward = ML2_LossBackwardCrossEntropy;

static constexpr ML2_Scalar learningRate = 0.01f;

static void ModelInit() {
    ML2_BatchInfo batchInfoTraining = ML2_BatchInfoMake(
        ML2_Matrices(trainingSamples, imageHeight, imageWidth, imageChannels), 
        ML2_Scalars(trainingSamples, outputs)
    );
    ML2_BatchInfo batchInfoTesting = ML2_BatchInfoMake(
        ML2_Matrices(testingSamples, imageHeight, imageWidth, imageChannels), 
        ML2_Scalars(testingSamples, outputs)
    );
    batchTraining = ML2_BatchNew(batchInfoTraining);
    batchTesting = ML2_BatchNew(batchInfoTesting);

    {
        FILE *mnist_train_input = nullptr;
        FILE *mnist_train_output = nullptr;
        ML2_HARD_ASSERT(mnist_train_input = fopen("data/mnist_train_input.bin", "rb"));
        ML2_HARD_ASSERT(mnist_train_output = fopen("data/mnist_train_output.bin", "rb"));
        ML2_LayerCacheMatrices batchTrainingInput = ML2_LayerCacheAsMatrices(batchTraining.input);
        ML2_LayerCacheScalars batchTrainingOutput = ML2_LayerCacheAsScalars(batchTraining.output);
        ML2_HARD_ASSERT(fread(batchTrainingInput.data.matrices, sizeof(*batchTrainingInput.data.matrices), trainingSamples * inputs, mnist_train_input) == trainingSamples * inputs);
        ML2_HARD_ASSERT(fread(batchTrainingOutput.data.scalars, sizeof(*batchTrainingOutput.data.scalars), trainingSamples * outputs, mnist_train_output) == trainingSamples * outputs);
        fclose(mnist_train_input);
        fclose(mnist_train_output);
    }
    {
        FILE *mnist_test_input = nullptr;
        FILE *mnist_test_output = nullptr;
        ML2_HARD_ASSERT(mnist_test_input = fopen("data/mnist_test_input.bin", "rb"));
        ML2_HARD_ASSERT(mnist_test_output = fopen("data/mnist_test_output.bin", "rb"));
        ML2_LayerCacheMatrices batchTestingInput = ML2_LayerCacheAsMatrices(batchTesting.input);
        ML2_LayerCacheScalars batchTestingOutput = ML2_LayerCacheAsScalars(batchTesting.output);
        ML2_HARD_ASSERT(fread(batchTestingInput.data.matrices, sizeof(*batchTestingInput.data.matrices), testingSamples * inputs, mnist_test_input) == testingSamples * inputs);
        ML2_HARD_ASSERT(fread(batchTestingOutput.data.scalars, sizeof(*batchTestingOutput.data.scalars), testingSamples * outputs, mnist_test_output) == testingSamples * outputs);
        fclose(mnist_test_input);
        fclose(mnist_test_output);
    }

    constexpr ML2_Int hidden1 = 2;
    constexpr ML2_Int hidden2 = 2;
    constexpr ML2_Int hidden3 = 4;
    ML2_Arch arch1 = ML2_ArchMake(
        ML2_Conv(
            imageChannels, hidden1,
            3, 3,
            1, 1,
            1, 1
        ),
        ML2_Sin(), // ML2_BatchNorm(),
        ML2_ReLU(),
        // [2]: Add me to [4]
        ML2_Conv(
            hidden1, hidden2,
            3, 3,
            1, 1,
            1, 1
        ),
        ML2_Sin(), // ML2_BatchNorm(),
        // [4]: Hi im [4]
        ML2_ReLU(),
        ML2_Sigmoid(), // ML2_MaxPool(2, 2, stride = 2),
        // [5]: Pass me to m2
        ML2_Conv(
            hidden2, hidden3,
            3, 3,
            1, 1,
            1, 1
        ),
        ML2_Sin(), // ML2_BatchNorm(),
        // [7]: Hi im [7]
        ML2_ReLU(),
        ML2_Sigmoid(), // ML2_MaxPool(2, 2, stride = 2),
        ML2_Sigmoid(), // ML2_AveragePool(2, 2, stride = 2),
        ML2_Flatten(),
        ML2_Linear(hidden3 * 28 * 28, 128),
        ML2_ReLU(),
        ML2_Linear(128, 10),
        ML2_Softmax(),
    );
    ML2_Arch arch2 = ML2_ArchMake(
        ML2_Conv(
            hidden2, hidden3,
            1, 1,
            0, 0,
            1, 1
        ),
        // [0]: Add me to [7]
    );

    m1 = ML2_ModelNew(arch1), m1Grad = ML2_ModelNew(arch1);
    mc1 = ML2_ModelCacheNew(arch1, batchInfoTraining), mc1Grad = ML2_ModelCacheNew(arch1, batchInfoTraining);
    m2 = ML2_ModelNew(arch2), m2Grad = ML2_ModelNew(arch2);
    
    ML2_OptimizerParameters op = ML2_Adam(learningRate, 0.9, 0.999);
    o1 = ML2_OptimizerNew(op, arch1);
    o2 = ML2_OptimizerNew(op, arch2);
}

static void ModelDeinit() {
    ML2_BatchDestroy(&batchTraining);
    ML2_BatchDestroy(&batchTesting);
    ML2_ModelDestroy(&m1), ML2_ModelDestroy(&m1Grad);
    ML2_ModelCacheDestroy(&mc1), ML2_ModelCacheDestroy(&mc1Grad);
    ML2_ModelDestroy(&m2), ML2_ModelDestroy(&m2Grad);
    ML2_OptimizerDestroy(&o1), ML2_OptimizerDestroy(&o2);
}

static void ModelClear() {
    ML2_ModelClear(m1Grad);
    ML2_ModelCacheClear(mc1);
    ML2_ModelCacheClear(mc1Grad);
}

static constexpr ML2_Int branchFrom1 = 2;
static constexpr ML2_Int branchTo1 = 4;
static constexpr ML2_Int branchFrom2 = 5;
static constexpr ML2_Int branchTo2 = 7;

static void ModelForward(ML2_LayerCache input) {
    ML2_ModelCacheClear(mc1);

    ML2_LayerForward(m1.layers[0], input, mc1.layers[0]);
    for (ML2_Int i = 1; i <= branchTo1; i++) {
        ML2_LayerForward(m1.layers[i], mc1.layers[i - 1], mc1.layers[i]);
    }

    ML2_LayerCacheSum(mc1.layers[branchTo1], mc1.layers[branchFrom1]);

    for (ML2_Int i = branchTo1 + 1; i <= branchTo2; i++) {
        ML2_LayerForward(m1.layers[i], mc1.layers[i - 1], mc1.layers[i]);
    }

    ML2_LayerForward(m2.layers[0], mc1.layers[branchFrom2], mc1.layers[branchTo2]);

    for (ML2_Int i = branchTo2 + 1; i <= m1.layerCount - 1; i++) {
        ML2_LayerForward(m1.layers[i], mc1.layers[i - 1], mc1.layers[i]);
    }
}

static void ModelBackward(ML2_LayerCache expectedOutput) {
    lossBackward(ML2_ModelCacheOutput(mc1), expectedOutput, ML2_ModelCacheOutput(mc1Grad));
    for (ML2_Int i = mc1.layerCount - 1; i >= branchTo2 + 1; i--) {
        ML2_LayerBackward(m1.layers[i], mc1.layers[i - 1], mc1Grad.layers[i]);
        ML2_LayerCacheBackward(m1.layers[i], mc1.layers[i - 1], mc1Grad.layers[i - 1], mc1.layers[i], mc1Grad.layers[i]);
    }

    ML2_LayerBackward(m2Grad.layers[0], mc1.layers[branchFrom2], mc1Grad.layers[branchTo2]);

    for (ML2_Int i = branchTo2; i >= branchTo1 + 1; i--) {
        ML2_LayerBackward(m1Grad.layers[i], mc1.layers[i - 1], mc1Grad.layers[i]);
        ML2_LayerCacheBackward(m1.layers[i], mc1.layers[i - 1], mc1Grad.layers[i - 1], mc1.layers[i], mc1Grad.layers[i]);
    }

    ML2_LayerCacheSum(mc1Grad.layers[branchFrom1], mc1Grad.layers[branchTo1]);

    for (ML2_Int i = branchTo1; i >= 1; i--) {
        ML2_LayerBackward(m1Grad.layers[i], mc1.layers[i - 1], mc1Grad.layers[i]);
        ML2_LayerCacheBackward(m1.layers[i], mc1.layers[i - 1], mc1Grad.layers[i - 1], mc1.layers[i], mc1Grad.layers[i]);
    }
    ML2_LayerBackward(m1Grad.layers[0], batchTraining.input, mc1Grad.layers[0]);
}

static void ModelOptimize() {
    ML2_ModelOptimize(m1, m1Grad, &o1);
    ML2_ModelOptimize(m2, m2Grad, &o2);
}

static ML2_Scalar ModelLoss() {
    // NOTE: both work the same
    return lossForward(ML2_ModelCacheOutput(mc1), batchTraining.output);
    // return ML2_ModelCacheLoss(mc1, batchTraining.output, lossForward);
}

int main() {
    ModelInit();

    constexpr int iterations = 10;
    int iterationsLen = (int)floor(log10(iterations)) + 1;

    ModelForward(batchTraining.input);
    ML2_Scalar loss = ModelLoss();
    printf("Loss[%*d] = %f\n", iterationsLen, 0, loss);
    for (int i = 1; i <= iterations; i++) {
        ModelClear();
        ModelForward(batchTraining.input);
        ModelBackward(batchTraining.output);
        ModelOptimize();
        loss = ModelLoss();
        printf("Loss[%*d] = %f\n", iterationsLen, i, loss);
    }

    ModelDeinit();
    return 0;
}