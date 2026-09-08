#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

int WindowWidth = 1200;
int WindowHeight = 600;
HDC hDoubleBufferDC = nullptr;
HBITMAP hDoubleBufferBitmap = nullptr;

// V RENDERING UTILITIES V

typedef uint32_t Color;

// WinGDI uses BGRA
#define RGBA(r, g, b, a) ((Color)(((a) << 24) | ((r) << 16) | ((g) << 8) | (b)))
#define R(c) (((c) >> 16) & 0xff)
#define G(c) (((c) >> 8) & 0xff)
#define B(c) ((c) & 0xff)
#define A(c) (((c) >> 24) & 0xff)

constexpr Color backgroundColor = RGBA(0x18, 0x18, 0x18, 0xff);
constexpr Color boundaryColor = RGBA(0x80, 0x80, 0x80, 0xff);
constexpr int boundaryWidth = 2;

Color *Pixels = nullptr;

Color *PixelsAt(int x, int y) {
    assert((0 <= x && x < WindowWidth && 0 <= y && y < WindowHeight));
    return &Pixels[WindowWidth * y + x];
}

void BlendColor(Color *c1, Color c2) {
    uint8_t R1 = R(*c1), G1 = G(*c1), B1 = B(*c1), A1 = A(*c1);

    uint8_t R2 = R(c2), G2 = G(c2), B2 = B(c2), A2 = A(c2);

    R1 = (R1 * (0xff - A2) + R2 * A2) / 0xff;
    G1 = (G1 * (0xff - A2) + G2 * A2) / 0xff;
    B1 = (B1 * (0xff - A2) + B2 * A2) / 0xff;
    A1 = (A1 * (0xff - A2) + A2) / 0xff;

    *c1 = RGBA(R1, G1, B1, A1);
}

typedef RECT Rect;

typedef struct {
    int x1, y1;
    int x2, y2;
} Line;

typedef struct {
    int x, y;
    int r;
} Circle;

int RectWidth(Rect rect) {
    return rect.right - rect.left;
}

int RectHeight(Rect rect) {
    return rect.bottom - rect.top;
}

bool AtBoundary(Rect boundary, int x, int y) {
    return (boundary.left <= x && x < boundary.right) && (boundary.top <= y && y < boundary.bottom);
}

void DrawPoint(Rect boundary, POINT point, Color color) {
    int x = point.x + boundary.left;
    int y = point.y + boundary.top;
    if (AtBoundary(boundary, x, y)) {
        BlendColor(PixelsAt(x, y), color);
    }
}

void DrawRect(Rect boundary, Rect rect, Color color) {
    for (int y = rect.top; y < rect.bottom; y++) {
        for (int x = rect.left; x < rect.right; x++) {
            int nx = x + boundary.left;
            int ny = y + boundary.top;
            if (AtBoundary(boundary, nx, ny)) {
                BlendColor(PixelsAt(nx, ny), color);
            }
        }
    }
}

void DrawBackground(Rect boundary, Color color) {
    DrawRect(boundary, boundary, color);
}

void DrawBoundary(Rect boundary, int width, Color color) {
    int left = boundary.left;
    int top = boundary.top;
    int right = boundary.right;
    int bottom = boundary.bottom;

    for (int y = top; y < top + width && y < bottom; y++) {
        for (int x = left; x < right; x++) {
            if (AtBoundary(boundary, x, y)) {
                BlendColor(PixelsAt(x, y), color);
            }
        }
    }

    for (int y = bottom - width; y < bottom; y++) {
        if (y < top) continue;
        for (int x = left; x < right; x++) {
            if (AtBoundary(boundary, x, y)) {
                BlendColor(PixelsAt(x, y), color);
            }
        }
    }

    for (int y = top + width; y < bottom - width; y++) {
        for (int x = left; x < left + width && x < right; x++) {
            if (AtBoundary(boundary, x, y)) {
                BlendColor(PixelsAt(x, y), color);
            }
        }
        for (int x = right - width; x < right; x++) {
            if (AtBoundary(boundary, x, y)) {
                BlendColor(PixelsAt(x, y), color);
            }
        }
    }
}

#define swap(T, a, b) \
    do {              \
        T t = a;      \
        a = b;        \
        b = t;        \
    } while (0)

void DrawLine(Rect boundary, Line line, int width, Color color) {
    int dx = line.x2 - line.x1;
    int dy = line.y2 - line.y1;
    if (!dx && !dy) return;

    int half = width / 2;

    if (abs(dx) > abs(dy)) {
        if (line.x1 > line.x2) {
            swap(int, line.x1, line.x2);
            swap(int, line.y1, line.y2);
        }

        for (int x = line.x1; x <= line.x2; ++x) {
            int y = (x - line.x1) * dy / dx + line.y1;

            for (int oy = -half; oy <= half; oy++) {
                for (int ox = -half; ox <= half; ox++) {
                    int nx = x + ox + boundary.left;
                    int ny = y + oy + boundary.top;
                    if (AtBoundary(boundary, nx, ny)) {
                        BlendColor(PixelsAt(nx, ny), color);
                    }
                }
            }
        }
    } else {
        if (line.y1 > line.y2) {
            swap(int, line.x1, line.x2);
            swap(int, line.y1, line.y2);
        }

        for (int y = line.y1; y <= line.y2; ++y) {
            int x = (y - line.y1) * dx / dy + line.x1;

            for (int oy = -half; oy <= half; oy++) {
                for (int ox = -half; ox <= half; ox++) {
                    int nx = x + ox + boundary.left;
                    int ny = y + oy + boundary.top;
                    if (AtBoundary(boundary, nx, ny)) {
                        BlendColor(PixelsAt(nx, ny), color);
                    }
                }
            }
        }
    }
}

void DrawCircle(Rect boundary, Circle c, Color color) {
    for (int dy = -c.r; dy <= c.r; dy++) {
        for (int dx = -c.r; dx <= c.r; dx++) {
            if (dx * dx + dy * dy <= c.r * c.r) {
                int x = c.x + dx;
                int y = c.y + dy;
                if (AtBoundary(boundary, x, y)) {
                    BlendColor(PixelsAt(x, y), color);
                }
            }
        }
    }
}

// ^ RENDERING UTILITIES ^

void DestroyDoubleBuffer(void) {
    SelectObject(hDoubleBufferDC, nullptr);
    DeleteDC(hDoubleBufferDC);
    DeleteObject(hDoubleBufferBitmap);
}

void NewDoubleBuffer(HDC hdc, int w, int h) {
    if (hDoubleBufferDC != nullptr) {
        DestroyDoubleBuffer();
    }
    hDoubleBufferDC = CreateCompatibleDC(hdc);
    // hDoubleBufferBitmap = CreateCompatibleBitmap(hdc, w, h);
    BITMAPINFO bmi = {
        .bmiHeader = {
            .biSize = sizeof(BITMAPINFOHEADER),
            .biWidth = w,
            .biHeight = -h,
            .biPlanes = 1,
            .biBitCount = 32,
            .biCompression = BI_RGB,
        },
    };
    hDoubleBufferBitmap = CreateDIBSection(hDoubleBufferDC, &bmi, DIB_RGB_COLORS, (void **)&Pixels, nullptr, 0);
    SelectObject(hDoubleBufferDC, hDoubleBufferBitmap);
}

void ResizeWindow(HWND hwnd) {
    Rect ClientRect;
    GetClientRect(hwnd, &ClientRect);
    WindowWidth = RectWidth(ClientRect);
    WindowHeight = RectHeight(ClientRect);
    HDC hdc = GetDC(hwnd);
    NewDoubleBuffer(hdc, WindowWidth, WindowHeight);
    ReleaseDC(hwnd, hdc);
}

constexpr int RenderTimerId = 0;
constexpr double FPS = 144.0;
constexpr double DeltaTime = 1.0 / FPS;

#define NO_MAIN
#include "mnist.c"
#undef NO_MAIN

static uint64_t epochs;
static bool paused;
static enum {
    TRAIN_MODE = 0,
    TEST_MODE,
    MODE_COUNT,
} drawMode;
static bool cycling;
static int sampleIndex;

constexpr int DIGITS_WIDTH = 5;
constexpr int DIGITS_HEIGHT = 7;

constexpr uint8_t DIGITS[10][DIGITS_WIDTH * DIGITS_HEIGHT] = {
    [0] = {
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 1, 1,
        1, 0, 1, 0, 1,
        1, 1, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
    },
    [1] = {
        0, 0, 1, 0, 0,
        0, 1, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        1, 1, 1, 1, 1,
    },
    [2] = {
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 1, 1, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
    },
    [3] = {
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 1, 1, 0,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
    },
    [4] = {
        0, 0, 0, 1, 1,
        0, 0, 1, 0, 1,
        0, 1, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
    },
    [5] = {
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
    },
    [6] = {
        0, 0, 1, 1, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
    },
    [7] = {
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
    },
    [8] = {
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
    },
    [9] = {
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 1, 1, 0, 0,
    },
};

void DrawLabel(Rect boundary, int digit, Color color) {
    assert(0 <= digit && digit <= 9);

    for (int y = 0; y < DIGITS_HEIGHT; y++) {
        for (int x = 0; x < DIGITS_WIDTH; x++) {
            if (DIGITS[digit][y * DIGITS_WIDTH + x] == 0) continue;
            int x1 = RectWidth(boundary) * x / DIGITS_WIDTH;
            int y1 = RectHeight(boundary) * y / DIGITS_HEIGHT;
            int x2 = RectWidth(boundary) * (x + 1) / DIGITS_WIDTH;
            int y2 = RectHeight(boundary) * (y + 1) / DIGITS_HEIGHT;
            Rect tile = {
                .left = x1,
                .top = y1,
                .right = x2,
                .bottom = y2,
            };
            DrawRect(boundary, tile, color);
        }
    }
}

// ML2_FN ML2_LayerCacheScalars ML2_ScalarsSample(ML2_LayerCacheScalars scalars, int sample) {
//     return (ML2_LayerCacheScalars){{1, scalars.info.scalars}, {ML2_ScalarsAt(scalars, sample, 0)}};
// }

// ML2_FN ML2_LayerCacheImages ML2_ImagesSample(ML2_LayerCacheImages images, int sample) {
//     return (ML2_LayerCacheImages){{1, images.info.channels, images.info.height, images.info.width}, {ML2_ImagesAt(images, sample, 0, 0, 0)}};
// }

// ML2_FN ML2_LayerCache ML2_LayerCacheSample(ML2_LayerCache cache, int sample) {
//     switch (cache.type) {
//         case ML2_LayerCacheTypeScalars: return (ML2_LayerCache){cache.type, .as.scalars = ML2_ScalarsSample(cache.as.scalars, sample)};
//         case ML2_LayerCacheTypeImages: return (ML2_LayerCache){cache.type, .as.images = ML2_ImagesSample(cache.as.images, sample)};
//         default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
//     }
// }

ML2_FN ML2_Scalar *ML2_ScalarsRaw(ML2_LayerCacheScalars scalars) {
    return ML2_ScalarsAt(scalars, 0, 0);
}

ML2_FN ML2_Scalar *ML2_VectorsRaw(ML2_LayerCacheVectors vectors) {
    return ML2_VectorsAt(vectors, 0, 0, 0);
}

ML2_FN ML2_Scalar *ML2_ImagesRaw(ML2_LayerCacheMatrices images) {
    return ML2_MatricesAt(images, 0, 0, 0, 0);
}

ML2_FN ML2_Scalar *ML2_LayerCacheRaw(ML2_LayerCache cache) {
    switch (cache.type) {
        case ML2_LayerCacheTypeScalars: return ML2_ScalarsRaw(cache.as.scalars);
        case ML2_LayerCacheTypeVectors: return ML2_VectorsRaw(cache.as.vectors);
        case ML2_LayerCacheTypeMatrices: return ML2_ImagesRaw(cache.as.matrices);
        default: ML2_UNREACHABLE("Unknown ML2_LayerCacheType");
    }
}

// this function is reused with a very tiny difference, so i put a pointer to use as a condition
void DrawLabelPredictionBars(Rect boundary, ML2_LayerCacheScalars *sampleTrainingOutput, ML2_LayerCache input) {
    static ML2_LayerCache output = {};
    static bool init = false;
    if (!init) {
        // MemLeak:
        output = ML2_LayerCacheNew(ML2_Scalars(1, outputs));
        init = true;
    }
    ML2_LayerCacheClear(output);
    ModelEval(modelCacheSample, input, output);
    ML2_LayerCacheScalars modelOutput = ML2_LayerCacheAsScalars(output);

    for (int i = 0; i < outputs; i++) {
        int x1 = RectWidth(boundary) * i / outputs;
        int x2 = RectWidth(boundary) * (i + 1) / outputs;

        int padFactor = 10;
        int pad = min(RectWidth(boundary) * padFactor / outputs / 100, RectHeight(boundary) * padFactor / 100);

        int fill = RectHeight(boundary) * *ML2_ScalarsAt(modelOutput, 0, i);
        Rect predictionBar = {
            .left = x1,
            .top = RectHeight(boundary) - fill,
            .right = x2,
            .bottom = RectHeight(boundary),
        };
        Color outputColor = (int)floorf(fminf(255.0f, *ML2_ScalarsAt(modelOutput, 0, i) * 255.0f));
        DrawRect(boundary, predictionBar, RGBA(0xff - outputColor, outputColor, 0, 0xff));

        Rect digitRect = {
            .left = x1 + boundary.left + pad,
            .top = boundary.top + pad,
            .right = x2 + boundary.left - pad,
            .bottom = boundary.bottom - pad,
        };
        Color digitValue = 0xff;
        if (sampleTrainingOutput) {
            digitValue = *ML2_ScalarsAt(*sampleTrainingOutput, 0, i) == 1.0f ? 0xff : 0x70;
        }
        DrawLabel(digitRect, i, RGBA(digitValue, digitValue, digitValue, 0xff));
    }
    for (int i = 0; i < outputs; i++) {
        int x2 = RectWidth(boundary) * (i + 1) / outputs;

        Line separator = {
            .x1 = x2,
            .y1 = 0,
            .x2 = x2,
            .y2 = RectHeight(boundary),
        };
        DrawLine(boundary, separator, boundaryWidth, boundaryColor);
    }
}

void DrawSampleImage(Rect boundary) {
    ML2_LayerCache sampleInput = ModelBatchTrainingInputSample(sampleIndex);
    ML2_Scalar *rawInput = ML2_LayerCacheRaw(sampleInput);

    for (int i = 0; i < inputs; i++) {
        int alpha = (int)floorf(fminf(255.0f, rawInput[i] * 255.0f));
        Color tileColor = RGBA(0xff, 0xff, 0xff, alpha);
        int x = i % imageWidth;
        int y = i / imageHeight;
        int x1 = RectWidth(boundary) * x / imageWidth;
        int y1 = RectHeight(boundary) * y / imageHeight;
        int x2 = RectWidth(boundary) * (x + 1) / imageWidth;
        int y2 = RectHeight(boundary) * (y + 1) / imageHeight;
        Rect tile = {
            .left = x1,
            .top = y1,
            .right = x2,
            .bottom = y2,
        };
        DrawRect(boundary, tile, tileColor);
    }
}

void DrawImagePrediction(Rect boundary) {
    ML2_LayerCache sampleTrainingInput = ModelBatchTrainingInputSample(sampleIndex);
    ML2_LayerCacheScalars sampleTrainingOutput = ModelBatchTrainingOutputSample(sampleIndex);
    ML2_LayerCacheCopy(inputSampleSingle, sampleTrainingInput);

    DrawLabelPredictionBars(boundary, &sampleTrainingOutput, inputSampleSingle);
}

uint8_t TestPixels[imageWidth * imageHeight] = {};
uint8_t *TestPixelsAt(int x, int y) {
    assert(0 <= x && x < imageWidth && 0 <= y && y < imageHeight);
    return &TestPixels[imageWidth * y + x];
}

void DrawImageTest(Rect boundary) {
    for (int y = 0; y < imageHeight; y++) {
        for (int x = 0; x < imageWidth; x++) {
            int x1 = RectWidth(boundary) * x / imageWidth;
            int y1 = RectHeight(boundary) * y / imageHeight;
            int x2 = RectWidth(boundary) * (x + 1) / imageWidth;
            int y2 = RectHeight(boundary) * (y + 1) / imageHeight;
            Rect tile = {
                .left = x1,
                .top = y1,
                .right = x2,
                .bottom = y2,
            };
            int value = *TestPixelsAt(x, y);
            DrawRect(boundary, tile, RGBA(value, value, value, 0xff));
        }
    }
}

void DrawImageTestPrediction(Rect boundary) {
    ML2_Scalar *rawInput = ML2_LayerCacheRaw(inputSampleSingle);

    for (int i = 0; i < imageWidth * imageHeight; i++) {
        rawInput[i] = (ML2_Scalar)TestPixels[i] / 255.0f;
    }

    DrawLabelPredictionBars(boundary, nullptr, inputSampleSingle);
}

enum : int {
    LossTraining = 0,
    LossTesting,
    LossCount,
};
struct {
    size_t start, end;
    ML2_Scalar data[100'000][LossCount];
} losses;

#define circular_size(ca) (sizeof((ca)->data) / sizeof(*(ca)->data))

#define circular_len(ca) ((ca)->start == (ca)->end - 1 ? circular_size(ca) - 1 : (ca)->end - (ca)->start)

#define circular_at(ca, i, j) ((ca)->data[((ca)->start + (i)) % circular_size(ca)][(j)])

#define circular_append(ca, x, y)                                                          \
    do {                                                                                   \
        (ca)->data[(ca)->end][LossTraining] = (x);                                         \
        (ca)->data[(ca)->end][LossTesting] = (y);                                          \
        (ca)->end = ((ca)->end + 1) % circular_size(ca);                                   \
        if ((ca)->end == (ca)->start) (ca)->start = ((ca)->start + 1) % circular_size(ca); \
    } while (0)

static Color Rainbow(float t) {
    assert(0.0f <= t && t <= 1.0f);

    float x = t * 6.0f;
    int i = (int)floorf(x);
    float u = x - i;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    switch (i) {
        case 0: r = 1.0f;     g = u;        break;
        case 1: r = 1.0f - u; g = 1.0f;     break;
        case 2: g = 1.0f;     b = u;        break;
        case 3: g = 1.0f - u; b = 1.0f;     break;
        case 4: r = u;        b = 1.0f;     break;
        case 5: r = 1.0f;     b = 1.0f - u; break;
        case 6: r = 1.0f;     b = 0.0f;     break;
    }
    uint8_t R = min(0xff, (int)floorf(r * 255.0f));
    uint8_t G = min(0xff, (int)floorf(g * 255.0f));
    uint8_t B = min(0xff, (int)floorf(b * 255.0f));

    return RGBA(R, G, B, 0xff);
}

static int lossSamples = circular_size(&losses);

static void DrawLossGraph(RECT boundary, bool loglog) {
    constexpr float padding = 0.05f;
    constexpr int axisThickness = 1;
    int width = RectWidth(boundary);
    int height = RectHeight(boundary);
    Color axisColor = loglog ? RGBA(0, 0xff, 0, 0xff) : RGBA(0xff, 0, 0, 0xff);
    Line timeAxis = {
        width * padding,
        height * (1.0f - padding),
        width * (1.0f - padding),
        height * (1.0f - padding),
    };
    DrawLine(boundary, timeAxis, axisThickness, axisColor);
    Line lossAxis = {
        width * padding,
        height * padding,
        width * padding,
        height * (1.0f - padding),
    };
    DrawLine(boundary, lossAxis, axisThickness, axisColor);
    if (losses.start == losses.end) return;

    int end = circular_len(&losses);
    int start = max(0, end - lossSamples);

    float max = circular_at(&losses, start, 0);
    float min = max;
    for (int i = start; i < end; i++) {
        for (int j = 0; j < LossCount; j++) {
            float loss = circular_at(&losses, i, j);
            if (loss > max) max = loss;
            if (loss < min) min = loss;
        }
    }

    for (int i = start; i < end - 1; i++) {
        for (int j = 0; j < LossCount; j++) {
            float loss1 = circular_at(&losses, i, j);
            float loss2 = circular_at(&losses, i + 1, j);
            float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
            if (loglog) {
                #define X(i) (((log10f((i) - start + 1) / log10f(end - start)) * (1.0f - 2.0f * padding) + padding) * width)
                #define Y(loss) (((1.0f - log10f((loss) / min) / log10f(max / min)) * (1.0f - 2.0f * padding) + padding) * height)
                x1 = X(i);
                y1 = Y(loss1);
                x2 = X(i + 1);
                y2 = Y(loss2);
                #undef X
                #undef Y
            } else {
                #define X(i) (((float)((i) - start) / (float)(end - 1 - start)) * (1.0f - 2.0f * padding) + padding) * width
                #define Y(loss) ((1.0f - (loss) / max) * (1.0f - 2.0f * padding) + padding) * height
                x1 = X(i);
                y1 = Y(loss1);
                x2 = X(i + 1);
                y2 = Y(loss2);
                #undef X
                #undef Y
            }
            Color color = {};
            static_assert(LossCount == 2);
            switch (j) {
                case LossTraining: {
                    color = Rainbow(((float)i + 0.5f - start) / (float)(end - 1 - start));
                } break;
                case LossTesting: {
                    color = Rainbow(fmodf(1.5f - ((float)i + 0.5f - start) / (float)(end - 1 - start), 1.0f));
                } break;
                default: ML2_UNREACHABLE();
            }
            DrawLine(boundary, (Line){x1, y1, x2, y2}, 1, color);
        }
    }
}

void RectShrinkToSquare(Rect *rect) {
    int diff = RectWidth(*rect) - RectHeight(*rect);
    if (diff > 0) {
        rect->left += diff / 2;
        rect->right -= diff / 2;
    } else if (diff < 0) {
        rect->top += -diff / 2;
        rect->bottom -= -diff / 2;
    }
}

Rect WindowRect = {};
Rect ImageRect = {};
Rect GraphRect = {};
Rect PredictionRect = {};

void CalculateBoundaries() {
    WindowRect = (Rect){0, 0, WindowWidth, WindowHeight};

    ImageRect = (Rect){
        .left = 0,
        .top = 0,
        .right = WindowWidth,
        .bottom = WindowHeight * 8 / 10,
    };
    RectShrinkToSquare(&ImageRect);

    PredictionRect = (Rect){
        .left = 0,
        .top = ImageRect.bottom,
        .right = WindowWidth,
        .bottom = WindowHeight,
    };
}

static bool loglog = false;

void Render() {
    CalculateBoundaries();
    DrawBackground(WindowRect, backgroundColor);

    DrawBackground(ImageRect, backgroundColor);
    DrawBackground(PredictionRect, backgroundColor);

    switch (drawMode) {
        case TRAIN_MODE: {
            int shiftToRight = WindowWidth - ImageRect.right;
            ImageRect.left += shiftToRight;
            ImageRect.right += shiftToRight;

            GraphRect = (Rect){0, 0, ImageRect.left, PredictionRect.top};
            DrawSampleImage(ImageRect);
            DrawLossGraph(GraphRect, loglog);
            DrawImagePrediction(PredictionRect);

            DrawBoundary(GraphRect, boundaryWidth, boundaryColor);
        } break;
        case TEST_MODE: {
            DrawImageTest(ImageRect);
            DrawImageTestPrediction(PredictionRect);
        } break;
        case MODE_COUNT: [[fallthrough]];
        default: ML2_UNREACHABLE("Unknown Mode");
    }

    DrawBoundary(ImageRect, boundaryWidth, boundaryColor);
    DrawBoundary(PredictionRect, boundaryWidth, boundaryColor);
}

DWORD UpdateLoop(void *param [[maybe_unused]]) {
    while (true) {
        ModelTrain();
        epochs++;
        ML2_Scalar lossTraining = ModelLossTraining();
        ML2_Scalar lossTesting = ModelLossTesting();
        circular_append(&losses, lossTraining, lossTesting);

        // if (ml.epochs % 1000 == 0)
        printf("Epochs: %llu, Sample: %d/%d, Loss: Training: %f, Testing: %f\n", epochs, sampleIndex + 1, trainingSamples, lossTraining, lossTesting);
    }
    return 0;
}

int emod(int x, int m) {
    return ((x % m) + m) % m;
}

constexpr int CycleTimerId = 1;
constexpr double CycleTimerFreq = 500.0;

static bool isMouseLeftDown;
static bool isMouseRightDown;

HANDLE UpdateThread = nullptr;

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_MOUSEWHEEL: {
            short delta = GET_WHEEL_DELTA_WPARAM(wParam);
            if (delta > 0) {
                if (lossSamples > 10) lossSamples -= lossSamples / 10;
            } else {
                lossSamples += lossSamples / 10;
            }
        } break;
        case WM_LBUTTONDOWN: {
            isMouseLeftDown = true;
        } break;
        case WM_LBUTTONUP: {
            isMouseLeftDown = false;
        } break;
        case WM_RBUTTONDOWN: {
            isMouseRightDown = true;
        } break;
        case WM_RBUTTONUP: {
            isMouseRightDown = false;
        } break;
        case WM_MOUSEMOVE: {
            if (isMouseLeftDown || isMouseRightDown) {
                int mouseX = LOWORD(lParam);
                int mouseY = HIWORD(lParam);
                if (AtBoundary(ImageRect, mouseX, mouseY)) {
                    int x = (mouseX - ImageRect.left) * imageWidth / RectWidth(ImageRect);
                    int y = (mouseY - ImageRect.top) * imageHeight / RectHeight(ImageRect);
                    if (isMouseLeftDown) {
                        *TestPixelsAt(x, y) = 0xff;
                    } else if (isMouseRightDown) {
                        *TestPixelsAt(x, y) = 0;
                    }
                    InvalidateRect(hwnd, nullptr, false);
                }
            }
        } break;
        case WM_KEYDOWN: {
            switch (wParam) {
                case ' ': {
                    paused = !paused;
                    if (paused) {
                        SuspendThread(UpdateThread);
                    } else {
                        ResumeThread(UpdateThread);
                    }
                } break;
                case 'E': {
                    cycling = !cycling;
                } break;
                case 'T': {
                    drawMode = (drawMode + 1) % MODE_COUNT;
                } break;
                case 'W': {
                    sampleIndex = emod(sampleIndex + 1, trainingSamples);
                } break;
                case 'S': {
                    sampleIndex = emod(sampleIndex - 1, trainingSamples);
                } break;
                case 'R': {
                    epochs = 0;
                    ModelRand();
                } break;
                case 'Z': {
                    memset(TestPixels, 0, sizeof(TestPixels));
                } break;
                case 'L': {
                    loglog = !loglog;
                } break;
                default: {
                    goto noRender;
                }
            }
            InvalidateRect(hwnd, nullptr, false);
        noRender:;
        } break;
        case WM_TIMER: {
            switch (wParam) {
                case RenderTimerId: {
                    InvalidateRect(hwnd, nullptr, false);
                } break;
                case CycleTimerId: {
                    if (cycling) {
                        sampleIndex = emod(sampleIndex + 1, trainingSamples);
                        InvalidateRect(hwnd, nullptr, false);
                    }
                } break;
            }
        } break;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            Rect ClientRect;
            GetClientRect(hwnd, &ClientRect);

            Render();

            BitBlt(
                hdc,
                0, 0,
                WindowWidth, WindowHeight,
                hDoubleBufferDC,
                0, 0,
                SRCCOPY
            );

            EndPaint(hwnd, &ps);
        } break;
        case WM_CREATE: {
            ResizeWindow(hwnd);

            SetTimer(hwnd, RenderTimerId, DeltaTime * 1000.0, nullptr);
            SetTimer(hwnd, CycleTimerId, CycleTimerFreq, nullptr);
        } break;
        case WM_SIZE: {
            ResizeWindow(hwnd);

            InvalidateRect(hwnd, nullptr, false);
        } break;
        case WM_CLOSE: {
            DestroyWindow(hwnd);
        } break;
        case WM_DESTROY: {
            DestroyDoubleBuffer();

            KillTimer(hwnd, RenderTimerId);
            KillTimer(hwnd, CycleTimerId);
            PostQuitMessage(0);
        } break;
        default: return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

const char WindowClassName[] = "MnistWindowClass";
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance [[maybe_unused]], LPSTR lpCmdLine [[maybe_unused]], int nCmdShow) {
    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);

    ModelInit();

    WNDCLASSEX wc = {
        .cbSize = sizeof(wc),
        .cbClsExtra = 0,
        .cbWndExtra = 0,
        .lpszClassName = WindowClassName,
        .lpfnWndProc = WndProc,
        .hInstance = hInstance,
        .style = 0,
        .lpszMenuName = nullptr,
        .hbrBackground = (HBRUSH)(COLOR_WINDOW + 1),
        .hIcon = LoadIcon(nullptr, IDI_APPLICATION),   // big icon
        .hIconSm = LoadIcon(nullptr, IDI_APPLICATION), // small icon
        .hCursor = LoadCursor(nullptr, IDC_ARROW),
    };
    if (!RegisterClassEx(&wc)) {
        MessageBox(nullptr, "Window Registration Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return -1;
    }

    DWORD dwExStyle = WS_EX_CLIENTEDGE;
    DWORD dwStyle = WS_OVERLAPPEDWINDOW;

    Rect WindowRect = {0, 0, WindowWidth, WindowHeight};
    AdjustWindowRectEx(&WindowRect, dwStyle, false, dwExStyle);
    
    HWND hwnd = CreateWindowEx(
        dwExStyle,
        WindowClassName,
        "Mnist",
        dwStyle,
        CW_USEDEFAULT, CW_USEDEFAULT,
        RectWidth(WindowRect), RectHeight(WindowRect),
        nullptr, nullptr, hInstance, nullptr
    );
    if (!hwnd) {
        MessageBox(nullptr, "Window Creation Failed!", "Error!", MB_ICONEXCLAMATION | MB_OK);
        return -1;
    }
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    UpdateThread = CreateThread(nullptr, 0, UpdateLoop, nullptr, 0, nullptr);

    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    SuspendThread(UpdateThread);
    CloseHandle(UpdateThread);
    UpdateThread = nullptr;
    ModelDeinit();

    return msg.wParam;
}