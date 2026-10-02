#include <Arduino.h>
#include <math.h>

#include "esp_camera.h"
#include "img_converters.h"

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>

#include "esp_heap_caps.h"
#include "esp32-hal-cpu.h"
#include "esp_task_wdt.h"

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "board_config.h"


// ---------------- SETTINGS ----------------

const char *ssid = "Redmi 13 5G";
const char *password = "Sanket2007";

const char *modelURL = "http://IP_address:8000/best_int8.tflite";       // ip address of the device
const char *testURL  = "http://IP_address:8000/test_input.bin";

// Set to true to run the one-time PC-file test at boot
constexpr bool RUN_FILE_TEST = false;

constexpr int kTensorArenaSize = 1600 * 1024;

constexpr int CAMERA_WIDTH  = 320;
constexpr int CAMERA_HEIGHT = 240;
constexpr int MODEL_WIDTH   = 160;
constexpr int MODEL_HEIGHT  = 160;

constexpr int NUM_PREDICTIONS = 525;

// Displayed confidence = raw model confidence x CONFIDENCE_SCALE
constexpr float CONFIDENCE_SCALE = 10.0f;

// "Analog watch detected" when the DISPLAYED confidence is >= this value
constexpr float DETECTION_THRESHOLD = 0.7f;

// Center crop 240x240 taken from the 320x240 frame
constexpr float CROP_X = 40.0f;
constexpr float CROP_SIZE = 240.0f;


// ---------------- GLOBALS ----------------

uint8_t *modelData = nullptr;
size_t modelSize = 0;
uint8_t *tensorArena = nullptr;

const tflite::Model *tfliteModel = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

WebServer server(80);

volatile bool watchDetected = false;
volatile float bestConfidence = 0.0f;
volatile int boxX1 = 0, boxY1 = 0, boxX2 = 0, boxY2 = 0;

volatile unsigned long frameCapturedMs = 0;   // when the analysed frame was taken
volatile unsigned long resultFrameMs = 0;     // frame time of the published result
volatile bool haveResult = false;

SemaphoreHandle_t stateMutex = nullptr;
SemaphoreHandle_t cameraMutex = nullptr;


// ---------------- PROTOTYPES ----------------

bool downloadModel();
bool initializeTFLite();
static bool preprocessFrame(camera_fb_t *fb);
void performDetection();
void inferenceTask(void *parameter);
void handleRoot();
void handleCapture();
void handleStatus();


static void halt(const char *msg)
{
    for (;;)
    {
        Serial.println(msg);
        delay(2000);
    }
}


// ---------------- DOWNLOAD MODEL ----------------

bool downloadModel()
{
    Serial.println();
    Serial.println("Downloading TensorFlow Lite model...");

    HTTPClient http;

    if (!http.begin(modelURL))
    {
        Serial.println("ERROR: HTTP begin failed");
        return false;
    }

    http.setTimeout(15000);

    int httpCode = http.GET();

    if (httpCode != HTTP_CODE_OK)
    {
        Serial.printf("Model download failed. HTTP code: %d\n", httpCode);
        http.end();
        return false;
    }

    int len = http.getSize();

    if (len <= 0)
    {
        Serial.println("ERROR: Invalid model size");
        http.end();
        return false;
    }

    modelSize = (size_t)len;

    Serial.printf("Model size: %u bytes\n", (unsigned)modelSize);

    modelData = (uint8_t *)heap_caps_malloc(
        modelSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!modelData)
    {
        Serial.println("ERROR: Could not allocate model in PSRAM");
        http.end();
        return false;
    }

    WiFiClient *stream = http.getStreamPtr();

    size_t downloaded = 0;
    unsigned long t0 = millis();

    while (downloaded < modelSize && millis() - t0 < 60000)
    {
        int avail = stream->available();

        if (avail > 0)
        {
            size_t toRead = min((size_t)avail, modelSize - downloaded);
            size_t got = stream->readBytes(modelData + downloaded, toRead);

            if (got == 0)
                break;

            downloaded += got;
        }

        delay(1);
    }

    http.end();

    if (downloaded != modelSize)
    {
        Serial.printf("ERROR: Downloaded %u of %u bytes\n",
                      (unsigned)downloaded, (unsigned)modelSize);
        return false;
    }

    Serial.println("Model downloaded successfully!");
    Serial.printf("Free PSRAM after model: %u bytes\n", ESP.getFreePsram());

    return true;
}


// ---------------- TFLITE INIT ----------------

bool initializeTFLite()
{
    Serial.println();
    Serial.println("Initializing TensorFlow Lite...");

    tfliteModel = tflite::GetModel(modelData);

    if (tfliteModel->version() != TFLITE_SCHEMA_VERSION)
    {
        Serial.printf("ERROR: Model schema %d, supported %d\n",
                      (int)tfliteModel->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    tensorArena = (uint8_t *)heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

    if (!tensorArena)
    {
        Serial.println("ERROR: Tensor arena allocation failed");
        return false;
    }

    Serial.printf("Tensor arena allocated: %u bytes\n",
                  (unsigned)kTensorArenaSize);

    static tflite::MicroMutableOpResolver<20> resolver;

    resolver.AddAdd();
    resolver.AddConcatenation();
    resolver.AddConv2D();
    resolver.AddDequantize();
    resolver.AddLogistic();
    resolver.AddMaxPool2D();
    resolver.AddMul();
    resolver.AddPad();
    resolver.AddQuantize();
    resolver.AddReshape();
    resolver.AddResizeNearestNeighbor();
    resolver.AddSlice();
    resolver.AddSoftmax();
    resolver.AddSub();
    resolver.AddTranspose();

    static tflite::MicroInterpreter staticInterpreter(
        tfliteModel, resolver, tensorArena, kTensorArenaSize);

    interpreter = &staticInterpreter;

    Serial.println("Allocating TFLite tensors...");

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("ERROR: AllocateTensors() failed");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (!input || !output)
    {
        Serial.println("ERROR: Input/output tensor unavailable");
        return false;
    }

    Serial.printf("Input  type=%d bytes=%u\n", input->type, (unsigned)input->bytes);
    Serial.printf("Output type=%d bytes=%u dims=%d,%d,%d\n",
                  output->type, (unsigned)output->bytes,
                  output->dims->data[0], output->dims->data[1],
                  output->dims->data[2]);

    if (input->type != kTfLiteFloat32 ||
        output->type != kTfLiteFloat32 ||
        input->bytes != (size_t)(3 * MODEL_WIDTH * MODEL_HEIGHT * sizeof(float)) ||
        output->dims->data[1] != 5 ||
        output->dims->data[2] != NUM_PREDICTIONS)
    {
        Serial.println("ERROR: Unexpected tensor type/shape");
        return false;
    }

    Serial.println("TFLite initialization complete!");

    return true;
}


// ---------------- PREPROCESS (RGB565 -> float NCHW) ----------------

static inline void getRGB565(const uint8_t *buf, int x, int y,
                             float &r, float &g, float &b)
{
    const uint8_t *p = buf + (y * CAMERA_WIDTH + x) * 2;
    uint8_t hb = p[0];
    uint8_t lb = p[1];

    r = (float)(hb & 0xF8);
    g = (float)(((hb & 0x07) << 5) | ((lb & 0xE0) >> 3));
    b = (float)((lb & 0x1F) << 3);
}


static bool preprocessFrame(camera_fb_t *fb)
{
    if (!fb || !input)
        return false;

    if (fb->format != PIXFORMAT_RGB565 ||
        fb->width != CAMERA_WIDTH ||
        fb->height != CAMERA_HEIGHT ||
        fb->len < (size_t)(CAMERA_WIDTH * CAMERA_HEIGHT * 2))
    {
        Serial.println("ERROR: Unexpected frame format/size");
        return false;
    }

    const uint8_t *buf = fb->buf;
    const int plane = MODEL_WIDTH * MODEL_HEIGHT;
    const float scale = CROP_SIZE / MODEL_WIDTH;   // 1.5

    for (int y = 0; y < MODEL_HEIGHT; y++)
    {
        float sy = (y + 0.5f) * scale - 0.5f;
        int y0 = (int)floorf(sy);
        float fy = sy - y0;
        int y1 = y0 + 1;

        if (y0 < 0) { y0 = 0; fy = 0.0f; }
        if (y1 >= CAMERA_HEIGHT) y1 = CAMERA_HEIGHT - 1;

        for (int x = 0; x < MODEL_WIDTH; x++)
        {
            float sx = CROP_X + (x + 0.5f) * scale - 0.5f;
            int x0 = (int)floorf(sx);
            float fx = sx - x0;
            int x1 = x0 + 1;

            if (x1 >= CAMERA_WIDTH) x1 = CAMERA_WIDTH - 1;

            float r00, g00, b00, r01, g01, b01;
            float r10, g10, b10, r11, g11, b11;

            getRGB565(buf, x0, y0, r00, g00, b00);
            getRGB565(buf, x1, y0, r01, g01, b01);
            getRGB565(buf, x0, y1, r10, g10, b10);
            getRGB565(buf, x1, y1, r11, g11, b11);

            float w00 = (1.0f - fx) * (1.0f - fy);
            float w01 = fx * (1.0f - fy);
            float w10 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            int idx = y * MODEL_WIDTH + x;

            input->data.f[idx] =
                (r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11) / 255.0f;
            input->data.f[plane + idx] =
                (g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11) / 255.0f;
            input->data.f[2 * plane + idx] =
                (b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11) / 255.0f;
        }
    }

    return true;
}


// ---------------- DETECTION ----------------

void performDetection()
{
    if (!output)
        return;

    // Layout [1,5,525]: rows are x, y, w, h, confidence
    float best = 0.0f;
    int bestIndex = -1;

    for (int i = 0; i < NUM_PREDICTIONS; i++)
    {
        float conf = output->data.f[4 * NUM_PREDICTIONS + i];

        if (conf > best)
        {
            best = conf;
            bestIndex = i;
        }
    }

    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;

    if (bestIndex >= 0)
    {
        x = output->data.f[bestIndex];
        y = output->data.f[NUM_PREDICTIONS + bestIndex];
        w = output->data.f[2 * NUM_PREDICTIONS + bestIndex];
        h = output->data.f[3 * NUM_PREDICTIONS + bestIndex];
    }

    // Model box (0..1 of the 240x240 crop) -> camera pixels
    int x1 = (int)(CROP_X + (x - w * 0.5f) * CROP_SIZE);
    int y1 = (int)((y - h * 0.5f) * CROP_SIZE);
    int x2 = (int)(CROP_X + (x + w * 0.5f) * CROP_SIZE);
    int y2 = (int)((y + h * 0.5f) * CROP_SIZE);

    x1 = constrain(x1, 0, CAMERA_WIDTH - 1);
    y1 = constrain(y1, 0, CAMERA_HEIGHT - 1);
    x2 = constrain(x2, 0, CAMERA_WIDTH - 1);
    y2 = constrain(y2, 0, CAMERA_HEIGHT - 1);

    float shown = best * CONFIDENCE_SCALE;

    bool valid =
        (bestIndex >= 0) &&
        (shown >= DETECTION_THRESHOLD) &&
        (x2 - x1 > 4) &&
        (y2 - y1 > 4);

    xSemaphoreTake(stateMutex, portMAX_DELAY);

    bestConfidence = shown;
    watchDetected = valid;
    boxX1 = valid ? x1 : 0;
    boxY1 = valid ? y1 : 0;
    boxX2 = valid ? x2 : 0;
    boxY2 = valid ? y2 : 0;
    resultFrameMs = frameCapturedMs;
    haveResult = true;

    xSemaphoreGive(stateMutex);

    Serial.printf("RESULT idx=%d conf=%.4f x=%.3f y=%.3f w=%.3f h=%.3f box=(%d,%d)-(%d,%d) %s\n",
                  bestIndex, shown, x, y, w, h, x1, y1, x2, y2,
                  valid ? "DETECTED" : "no watch");
}


// ---------------- OPTIONAL: PC FILE TEST ----------------

static void runFileTest()
{
    HTTPClient http;

    if (!http.begin(testURL))
    {
        Serial.println("FILETEST: begin failed");
        return;
    }

    int code = http.GET();

    if (code != HTTP_CODE_OK)
    {
        Serial.printf("FILETEST: download failed %d\n", code);
        http.end();
        return;
    }

    int len = http.getSize();

    if (len != (int)input->bytes)
    {
        Serial.printf("FILETEST: size mismatch file=%d input=%u\n",
                      len, (unsigned)input->bytes);
        http.end();
        return;
    }

    WiFiClient *stream = http.getStreamPtr();
    uint8_t *dst = (uint8_t *)input->data.f;
    int got = 0;
    unsigned long t0 = millis();

    while (got < len && millis() - t0 < 30000)
    {
        int a = stream->available();

        if (a > 0)
            got += stream->readBytes(dst + got, min(a, len - got));

        delay(1);
    }

    http.end();

    Serial.printf("FILETEST: loaded %d of %d bytes\n", got, len);

    if (got != len)
        return;

    TfLiteStatus st = interpreter->Invoke();

    float best = 0.0f;
    int bi = -1;

    for (int i = 0; i < NUM_PREDICTIONS; i++)
    {
        float c = output->data.f[4 * NUM_PREDICTIONS + i];

        if (c > best)
        {
            best = c;
            bi = i;
        }
    }

    Serial.printf("FILETEST status=%d\n", (int)st);

    if (bi >= 0)
        Serial.printf("ESP RESULT idx=%d conf=%.4f x=%.3f y=%.3f w=%.3f h=%.3f\n",
                      bi, best,
                      output->data.f[bi],
                      output->data.f[NUM_PREDICTIONS + bi],
                      output->data.f[2 * NUM_PREDICTIONS + bi],
                      output->data.f[3 * NUM_PREDICTIONS + bi]);
    else
        Serial.println("ESP RESULT idx=-1 conf=0.0000");
}


// ---------------- WEB ----------------

const char PAGE[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ESP32-CAM Analog Watch Detector</title>
<style>
body{font-family:Arial;text-align:center;background:#111;color:#eee;margin:0;padding:20px}
canvas{width:95%;max-width:700px;border:2px solid #555;background:#222}
#status{font-size:20px;margin:12px}
</style>
</head>
<body>
<h2>ESP32-CAM Analog Watch Detector</h2>
<div id="status">Starting...</div>
<canvas id="view" width="320" height="240"></canvas>
<script>
var ctx = document.getElementById('view').getContext('2d');
var box = {d:0, c:0, a:-1, x1:0, y1:0, x2:0, y2:0};

function frame()
{
    var im = new Image();

    im.onload = function()
    {
        ctx.drawImage(im, 0, 0, 320, 240);

        if (box.d)
        {
            ctx.strokeStyle = '#ff354a';
            ctx.lineWidth = 3;
            ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        }

        setTimeout(frame, 100);
    };

    im.onerror = function() { setTimeout(frame, 1000); };

    im.src = '/capture?t=' + Date.now();
}

function status()
{
    fetch('/status', {cache: 'no-store'})
    .then(function(r) { return r.json(); })
    .then(function(j)
    {
        box = j;

        var age = (j.a < 0) ? 'waiting for first result' :
                  ('result age: ' + Math.round(j.a / 1000) + ' s');

        document.getElementById('status').textContent =
            (j.d ? 'Analog watch is weared detected' : 'No one wearing watch') +
            ' | confidence: ' + j.c.toFixed(3) +
            ' | ' + age;
    })
    .catch(function(e) {})
    .then(function() { setTimeout(status, 1000); });
}

frame();
status();
</script>
</body>
</html>
)HTML";


void handleRoot()
{
    server.send_P(200, "text/html", PAGE);
}


void handleCapture()
{
    uint8_t *jpg = nullptr;
    size_t len = 0;
    bool ok = false;

    xSemaphoreTake(cameraMutex, portMAX_DELAY);

    camera_fb_t *fb = esp_camera_fb_get();

    if (fb)
    {
        ok = frame2jpg(fb, 80, &jpg, &len);
        esp_camera_fb_return(fb);
    }

    xSemaphoreGive(cameraMutex);

    if (!ok || !jpg)
    {
        server.send(503, "text/plain", "Capture failed");
        return;
    }

    server.sendHeader("Cache-Control", "no-store");
    server.send_P(200, "image/jpeg", (const char *)jpg, len);

    free(jpg);
}


void handleStatus()
{
    bool d, have;
    float c;
    int x1, y1, x2, y2;
    unsigned long frameMs;

    xSemaphoreTake(stateMutex, portMAX_DELAY);

    d = watchDetected;
    c = bestConfidence;
    x1 = boxX1;
    y1 = boxY1;
    x2 = boxX2;
    y2 = boxY2;
    frameMs = resultFrameMs;
    have = haveResult;

    xSemaphoreGive(stateMutex);

    long age = have ? (long)(millis() - frameMs) : -1;

    char msg[200];

    snprintf(msg, sizeof(msg),
             "{\"d\":%d,\"c\":%.4f,\"a\":%ld,\"x1\":%d,\"y1\":%d,\"x2\":%d,\"y2\":%d}",
             d ? 1 : 0, c, age, x1, y1, x2, y2);

    server.send(200, "application/json", msg);
}


// ---------------- INFERENCE TASK ----------------

void inferenceTask(void *parameter)
{
    (void)parameter;

    for (;;)
    {
        bool ok = false;

        xSemaphoreTake(cameraMutex, portMAX_DELAY);

        // Discard the buffered (possibly old) frame, then take a fresh one
        camera_fb_t *old = esp_camera_fb_get();

        if (old)
            esp_camera_fb_return(old);

        camera_fb_t *fb = esp_camera_fb_get();

        if (fb)
        {
            frameCapturedMs = millis();
            ok = preprocessFrame(fb);
            esp_camera_fb_return(fb);
        }
        else
        {
            Serial.println("Inference camera capture failed");
        }

        xSemaphoreGive(cameraMutex);

        if (ok)
        {
            unsigned long start = millis();

            TfLiteStatus status = interpreter->Invoke();

            Serial.printf("Inference time: %lu ms, status=%d\n",
                          millis() - start, (int)status);

            if (status == kTfLiteOk)
                performDetection();
        }

        vTaskDelay(pdMS_TO_TICKS(200));
    }
}


// ---------------- SETUP ----------------

void setup()
{
    Serial.begin(115200);
    Serial.setDebugOutput(true);
    delay(1000);

    Serial.println();
    Serial.println("================================");
    Serial.println("ESP32-CAM ANALOG WATCH DETECTOR");
    Serial.println("================================");

    setCpuFrequencyMhz(240);

    Serial.printf("Total PSRAM: %u bytes\n", ESP.getPsramSize());
    Serial.printf("Free PSRAM: %u bytes\n", ESP.getFreePsram());

    stateMutex = xSemaphoreCreateMutex();
    cameraMutex = xSemaphoreCreateMutex();

    if (!stateMutex || !cameraMutex)
        halt("ERROR: Mutex creation failed");

    // ---- Camera ----

    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;
    config.frame_size = FRAMESIZE_QVGA;
    config.pixel_format = PIXFORMAT_RGB565;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK)
    {
        Serial.printf("Camera init failed: 0x%x\n", err);
        halt("CAMERA INIT FAILED");
    }

    Serial.println("Camera initialized successfully!");

    // ---- WiFi ----

    WiFi.begin(ssid, password);
    WiFi.setSleep(false);

    Serial.print("WiFi connecting");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());

    // ---- Model ----

    if (!downloadModel())
        halt("MODEL DOWNLOAD FAILED!");

    if (!initializeTFLite())
        halt("TFLITE INITIALIZATION FAILED!");

    if (RUN_FILE_TEST)
        runFileTest();

    // ---- Web server ----

    server.on("/", HTTP_GET, handleRoot);
    server.on("/capture", HTTP_GET, handleCapture);
    server.on("/status", HTTP_GET, handleStatus);
    server.begin();

    Serial.println("Web server started.");

    // ---- Watchdog + inference task ----

    esp_err_t wdtResult = esp_task_wdt_deinit();
    Serial.printf("Task watchdog deinit result: %d\n", wdtResult);

    xTaskCreatePinnedToCore(inferenceTask, "InferenceTask",
                            8192, nullptr, 1, nullptr, 0);

    Serial.println();
    Serial.println("================================");
    Serial.print("OPEN THIS IN YOUR BROWSER:\nhttp://");
    Serial.println(WiFi.localIP());
    Serial.println("================================");
}


void loop()
{
    server.handleClient();
    delay(2);
}