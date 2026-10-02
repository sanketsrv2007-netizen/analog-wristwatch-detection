#include <stdint.h>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_camera.h"

#include "img_converters.h"
#include "esp_heap_caps.h"

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

static const char *TAG = "WATCH";

// ============================================================
// YOLO detection structures
// ============================================================

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
};

// ============================================================
// Sigmoid
// ============================================================

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================
// Dequantize INT8 value
//
// Real value = quantized value * 2^exponent
// ============================================================

static inline float dequantize_int8(int8_t value, int exponent)
{
    return (float)value * powf(2.0f, (float)exponent);
}

// ============================================================
// DFL decoding
//
// 16 bins are used for each side of the bounding box.
// The 16 values are converted to probabilities using
// softmax, then the expected value is calculated.
// ============================================================

static float decode_dfl(
    const int8_t *data,
    int offset,
    int exponent)
{
    float values[16];

    float max_value = -1000000.0f;

    for (int i = 0; i < 16; i++)
    {
        values[i] =
            dequantize_int8(
                data[offset + i],
                exponent
            );

        if (values[i] > max_value)
        {
            max_value = values[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < 16; i++)
    {
        values[i] = expf(values[i] - max_value);
        sum += values[i];
    }

    float result = 0.0f;

    for (int i = 0; i < 16; i++)
    {
        float probability = values[i] / sum;
        result += probability * (float)i;
    }

    return result;
}

// ============================================================
// Decode one YOLO output scale
//
// box tensor:
//   [1, H, W, 64]
//   64 = 4 sides * 16 DFL bins
//
// score tensor:
//   [1, H, W, 1]
//
// stride:
//   8, 16 or 32
// ============================================================

static void decode_yolo_scale(
    dl::TensorBase *box_tensor,
    dl::TensorBase *score_tensor,
    int grid_h,
    int grid_w,
    int stride,
    std::vector<Detection> &detections)
{
    const int8_t *box_data =
        box_tensor->get_element_ptr<int8_t>();

    const int8_t *score_data =
        score_tensor->get_element_ptr<int8_t>();

    if (box_data == nullptr || score_data == nullptr)
    {
        ESP_LOGE(TAG, "Null YOLO output tensor!");
        return;
    }

    int box_exponent =
        box_tensor->get_exponent();

    int score_exponent =
        score_tensor->get_exponent();

    const float score_threshold = 0.25f;

    ESP_LOGI(
    TAG,
    "Score tensor: exponent=%d, first values=%d %d %d %d %d",
    score_exponent,
    (int)score_data[0],
    (int)score_data[1],
    (int)score_data[2],
    (int)score_data[3],
    (int)score_data[4]
);

    float max_confidence = 0.0f;

    for (int y = 0; y < grid_h; y++)
    {
        for (int x = 0; x < grid_w; x++)
        {
            int cell_index =
                y * grid_w + x;

            // --------------------------------------------
            // Decode confidence
            // --------------------------------------------

            float score_logit =
                dequantize_int8(
                    score_data[cell_index],
                    score_exponent
                );

            float confidence =
                sigmoid(score_logit);

            if (confidence > max_confidence)
            {
                max_confidence = confidence;
            }

            if (confidence < score_threshold)
            {
                continue;
            }

            // --------------------------------------------
            // Each cell contains:
            //
            // left   = 0..15
            // top    = 16..31
            // right  = 32..47
            // bottom = 48..63
            // --------------------------------------------

            int box_offset =
                cell_index * 64;

            float left =
                decode_dfl(
                    box_data,
                    box_offset + 0,
                    box_exponent
                );

            float top =
                decode_dfl(
                    box_data,
                    box_offset + 16,
                    box_exponent
                );

            float right =
                decode_dfl(
                    box_data,
                    box_offset + 32,
                    box_exponent
                );

            float bottom =
                decode_dfl(
                    box_data,
                    box_offset + 48,
                    box_exponent
                );

            // --------------------------------------------
            // Ultralytics anchor convention:
            //
            // anchor = cell + 0.5
            // --------------------------------------------

            float anchor_x =
                ((float)x + 0.5f);

            float anchor_y =
                ((float)y + 0.5f);

            // DFL distances are measured in grid units.
            // Convert them to input-image pixels.
            // --------------------------------------------

            float x1 =
                (anchor_x - left) * stride;

            float y1 =
                (anchor_y - top) * stride;

            float x2 =
                (anchor_x + right) * stride;

            float y2 =
                (anchor_y + bottom) * stride;

            // --------------------------------------------
            // Clamp to 320x320 model image
            // --------------------------------------------

            x1 = std::max(0.0f, std::min(320.0f, x1));
            y1 = std::max(0.0f, std::min(320.0f, y1));
            x2 = std::max(0.0f, std::min(320.0f, x2));
            y2 = std::max(0.0f, std::min(320.0f, y2));

            if (x2 <= x1 || y2 <= y1)
            {
                continue;
            }

            Detection detection;

            detection.x1 = x1;
            detection.y1 = y1;
            detection.x2 = x2;
            detection.y2 = y2;
            detection.confidence = confidence;

            detections.push_back(detection);
        }
    }

    ESP_LOGI(
    TAG,
    "Scale %dx%d stride %d: max confidence = %.4f",
    grid_h,
    grid_w,
    stride,
    max_confidence
);

}

// ============================================================
// AI-Thinker ESP32-CAM pins
// ============================================================

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ============================================================
// ESP-DL model
// ============================================================

extern const uint8_t model_start[] asm("_binary_best_espdl_start");

// ============================================================
// Camera initialization
// ============================================================

static esp_err_t init_camera()
{
    camera_config_t config = {};

    config.pin_pwdn       = PWDN_GPIO_NUM;
    config.pin_reset      = RESET_GPIO_NUM;
    config.pin_xclk       = XCLK_GPIO_NUM;

    config.pin_sccb_sda   = SIOD_GPIO_NUM;
    config.pin_sccb_scl   = SIOC_GPIO_NUM;

    config.pin_d7         = Y9_GPIO_NUM;
    config.pin_d6         = Y8_GPIO_NUM;
    config.pin_d5         = Y7_GPIO_NUM;
    config.pin_d4         = Y6_GPIO_NUM;
    config.pin_d3         = Y5_GPIO_NUM;
    config.pin_d2         = Y4_GPIO_NUM;
    config.pin_d1         = Y3_GPIO_NUM;
    config.pin_d0         = Y2_GPIO_NUM;

    config.pin_vsync      = VSYNC_GPIO_NUM;
    config.pin_href       = HREF_GPIO_NUM;
    config.pin_pclk       = PCLK_GPIO_NUM;

    config.xclk_freq_hz   = 20000000;
    config.ledc_timer     = LEDC_TIMER_0;
    config.ledc_channel   = LEDC_CHANNEL_0;

    config.pixel_format   = PIXFORMAT_RGB565;
    config.frame_size     = FRAMESIZE_QVGA;

    config.jpeg_quality   = 12;

    config.fb_count       = 1;
    config.fb_location    = CAMERA_FB_IN_PSRAM;
    config.grab_mode      = CAMERA_GRAB_WHEN_EMPTY;

    ESP_LOGI(TAG, "Initializing camera...");

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera initialization failed: 0x%x", err);
        return err;
    }

    ESP_LOGI(TAG, "Camera initialized successfully!");

    return ESP_OK;
}

// ============================================================
// Convert RGB565 pixel to RGB888
// ============================================================

static inline void rgb565_to_rgb888(
    uint16_t pixel,
    uint8_t &r,
    uint8_t &g,
    uint8_t &b)
{
    r = ((pixel >> 11) & 0x1F) << 3;
    g = ((pixel >> 5)  & 0x3F) << 2;
    b = (pixel & 0x1F) << 3;

    // Fill the lower bits for better approximation.
    r |= r >> 5;
    g |= g >> 6;
    b |= b >> 5;
}

// ============================================================
// Main
// ============================================================

extern "C" void app_main()
{
    ESP_LOGI(TAG, "Starting camera + ESP-DL inference test...");

    // --------------------------------------------------------
    // Camera
    // --------------------------------------------------------

    if (init_camera() != ESP_OK)
    {
        ESP_LOGE(TAG, "Camera setup failed!");
        return;
    }

    // --------------------------------------------------------
    // Load ESP-DL model
    // --------------------------------------------------------

    dl::Model *model = new dl::Model(
        (const char *)model_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );

    if (model == nullptr)
    {
        ESP_LOGE(TAG, "Failed to create ESP-DL model!");
        return;
    }

    ESP_LOGI(TAG, "ESP-DL model loaded successfully!");

    // --------------------------------------------------------
    // Get model input
    // --------------------------------------------------------

    dl::TensorBase *model_input = model->get_input();

    if (model_input == nullptr)
    {
        ESP_LOGE(TAG, "Could not get model input!");
        return;
    }

    const std::vector<int> &shape = model_input->get_shape();

    ESP_LOGI(TAG, "Model input shape:");

    for (int i = 0; i < (int)shape.size(); i++)
    {
        ESP_LOGI(TAG, "  shape[%d] = %d", i, shape[i]);
    }

    ESP_LOGI(
        TAG,
        "Model input dtype: %s",
        model_input->get_dtype_string()
    );

    ESP_LOGI(
        TAG,
        "Model input exponent: %d",
        model_input->get_exponent()
    );

    ESP_LOGI(
        TAG,
        "Model input bytes: %d",
        model_input->get_bytes()
    );


    // --------------------------------------------------------
// RGB888 conversion buffer
// --------------------------------------------------------

uint8_t *rgb888_buffer =
    (uint8_t *)heap_caps_malloc(
        320 * 240 * 3,
        MALLOC_CAP_SPIRAM
    );

if (rgb888_buffer == nullptr)
{
    ESP_LOGE(TAG, "Failed to allocate RGB888 buffer!");
    return;
}

ESP_LOGI(TAG, "RGB888 buffer allocated successfully!");



    // --------------------------------------------------------
    // Main loop
    // --------------------------------------------------------

    while (true)
    {
        camera_fb_t *fb = esp_camera_fb_get();

        if (fb == nullptr)
        {
            ESP_LOGE(TAG, "Camera capture failed!");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        ESP_LOGI(
            TAG,
            "Frame captured: %dx%d, bytes=%u",
            fb->width,
            fb->height,
            (unsigned int)fb->len
        );


        // ----------------------------------------------------
// Convert RGB565 camera frame to RGB888
// using ESP32-camera's official converter.
// ----------------------------------------------------

if (!fmt2rgb888(
        fb->buf,
        fb->len,
        PIXFORMAT_RGB565,
        rgb888_buffer))
{
    ESP_LOGE(TAG, "RGB565 -> RGB888 conversion failed!");
    esp_camera_fb_return(fb);
    continue;
}


        // ----------------------------------------------------
        // Verify that the model input is what we expect.
        // ----------------------------------------------------

        if (shape.size() != 4)
        {
            ESP_LOGE(TAG, "Unexpected model input dimensions!");
            esp_camera_fb_return(fb);
            break;
        }

        // ----------------------------------------------------
        // Fill model input.
        //
        // The ESP-DL importer uses NHWC for this model:
        //
        // [1, 320, 320, 3]
        //
        // Camera is:
        //
        // 240 x 240 RGB565
        //
        // Since both are square, this first test uses
        // nearest-neighbor resize.
        // ----------------------------------------------------

        int input_h = shape[1];
        int input_w = shape[2];
        int input_c = shape[3];

        if (input_h != 320 ||
            input_w != 320 ||
            input_c != 3)
        {
            ESP_LOGE(
                TAG,
                "Unexpected input layout: [%d,%d,%d,%d]",
                shape[0],
                shape[1],
                shape[2],
                shape[3]
            );

            esp_camera_fb_return(fb);
            break;
        }

        int8_t *input_data = model_input->get_element_ptr<int8_t>();

        if (input_data == nullptr)
        {
            ESP_LOGE(TAG, "Model input data pointer is null!");
            esp_camera_fb_return(fb);
            break;
        }

        const int exponent = model_input->get_exponent();

        const float scale = powf(2.0f, (float)exponent);

        
// --------------------------------------------------------
// Ultralytics-style letterbox preprocessing
// Camera: 320x240
// Model : 320x320
// --------------------------------------------------------

const int src_w = fb->width;
const int src_h = fb->height;

const float resize_scale = fminf(
    (float)input_w / (float)src_w,
    (float)input_h / (float)src_h
);

const int resized_w = (int)roundf(src_w * resize_scale);
const int resized_h = (int)roundf(src_h * resize_scale);

const int pad_x = (input_w - resized_w) / 2;
const int pad_y = (input_h - resized_h) / 2;

// YOLO/Ultralytics letterbox padding value
const uint8_t pad_value = 114;

// Model quantization
const float scale_q = powf(2.0f, (float)exponent);

for (int y = 0; y < input_h; y++)
{
    for (int x = 0; x < input_w; x++)
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;

        // Inside the resized camera image
        if (x >= pad_x &&
            x < pad_x + resized_w &&
            y >= pad_y &&
            y < pad_y + resized_h)
        {
            int resized_x = x - pad_x;
            int resized_y = y - pad_y;

            int src_x = (int)(resized_x / resize_scale);
            int src_y = (int)(resized_y / resize_scale);

            if (src_x >= src_w)
                src_x = src_w - 1;

            if (src_y >= src_h)
                src_y = src_h - 1;

            int src_index =
                (src_y * src_w + src_x) * 3;

            r = rgb888_buffer[src_index + 0];
            g = rgb888_buffer[src_index + 1];
            b = rgb888_buffer[src_index + 2];
        }
        else
        {
            // Letterbox padding
            r = pad_value;
            g = pad_value;
            b = pad_value;
        }

        // Normalize exactly like ToTensor()
        float rf = r / 255.0f;
        float gf = g / 255.0f;
        float bf = b / 255.0f;

        // Quantize to INT8
        int qr = (int)roundf(rf / scale_q);
        int qg = (int)roundf(gf / scale_q);
        int qb = (int)roundf(bf / scale_q);

        // INT8 clamp
        qr = std::max(-128, std::min(127, qr));
        qg = std::max(-128, std::min(127, qg));
        qb = std::max(-128, std::min(127, qb));

        int index =
            (y * input_w + x) * input_c;

        input_data[index + 0] = (int8_t)qr;
        input_data[index + 1] = (int8_t)qg;
        input_data[index + 2] = (int8_t)qb;
    }
}

// Copy the camera image into the letterboxed area.
for (int y = 0; y < resized_h; y++)
{
    int src_y =
        (int)((float)y / resize_scale);

    src_y =
        std::max(0, std::min(src_h - 1, src_y));

    for (int x = 0; x < resized_w; x++)
    {
        int src_x =
            (int)((float)x / resize_scale);

        src_x =
            std::max(0, std::min(src_w - 1, src_x));

        int src_index =
    (src_y * src_w + src_x) * 3;

uint8_t r = rgb888_buffer[src_index + 0];
uint8_t g = rgb888_buffer[src_index + 1];
uint8_t b = rgb888_buffer[src_index + 2];



        float rf = r / 255.0f;
        float gf = g / 255.0f;
        float bf = b / 255.0f;

        int dst_x = x + pad_x;
        int dst_y = y + pad_y;

        int index =
            (dst_y * input_w + dst_x) * 3;

        int qr = (int)roundf(rf / scale);
        int qg = (int)roundf(gf / scale);
        int qb = (int)roundf(bf / scale);

        qr = std::max(-128, std::min(127, qr));
        qg = std::max(-128, std::min(127, qg));
        qb = std::max(-128, std::min(127, qb));

        input_data[index + 0] = (int8_t)qr;
        input_data[index + 1] = (int8_t)qg;
        input_data[index + 2] = (int8_t)qb;
    }
}

        esp_camera_fb_return(fb);

        // ----------------------------------------------------
        // Run model
        // ----------------------------------------------------

        ESP_LOGI(TAG, "Running ESP-DL model...");


        model->run(dl::RUNTIME_MODE_MULTI_CORE);

        ESP_LOGI(TAG, "ESP-DL inference completed!");



        // ----------------------------------------------------
// YOLO DFL decoding
// ----------------------------------------------------

auto outputs = model->get_outputs();

std::vector<Detection> detections;

decode_yolo_scale(
    outputs.at("box0"),
    outputs.at("score0"),
    40,
    40,
    8,
    detections
);

decode_yolo_scale(
    outputs.at("box1"),
    outputs.at("score1"),
    20,
    20,
    16,
    detections
);

decode_yolo_scale(
    outputs.at("box2"),
    outputs.at("score2"),
    10,
    10,
    32,
    detections
);

ESP_LOGI(
    TAG,
    "YOLO candidates above threshold: %d",
    (int)detections.size()
);

for (int i = 0; i < (int)detections.size(); i++)
{
    ESP_LOGI(
    TAG,
    "Candidate %d: conf=%.3f box=(%.1f, %.1f, %.1f, %.1f)",
        i,
        detections[i].confidence,
        detections[i].x1,
        detections[i].y1,
        detections[i].x2,
        detections[i].y2
    );
}

    

        vTaskDelay(pdMS_TO_TICKS(1000));
    }


    
}