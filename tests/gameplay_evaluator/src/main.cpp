#include <iostream>
#include <cstdlib>
#include <cstring>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>

extern "C"
{
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

#include "utec/nn/neural_network.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_dense.h"

#include <opencv2/opencv.hpp>

#if defined(USE_X11)
#include <X11/Xlib.h>
#include <X11/extensions/XTest.h>
#include <X11/keysym.h>
#endif

#if defined(USE_WIN32)
#include <windows.h>
#endif

using namespace utec::neural_network;

#if defined(USE_X11)
std::array<int, 9> keySyms = {XK_C, XK_F, XK_V, XK_G, XK_B, XK_H, XK_N, XK_J, XK_M};
#elif defined(USE_WIN32)
std::array<WORD, 9> keySyms = {
    'C', 'F', 'V', 'G', 'B', 'H', 'N', 'J', 'M'};
#else
std::array<int, 9> keySyms = {1, 2, 3, 4, 5, 6, 7, 8, 9};
#endif
std::array<int, 9> elements = {1, 2, 3, 4, 5, 6, 7, 8, 9};
std::array<std::string, 512> combinations;
#if defined(USE_X11)
using KeyType = KeySym;
#elif defined(USE_WIN32)
using KeyType = WORD;
#else
using KeyType = int;
#endif
std::unordered_map<std::string, std::unordered_set<KeyType>> keyCombos;

int res_index = 0;
int lastcombo = 511;

void print_usage(const char *prog)
{
    std::cout << "Usage:\n"
              << "  " << prog << "              # X11capture (Linux)\n"
              << "  " << prog << " --kms [drm_device] [plane_id]  # KMS-DRM capture\n"
              << "  " << prog << " (Windows/macOS)             # auto}\n";
}

void generate_combos(const std::array<int, 9> &elms, int r, int st, std::vector<int> &cur)
{
    if (cur.size() == r)
    {
        std::string number_str;
        std::unordered_set<KeyType> keys;

        for (int num : cur)
        {
            number_str += std::to_string(num);
            keys.insert(keySyms[num - 1]); // Convert 1-based to 0-based index
        }

        combinations[res_index++] = number_str;
        keyCombos[number_str] = std::move(keys);
        return;
    }

    for (int i = st; i < elms.size(); ++i)
    {
        cur.push_back(elms[i]);
        generate_combos(elms, r, i + 1, cur);
        cur.pop_back();
    }
}

#if defined(USE_WIN32)
void sendWinKey(WORD vk, bool down)
{
    INPUT inp = {};
    inp.type = INPUT_KEYBOARD;
    inp.ki.wVk = vk;
    inp.ki.dwFlags = down ? 0 : KEYEVENTF_KEYUP;
    UINT sent = SendInput(1, &inp, sizeof(INPUT));
    if (sent == 0)
        std::cout << "SendInput failed: " << GetLastError() << std::endl;
}
#endif

template <typename T>
void evaluatePress(Tensor<T, 2> &image, NeuralNetwork<T> &net)
{
    auto final_pred = net.predict(image);
    size_t pred = 0;
    auto maxv = final_pred(0, 0);

    {
#pragma omp parallel for
        for (size_t j = 1; j < final_pred.shape()[1]; ++j)
        {
            if (final_pred(0, j) > maxv)
            {
                maxv = final_pred(0, j);
                pred = j;
            }
        }
    }

    if (lastcombo != pred)
    {
        if (combinations[lastcombo] != "0" || combinations[pred] == combinations[lastcombo])
        {
#if defined(USE_X11)
            for (KeySym key : keyCombos[combinations[lastcombo]])
            {
                KeyCode keycode = XKeysymToKeycode(display, key);
                XTestFakeKeyEvent(display, keycode, False, 0);
            }
            XFlush(display);
#elif defined(USE_WIN32)
            if (combinations[lastcombo] != "0" || combinations[pred] == combinations[lastcombo])
            {
                for (WORD vk : keyCombos[combinations[lastcombo]])
                    sendWinKey(vk, false);
            }
#else
            std::cout << "Not implemented on this platform yet." << std::endl;
#endif
        }
        if (combinations[pred] != "0" && combinations[pred] != combinations[lastcombo])
        {
#if defined(USE_X11)
            for (KeySym key : keyCombos[combinations[pred]])
            {
                KeyCode keycode = XKeysymToKeycode(display, key);
                XTestFakeKeyEvent(display, keycode, True, 0);
            }
            XFlush(display);
#elif defined(USE_WIN32)
            if (combinations[pred] != "0" && combinations[pred] != combinations[lastcombo])
            {
                for (WORD vk : keyCombos[combinations[pred]])
                    sendWinKey(vk, true);
            }
#else
            std::cout << "Not implemented on this platform yet." << std::endl;
#endif
        }
    }
    lastcombo = pred;

    std::cout << "Predicted: " << combinations[pred] << std::endl;
}

int main(int argc, char *argv[])
{
    std::string fmtName, devName;
    AVDictionary *options = nullptr;

    avdevice_register_all();

#if defined(_WIN32)
    fmtName = "gdigrab";
    devName = "desktop";
#elif defined(__APPLE__)
    fmtName = "avfoundation";
    devName = "Capture screen 0";
#elif defined(__linux__)
    bool useKMS = (argc > 1 && std::string(argv[1]) == "--kms");
    if (useKMS)
    {
        fmtName = "kmsgrab";
        devName = (argc > 2 ? argv[2] : "/dev/dri/card0");
        if (argc > 3)
        {
            av_dict_set(&options, "plane_id", argv[3], 0);
        }
        // Permissions: CAP_SYS_ADMIN or root
        // e.g. sudo setcap cap_sys_admin+ep /path/to/your_app
    }
    else
    {
        fmtName = "x11grab";
        devName = ":0.0+0,0";
    }
#else
    std::cerr << "Unsupported platform.\n";
    print_usage(argv[0]);
    return -1;
#endif

    // Common options
    av_dict_set(&options, "framerate", "60", 0);
    av_dict_set(&options, "video_size", "1920x1080", 0);

    const AVInputFormat *inFmt = av_find_input_format(fmtName.c_str());
    if (!inFmt)
    {
        std::cerr << "Input format not found: " << fmtName << "\n";
        return -1;
    }

    AVFormatContext *fmtCtx = nullptr;
    if (avformat_open_input(&fmtCtx, devName.c_str(), inFmt, &options) < 0)
    {
        std::cerr << "Could not open device: " << devName << "\n";
        if (fmtName == "kmsgrab")
            std::cerr << "Ensure you have CAP_SYS_ADMIN or run as root, and a valid plane_id.\n";
        return -1;
    }

    if (avformat_find_stream_info(fmtCtx, nullptr) < 0)
    {
        std::cerr << "Could not get stream info.\n";
        avformat_close_input(&fmtCtx);
        return -1;
    }

    int vidIdx = -1;
    for (unsigned i = 0; i < fmtCtx->nb_streams; ++i)
    {
        if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            vidIdx = i;
            break;
        }
    }
    if (vidIdx < 0)
    {
        std::cerr << "No video stream found.\n";
        avformat_close_input(&fmtCtx);
        return -1;
    }

    AVCodecParameters *cpar = fmtCtx->streams[vidIdx]->codecpar;
    const AVCodec *codec = avcodec_find_decoder(cpar->codec_id);
    AVCodecContext *cctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(cctx, cpar);
    avcodec_open2(cctx, codec, nullptr);

    SwsContext *swsCtx = sws_getContext(
        cctx->width, cctx->height, cctx->pix_fmt,
        cctx->width, cctx->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frm = av_frame_alloc();
    AVFrame *bgr = av_frame_alloc();
    int bufSize = av_image_get_buffer_size(AV_PIX_FMT_BGR24, cctx->width, cctx->height, 1);
    uint8_t *buf = (uint8_t *)av_malloc(bufSize);
    av_image_fill_arrays(bgr->data, bgr->linesize, buf, AV_PIX_FMT_BGR24, cctx->width, cctx->height, 1);

    Tensor<float, 2> imageData(1, 154 * 13);

    std::vector<int> current;
    for (int r = 1; r <= 9; ++r)
    {
        generate_combos(elements, r, 0, current);
    }

#if defined(USE_X11)
    display = XOpenDisplay(nullptr);
    if (!display)
    {
        std::cerr << "Cannot open display\n";
        return 1;
    }
#endif

    combinations[511] = "0";
    combinations[lastcombo] = "0";
    keyCombos["0"] = {0};

    auto init_w = [&](auto &W)
    {
        std::mt19937 gen(42);
        float fan_in = W.shape()[1];
        float fan_out = W.shape()[0];
        float scale = std::sqrt(2.0f / (fan_in + fan_out));
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto &v : W)
            v = dist(gen);
    };

    auto init_b = [](auto &B)
    {
        for (auto &val : B)
            val = 0.0f;
    };

    NeuralNetwork<float> net;

    net.add_layer(std::make_unique<Dense<float>>(154 * 13, 128, init_w, init_b));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(128, 64, init_w, init_b));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(64, 512, init_w, init_b));
    net.add_layer(std::make_unique<Softmax<float>>());

    std::string custom_path = "";
    do
    {
        std::cout << "Input full path: " << std::endl;
        std::getline(std::cin, custom_path);
        if (std::filesystem::is_directory(custom_path) || !std::filesystem::exists(custom_path))
        {
            std::cout << "Invalid path. Try again." << std::endl;
        }
    } while (custom_path.empty());

    std::cout << "Reading model from file... ";
    net.load(custom_path);
    std::cout << "Done." << std::endl;

    bool printRaw = false;
    bool printFrame = false;
    bool notPopn10 = false;

    bool judgingPos = false;
    bool customPos = false;
    bool judgingX = false;
    int customX = -1, customY = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (judgingPos == true) {
            int data = stoi(arg);
            if (judgingX) {
                customX = data;
                judgingX = false;
            } else {
                customY = data;
                judgingPos = false;
            }
        }

        if (arg == "--printraw") printRaw = true;
        else if (arg == "--printframe") printFrame = true;
        else if (arg == "--not10") notPopn10 = true;
        else if (arg == "--custompos") {
            customPos = true;
            judgingPos = true;
            judgingX = true;
        }
    }

    if (customX == -1 || customY == -1)
    {
        if (customX == -1) customX = 651;
        if (customY == -1) customY = 736;
    }

    // Now you can use the flags
    if (printRaw) std::cout << "Raw frame printing enabled" << std::endl;
    if (printFrame) std::cout << "Modified frame printing enabled" << std::endl;
    if (notPopn10) std::cout << "Marked as not Pop'n Music 10" << std::endl;
    if (customPos) std::cout << "Custom position: " << customX << " " << customY << std::endl;

    std::cout << "Starting capture (" << fmtName << ") on device " << devName << "... Press ESC to quit.\n";
    while (av_read_frame(fmtCtx, pkt) >= 0)
    {
        if (pkt->stream_index == vidIdx)
        {
            if (avcodec_send_packet(cctx, pkt) == 0)
            {
                while (avcodec_receive_frame(cctx, frm) == 0)
                {
                    sws_scale(swsCtx, frm->data, frm->linesize, 0, cctx->height, bgr->data, bgr->linesize);
                    cv::Mat mat(cctx->height, cctx->width, CV_8UC3, bgr->data[0], bgr->linesize[0]);
                    cv::Mat gray, res;
                    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
                    cv::Rect roi(customX, customY, 616, 52);
                    if (notPopn10) roi.y -= 38;
                    cv::Mat crop = gray(roi);
                    cv::resize(crop, res, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
                    {
                        #pragma omp parallel for
                        for (size_t i = 0; i < 154 * 13; i++)
                            imageData(0, i) = res.data[i];
                    }
                    if (printFrame) cv::imwrite("./frame.png", res);
                    if (printRaw) cv::imwrite("./frame_raw.png", mat(roi));
                    evaluatePress(imageData, net);
                }
            }
        }
        av_packet_unref(pkt);
    }

end:
    av_free(buf);
    av_frame_free(&frm);
    av_frame_free(&bgr);
    av_packet_free(&pkt);
    sws_freeContext(swsCtx);
    avcodec_free_context(&cctx);
    avformat_close_input(&fmtCtx);
#if defined(USE_X11)
    XFlush(display);
    XCloseDisplay(display);
#endif
    return 0;
}
