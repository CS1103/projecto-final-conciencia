#include <iostream>
#include <fstream>
#include <filesystem>

#include <string>
#include <vector>
#include <random>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

#define DIFFICULTY_LEVELS 9
#define IMAGE_COUNT_DIFFI 10

cv::Mat background, kun_red, kun_blue, kun_white, kun_yellow, kun_green;

std::vector<int> elements = {1,2,3,4,5,6,7,8,9};
std::vector<int32_t> combinations;
std::random_device rd;
std::mt19937 gen(rd());

int kun_positions[9] = {0, 39, 68, 107, 136, 175, 204, 243, 272};

std::vector<int> rand_sample(const std::vector<int> &pop, size_t k){
    std::vector<int> result = pop;
    std::shuffle(result.begin(), result.end(), gen);

    result.resize(k);
    return result;
}

std::vector<int> combo_fromstr(const std::string &str) {
    std::vector<int> res;
    for (const char &a : str) res.push_back(a - '0');
    return res;
}

void generate_combos(const std::vector<int> elms, int r, int st, std::vector<int> &cur, std::vector<int32_t> &res) {
    if (cur.size() == r) {
        std::string number_str;
        for (int num : cur) number_str += std::to_string(num);
        res.push_back(std::stoi(number_str));
        return;
    }

    for (int i = st; i < elements.size(); ++i) {
        cur.push_back(elements[i]);
        generate_combos(elements, r, i + 1, cur, res);
        cur.pop_back();
    }    
}

void png_overlay(const cv::Mat &src, cv::Mat &dst, int x, int y) {
    for (int yy = 0; yy < src.rows; ++yy) {
        int dy = yy + y;
        if (dy < 0 || dy >= dst.rows) continue;
        for (int xx = 0; xx < src.cols; ++xx) {
            int dx = xx + x;
            if (dx < 0 || dx >= dst.cols) continue;

            cv::Vec4b s = src.at<cv::Vec4b>(yy, xx);
            float alpha = s[3] / 255.0f;
            if (alpha <= 0) continue;

            cv::Vec4b &d = dst.at<cv::Vec4b>(dy, dx);
            for (int c = 0; c < 3; ++c) {
                d[c] = uchar(s[c] * alpha + d[c] * (1.0f - alpha));
            }
            d[3] = uchar((alpha + d[3]/255.0f * (1.0f - alpha)) * 255);
        }
    }
}

void generate_pos(cv::Mat &image, bool garb = false, const std::string &num = ""){
    std::vector<int> picks;
    std::uniform_int_distribution<> am_notes(1,5);
    if (garb) picks = rand_sample(elements, (size_t)am_notes(gen));
    else picks = combo_fromstr(num);
    for (int &i : picks){
        std::uniform_int_distribution<> am_positions(10, 15);
        int combo_x = kun_positions[i-1], combo_y = garb ? 13-am_positions(gen) : 13;
        switch (i) {
            case 1: case 9: png_overlay(kun_white,  image, combo_x, combo_y); break;
            case 4: case 6: png_overlay(kun_blue,   image, combo_x, combo_y); break;
            case 3: case 7: png_overlay(kun_green,  image, combo_x, combo_y); break;
            case 2: case 8: png_overlay(kun_yellow, image, combo_x, combo_y); break;
            case 5: png_overlay(kun_red,    image, combo_x, combo_y); break;
            default: break;
        }
    }
}

int main(int, char**){
    cv::Mat bg_bgr = cv::imread("./assets/empty_template_smaller.png", cv::IMREAD_COLOR);
    cv::cvtColor(bg_bgr, background, cv::COLOR_BGR2BGRA);
    kun_red = cv::imread("./assets/kuns/red.png", cv::IMREAD_UNCHANGED);
    kun_blue = cv::imread("./assets/kuns/blue.png", cv::IMREAD_UNCHANGED);
    kun_green = cv::imread("./assets/kuns/green.png", cv::IMREAD_UNCHANGED);
    kun_yellow = cv::imread("./assets/kuns/yellow.png", cv::IMREAD_UNCHANGED);
    kun_white = cv::imread("./assets/kuns/white.png", cv::IMREAD_UNCHANGED);

    for (int r = 1; r <= 9; ++r) {
        std::vector<int> current;
        generate_combos(elements, r, 0, current, combinations);
    }

    combinations.push_back(0);
    std::vector<cv::Mat> images, testimages;

    if (!std::filesystem::is_directory("../results") || !std::filesystem::exists("../results")) {
        std::filesystem::create_directory("../results"); // results folder
    }

    if (!std::filesystem::is_directory("../results/images") || !std::filesystem::exists("../results/images")) {
        std::filesystem::create_directory("../results/images"); // results folder
    }

    for (int32_t &c : combinations) {
        std::string strnum = std::to_string(c);
        for (int d = 0; d < DIFFICULTY_LEVELS; d++) {
            for (int i = 0; i < IMAGE_COUNT_DIFFI + 10; i++){
                cv::Mat res = background.clone();
                if (i != 0) generate_pos(res, true, strnum);
                generate_pos(res, false, strnum);
                std::string fn = "../results/images/" + strnum + "-" + std::to_string(d+1) + "-" + std::to_string((i-15)+1) + ".png";
                cv::Mat endres, smallres;
                cv::cvtColor(res, endres, cv::COLOR_BGRA2GRAY);
                cv::resize(endres, smallres, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
                if (i < 10) images.push_back(smallres);
                else if (i < 15) testimages.push_back(smallres);
                else cv::imwrite(fn, smallres);
            }
        }
    }

    std::ofstream out("../results/images.bin", std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open output file!" << std::endl;
        return 1;
    }

    std::ofstream testout("../results/test_images.bin", std::ios::binary);
    if (!testout) {
        std::cerr << "Failed to open test output file!" << std::endl;
        return 1;
    }

    std::ofstream label_out("../results/labels.bin", std::ios::binary);
    if (!label_out) {
        std::cerr << "Failed to open label output file!" << std::endl;
        return 1;
    }

    std::ofstream testlabel_out("../results/test_labels.bin", std::ios::binary);
    if (!testlabel_out) {
        std::cerr << "Failed to open label output file!" << std::endl;
        return 1;
    }

    uint32_t width = images[0].cols;
    uint32_t height = images[0].rows;
    // ^^ todas las imagenes son del mismo tamaño, por lo que podemos hacer que los tests tambien usen esta referencia
    uint32_t count = static_cast<uint32_t>(images.size());
    uint32_t testcount = static_cast<uint32_t>(testimages.size());
    

    out.write(reinterpret_cast<const char*>(&width), sizeof(width));
    out.write(reinterpret_cast<const char*>(&height), sizeof(height));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    label_out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    testout.write(reinterpret_cast<const char*>(&width), sizeof(width));
    testout.write(reinterpret_cast<const char*>(&height), sizeof(height));
    testout.write(reinterpret_cast<const char*>(&testcount), sizeof(testcount));
    testlabel_out.write(reinterpret_cast<const char*>(&testcount), sizeof(testcount));

    for (const auto& img : images) out.write(reinterpret_cast<const char*>(img.data), width * height);
    for (const auto& img : testimages) testout.write(reinterpret_cast<const char*>(img.data), width * height);
    for (int32_t &c : combinations) {
        for (int d = 0; d < DIFFICULTY_LEVELS; d++) {
            for (int i = 0; i < IMAGE_COUNT_DIFFI + 5; i++) {
                if (i < 10) label_out.write(reinterpret_cast<const char*>(&c), sizeof(c));
                else testlabel_out.write(reinterpret_cast<const char*>(&c), sizeof(c));
            }
        }
    }

    out.close();
    label_out.close();

    testout.close();
    testlabel_out.close();
    std::cout << "Exported " << count << " images and " << testcount << " test images (" << width << "x" << height << ") to binary." << std::endl;
    std::cout << "Done." << std::endl;
    return 0;
}