#include <iostream>
#include <fstream>
#include <filesystem>

#include <string>
#include <vector>
#include <random>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#define DIFFICULTY_LEVELS 9
#define IMAGE_COUNT_DIFFI 20

cv::Mat background, kun_red, kun_blue, kun_white, kun_yellow, kun_green, halo_big, halo_small;

std::vector<int> elements = {1,2,3,4,5,6,7,8,9};
std::vector<int32_t> combinations;
std::random_device rd;
std::mt19937 gen(rd());

int halo_positions[9] = {-2, 37, 68, 107, 139, 179, 211, 243, 283};
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
 
void png_overlay(const cv::Mat& srcRGBA, cv::Mat& dstBGRA, int x, int y) {
    if (srcRGBA.empty() || dstBGRA.empty()) return;

    cv::Rect dst_rect(x, y, srcRGBA.cols, srcRGBA.rows);
    cv::Rect dst_roi = dst_rect & cv::Rect(0, 0, dstBGRA.cols, dstBGRA.rows);
    
    if (dst_roi.empty()) return;

    // calculamos la region aqui
    cv::Rect src_roi(dst_roi.x - x, dst_roi.y - y, dst_roi.width, dst_roi.height);
    cv::Mat src = srcRGBA(src_roi);
    cv::Mat dst = dstBGRA(dst_roi);

    // como aun estamos usando colores aqui, hacemos split a los canales
    std::vector<cv::Mat> src_channels, dst_channels;
    cv::split(src, src_channels);
    cv::split(dst, dst_channels);
    
    cv::Mat alpha;
    src_channels[3].convertTo(alpha, CV_32FC1, 1.0/255.0); // no sabemos el tipo adecuado para el canal, lo dejamos asi
    cv::Mat inverse_alpha = 1.0 - alpha;

    for (int c = 0; c < 3; c++) {
        cv::Mat src_float, dst_float;
        src_channels[c].convertTo(src_float, CV_32FC1);
        dst_channels[c].convertTo(dst_float, CV_32FC1);

        cv::Mat blended = src_float.mul(alpha) + dst_float.mul(inverse_alpha);
        blended.convertTo(dst_channels[c], CV_8UC1);
    }
    cv::merge(dst_channels, dst);
}

void generate_halo(cv::Mat &image, const std::string &pat = ""){
    std::vector<int> picks;
    std::uniform_int_distribution<> am_notes(1,8);
    if(!pat.empty()) picks = rand_sample(elements, (size_t)am_notes(gen));
    else picks = combo_fromstr(pat);
    for (int &i : picks){
        cv::Mat halo = i == 4 || i == 6 || i == 2 || i == 8 ? halo_small : halo_big;
        int combo_x = halo_positions[i-1], combo_y = 0;
        png_overlay(halo, image, combo_x, combo_y);
    }
}

void generate_pos(cv::Mat &image, bool garb = false, const std::string &num = ""){
    std::vector<int> picks;
    std::uniform_int_distribution<> am_notes(1,6);
    if (garb) picks = rand_sample(elements, (size_t)am_notes(gen));
    else picks = combo_fromstr(num);
    for (int &i : picks){
        std::uniform_int_distribution<> am_positions(10, 20);
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

int main(int argc, char *argv[]){
    cv::Mat baseBg = cv::imread("./assets/empty_template_smaller.png", cv::IMREAD_UNCHANGED);
    kun_red = cv::imread("./assets/kuns/red.png", cv::IMREAD_UNCHANGED);
    kun_blue = cv::imread("./assets/kuns/blue.png", cv::IMREAD_UNCHANGED);
    kun_green = cv::imread("./assets/kuns/green.png",cv::IMREAD_UNCHANGED);
    kun_yellow = cv::imread("./assets/kuns/yellow.png", cv::IMREAD_UNCHANGED);
    kun_white = cv::imread("./assets/kuns/white.png", cv::IMREAD_UNCHANGED);

    cv::Mat bg_popn8 = cv::imread("./assets/empty_pop8.png", cv::IMREAD_UNCHANGED);
    cv::Mat bg_popn9 = cv::imread("./assets/empty_pop9.png", cv::IMREAD_UNCHANGED);
    cv::Mat bg_popn11 = cv::imread("./assets/empty_pop11.png", cv::IMREAD_UNCHANGED);

    background = baseBg.clone();

    if (argc > 1){
        if (std::string(argv[1]) == "--priority8") {
            background = bg_popn8.clone();
            bg_popn8 = baseBg.clone();
        } else if (std::string(argv[1]) == "--priority9") {
            background = bg_popn9.clone();
            bg_popn9 = baseBg.clone();
        } else if (std::string(argv[1]) == "--priority11") {
            background = bg_popn11.clone();
            bg_popn11 = baseBg.clone();
        }
    }

    //posiblemente podriamos shufflear el orden por cada combinacion
    std::vector<cv::Mat> bgVariants = {
        background,
        bg_popn8,
        bg_popn8,
        bg_popn9,
        bg_popn11
    };

    halo_big = cv::imread("./assets/halo.png", cv::IMREAD_UNCHANGED);
    halo_small = cv::imread("./assets/halo_smaller.png", cv::IMREAD_UNCHANGED);

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

    cv::Mat res, endres, smallres;
    res.create(background.size(), background.type());
    endres.create(background.rows, background.cols, CV_8UC1);
    smallres.create(background.rows/2, background.cols/2, CV_8UC1);

    for (int32_t &c : combinations) {
        std::shuffle(bgVariants.begin(), bgVariants.end(), gen);
        std::string strnum = std::to_string(c);
        for (int d = 0; d < DIFFICULTY_LEVELS; d++) {
            for (int i = 0; i < IMAGE_COUNT_DIFFI + 10; i++){
                int ran = i % 5;
                bgVariants[ran].copyTo(res);

                if (d > 5) {
                    if (i % 2 == 0) generate_halo(res);
                    else if (c != 0) generate_halo(res, strnum);
                }
                if (i != 0 && i != 10 && i != 15) generate_pos(res, true, strnum);
                generate_pos(res, false, strnum);

                cv::cvtColor(res, endres, cv::COLOR_BGRA2GRAY);
                if (i % 2 == 0) {
                    double shift = std::uniform_real_distribution<>(15,50)(gen);
                    if (i % 4 == 0) endres -= shift;
                    else endres += shift;
                }
                
                cv::resize(endres, smallres, smallres.size(), 0, 0, cv::INTER_AREA);
                std::string fn = "../results/images/" + strnum + "-" + std::to_string(d+1) + "-" + std::to_string((i-25)+1) + ".png";

                if (i < 20) images.push_back(smallres);
                else if (i < 25) testimages.push_back(smallres);
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
                if (i < 20) label_out.write(reinterpret_cast<const char*>(&c), sizeof(c));
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