#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

namespace popn::utils {
    const static std::string gscale = " .,:oOX#$@";

    static void print_ascii(cv::Mat image, int wh){
        for (size_t i = 0; i < wh; i++) {
            int char_index = static_cast<int>((static_cast<double>(image.data[i]) / 255) * (gscale.length() - 1));
            std::cout << gscale[char_index];
            if ((i + 1) % 154 == 0) std::cout << std::endl;
        }
    }
}