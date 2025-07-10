#include <iostream>
#include <vector>
#include <array>

namespace popn::utils {
    const static std::array<int, 9> availableNumbers = {1,2,3,4,5,6,7,8,9};

    static void generate_combos(int r, int st, std::vector<int> &cur, std::vector<int32_t> &res) {
        if (cur.size() == r) {
            std::string number_str;
            for (int num : cur) number_str += std::to_string(num);
            res.push_back(std::stoi(number_str));
            return;
        }

        for (int i = st; i < availableNumbers.size(); ++i) {
            cur.push_back(availableNumbers[i]);
            generate_combos(r, i + 1, cur, res);
            cur.pop_back();
        }
    }
}