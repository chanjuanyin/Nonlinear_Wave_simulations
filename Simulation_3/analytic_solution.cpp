#include <bits/stdc++.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <istream>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <xtensor/xarray.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
// #include <xtensor/xjson.hpp>

using namespace std;
typedef long long ll;
typedef pair<int,int> PP;
typedef double ld;
const double eps=1e-6;

xt::xarray<double> fix_x_varies_t(double x, double lower_bound, double upper_bound, double delta_t_) {
    xt::xarray<double> arr = xt::arange(lower_bound, upper_bound + delta_t_, delta_t_);
    xt::xarray<double> arr2 = 4 / (pow(x + sqrt(2) * arr, 2)-4);
    return arr2;
} 

int main()
{
    string directoryPath = "../Simulation_3/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    double x = 9;
    xt::xarray<double> arr;
    arr = fix_x_varies_t(x, 0., 10., 0.01);
    arr = arr.reshape({1, static_cast<int> ((10 - 0)/0.01) + 1});
    string file_name = "../Simulation_3/output/analytic.csv";
    std::ofstream out_file(file_name);
    xt::dump_csv(out_file, arr);
    cout << "x = "<< x << ", successfully exported to csv!" << endl;
    return 0;
}