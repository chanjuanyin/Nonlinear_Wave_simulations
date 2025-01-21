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
#include <complex>  // Include complex header for complex numbers
// #include <xtensor/xjson.hpp>

using namespace std;
typedef long long ll;
typedef pair<int,int> PP;
typedef double ld;
const double eps=1e-6;

xt::xarray<double> fix_x_varies_t(double x, double lower_bound, double upper_bound, double delta_t_) {
    xt::xarray<double> arr = xt::arange(lower_bound, upper_bound + delta_t_, delta_t_);
    double c = 2.;
    double d = 1. / pow(pow(c,2)-1,1./2.);
    xt::xarray<double> arr2 = tanh( pow(2,-1./2.) * d * (x - c * arr) );
    return arr2;
} 

int main()
{
    string directoryPath = "../Simulation_7/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }
    double x = -1.;
    xt::xarray<double> arr;
    arr = fix_x_varies_t(x, 0., 3., 0.01);
    arr = arr.reshape({1, static_cast<int> ((3 - 0)/0.01) + 1});
    string file_name = "../Simulation_7/output/analytic.csv";
    std::ofstream out_file(file_name);
    xt::dump_csv(out_file, arr);
    cout << "x = "<< x << ", successfully exported to csv!" << endl;
    return 0;
}