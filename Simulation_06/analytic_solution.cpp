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

std::pair<xt::xarray<double>, xt::xarray<double>> fix_x_varies_t(double x_1, double x_2, double lower_bound, double upper_bound, double delta_t_) {
    xt::xarray<double> arr = xt::arange(lower_bound, upper_bound + delta_t_, delta_t_);
    complex<double> imag_i(0, 1);    // imag_i = i 
    
    // Calculate the complex result
    xt::xarray<complex<double>> arr2 = tanh( (1/sqrt(6)) * ( (imag_i/sqrt(2)) * (x_1 + x_2) - 2. * arr)); // u(t,(x_1,x_2)) = tanh(\frac{1}{\sqrt{6}}(\frac{i}{\sqrt{2}}(x_1+x_2)-2t))

    // Convert the complex result to real by taking the real and imaginary part
    xt::xarray<double> real_arr(arr2.shape());
    xt::xarray<double> imag_arr(arr2.shape());
    for (size_t i = 0; i < arr2.size(); ++i) {
        real_arr[i] = std::real(arr2[i]);  // Extract imaginary part
        imag_arr[i] = std::imag(arr2[i]);  // Extract imaginary part
    }
    
    // Return both arrays as a pair
    return std::make_pair(real_arr, imag_arr);
}

int main()
{
    string directoryPath = "../Simulation_06/results";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }
    
    double x_1 = -1.;
    double x_2 = -1.;
    std::pair<xt::xarray<double>, xt::xarray<double>> arr;
    arr = fix_x_varies_t(x_1, x_2, 0., 3., 0.01);

    // Reshape each array in the pair separately
    arr.first = arr.first.reshape({1, static_cast<int>((3 - 0) / 0.01) + 1});
    arr.second = arr.second.reshape({1, static_cast<int>((3 - 0) / 0.01) + 1});
    
    string file_name = "../Simulation_06/results/analytic.csv";
    std::ofstream out_file(file_name);
    
    // Write the real and imaginary arrays to the CSV file separately
    xt::dump_csv(out_file, arr.first);  // Real part
    xt::dump_csv(out_file, arr.second); // Imaginary part
    
    cout << "x_1 = "<< x_1 << ", x_2 = "<< x_2 << ", successfully exported to csv!" << endl;
    return 0;
}