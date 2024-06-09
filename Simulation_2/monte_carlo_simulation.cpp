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
#include <random>
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

double simulate_recursion(double t, double x, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(-1., 1.);
    std::uniform_real_distribution<> radiusDistribution(0., 1.);
    std::uniform_real_distribution<> angleDistribution(0., 360.);

    double tau = exponential_distribution(generator);
    if (tau<t) {
        double x_tilda = tau * uniform_distribution(generator);
        double v1 = simulate_recursion(t-tau, x-x_tilda, lambda);
        double v2 = simulate_recursion(t-tau, x-x_tilda, lambda);
        double v3 = simulate_recursion(t-tau, x-x_tilda, lambda);
        return (1./lambda) * exp(lambda * tau) * tau * v1 * v2 * v3;
    }
    else {
        double eta = uniform_distribution(generator);
        if (eta < 0) {
            return exp(lambda * t) * 2 / sqrt(2) * (pow(x+t, -1) + pow(x-t, -1));
        }
        else {
            return exp(lambda * t) * 2 * (pow(x+t, -1) - pow(x-t, -1));
        }
    }
}

double simulate(double t, double x, double lambda, int total_sims) {
    xt::xarray<double> arr = xt::zeros<double>({total_sims});
    
    for (int i=0; i<total_sims; i++) {
        arr[i] = simulate_recursion(t, x, lambda);
    }
    double avg = xt::average(arr)[0];
    return avg;
}

int main()
{
    string directoryPath = "../Simulation_2/output";
    if (!std::__fs::filesystem::exists(directoryPath)) {
        std::__fs::filesystem::create_directory(directoryPath);
    }
    double x = 6;
    double lambda = 1.;
    int num_estimations = 51;
    xt::xarray<double> arr = xt::zeros<double>({1, num_estimations});
    for (int k = 0; k < num_estimations; k++) {
        double t = static_cast<double>(k);
        t /= 10.;
        auto start_time = std::chrono::high_resolution_clock::now();
        double estimated_value = simulate(t, x, lambda, 100000);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
            << elapsed_time.count() << " seconds."<< endl;
        xt::view(arr, 0, k) = estimated_value;
        string file_name = "../Simulation_2/output/monte_carlo.csv";
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "x = "<< x << ", t = " << t << ", successfully exported to csv!" << endl;
    }
    return 0;
}