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

using namespace std;
typedef long long ll;
typedef pair<int,int> PP;
typedef double ld;
const double eps=1e-6;

// Number of simulations
const int NUM_SIMULATIONS = 10000000;
// Number of threads
const int NUM_THREADS = 160;

// Simulation function that each thread will run
double simulate_recursion(double t, double x, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(-1., 1.);
    std::uniform_real_distribution<> radiusDistribution(0., 1.);
    std::uniform_real_distribution<> angleDistribution(0., 360.);

    double tau = exponential_distribution(generator);
    if (tau < t) {
        double x_tilda = tau * uniform_distribution(generator);
        double v1 = simulate_recursion(t - tau, x - x_tilda, lambda);
        double v2 = simulate_recursion(t - tau, x - x_tilda, lambda);
        return (1. / lambda) * exp(lambda * tau) * tau * v1 * v2;
    } else {
        double eta = uniform_distribution(generator);
        if (eta < 0) {
            return exp(lambda * t) * 2 * 3 * (pow(x + t, -2) + pow(x - t, -2));
        } else {
            return exp(lambda * t) * 2 * 3 * sqrt(2) * (pow(x + t, -2) - pow(x - t, -2));
        }
    }
}

void run_simulation(int start, int end, double t, double x, double lambda, xt::xarray<double>& results) {
    for (int i = start; i < end; ++i) {
        results[i] = simulate_recursion(t, x, lambda);
    }
}

double simulate(double t, double x, double lambda, int total_sims) {
    xt::xarray<double> results = xt::zeros<double>({total_sims});

    // Create a vector to hold the threads
    vector<thread> threads;

    // Determine the workload for each thread
    int simulations_per_thread = total_sims / NUM_THREADS;

    // Launch threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * simulations_per_thread;
        int end = (i == NUM_THREADS - 1) ? total_sims : start + simulations_per_thread;
        threads.emplace_back(run_simulation, start, end, t, x, lambda, std::ref(results));
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Calculate the average result
    double avg = xt::mean(results)[0];
    return avg;
}

int main()
{
    string directoryPath = "../Simulation_1/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }
    double x = 3;
    double lambda = .25;
    int num_estimations = 21;
    xt::xarray<double> arr = xt::zeros<double>({1, num_estimations});
    string file_name = "../Simulation_1/output/monte_carlo.csv";

    for (int k = 0; k < num_estimations; k++) {
        double t = static_cast<double>(k);
        t /= 10.;
        auto start_time = std::chrono::high_resolution_clock::now();
        double estimated_value = simulate(t, x, lambda, NUM_SIMULATIONS);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
            << elapsed_time.count() << " seconds."<< endl;

        // Update the value in the array and write to CSV
        xt::view(arr, 0, k) = estimated_value;
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "Updated CSV with x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << endl;
    }

    return 0;
}
