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

complex<double> sech(complex<double> x) {
    return 2.0 / (exp(x) + exp(-x));
}

// Simulation function that each thread will run
complex<double> simulate_recursion(double t, complex<double> x, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(0., 1.);

    double tau = exponential_distribution(generator);
    if (tau < t) {
        double J = uniform_distribution(generator);
        if (J < 1./2.) {
            std::uniform_real_distribution<> time_distribution(-tau, tau);
            double y = time_distribution(generator);
            complex<double> imag_i(0, 1);  // imag_i = i 
            complex<double> v = simulate_recursion(t - tau, x + imag_i * y, lambda);
            return - 2. * tau / lambda * exp(lambda * tau) * v;
        } else {
            std::uniform_real_distribution<> time_distribution(-tau, tau);
            double y = time_distribution(generator);
            complex<double> imag_i(0, 1);  // imag_i = i 
            complex<double> v1 = simulate_recursion(t - tau, x + imag_i * y, lambda);
            complex<double> v2 = simulate_recursion(t - tau, x + imag_i * y, lambda);
            complex<double> v3 = simulate_recursion(t - tau, x + imag_i * y, lambda);
            return 2. * tau / lambda * exp(lambda * tau) * v1 * v2 * v3;
        }
    } else {
        double c = 2.;
        double d = 1. / pow(pow(c,2)-1,1./2.);
        double eta = uniform_distribution(generator);
        complex<double> neg_i(0, -1);  // neg_i = -i 
        if (eta < 1./3.) {
            return 3./2. * exp(lambda * t) * tanh( - pow(2,-1./2.) * d * (neg_i * x + t) );
        } else if (eta >= 1./3. && eta < 2./3.) {
            return 3./2. * exp(lambda * t) * tanh( - pow(2,-1./2.) * d * (neg_i * x - t) );
        } else {
            std::uniform_real_distribution<> time_distribution(-t, t);
            double y = time_distribution(generator);
            return - 3. * pow(2,-1./2.) * exp(lambda * t) * t * d * c * pow( sech( - pow(2,-1./2.) * d * (neg_i * x + y) ) , 2 );
        }
    }
}

void run_simulation(int start, int end, double t, double x, double lambda, xt::xarray<double>& results_real, xt::xarray<double>& results_imaginary) {
    for (int i = start; i < end; ++i) {
        complex<double> result = simulate_recursion(t, x, lambda);
        results_real[i] = real(result);
        results_imaginary[i] = imag(result);
    }
}

std::pair<double, double> simulate(double t, double x, double lambda, int total_sims) {
    xt::xarray<double> results_real = xt::zeros<double>({total_sims});
    xt::xarray<double> results_imaginary = xt::zeros<double>({total_sims});

    // Create a vector to hold the threads
    vector<thread> threads;

    // Determine the workload for each thread
    int simulations_per_thread = total_sims / NUM_THREADS;

    // Launch threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * simulations_per_thread;
        int end = (i == NUM_THREADS - 1) ? total_sims : start + simulations_per_thread;
        threads.emplace_back(run_simulation, start, end, t, x, lambda, std::ref(results_real), std::ref(results_imaginary));
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Calculate the average result
    double avg_real = xt::mean(results_real)[0];
    double avg_imaginary = xt::mean(results_imaginary)[0];

    // Return both averages as a pair
    return std::make_pair(avg_real, avg_imaginary);
}

int main()
{
    string directoryPath = "../Simulation_8/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }
    double x = -1.;
    double lambda = 0.25;
    int num_estimations = 31;
    xt::xarray<double> arr = xt::zeros<double>({2, num_estimations});
    string file_name = "../Simulation_8/output/monte_carlo.csv";
    // string file_name = "../Simulation_7/output/monte_carlo_2.csv";

    for (int k = 0; k < num_estimations; k++) {
        double t = static_cast<double>(k);
        t /= 10.;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::pair<double, double> estimated_value = simulate(t, x, lambda, NUM_SIMULATIONS);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "x = " << x << ", t = " << t << ", estimated_value u(t,x) = " << estimated_value.first 
                << " + " << estimated_value.second << " * i, Execution time: " << elapsed_time.count() 
                << " seconds."<< endl;

        // Update the value in the array and write to CSV
        xt::view(arr, 0, k) = estimated_value.first;
        xt::view(arr, 1, k) = estimated_value.second;
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "Updated CSV file name: '" << file_name << "'." << endl;
    }

    return 0;
}
