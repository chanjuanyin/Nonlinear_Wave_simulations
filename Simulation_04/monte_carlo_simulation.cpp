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
const int NUM_THREADS = 120;

complex<double> sech(complex<double> x) {
    return 2.0 / (exp(x) + exp(-x));
}

// Simulation function that each thread will run
complex<double> simulate_recursion(double t, complex<double> x, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(-1., 1.);

    double tau = exponential_distribution(generator);
    if (tau < t) {
        double J = uniform_distribution(generator);
        double Y = tau * uniform_distribution(generator);
        complex<double> imag_i(0, 1);  // imag_i = i 
        if (J < 0) {
            complex<double> chi_1 = simulate_recursion(t - tau, x + imag_i * Y, lambda);
            return (tau / lambda) * exp(lambda * tau) * (-2) * chi_1;
        } else {
            complex<double> chi_1 = simulate_recursion(t - tau, x + imag_i * Y, lambda);
            complex<double> chi_2 = simulate_recursion(t - tau, x + imag_i * Y, lambda);
            complex<double> chi_3 = simulate_recursion(t - tau, x + imag_i * Y, lambda);
            return (tau / lambda) * exp(lambda * tau) * 2 * chi_1 * chi_2 * chi_3;
        }
    } else {
        complex<double> imag_i(0, 1);    // imag_i = i
        double Y = t * uniform_distribution(generator);
        return exp(lambda * t) * ( (1./2.) * tanh( (1./sqrt(6)) * imag_i * (x + imag_i * t) ) + (1./2.) * tanh( (1./sqrt(6)) * imag_i * (x - imag_i * t) ) + t * (-sqrt(2./3.)) * pow(sech( (1./sqrt(6)) * imag_i * (x + imag_i * Y) ), 2) );
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
    string directoryPath = "../Simulation_04/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }

    // Create output directory if not exists for the Monte Carlo results
    string resultsDirectory = "../Simulation_04/results";
    if (!std::filesystem::exists(resultsDirectory)) {
        std::filesystem::create_directories(resultsDirectory);
    }

    // Open log file for stdout (normal output)
    freopen("../Simulation_04/output/my_output.out", "w", stdout);
    // Open log file for stderr (error messages)
    freopen("../Simulation_04/output/my_error.err", "w", stderr);

    try{
        double x = -1.;
        double lambda = 0.25;
        int num_estimations = 31;
        xt::xarray<double> arr = xt::zeros<double>({2, num_estimations});
        string file_name = "../Simulation_04/results/monte_carlo.csv";

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
            cout.flush();  // Ensure that output is written immediately to the file
    
            // Update the value in the array and write to CSV
            xt::view(arr, 0, k) = estimated_value.first;
            xt::view(arr, 1, k) = estimated_value.second;
            std::ofstream out_file(file_name);
            xt::dump_csv(out_file, arr);
            cout << "Updated CSV file name: '" << file_name << "'." << endl;
            cout.flush();  // Ensure that output is written immediately to the file
        }
    }
    catch (const std::exception& e) {
        cerr << "An exception occurred: " << e.what() << endl;  // Print exception message to stderr
        cerr.flush();  // Ensure that the error message is written to the error log
    }
    catch (...) {
        cerr << "An unknown error occurred." << endl;  // Catch any other errors and print to stderr
        cerr.flush();
    }

    // Close the log files
    fclose(stdout);
    fclose(stderr);

    return 0;
}