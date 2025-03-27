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

// Simulation function that each thread will run
double simulate_recursion(double t, double x, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(-1., 1.);

    double tau = exponential_distribution(generator);
    if (tau < t) {
        double J = uniform_distribution(generator);
        double Y = tau * uniform_distribution(generator);
        if (J < 0) {
            double v1 = simulate_recursion(t - tau, x + Y, lambda);
            double v2 = simulate_recursion(t - tau, x + Y, lambda);
            return (tau / lambda) * exp(lambda * tau) * 3 * v1 * v2;
        } else {
            double v1 = simulate_recursion(t - tau, x + Y, lambda);
            double v2 = simulate_recursion(t - tau, x + Y, lambda);
            double v3 = simulate_recursion(t - tau, x + Y, lambda);
            return (tau / lambda) * exp(lambda * tau) * 4 * v1 * v2 * v3;
        }
    } else {
        double Y = t * uniform_distribution(generator);
        return exp(lambda * t) * ( (1./2.) * 4 / (pow(x + t, 2) - 4) + (1./2.) * 4 / (pow(x - t, 2) - 4) + t * (-8 * sqrt(2) * (x+Y)) / pow( pow(x+Y, 2) - 4, 2));
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
    string directoryPath = "../Simulation_03/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }

    // Create output directory if not exists for the Monte Carlo results
    string resultsDirectory = "../Simulation_03/results";
    if (!std::filesystem::exists(resultsDirectory)) {
        std::filesystem::create_directories(resultsDirectory);
    }

    // Open log file for stdout (normal output)
    freopen("../Simulation_03/output/my_output.out", "w", stdout);
    // Open log file for stderr (error messages)
    freopen("../Simulation_03/output/my_error.err", "w", stderr);

    try {
        double x = 9;
        double lambda = 1.;
        int num_estimations = 51;
        xt::xarray<double> arr = xt::zeros<double>({1, num_estimations});
        string file_name = "../Simulation_03/results/monte_carlo.csv"; // Monte Carlo results file

        for (int k = 0; k < num_estimations; k++) {
            double t = static_cast<double>(k);
            t /= 10.;
            auto start_time = std::chrono::high_resolution_clock::now();
            double estimated_value = simulate(t, x, lambda, NUM_SIMULATIONS);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            cout << "x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
                 << elapsed_time.count() << " seconds." << endl;
            cout.flush();  // Ensure that output is written immediately to the file

            // Update the value in the array and write to CSV
            xt::view(arr, 0, k) = estimated_value;
            std::ofstream out_file(file_name);
            xt::dump_csv(out_file, arr); // Write the results into the CSV file
            cout << "Updated CSV with x = " << x << ", t = " << t << ", estimated_value = " << estimated_value << endl;
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
