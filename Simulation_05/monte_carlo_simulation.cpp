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
double simulate_recursion(double t, double x_1, double x_2, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(0., 1.);
    std::uniform_real_distribution<> angle_distribution(0., 2*M_PI);

    double tau = exponential_distribution(generator);
    if (tau < t) {
        double theta  = angle_distribution(generator);
        double p = uniform_distribution(generator);
        double R = tau * (pow(1 - pow(1 - p, 2.0), 0.5));
        double chi_1 = simulate_recursion(t - tau, x_1 + R * cos(theta), x_2 + R * sin(theta), lambda);
        double chi_2 = simulate_recursion(t - tau, x_1 + R * cos(theta), x_2 + R * sin(theta), lambda);
        return (tau / lambda) * exp(lambda * tau) * chi_1 * chi_2;
    } else {
        double theta  = angle_distribution(generator);
        double p = uniform_distribution(generator);
        double R = t * (pow(1 - pow(1 - p, 2.0), 0.5));
        return exp(lambda * t) * (6 / pow(x_1 + R * cos(theta) + x_2 + R * sin(theta), 2) + (R * cos(theta) + R * sin(theta)) * (-12 / pow(x_1 + R * cos(theta) + x_2 + R * sin(theta), 3)) + t * (-12) * sqrt(3) / pow(x_1 + R * cos(theta) + x_2 + R * sin(theta), 3));
    }
}

void run_simulation(int start, int end, double t, double x_1, double x_2, double lambda, xt::xarray<double>& results) {
    for (int i = start; i < end; ++i) {
        results[i] = simulate_recursion(t, x_1, x_2, lambda);
    }
}

double simulate(double t, double x_1, double x_2, double lambda, int total_sims) {
    xt::xarray<double> results = xt::zeros<double>({total_sims});

    // Create a vector to hold the threads
    vector<thread> threads;

    // Determine the workload for each thread
    int simulations_per_thread = total_sims / NUM_THREADS;

    // Launch threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * simulations_per_thread;
        int end = (i == NUM_THREADS - 1) ? total_sims : start + simulations_per_thread;
        threads.emplace_back(run_simulation, start, end, t, x_1, x_2, lambda, std::ref(results));
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
    string directoryPath = "../Simulation_05/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }

    // Create output directory if not exists for the Monte Carlo results
    string resultsDirectory = "../Simulation_05/results";
    if (!std::filesystem::exists(resultsDirectory)) {
        std::filesystem::create_directories(resultsDirectory);
    }

    // Open log file for stdout (normal output)
    freopen("../Simulation_05/output/my_output.out", "w", stdout);
    // Open log file for stderr (error messages)
    freopen("../Simulation_05/output/my_error.err", "w", stderr);

    try {
        double x_1 = 4;
        double x_2 = 4;
        double lambda = 1.;
        int num_estimations = 41;
        xt::xarray<double> arr = xt::zeros<double>({1, num_estimations});
        string file_name = "../Simulation_05/results/monte_carlo.csv"; // Monte Carlo results file

        for (int k = 0; k < num_estimations; k++) {
            double t = static_cast<double>(k);
            t /= 10.;
            auto start_time = std::chrono::high_resolution_clock::now();
            double estimated_value = simulate(t, x_1, x_2, lambda, NUM_SIMULATIONS);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            cout << "x_1 = " << x_1 << ", x_2 = " << x_2 << ", t = " << t << ", estimated_value = " << estimated_value << ", Execution time: " 
                 << elapsed_time.count() << " seconds." << endl;
            cout.flush();  // Ensure that output is written immediately to the file

            // Update the value in the array and write to CSV
            xt::view(arr, 0, k) = estimated_value;
            std::ofstream out_file(file_name);
            xt::dump_csv(out_file, arr); // Write the results into the CSV file
            cout << "Updated CSV with x_1 = " << x_1 << ", x_2 = " << x_2 << ", t = " << t << ", estimated_value = " << estimated_value << endl;
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
