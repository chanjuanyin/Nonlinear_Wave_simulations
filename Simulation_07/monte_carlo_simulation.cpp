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
complex<double> simulate_recursion(double t, complex<double> x_1, complex<double> x_2, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(0., 1.);
    std::uniform_real_distribution<> angle_distribution(0., 2*M_PI);

    double tau = exponential_distribution(generator);
    if (tau < t) {
        complex<double> imag_i(0, 1);    // imag_i = i
        double theta  = angle_distribution(generator);
        double p = uniform_distribution(generator);
        double R = tau * (pow(1 - pow(1 - p, 2.0), 0.5));
        double J = uniform_distribution(generator);
        if (J<0.5) {
            complex<double> chi_1 = simulate_recursion(t - tau, x_1 + imag_i * R * cos(theta), x_2 + imag_i * R * sin(theta), lambda);
            return (tau / lambda) * exp(lambda * tau) * (-2.) * chi_1;
        }
        else{
            complex<double> chi_1 = simulate_recursion(t - tau, x_1 + imag_i * R * cos(theta), x_2 + imag_i * R * sin(theta), lambda);
            complex<double> chi_2 = simulate_recursion(t - tau, x_1 + imag_i * R * cos(theta), x_2 + imag_i * R * sin(theta), lambda);
            complex<double> chi_3 = simulate_recursion(t - tau, x_1 + imag_i * R * cos(theta), x_2 + imag_i * R * sin(theta), lambda);
            return (tau / lambda) * exp(lambda * tau) * (-2.) * chi_1 * chi_2 * chi_3;
        }
    } else {
        complex<double> imag_i(0, 1);    // imag_i = i
        double theta  = angle_distribution(generator);
        double p = uniform_distribution(generator);
        double R = t * (pow(1 - pow(1 - p, 2.0), 0.5));
        return exp(lambda * t) * ( sin(M_PI*(x_1 + imag_i * R * cos(theta)))*sin(M_PI*(x_2 + imag_i * R * sin(theta))) + imag_i * R * cos(theta) * M_PI * cos(M_PI*(x_1+imag_i*R*cos(theta)))*sin(M_PI*(x_2+imag_i*R*sin(theta))) + imag_i * R * sin(theta) * M_PI * sin(M_PI*(x_1+imag_i*R*cos(theta)))*cos(M_PI*(x_2+imag_i*R*sin(theta))) - t * sin(M_PI*(x_1+imag_i*R*cos(theta))) * sin(M_PI*(x_2+imag_i*R*sin(theta))));
    }
}

void run_simulation(int start, int end, double t, double x_1, double x_2, double lambda, xt::xarray<double>& results_real, xt::xarray<double>& results_imaginary) {
    for (int i = start; i < end; ++i) {
        complex<double> result = simulate_recursion(t, x_1, x_2, lambda);
        results_real[i] = real(result);
        results_imaginary[i] = imag(result);
    }
}

std::pair<double, double> simulate(double t, double x_1, double x_2, double lambda, int total_sims) {
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
        threads.emplace_back(run_simulation, start, end, t, x_1, x_2, lambda, std::ref(results_real), std::ref(results_imaginary));
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
    string directoryPath = "../Simulation_07/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }

    // Create output directory if not exists for the Monte Carlo results
    string resultsDirectory = "../Simulation_07/results";
    if (!std::filesystem::exists(resultsDirectory)) {
        std::filesystem::create_directories(resultsDirectory);
    }

    // Open log file for stdout (normal output)
    freopen("../Simulation_07/output/my_output.out", "w", stdout);
    // Open log file for stderr (error messages)
    freopen("../Simulation_07/output/my_error.err", "w", stderr);

    try{
        double X_1 = 1.;
        double X_2 = 1.;
        double t = 0.5;
        double lambda = 0.25;
        int num_estimations = 100; // Should be 100 but we try not to be so aggressive at start
        xt::xarray<double> arr_real = xt::zeros<double>({num_estimations + 1, num_estimations + 1});
        xt::xarray<double> arr_imag = xt::zeros<double>({num_estimations + 1, num_estimations + 1});
        string file_name_real = "../Simulation_07/results/monte_carlo_real.csv";
        string file_name_imag = "../Simulation_07/results/monte_carlo_imag.csv";

        for (int k = 0; k <= num_estimations; k++) {
            for (int l = 0; l <= num_estimations; l++) {
                double x_1 = static_cast<double>(k);
                x_1 = X_1 / num_estimations * x_1;
                double x_2 = static_cast<double>(l);
                x_2 = X_2 / num_estimations * x_2;

                auto start_time = std::chrono::high_resolution_clock::now();
                std::pair<double, double> estimated_value = simulate(t, x_1, x_2, lambda, NUM_SIMULATIONS);
                auto end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_time = end_time - start_time;
                std::cout << "x_1 = " << x_1 << ", x_2 = " << x_2 << ", t = " << t << ", estimated_value u(t,(x_1,x_2)) = " 
                    << estimated_value.first << " + " << estimated_value.second 
                    << " * i, Execution time: " << elapsed_time.count() << " seconds."<< std::endl;

                // Update the value in the array and write to CSV
                xt::view(arr_real, k, l) = estimated_value.first;
                xt::view(arr_imag, k, l) = estimated_value.second;
                std::ofstream out_file_real(file_name_real);
                std::ofstream out_file_imag(file_name_imag);
                xt::dump_csv(out_file_real, arr_real);
                xt::dump_csv(out_file_imag, arr_imag);
                std::cout << "Updated CSV file name: '" << file_name_real << "'." << std::endl;
                std::cout << "Updated CSV file name: '" << file_name_imag << "'." << std::endl;
            }
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