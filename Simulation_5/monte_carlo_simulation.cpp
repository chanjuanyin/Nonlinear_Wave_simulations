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
complex<double> simulate_recursion(double t, complex<double> x, complex<double> y, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(0., 1.);
    std::uniform_real_distribution<> radiusDistribution(0., 1.);
    std::uniform_real_distribution<> angleDistribution(0., 2*M_PI);

    double tau = exponential_distribution(generator);
    if (tau < t) { // recursion
        double J = uniform_distribution(generator);
        if (J < 1./2.) {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = tau * pow(1 - pow(1 - p, 2.0), 0.5);
            complex<double> imag_i(0, 1);  // imag_i = i 
            complex<double> u = simulate_recursion(t - tau, x - imag_i * r * cos(theta), y - imag_i * r * sin(theta), lambda);
            return - 2 * tau / lambda * exp(lambda * tau) * u;
        } else {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = tau * pow(1 - pow(1 - p, 2.0), 0.5);
            complex<double> imag_i(0, 1);  // imag_i = i 
            complex<double> u_1 = simulate_recursion(t - tau, x - imag_i * r * cos(theta), y - imag_i * r * sin(theta), lambda);
            complex<double> u_2 = simulate_recursion(t - tau, x - imag_i * r * cos(theta), y - imag_i * r * sin(theta), lambda);
            complex<double> u_3 = simulate_recursion(t - tau, x - imag_i * r * cos(theta), y - imag_i * r * sin(theta), lambda);
            return - 2 * tau / lambda * exp(lambda * tau) * u_1 * u_2 * u_3;
        }
    } else { // end condition
        double eta = uniform_distribution(generator);
        if (eta < 1./3.) {
            return 3 * exp(lambda * t) * sin(M_PI * x) * sin(M_PI * y);
        } else if ((eta >= 1./3.) && (eta < 1./3.)) {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = t * sin(M_PI / 2 * p);
            complex<double> imag_i(0, 1);  // imag_i = i 
            return 3 * exp(lambda * t) * (M_PI * M_PI / 2 * t * imag_i) * ( cos(theta) * cos(M_PI*(x-imag_i*r*cos(theta))) * sin(M_PI*(y-imag_i*r*sin(theta))) + sin(theta) * sin(M_PI*(x-imag_i*r*cos(theta))) * cos(M_PI*(y-imag_i*r*sin(theta))) );
        } else {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = t * pow(1 - pow(1 - p, 2.0), 0.5);
            complex<double> imag_i(0, 1);  // imag_i = i 
            return - 3 * exp(lambda * t) * t * sin(M_PI*(x-imag_i*r*cos(theta))) * sin(M_PI*(y-imag_i*r*sin(theta)));
        }
    }
}

void run_simulation(int start, int end, double t, double x, double y, double lambda, xt::xarray<double>& results_real, xt::xarray<double>& results_imaginary) {
    for (int i = start; i < end; ++i) {
        complex<double> result = simulate_recursion(t, x, y, lambda);
        results_real[i] = real(result);
        results_imaginary[i] = imag(result);
    }
}

std::pair<double, double> simulate(double t, double x, double y, double lambda, int total_sims) {
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
        threads.emplace_back(run_simulation, start, end, t, x, y, lambda, std::ref(results_real), std::ref(results_imaginary));
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
    string directoryPath = "../Simulation_5/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }
    double X = 1.;
    double Y = 1.;
    double t = 0.5;
    double lambda = 1.;
    int num_estimations = 5; // Should be 101 but we try not to be so aggressive at start
    xt::xarray<double> arr_real = xt::zeros<double>({num_estimations + 1, num_estimations + 1});
    xt::xarray<double> arr_imag = xt::zeros<double>({num_estimations + 1, num_estimations + 1});
    string file_name_real = "../Simulation_5/output/monte_carlo_real.csv";
    string file_name_imag = "../Simulation_5/output/monte_carlo_imag.csv";

    for (int k = 0; k <= num_estimations; k++) {
        for (int l = 0; l <= num_estimations; l++) {
            double x = static_cast<double>(k);
            x = X / num_estimations * x;
            double y = static_cast<double>(l);
            y = X / num_estimations * y;

            auto start_time = std::chrono::high_resolution_clock::now();
            std::pair<double, double> estimated_value = simulate(t, x, y, lambda, NUM_SIMULATIONS);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            std::cout << "x = " << x << ", y = " << y << ", t = " << t << ", estimated_value u(t,x) = " 
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

    return 0;
}
