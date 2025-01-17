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
double simulate_recursion(double t, double x, double y, double lambda) {
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
            double u = simulate_recursion(t - tau, x - r * cos(theta), y - r * sin(theta), lambda);
            return - 2 * tau / lambda * exp(lambda * tau) * u;
        } else {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = tau * pow(1 - pow(1 - p, 2.0), 0.5);
            double u_1 = simulate_recursion(t - tau, x - r * cos(theta), y - r * sin(theta), lambda);
            double u_2 = simulate_recursion(t - tau, x - r * cos(theta), y - r * sin(theta), lambda);
            double u_3 = simulate_recursion(t - tau, x - r * cos(theta), y - r * sin(theta), lambda);
            return - 2 * tau / lambda * exp(lambda * tau) * u_1 * u_2 * u_3;
        }
    } else { // end condition
        double eta = uniform_distribution(generator);
        if (eta < 1./3.) {
            return 3 * exp(lambda * t) * sinh(M_PI * x) * sinh(M_PI * y);
        } else if ((eta >= 1./3.) && (eta < 1./3.)) {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = t * sin(M_PI / 2 * p);
            return 3 * exp(lambda * t) * (M_PI * M_PI / 2 * t) * ( cos(theta) * cosh(M_PI*(x-r*cos(theta))) * sinh(M_PI*(y-r*sin(theta))) + sin(theta) * sinh(M_PI*(x-r*cos(theta))) * cosh(M_PI*(y-r*sin(theta))) );
        } else {
            double theta = angleDistribution(generator);
            double p = radiusDistribution(generator);
            double r = t * pow(1 - pow(1 - p, 2.0), 0.5);
            return 3 * exp(lambda * t) * t * sinh(M_PI*(x-r*cos(theta))) * sinh(M_PI*(y-r*sin(theta)));
        }
    }
    return 0.;
}

void run_simulation(int start, int end, double t, double x, double y, double lambda, xt::xarray<double>& results) {
    for (int i = start; i < end; ++i) {
        results[i] = simulate_recursion(t, x, y, lambda);
    }
}

double simulate(double t, double x, double y, double lambda, int total_sims) {
    xt::xarray<double> results = xt::zeros<double>({total_sims});

    // Create a vector to hold the threads
    vector<thread> threads;

    // Determine the workload for each thread
    int simulations_per_thread = total_sims / NUM_THREADS;

    // Launch threads
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * simulations_per_thread;
        int end = (i == NUM_THREADS - 1) ? total_sims : start + simulations_per_thread;
        threads.emplace_back(run_simulation, start, end, t, x, y, lambda, std::ref(results));
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
    string directoryPath = "../Simulation_5/output";
    if (!std::filesystem::exists(directoryPath)) {
        std::filesystem::create_directories(directoryPath);
    }
    double X = 1.;
    double Y = 1.;
    double t = 0.5;
    double lambda = 3.0;
    int num_estimations = 5; // Should be 101 but we try not to be so aggressive at start
    xt::xarray<double> arr = xt::zeros<double>({num_estimations + 1, num_estimations + 1});
    string file_name = "../Simulation_5/output/monte_carlo.csv";

    for (int k = 0; k <= num_estimations; k++) {
        for (int l = 0; l <= num_estimations; l++) {
            double x = static_cast<double>(k);
            x = X / num_estimations * x;
            double y = static_cast<double>(l);
            y = X / num_estimations * y;

            auto start_time = std::chrono::high_resolution_clock::now();
            double estimated_value = simulate(t, x, y, lambda, NUM_SIMULATIONS);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_time = end_time - start_time;
            cout << "x = " << x << ", y = " << y << ", t = " << t << ", estimated_value u(t,x,y) = " << estimated_value << ", Execution time: " 
                << elapsed_time.count() << " seconds."<< endl;

            // Update the value in the array and write to CSV
            xt::view(arr, k, l) = estimated_value;
            std::ofstream out_file(file_name);
            xt::dump_csv(out_file, arr);
            cout << "Updated CSV file name: '" << file_name << "'." << endl;
        }
    }

    return 0;
}
