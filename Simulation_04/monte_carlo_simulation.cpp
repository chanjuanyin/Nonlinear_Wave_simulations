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
const int NUM_THREADS = 102;

complex<double> sech(complex<double> x) {
    return 2.0 / (exp(x) + exp(-x));
}

complex<double> phi(complex<double> z) {
    return tanh( (complex<double>(0., 1.)/sqrt(6)) * z );
}

complex<double> psi(complex<double> z) {
    return -sqrt(2./3.) * pow(sech( (complex<double>(0., 1.)/sqrt(6)) * z ), 2);
}

// Simulation function that each thread will run
complex<double> simulate_recursion(complex<double> z, double t, complex<double> c, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(0., 1.);

    double tau = exponential_distribution(generator);
    double p = uniform_distribution(generator);
    if (tau > t) {
        return exp(lambda*t) * ( phi(z+c*t)/2. + phi(z-c*t)/2. + t*psi(z+c*t*(2.*p-1)) );
    } else {
        double J_draw = uniform_distribution(generator);
        int J;
        complex<double> aJ;
        double q_J = 1/2.;
        if (J_draw < 1./2.) {
            J = 1;
            aJ = complex<double>(-1., 0.);
        } else {
            J = 3;
            aJ = complex<double>(1., 0.);
        }
        complex<double> H = complex<double>(1., 0.);
        for (int l = 0; l < J; l++) {
            H = H * simulate_recursion(z+c*tau*(2.*p-1),t-tau, c, lambda);
        }
        return exp(lambda*tau) * (tau / lambda) * (aJ/q_J) * H;
    } 
}

void run_simulation(int start, int end, complex<double> z, double t, complex<double> c, double lambda, xt::xarray<double>& results_real, xt::xarray<double>& results_imaginary) {
    for (int i = start; i < end; ++i) {
        complex<double> result = simulate_recursion(z, t, c, lambda);
        results_real[i] = real(result);
        results_imaginary[i] = imag(result);
    }
}

std::pair<double, double> simulate(complex<double> z, double t, complex<double> c, double lambda, int total_sims) {
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
        threads.emplace_back(run_simulation, start, end, z, t, c, lambda, std::ref(results_real), std::ref(results_imaginary));
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

    // Create output directory if not exists for the Monte Carlo results
    string resultsDirectory = "../Simulation_04/results";
    if (!std::filesystem::exists(resultsDirectory)) {
        std::filesystem::create_directories(resultsDirectory);
    }

    complex<double> z = complex<double>(-1., 0.);
    complex<double> c = complex<double>(0., 1.); // c = i
    double lambda = 0.4185722; // Choose lambda = t_0, please refer to the Plot_history.ipynb Jupyter Notebook for details of how to calculate maximum t_0 (existence time of classical solution)
    int num_estimations = 31;
    xt::xarray<double> arr = xt::zeros<double>({2, num_estimations});
    string file_name = "../Simulation_04/results/monte_carlo.csv";

    for (int k = 0; k < num_estimations; k++) {
        double t = static_cast<double>(k);
        t /= 10.;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::pair<double, double> estimated_value = simulate(z, t, c, lambda, NUM_SIMULATIONS);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "z = " << z.real() << " + " << z.imag() << " * i, t = " << t << ", estimated_value u(z,t) = " 
            << estimated_value.first << " + " << estimated_value.second << " * i, Execution time: " 
            << elapsed_time.count() << " seconds."<< endl;

        // Update the value in the array and write to CSV
        xt::view(arr, 0, k) = estimated_value.first;
        xt::view(arr, 1, k) = estimated_value.second;
        std::ofstream out_file(file_name);
        xt::dump_csv(out_file, arr);
        cout << "Updated CSV file name: '" << file_name << "'." << endl;
    }

    return 0;
}