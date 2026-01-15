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

complex<double> phi(complex<double> z_1, complex<double> z_2) {
    return 6. / pow(z_1 + z_2, 2.);
}

complex<double> partial_z_1_phi(complex<double> z_1, complex<double> z_2) {
    return -12. / pow(z_1 + z_2, 3.);
}

complex<double> partial_z_2_phi(complex<double> z_1, complex<double> z_2) {
    return -12. / pow(z_1 + z_2, 3.);
} // Note: in our case \partial_z_1 phi = \partial_z_2 phi

complex<double> psi(complex<double> z_1, complex<double> z_2) {
    return -12. * sqrt(3.) / pow(z_1 + z_2, 3.);
}

// Simulation function that each thread will run
complex<double> simulate_recursion(complex<double> z_1, complex<double> z_2, double t, complex<double> c, double lambda) {
    std::random_device random_seed;  // Obtain a random seed from the hardware
    std::mt19937 generator(random_seed()); // Standard Mersenne Twister engine
    std::exponential_distribution<double> exponential_distribution(lambda);
    std::uniform_real_distribution<> uniform_distribution(0., 1.);
    std::uniform_real_distribution<> angle_distribution(0., 2*M_PI);

    double tau = exponential_distribution(generator);
    double p = uniform_distribution(generator);
    double theta = angle_distribution(generator);
    if (tau > t) {
        double R = t * sqrt(1-pow((1-p), 2));
        complex<double> y_1 = c * R * cos(theta);
        complex<double> y_2 = c * R * sin(theta);
        complex<double> I_1 = phi(z_1 + y_1, z_2 + y_2);
        complex<double> I_2 = y_1 * partial_z_1_phi(z_1 + y_1, z_2 + y_2) + y_2 * partial_z_2_phi(z_1 + y_1, z_2 + y_2);
        complex<double> I_3 = t * psi(z_1 + y_1, z_2 + y_2);
        return exp(lambda*t) * ( I_1 + I_2 + I_3 );
    } else {
        double R = tau * sqrt(1-pow((1-p), 2));
        complex<double> y_1 = c * R * cos(theta);
        complex<double> y_2 = c * R * sin(theta);
        int J = 2;
        double a_J = 1.0;
        double q_J = 1.0;
        complex<double> H = complex<double>(1., 0.);
        for (int l = 0; l < J; l++) {
            H = H * simulate_recursion(z_1+y_1, z_2+y_2, t-tau, c, lambda);
        }
        return exp(lambda*tau) * (tau / lambda) * (a_J/q_J) * H;
    } 
}

void run_simulation(int start, int end, complex<double> z_1, complex<double> z_2, double t, complex<double> c, double lambda, xt::xarray<double>& results_real, xt::xarray<double>& results_imaginary) {
    for (int i = start; i < end; ++i) {
        complex<double> result = simulate_recursion(z_1, z_2, t, c, lambda);
        results_real[i] = real(result);
        results_imaginary[i] = imag(result);
    }
}

std::pair<double, double> simulate(complex<double> z_1, complex<double> z_2, double t, complex<double> c, double lambda, int total_sims) {
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
        threads.emplace_back(run_simulation, start, end, z_1, z_2, t, c, lambda, std::ref(results_real), std::ref(results_imaginary));
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
    string resultsDirectory = "../Simulation_05/results";
    if (!std::filesystem::exists(resultsDirectory)) {
        std::filesystem::create_directories(resultsDirectory);
    }

    complex<double> z_1 = complex<double>(4., 0.); // z_1 = 4 + 0i
    complex<double> z_2 = complex<double>(4., 0.); // z_2 = 4 + 0i
    complex<double> c = complex<double>(1., 0.); // c = 1 + 0i
    double lambda = 1.;
    int num_estimations = 41;
    xt::xarray<double> arr = xt::zeros<double>({2, num_estimations});
    string file_name = "../Simulation_05/results/monte_carlo.csv";

    for (int k = 0; k < num_estimations; k++) {
        double t = static_cast<double>(k);
        t /= 10.;
        auto start_time = std::chrono::high_resolution_clock::now();
        std::pair<double, double> estimated_value = simulate(z_1, z_2, t, c, lambda, NUM_SIMULATIONS);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = end_time - start_time;
        cout << "z_1 = " << z_1.real() << " + " << z_1.imag() << " * i, z_2 = " << z_2.real() 
            << " + " << z_2.imag() << " * i, t = " << t << ", estimated_value u(z,t) = " 
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
