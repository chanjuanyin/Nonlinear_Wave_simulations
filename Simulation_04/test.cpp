#include <bits/stdc++.h>
#include <iostream>
#include <cmath>
#include <complex>  // Include complex header for complex numbers

using namespace std;

int main()
{
    // Create complex numbers
    complex<double> num1(1.0, 2.0);  // 1 + 2i
    complex<double> num2(3.0, 4.0);  // 3 + 4i

    // Perform some complex number calculations
    complex<double> result = num1 * num2;  // Multiply the two complex numbers

    // Extract and print the real and imaginary parts
    double real_part = real(result);
    double imag_part = imag(result);

    cout << "Real part: " << real_part << endl;
    cout << "Imaginary part: " << imag_part << endl;

    // Perform trigonometric operations with complex numbers
    complex<double> trig_result = exp(complex<double>(0, M_PI));  // e^(i*pi) = -1
    cout << "exp(i*pi) = " << trig_result << endl;

    // Extract and print the real and imaginary parts of the result
    cout << "Real part of exp(i*pi): " << real(trig_result) << endl;
    cout << "Imaginary part of exp(i*pi): " << imag(trig_result) << endl;

    return 0;
}
