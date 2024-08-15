cmake_minimum_required(VERSION 3.18..3.20)

project(mymodule)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the xtensor package
find_package(xtensor REQUIRED)

# Add executable targets
add_executable(Simulation_1_analytic_solution Simulation_1/analytic_solution.cpp)
add_executable(Simulation_1_monte_carlo_simulation Simulation_1/monte_carlo_simulation.cpp)
add_executable(Simulation_2_analytic_solution Simulation_2/analytic_solution.cpp)
add_executable(Simulation_2_monte_carlo_simulation Simulation_2/monte_carlo_simulation.cpp)
add_executable(Simulation_3_analytic_solution Simulation_3/analytic_solution.cpp)
add_executable(Simulation_3_monte_carlo_simulation Simulation_3/monte_carlo_simulation.cpp)
add_executable(Simulation_4_analytic_solution Simulation_4/analytic_solution.cpp)
add_executable(Simulation_4_monte_carlo_simulation Simulation_4/monte_carlo_simulation.cpp)

# Link xtensor library
target_link_libraries(Simulation_1_analytic_solution PUBLIC xtensor pthread)
target_link_libraries(Simulation_1_monte_carlo_simulation PUBLIC xtensor pthread)
target_link_libraries(Simulation_2_analytic_solution PUBLIC xtensor pthread)
target_link_libraries(Simulation_2_monte_carlo_simulation PUBLIC xtensor pthread)
target_link_libraries(Simulation_3_analytic_solution PUBLIC xtensor pthread)
target_link_libraries(Simulation_3_monte_carlo_simulation PUBLIC xtensor pthread)
target_link_libraries(Simulation_4_analytic_solution PUBLIC xtensor pthread)
target_link_libraries(Simulation_4_monte_carlo_simulation PUBLIC xtensor pthread)

# Link the filesystem library if necessary (GCC < 9 might need this)
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(Simulation_1_analytic_solution PUBLIC stdc++fs)
        target_link_libraries(Simulation_1_monte_carlo_simulation PUBLIC stdc++fs)
        target_link_libraries(Simulation_2_analytic_solution PUBLIC stdc++fs)
        target_link_libraries(Simulation_2_monte_carlo_simulation PUBLIC stdc++fs)
        target_link_libraries(Simulation_3_analytic_solution PUBLIC stdc++fs)
        target_link_libraries(Simulation_3_monte_carlo_simulation PUBLIC stdc++fs)
        target_link_libraries(Simulation_4_analytic_solution PUBLIC stdc++fs)
        target_link_libraries(Simulation_4_monte_carlo_simulation PUBLIC stdc++fs)
    endif()
endif()
