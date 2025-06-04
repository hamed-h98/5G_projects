#include "OFDMTransmitter.hpp"
#include <iostream>

int main() {
    // Parameters: fftSize, cpLength, numSymbols, numActive
    OFDMTransmitter tx(720, 80, 20, 109);

    // Generate OFDM signal
    auto signal = tx.generateOFDMSignal();

    // Save to binary file (for SDR use)
    tx.saveToFile("ofdm_output.dat", signal);


    std::cout << "OFDM signal generated and saved to ofdm_output.dat\n";
    return 0;
}

// run in terminal:

// g++ -std=c++17 main.cpp OFDMTransmitter.cpp \
//     -I/opt/homebrew/opt/fftw/include \
//     -L/opt/homebrew/opt/fftw/lib \
//     -lfftw3f -o ofdm_tx

// ./ofdm_tx

