#include "OFDMTransmitter.hpp"
#include <cmath>
#include <random>
#include <fftw3.h>  // Fast IFFT
#include <fstream>
#include <stdexcept>

// on terminal: brew install fftw


OFDMTransmitter::OFDMTransmitter(int fftSize, int cpLength, int numSymbols, int numActive)
    : fftSize(fftSize), cpLength(cpLength), numSymbols(numSymbols), numActive(numActive) {
    dcIndex = fftSize / 2;
    int half = numActive / 2;
    for (int i = dcIndex - half; i <= dcIndex + half; ++i) {
        if (i != dcIndex) activeSubcarriers.push_back(i);
    }
}

std::vector<int> OFDMTransmitter::generateBits() {
    std::vector<int> bits(numSymbols * numActive * 2);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> bit(0, 1);
    for (int& b : bits) b = bit(rng);
    return bits;
}

std::vector<std::complex<float>> OFDMTransmitter::mapToQPSK(const std::vector<int>& bits) {
    std::vector<std::complex<float>> symbols;
    for (size_t i = 0; i < bits.size(); i += 2) {
        float real = 2 * bits[i] - 1;
        float imag = 2 * bits[i + 1] - 1;
        symbols.emplace_back(real, imag);
    }
    return symbols;
}

std::vector<std::complex<float>> OFDMTransmitter::performIFFT(const std::vector<std::complex<float>>& freqData) {
    std::vector<std::complex<float>> timeData(fftSize);
    fftwf_complex* in = reinterpret_cast<fftwf_complex*>(const_cast<std::complex<float>*>(freqData.data()));
    fftwf_complex* out = reinterpret_cast<fftwf_complex*>(timeData.data());
    fftwf_plan plan = fftwf_plan_dft_1d(fftSize, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    for (auto& x : timeData) x *= std::sqrt(fftSize);
    return timeData;
}

std::vector<std::complex<float>> OFDMTransmitter::generateOFDMSignal() {
    auto bits = generateBits();
    auto qpsk = mapToQPSK(bits);
    std::vector<std::complex<float>> ofdmSignal;

    for (int i = 0; i < numSymbols; ++i) {
        std::vector<std::complex<float>> freqData(fftSize, 0.0f);
        auto start = i * numActive;
        for (int j = 0; j < numActive; ++j) {
            freqData[activeSubcarriers[j]] = qpsk[start + j];
        }

        auto timeData = performIFFT(freqData);
        std::vector<std::complex<float>> ofdmSymbol(cpLength + fftSize);
        std::copy(timeData.end() - cpLength, timeData.end(), ofdmSymbol.begin());
        std::copy(timeData.begin(), timeData.end(), ofdmSymbol.begin() + cpLength);
        ofdmSignal.insert(ofdmSignal.end(), ofdmSymbol.begin(), ofdmSymbol.end());
    }

    return ofdmSignal;
}


void OFDMTransmitter::saveToFile(const std::string& filename, const std::vector<std::complex<float>>& signal) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write header (optional)
    file << "real,imag\n";

    for (const auto& sample : signal) {
        file << sample.real() << "," << sample.imag() << "\n";
    }

    file.close();
}
