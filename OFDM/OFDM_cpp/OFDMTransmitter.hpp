#ifndef OFDM_TRANSMITTER_HPP
#define OFDM_TRANSMITTER_HPP

#include <vector>
#include <complex>

class OFDMTransmitter {
public:
    OFDMTransmitter(int fftSize, int cpLength, int numSymbols, int numActive);
    std::vector<std::complex<float>> generateOFDMSignal();

    void saveToFile(const std::string& filename, const std::vector<std::complex<float>>& signal);

private:
    int fftSize, cpLength, numSymbols, numActive;
    int dcIndex;
    std::vector<int> activeSubcarriers;

    std::vector<int> generateBits();
    std::vector<std::complex<float>> mapToQPSK(const std::vector<int>& bits);
    std::vector<std::complex<float>> performIFFT(const std::vector<std::complex<float>>& freqData);
};

#endif
