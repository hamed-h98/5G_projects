#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <fstream>
#include <fftw3.h>  // Fast IFFT

using namespace std;

const int ifft_size = 720;
const int cp_length = 80;
const int num_symbols = 20;
const int mod_order = 4; // QPSK
const int dc_index = ifft_size / 2;

// Helper function to generate random bits
vector<int> generate_random_bits(int num_bits) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    vector<int> bits(num_bits);
    for (int &bit : bits)
        bit = dis(gen);
    return bits;
}

// Map 2 bits to QPSK symbol
vector<complex<float>> bits_to_qpsk(const vector<int> &bits) {
    vector<complex<float>> qpsk_data;
    for (size_t i = 0; i < bits.size(); i += 2) {
        float real = 2 * bits[i] - 1;
        float imag = 2 * bits[i + 1] - 1;
        qpsk_data.emplace_back(real, imag);
    }
    return qpsk_data;
}

int main() {
    int num_active = 109;
    int half = num_active / 2;

    vector<int> active_subcarriers;
    for (int i = dc_index - half; i <= dc_index + half; ++i) {
        if (i != dc_index) active_subcarriers.push_back(i);
    }
    num_active = active_subcarriers.size(); // should now be 108

    int bits_per_symbol = log2(mod_order);
    int total_bits = num_active * num_symbols * bits_per_symbol;

    vector<int> bits = generate_random_bits(total_bits);
    vector<complex<float>> qpsk_data = bits_to_qpsk(bits);

    vector<complex<float>> ofdm_signal;

    for (int i = 0; i < num_symbols; i++) {
        vector<complex<float>> freq_data(ifft_size, 0);
        int idx_start = i * num_active;
        for (int j = 0; j < num_active; j++) {
            freq_data[active_subcarriers[j]] = qpsk_data[idx_start + j];
        }
        freq_data[dc_index] = 0;

        // IFFT
        vector<complex<float>> time_data(ifft_size);
        fftwf_plan plan = fftwf_plan_dft_1d(ifft_size,
            reinterpret_cast<fftwf_complex*>(freq_data.data()),
            reinterpret_cast<fftwf_complex*>(time_data.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        for (auto &x : time_data) x *= sqrt(ifft_size);

        // Add CP
        vector<complex<float>> ofdm_symbol;
        ofdm_symbol.insert(ofdm_symbol.end(), time_data.end() - cp_length, time_data.end());
        ofdm_symbol.insert(ofdm_symbol.end(), time_data.begin(), time_data.end());

        ofdm_signal.insert(ofdm_signal.end(), ofdm_symbol.begin(), ofdm_symbol.end());
    }

    // Normalize to unit power
    float power = 0;
    for (const auto &x : ofdm_signal) power += norm(x);
    power /= ofdm_signal.size();
    float norm_factor = sqrt(power);
    for (auto &x : ofdm_signal) x /= norm_factor;

    // Add noise
    float noise_snr_db = 30;
    float noise_power = 1.0f / pow(10.0f, noise_snr_db / 10.0f);
    float noise_std = sqrt(noise_power);
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<float> dist(0.0, noise_std / sqrt(2));

    vector<complex<float>> ofdm_signal_noisy(ofdm_signal.size());
    for (size_t i = 0; i < ofdm_signal.size(); ++i) {
        complex<float> noise(dist(gen), dist(gen));
        ofdm_signal_noisy[i] = ofdm_signal[i] + noise;
    }

    // Add noise padding
    int left_pad = 600, right_pad = 1000;
    vector<complex<float>> noise_before(left_pad), noise_after(right_pad);
    for (auto &x : noise_before) x = complex<float>(dist(gen), dist(gen));
    for (auto &x : noise_after)  x = complex<float>(dist(gen), dist(gen));

    vector<complex<float>> final_signal;
    final_signal.insert(final_signal.end(), noise_before.begin(), noise_before.end());
    final_signal.insert(final_signal.end(), ofdm_signal_noisy.begin(), ofdm_signal_noisy.end());
    final_signal.insert(final_signal.end(), noise_after.begin(), noise_after.end());

    // Save to file (CSV)
    ofstream file("final_signal.csv");
    file << "real,imag\n";
    for (const auto &s : final_signal) {
        file << s.real() << "," << s.imag() << "\n";
    }
    file.close();

    cout << "OFDM signal generated and saved to final_signal.csv" << endl;
    return 0;
}
