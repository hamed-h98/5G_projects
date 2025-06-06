clc
clear
close all 
% Parameters
ifft_size = 720; % FFT size, or number of subcarriers 
cp_length = 80;
Ld = ifft_size;
Ls = Ld + cp_length;
num_symbols = 20; 
% time domain: each symbol carries 720 subcarriers at time sample i, where i = 1:20
mod_order = 4; % QPSK
dc_index = ifft_size / 2 + 1;

% Number of active subcarriers
num_active = 109; % number of active subcarriers
half = floor(num_active / 2);
active_subcarriers = (dc_index - half):(dc_index + half);  % symmetrical around DC
active_subcarriers(active_subcarriers == dc_index) = [];   % remove DC

%%%%%%%%%%%%% Generate random QPSK data %%%%%%%%%%%%%%
bits_per_symbol = log2(mod_order); % 2 bits per symbol for QPSK 

total_bits = num_active * num_symbols * bits_per_symbol; 
% 109 active subcarriers * 20 time samples * 2 bits per symbol = total bits

bits = randi([0 1], total_bits, 1); % random bits generation [1x4360] bits 
symbols = reshape(bits, [], bits_per_symbol); % 2 bits per symbol for QPSK

%%%%%%%%%% OFDM Signal generation %%%%%%%%%%%
ofdm_signal = [];
symbol_matrix = zeros(num_symbols, ifft_size); % 20 x 720 
% Ensuring num_active is consistent with active_subcarriers length (109 -> 108 because of DC null)
num_active = length(active_subcarriers);  

for i = 1:num_symbols
    freq_data = zeros(ifft_size, 1);
    idx_start = (i - 1) * num_active + 1;
    idx_end = i * num_active;
    this_symbol = qpsk_data(idx_start:idx_end);
    freq_data(active_subcarriers) = this_symbol;  % Map to subcarriers
    freq_data(dc_index) = 0;                      % Explicit DC null
    symbol_matrix(i, :) = freq_data;              % For visualization
    time_data = ifft(ifftshift(freq_data)) * sqrt(ifft_size);
    cp = time_data(end - cp_length + 1 : end);
    ofdm_symbol = [cp; time_data];
    ofdm_signal = [ofdm_signal; ofdm_symbol]; % vertical concatenation, for appending this symbol below the existing signal.
end

% Normalize final signal to max amplitude = 3
ofdm_signal = ofdm_signal / sqrt(mean(abs(ofdm_signal).^2));  % Normalize to unit power

% Add in-band noise
% Set stronger noise level
noise_snr_db = 30;  % Try 5 or 10 dB SNR for visible noise floor
signal_power = mean(abs(ofdm_signal).^2);
noise_power = signal_power / (10^(noise_snr_db/10));

% Add broadband complex Gaussian noise
inband_noise = sqrt(noise_power/2) * (randn(length(ofdm_signal),1) + 1j*randn(length(ofdm_signal),1));
ofdm_signal_noisy = ofdm_signal + inband_noise;

% Add noise padding before and after
left_pad = 600;
right_pad = 1000;
noise_std = sqrt(noise_power);
noise_before = noise_std * (randn(left_pad,1) + 1j*randn(left_pad,1));
noise_after  = noise_std * (randn(right_pad,1) + 1j*randn(right_pad,1));

% Final signal
final_signal = [noise_before; ofdm_signal_noisy; noise_after];

% Save
save('OFDM_Rx_Signal.mat', 'final_signal');

% === Visualization: Subcarrier Power ===
% Remove CP and reshape
X_matrix = reshape(ofdm_signal_noisy, Ls, []).';
X_no_cp = X_matrix(:, cp_length+1:end);
X_fft = fft(X_no_cp, [], 2);
X_fft_shifted = fftshift(X_fft, 2);
% power_matrix = abs(X_fft_shifted).^2;

% Reshape OFDM symbols
num_symbols_actual = length(ofdm_signal) / Ls;
X_matrix = reshape(ofdm_signal_noisy, Ls, num_symbols_actual).';

% Remove CP
X_no_cp = X_matrix(:, cp_length+1:end);  % shape: [num_symbols, Ld]

% FFT of each symbol
X_fft = fftshift(fft(X_no_cp, [], 2), 2);  % shape: [num_symbols, Ld]
power_matrix = abs(X_fft).^2;             % shape: [num_symbols, Ld]

% Pad to full subcarrier length (optional)
full_power = zeros(num_symbols_actual, ifft_size);
full_power(:, :) = power_matrix;  % Since Ld = ifft_size

figure(1) 
plot(abs(final_signal));
title('Received OFDM Signal with Noise');

figure(2);
power_db = 10*log10(power_matrix + 1e-12);  % avoid log(0)
imagesc(0:num_symbols-1, 1:ifft_size, power_db.');
colormap(jet);  % better dynamic color range
caxis([max(power_db(:)) - 40, max(power_db(:))]);  % show 40 dB range
axis xy;
colormap(gray);
title('Subcarrier Power (Linear Scale)');
xlabel('Symbol Index');
ylabel('Subcarrier Index');
hold on;
% Active subcarriers
% lower_edge = 320;
% upper_edge = 400;
% yline(upper_edge, 'r--', 'Upper Edge Sugcarrier');
% yline(lower_edge, 'r--', 'Lower Edge Subcarrier');
xlim([0 num_symbols_actual - 1]);
ylim([0 ifft_size]);
legend('Guard Band Limits');

N = length(final_signal);

% Define maximum lag to compute autocorrelation
max_lag = floor(N/2);

% Compute autocorrelation at different lags (shifts)
[Rxx, lags] = xcorr(final_signal, max_lag, 'coeff');

% Consider only positive lags
Rxx_positive = Rxx(lags >= 0);
lags_positive = lags(lags >= 0);

% Plotting Autocorrelation
figure(3);
plot(lags_positive, abs(Rxx_positive));
title('Autocorrelation of the Received Signal');
xlabel('Lag');
ylabel('Autocorrelation Coefficient');

freq_axis = -ifft_size/2 : ifft_size/2 - 1;
avg_power = mean(abs(X_fft_shifted).^2, 1);  % average over symbols

% Plot average PSD
avg_power = mean(power_matrix, 1);
freq_axis = -ifft_size/2 : ifft_size/2 - 1;
figure(4);
plot(freq_axis, avg_power, 'LineWidth', 1.2);
xlabel('Subcarrier Index (centered)');
ylabel('Power');
title('Power Spectrum of OFDM with Noise Floor');
grid on;

% Plot constellation for all symbols
figure(5);
scatter(real(rx_symbols_flat), imag(rx_symbols_flat), 10, 'filled');
title('QPSK Constellation after OFDM Demodulation');
xlabel('In-phase');
ylabel('Quadrature');
grid on;
axis equal;
xlim([-2 2] * sqrt(mean(abs(rx_symbols_flat).^2)));  % scale according to average power
ylim([-2 2] * sqrt(mean(abs(rx_symbols_flat).^2)));



