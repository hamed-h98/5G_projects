## OFDM Signal Generation, Noise Injection, and Analysis in MATLAB

The Live MATLAB script `OFDM_Tx.mlx`, simulates a realistic OFDM signal transmission system with theoretical explanations, including QPSK modulation, OFDM generation, cyclic prefix insertion, noise padding, in-band noise injection, and visualization of subcarrier power and signal characteristics.

---

## **Features**

* **QPSK Modulation** over 109 active subcarriers (excluding DC)
* **IFFT-based OFDM symbol generation** with cyclic prefix (CP)
* **Preamble symbol generation** for synchronization
* **Customizable noise injection** with a controllable SNR
* **Addition of Gaussian noise padding** (guard intervals) before and after the signal
* **Power spectrum, subcarrier visualization, and autocorrelation plots**

---

## **Parameters & Settings**

| Parameter       | Value    | Description                             |
| --------------- | -------- | --------------------------------------- |
| `ifft_size`     | 720      | Number of IFFT points (FFT size)        |
| `cp_length`     | 80       | Cyclic prefix length                    |
| `mod_order`     | 4        | Modulation order (QPSK)                 |
| `num_symbols`   | 20       | Number of OFDM symbols                  |
| `num_active`    | 109      | Number of active subcarriers            |
| `SNR (in-band)` | 30 dB    | Signal-to-noise ratio for in-band noise |
| `Noise padding` | 600/1000 | Pre-/Post-padding with Gaussian noise   |

---

## **Main Steps**

1. **Active Subcarrier Selection**
   The subcarriers are symmetrically selected around the DC index (which is nulled).

2. **Random QPSK Data Generation**
   Generates QPSK symbols for all OFDM symbols.

3. **Preamble Generation**
   One known QPSK symbol inserted at the beginning for synchronization (optional).

4. **OFDM Symbol Construction**

   * QPSK symbols are mapped to active subcarriers.
   * IFFT is applied to obtain time-domain symbols.
   * Cyclic prefix is added.
   * All symbols are concatenated.

5. **Noise Modeling**

   * In-band complex Gaussian noise is added based on a target SNR.
   * Additional noise is appended before and after the OFDM signal to simulate guard bands.

6. **Signal Normalization**
   Power of the final signal is normalized to unit power.

7. **Saving Output**
   The final noisy OFDM signal is saved as `OFDM_Rx_Signal.mat`.

---

## **Visualizations**

* **Raw Signal Plot**: Time-domain magnitude of the full signal with noise.
* **Autocorrelation Plot**: Shows time-domain signal periodicity and structure.
* **Subcarrier Power Spectrogram**: 2D heatmap showing the power of each subcarrier across OFDM symbols.
* **Power Spectral Density (PSD)**: Plots average power across subcarriers.

---

### **Applications**

* Receiver synchronization and CP estimation experiments
* Channel estimation and equalization testbed
* Machine learning dataset generation for OFDM tasks
* Realistic SNR testing scenarios with controlled noise

---

## OFDM Signal Processing (Receiver Side, Python Implementation)

`OFDM_Rx.ipynb`

A complete **baseband receiver-side analysis** of an OFDM signal:

* Synchronization via energy threshold and correlation
* CP detection to validate symbol structure
* FFT demodulation for symbol extraction
* Visualization of constellation and power spectrum
* Subcarrier diagnostics for quality inspection

Unkown parameters: 

- CP length
- FFT length
- Symbol start index
- Symbol stop index
- Null subcarriers


This framework is valuable for evaluating **modulation integrity**, **timing accuracy**, and **subcarrier behavior** in both simulated and real-world OFDM systems.


### 1. **Signal Preprocessing**

* Load the received signal (input data)
* Define OFDM parameters:
  * Estimate symbol start index 
  * Estimate symbol stop index 
  * Estimate cyclic prefix length for symbols 
  * Estimate total number of symbols

### 2. **Autocorrelation for Rough Symbol Start**

* Compute autocorrelation of the received signal to identify repeating structure.
* Detect the **initial start index** of OFDM symbols based on energy thresholding.
* Use a moving window to refine this estimate.


### 3. **Cyclic Prefix (CP) Correlation-Based Detection**

* Sweep over candidate start indices around the rough estimate.
* For each, check the correlation between CP and end of symbols.
* Select the start index that gives the **maximum average CP correlation**.


### 4. **Symbol Extraction and FFT**

* From the chosen start index, extract all OFDM symbols.
* Remove the cyclic prefix from each.
* Perform FFT on each symbol to transform to the frequency domain.
* Apply `fftshift` to center the DC subcarrier.


### 5. **Constellation Plotting**

* Flatten all subcarriers from all symbols into a single array.
* Plot real vs. imaginary values of subcarriers to visualize:

  * Identify modulation type (e.g., QPSK, QAM)
  * Analyze signal quality and distortions

### 6. **Subcarrier Power Analysis**

* Calculate power per subcarrier for each symbol.
* Identify **low-power (null or attenuated)** subcarriers.
* Use `stem` plots to highlight which subcarriers fall below a threshold (e.g., power < threshold).
* Overplot known null subcarriers for verification.


---
