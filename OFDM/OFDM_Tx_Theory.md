# Theoretical Foundations of OFDM Signal Generation with QPSK

This document explains the theoretical principles and mathematical formulations behind the OFDM signal generation, modulation, and noise modeling used in the system.

---

## 1. OFDM

OFDM divides the total channel bandwidth into multiple orthogonal subcarriers. Each subcarrier carries a portion of the data in parallel, making the system resilient to multipath fading.

Let:
- \( N \): IFFT size (number of subcarriers)
- \( X[k] \): QPSK symbol on the \(k^\text{th}\) subcarrier
- \( x[n] \): Time-domain OFDM signal

The time-domain signal is obtained using the inverse FFT:

$$
x[n] = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} X[k] \cdot e^{j \frac{2\pi kn}{N}}, \quad n = 0, 1, \dots, N-1
$$

---

## 2. Subcarrier Allocation and DC Nulling

- Active subcarriers are chosen symmetrically around the center.
- The DC subcarrier (at \(k = N/2\)) is set to zero to eliminate carrier leakage.
- Guard bands are left unused to avoid spectral leakage into adjacent channels.

---

## 3. QPSK Modulation

QPSK (Quadrature Phase Shift Keying) encodes 2 bits per symbol using orthogonal in-phase (I) and quadrature (Q) components.

QPSK symbol set:

$$
s = \frac{1}{\sqrt{2}} \left( \pm1 \pm j \right)
$$

| Bits | Symbol |
|------|--------|
| 00   | $ \frac{1 + j}{\sqrt{2}} $ |
| 01   | $ \frac{-1 + j}{\sqrt{2}} $ |
| 10   | $ \frac{1 - j}{\sqrt{2}} $ |
| 11   | $ \frac{-1 - j}{\sqrt{2}} $ |

The symbols are normalized to have unit average power:
$$
E[|s|^2] = 1
$$

---

## 4. Cyclic Prefix (CP)

To combat Inter-Symbol Interference (ISI), a Cyclic Prefix of length $L_{\text{CP}}$ is added by copying the last \( L_{\text{CP}} \) samples of the time-domain symbol and appending them to the front.

$$
x_{\text{cp}}[n] =
\begin{cases}
x[n + N - L_{\text{CP}}], & 0 \leq n < L_{\text{CP}} \\
x[n - L_{\text{CP}}], & L_{\text{CP}} \leq n < N + L_{\text{CP}}
\end{cases}
$$

The total OFDM symbol length becomes \( L_s = N + L_{\text{CP}} \).

---

## 5. Complete OFDM Frame

The transmit signal is constructed by concatenating multiple CP-extended OFDM symbols:

$$
x_{\text{total}}[n] = \left[ \text{CP}_1 + x_1[n],\ \text{CP}_2 + x_2[n],\ \dots \right]
$$

---

## 6. Noise Modeling: Additive White Gaussian Noise (AWGN)

AWGN simulates channel impairments. It is added with a controlled Signal-to-Noise Ratio (SNR):

- Target SNR in dB:
  $$
  \text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right)
  $$
- Convert to linear scale:
  $$
  \text{SNR}_{\text{linear}} = 10^{\text{SNR}_{\text{dB}} / 10}
  $$
- Noise power:
  $$
  P_{\text{noise}} = \frac{P_{\text{signal}}}{\text{SNR}_{\text{linear}}}
  $$
- AWGN samples:
  $$
  w[n] = \sqrt{\frac{P_{\text{noise}}}{2}} \cdot \left( \mathcal{N}(0,1) + j \mathcal{N}(0,1) \right)
  $$
- Final noisy signal:
  $$
  x_{\text{noisy}}[n] = x_{\text{total}}[n] + w[n]
  $$

---

## 7. Noise Padding

To emulate practical signal capture scenarios (e.g., over-the-air or burst transmissions), noise-only regions are added before and after the useful signal.

$$
x_{\text{final}}[n] = \left[ w_{\text{before}},\ x_{\text{noisy}}[n],\ w_{\text{after}} \right]
$$

This helps in testing synchronization, thresholding, and signal detection techniques.

---
