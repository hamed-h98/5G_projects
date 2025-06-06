`L2_multi_thread.cpp`

---

## Summary
Simulation of a **simplified 5G Layer-2 protocol stack**, including:

* **PDCP (Packet Data Convergence Protocol)**
* **RLC (Radio Link Control)**
* **MAC (Medium Access Control)**

The simulator mimics packet flow **from PDCP to MAC**, with a focus on:

* Multi-threaded traffic generation (uplink/downlink)
* Queued packet scheduling
* MAC-layer retransmissions (HARQ-like behavior)
* Thread-safe design using C++11 primitives

---

## Theoretical Model of 5G Layer-2 Stack

| Layer    | Function                                          |
| -------- | ------------------------------------------------- |
| **PDCP** | Packet reordering, header compression, ciphering  |
| **RLC**  | Segmentation, reassembly, retransmission (ARQ)    |
| **MAC**  | Scheduling, multiplexing, HARQ, buffer management |

This simulation simplifies these by simulating flow and retries, not the actual segmentation or encryption.

---

### `Packet` Struct

Each packet carries:

* `id`: unique (atomic counter)
* `size`, `priority`: for scheduling
* `direction`: Uplink or Downlink
* `retries`: for MAC retransmission
* `arrivalTime`: useful for latency stats (future use)

```cpp
Packet(size_t s, int p, Direction d)
```

---

### `ThreadSafeQueue<T>` (template)

A custom **priority queue with mutex and condition variable** for safe multi-threaded use.

Key Methods:

* `push`: enqueue a packet
* `pop`: dequeue the highest-priority packet (if available)
* `empty`: check if the queue is empty

Used separately for:

* `uplinkQueue`
* `downlinkQueue`

---

### `MAC_Layer` Class

Simulates **MAC behavior**, including:

* Transmit buffer with limited size (`maxBufferSize = 10`)
* Randomized channel simulation:

  * Success with 80% probability
  * Retries up to 3 times (like HARQ)
* Calls back to `RLC_Layer::receivePacket()` on success

```cpp
if (dis(gen) < 0.8) { // Success with 80% probability
```

---

### `RLC_Layer` and `PDCP_Layer` Classes

Simple forwarding:

* **RLC**: Forwards down to MAC and receives packets back
* **PDCP**: Entry point and exit point of the simulated stack

In real systems:

* RLC would handle segmentation/ARQ
* PDCP would do ciphering and integrity protection

Here they **just log** packet flow.

---

### `producerFunction()`

Each producer:

* Generates packets with:

  * Size: random between 1200–1600 bytes
  * Priority: random \[0–9]
* Pushes to uplink/downlink queues every 100–200 ms

This mimics **application traffic generation**.

---

### `consumerFunction()`

Each consumer:

* Pops packets from a queue
* Calls `mac.sendPacket()` to simulate MAC-layer transmission

Runs while:

* Simulation is active
* Or packets still exist in queues

---

### `main()`

Creates the full stack:

* Sets up connections between PDCP → RLC → MAC

Spawns:

* **2 Uplink producers**
* **1 Downlink producer**
* **2 Uplink consumers**

Runs for **10 seconds**, then shuts down gracefully.

---

## Future Considerations

* Adding latency/statistics logging
* Visualizing queues over time
* QoS class-based scheduling
* Timer-based retransmission (vs. instant retry)
* True HARQ state machine
* Integrate this with a physical SDR

---

This code is a **clean simulation** of basic Layer-2 packet handling and for:

* **Protocol stack debugging**
* **Scheduling algorithm evaluation**
* **Embedded systems logic prototyping** (e.g., C++ for SDR/FPGA integration)

