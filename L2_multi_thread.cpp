/*
C++ Multi-threaded Layer-2 Protocol Stack Simulator (PDCP–RLC–MAC) 
/*

===============================================================
  Layer-2 Protocol Stack Simulator (PDCP–RLC–MAC) – C++ (Multithreaded)
===============================================================

This simulation models a simplified 5G Layer-2 protocol stack 
consisting of:
  - PDCP (Packet Data Convergence Protocol)
  - RLC (Radio Link Control)
  - MAC (Medium Access Control)

Features:
----------
- Multithreaded architecture:
    • Producer threads generate Uplink/Downlink packets.
    • Consumer threads simulate MAC-layer transmission.

- Thread-safe queues:
    • Uplink and Downlink traffic are managed using synchronized 
      priority queues for realistic packet scheduling.

- MAC Layer simulation:
    • Retransmission logic using retry count (HARQ-like).
    • Random packet success/failure based on a simulated channel.
    • Packet drop handling on buffer overflow or max retries.

- RLC Layer:
    • Forwards packets to MAC and receives them back.
    • Sends completed packets up to the PDCP layer.

- PDCP Layer:
    • Logs final reception of packets.

- Clean termination:
    • Consumer threads exit once the simulation flag is cleared 
      and queues are drained.

Future Extensions:
------------------
- Logging packet latency and retry statistics to CSV.
- Support for QoS class tagging and scheduling.
- Advanced HARQ and timer-based retransmission.

*/


#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <string>
#include <atomic>
#include <random>
#include <thread>
#include <deque>
#include <vector>
#include <memory>

enum class Direction { Uplink, Downlink };
std::atomic<int> globalPacketId{0};

struct Packet {
    int id;
    size_t size;
    int priority;
    Direction direction;
    int retries;
    std::chrono::steady_clock::time_point arrivalTime;

    Packet(size_t s, int p, Direction d)
        : id(globalPacketId++), size(s), priority(p), direction(d), retries(0),
          arrivalTime(std::chrono::steady_clock::now()) {}
};

struct PacketComparator {
    bool operator()(const Packet& a, const Packet& b) {
        return a.priority < b.priority;
    }
};

template <typename T>
class ThreadSafeQueue {
private:
    std::priority_queue<T, std::vector<T>, PacketComparator> queue;
    std::mutex mtx;
    std::condition_variable cv;

public:
    void push(const T& item) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push(item);
        }
        cv.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        if (queue.empty()) return false;
        item = queue.top();
        queue.pop();
        return true;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
};

ThreadSafeQueue<Packet> uplinkQueue;
ThreadSafeQueue<Packet> downlinkQueue;
std::atomic<bool> simulationRunning{true};

class PDCP_Layer; // Forward declaration only
class RLC_Layer;

class MAC_Layer {
private:
    std::shared_ptr<RLC_Layer> rlc;
    std::deque<Packet> buffer;
    const size_t maxBufferSize = 10;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;

public:
    MAC_Layer() : gen(rd()), dis(0.0, 1.0) {}

    void setRLC(const std::shared_ptr<RLC_Layer>& r) {
        rlc = r;
    }

    void sendPacket(Packet pkt);
};

class RLC_Layer {
private:
    std::shared_ptr<MAC_Layer> mac;
    std::shared_ptr<PDCP_Layer> pdcp;

public:
    void setMAC(const std::shared_ptr<MAC_Layer>& m) {
        mac = m;
    }

    void setPDCP(const std::shared_ptr<PDCP_Layer>& p) {
        pdcp = p;
    }

    void sendPacket(const Packet& pkt) {
        std::cout << "[RLC] Forwarding packet ID: " << pkt.id << "\n";
        mac->sendPacket(pkt);
    }

    void receivePacket(const Packet& pkt);
};

class PDCP_Layer {
private:
    std::shared_ptr<RLC_Layer> rlc;

public:
    void setRLC(const std::shared_ptr<RLC_Layer>& r) {
        rlc = r;
    }

    void sendPacket(const Packet& pkt) {
        std::cout << "[PDCP] Sending packet ID: " << pkt.id << "\n";
        rlc->sendPacket(pkt);
    }

    void receivePacket(const Packet& pkt) {
        std::cout << "[PDCP] Received packet ID: " << pkt.id << "\n";
    }
};

void RLC_Layer::receivePacket(const Packet& pkt) {
    std::cout << "[RLC] Received packet ID: " << pkt.id << "\n";
    if (pdcp) pdcp->receivePacket(pkt);
}

void MAC_Layer::sendPacket(Packet pkt) {
    if (buffer.size() >= maxBufferSize) {
        std::cout << "[MAC] Buffer full! Dropping packet ID: " << pkt.id << "\n";
        return;
    }

    buffer.push_back(pkt);

    while (!buffer.empty()) {
        Packet current = buffer.front();
        buffer.pop_front();

        std::cout << "[MAC] Transmitting packet ID: " << current.id << " (Try #" << current.retries + 1 << ")\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        if (dis(gen) < 0.8) {
            std::cout << "[MAC] Success for packet ID: " << current.id << "\n";
            rlc->receivePacket(current);
        } else {
            std::cout << "[MAC] Failure for packet ID: " << current.id << "\n";
            current.retries++;
            if (current.retries < 3)
                buffer.push_back(current);
            else
                std::cout << "[MAC] Dropped after max retries: " << current.id << "\n";
        }
    }
}

void producerFunction(Direction dir, int producerId) {
    while (simulationRunning) {
        Packet pkt(1200 + rand() % 400, rand() % 10, dir);
        std::cout << "[Producer-" << producerId << "] Created packet ID: " << pkt.id << "\n";

        if (dir == Direction::Uplink)
            uplinkQueue.push(pkt);
        else
            downlinkQueue.push(pkt);

        std::this_thread::sleep_for(std::chrono::milliseconds(100 + rand() % 100));
    }
}

void consumerFunction(MAC_Layer& mac, Direction dir, int consumerId) {
    while (simulationRunning || !uplinkQueue.empty() || !downlinkQueue.empty()) {
        Packet pkt(0, 0, dir);
        bool gotPacket = false;

        if (dir == Direction::Uplink)
            gotPacket = uplinkQueue.pop(pkt);
        else
            gotPacket = downlinkQueue.pop(pkt);

        if (gotPacket) {
            std::cout << "[Consumer-" << consumerId << "] Processing packet ID: " << pkt.id << "\n";
            mac.sendPacket(pkt);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            if (!simulationRunning) break;
        }
    }
    std::cout << "[Consumer-" << consumerId << "] Exiting.\n";
}

int main() {
    auto mac = std::make_shared<MAC_Layer>();
    auto rlc = std::make_shared<RLC_Layer>();
    auto pdcp = std::make_shared<PDCP_Layer>();

    mac->setRLC(rlc);
    rlc->setMAC(mac);
    rlc->setPDCP(pdcp);
    pdcp->setRLC(rlc);

    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;

    for (int i = 0; i < 2; ++i)
        producers.emplace_back(producerFunction, Direction::Uplink, i);
    for (int i = 0; i < 1; ++i)
        producers.emplace_back(producerFunction, Direction::Downlink, i + 2);

    for (int i = 0; i < 2; ++i)
        consumers.emplace_back(consumerFunction, std::ref(*mac), Direction::Uplink, i);

    std::this_thread::sleep_for(std::chrono::seconds(10));
    simulationRunning = false;

    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();

    std::cout << "Simulation finished.\n";
    return 0;
}
