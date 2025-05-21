#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>

// simulation of packets arriving, stored in a queue, and processed by worker threads:

// Producer-Consumer model (packet generator and packet processor)
// Threads for parallel simulation
// Mutexes and condition variables for safe queue handling
// c++20

// One producer thread creates packets at random sizes and pushes them into a thread-safe queue.
// Two consumer threads process packets from the queue in parallel.
// Synchronization is handled by mutex + condition_variable to prevent race conditions.
// Graceful shutdown when the producer finishes all packets.


// Packet structure
struct Packet {
    int id;
    int size; // Size in bytes
};

std::queue<Packet> packetQueue;
std::mutex queueMutex;
std::condition_variable packetAvailable;
bool doneProducing = false;

void packetProducer(int totalPackets) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> sizeDist(64, 1500); // Typical Ethernet frame sizes

    for (int i = 1; i <= totalPackets; ++i) {
        Packet pkt { i, sizeDist(gen) };

        {
            std::lock_guard<std::mutex> lock(queueMutex);
            packetQueue.push(pkt);
            std::cout << "[Producer] Packet " << pkt.id << " produced (size: " << pkt.size << " bytes)\n";
        }

        packetAvailable.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // simulate packet arrival interval
    }

    // Signal consumer threads to finish
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        doneProducing = true;
    }
    packetAvailable.notify_all();
}

void packetConsumer(int consumerId) {
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        packetAvailable.wait(lock, [] {
            return !packetQueue.empty() || doneProducing;
        });

        if (!packetQueue.empty()) {
            Packet pkt = packetQueue.front();
            packetQueue.pop();
            lock.unlock(); // Unlock early to allow producer to add more packets

            std::cout << "[Consumer " << consumerId << "] Processing packet " << pkt.id << " (size: " << pkt.size << " bytes)\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(150)); // simulate processing time

        } else if (doneProducing) {
            break; // No more packets will arrive, exit thread
        }
    }

    std::cout << "[Consumer " << consumerId << "] Exiting.\n";
}

int main() {
    const int totalPackets = 10;
    const int consumerThreads = 2;

    std::thread producer(packetProducer, totalPackets);
    std::vector<std::thread> consumers;

    for (int i = 0; i < consumerThreads; ++i) {
        consumers.emplace_back(packetConsumer, i + 1);
    }

    producer.join();
    for (auto& consumer : consumers) {
        consumer.join();
    }

    std::cout << "Simulation complete!\n";
    return 0;
}
