#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>


// multi-threaded simulator of the 5G Layer-2 stack, focusing on PDCP, RLC, and MAC layers.

// The simulator creates packets with random priorities and directions (uplink or downlink), 
// manages them in separate priority queues, and uses multiple threads to process them in parallel. 
// It includes key telecom mechanisms like HARQ retransmissions, buffer overflow handling, and packet loss simulation. 
// Packet processing decisions follow proportional fair scheduling, ensuring higher priority packets are served first.

// Finally, it logs detailed packet-level statistics to a CSV file, including latency and retransmission counts, 
// so I can analyze system performance after the simulation.


// Packet structure
struct Packet {
    int id;
    int size; // bytes
    int priority; // 1–5 (higher = better)
    int retransmissions = 0; // HARQ
    std::chrono::steady_clock::time_point arrivalTime;
};

// Comparator for priority queue (max-heap)
struct PacketComparator {
    bool operator()(const Packet& a, const Packet& b) {
        return a.priority < b.priority;
    }
};

// Global queues: Simulate RLC/MAC buffers
const size_t BUFFER_LIMIT = 10; // buffer overflow limit

std::priority_queue<Packet, std::vector<Packet>, PacketComparator> uplinkQueue;
std::priority_queue<Packet, std::vector<Packet>, PacketComparator> downlinkQueue;

std::mutex uplinkMutex, downlinkMutex, statsMutex;
std::condition_variable packetAvailable;

// Simulation flags and stats
bool doneProducing = false;
int packetsProcessed = 0;
int packetsLost = 0;
int packetsRetransmitted = 0;
std::ofstream logFile("packet_log.csv");

// Random number generators
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> packetLossDist(0.0, 1.0);
std::uniform_int_distribution<> sizeDist(64, 1500);
std::uniform_int_distribution<> priorityDist(1, 5);
std::uniform_int_distribution<> directionDist(0, 1);

void packetProducer(int totalPackets) {
    for (int i = 1; i <= totalPackets; ++i) {
        Packet pkt { i, sizeDist(gen), priorityDist(gen), 0, std::chrono::steady_clock::now() };
        bool uplink = directionDist(gen);

        if (uplink) {
            std::lock_guard<std::mutex> lock(uplinkMutex);
            if (uplinkQueue.size() < BUFFER_LIMIT) {
                uplinkQueue.push(pkt);
                std::cout << "[PDCP] Uplink Packet " << pkt.id << " generated.\n";
            } else {
                std::cout << "[PDCP] Uplink Packet " << pkt.id << " dropped (buffer overflow).\n";
                std::lock_guard<std::mutex> statsLock(statsMutex);
                ++packetsLost;
            }
        } else {
            std::lock_guard<std::mutex> lock(downlinkMutex);
            if (downlinkQueue.size() < BUFFER_LIMIT) {
                downlinkQueue.push(pkt);
                std::cout << "[PDCP] Downlink Packet " << pkt.id << " generated.\n";
            } else {
                std::cout << "[PDCP] Downlink Packet " << pkt.id << " dropped (buffer overflow).\n";
                std::lock_guard<std::mutex> statsLock(statsMutex);
                ++packetsLost;
            }
        }

        packetAvailable.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate packet arrival rate
    }

    {
        std::lock_guard<std::mutex> uplinkLock(uplinkMutex);
        std::lock_guard<std::mutex> downlinkLock(downlinkMutex);
        doneProducing = true;
    }
    packetAvailable.notify_all();
}

void processPacket(Packet pkt, const std::string& direction, int consumerId) {
    auto now = std::chrono::steady_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - pkt.arrivalTime).count();

    // Simulate random packet loss
    bool packetLost = (packetLossDist(gen) < 0.1); // 10% chance of packet loss

    if (packetLost) {
        if (pkt.retransmissions < 3) {
            Packet retransmitPkt = pkt;
            retransmitPkt.retransmissions++;
            retransmitPkt.arrivalTime = std::chrono::steady_clock::now(); // Update timestamp

            std::cout << "[RLC] Packet " << pkt.id << " (" << direction << ") lost. Retransmitting (attempt " << retransmitPkt.retransmissions << ").\n";
            {
                std::lock_guard<std::mutex> statsLock(statsMutex);
                ++packetsRetransmitted;
            }

            if (direction == "Uplink") {
                std::lock_guard<std::mutex> lock(uplinkMutex);
                uplinkQueue.push(retransmitPkt);
            } else {
                std::lock_guard<std::mutex> lock(downlinkMutex);
                downlinkQueue.push(retransmitPkt);
            }

            packetAvailable.notify_all();
        } else {
            std::cout << "[MAC] Packet " << pkt.id << " (" << direction << ") lost after " << pkt.retransmissions << " retransmissions.\n";
            std::lock_guard<std::mutex> statsLock(statsMutex);
            ++packetsLost;
        }
        return;
    }


    if (packetLost) {
        std::cout << "[MAC] Packet " << pkt.id << " (" << direction << ") lost after " << pkt.retransmissions << " retransmissions.\n";
        std::lock_guard<std::mutex> statsLock(statsMutex);
        ++packetsLost;
    } else {
        std::cout << "[MAC] Consumer " << consumerId << " processed " << direction << " Packet " << pkt.id
                  << " (priority: " << pkt.priority
                  << ", size: " << pkt.size
                  << ", latency: " << latency
                  << "ms, retransmissions: " << pkt.retransmissions << ")\n";

        std::lock_guard<std::mutex> lock(statsMutex);
        ++packetsProcessed;
        logFile << pkt.id << "," << direction << "," << pkt.priority << "," << pkt.size
                << "," << latency << "," << pkt.retransmissions << "\n";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(150)); // Simulate processing time
}

void proportionalFairScheduler(int consumerId) {
    while (true) {
        // Step 1: Wait until packets are available or production is done
        {
            std::unique_lock<std::mutex> lock(statsMutex);
            packetAvailable.wait(lock, [] {
                return !uplinkQueue.empty() || !downlinkQueue.empty() || doneProducing;
            });
            // Lock is released at the end of this block
        }

        // Step 2: Select a packet to process (without holding statsMutex)
        Packet pkt;
        std::string direction;
        bool hasPacket = false;

        {
            std::lock_guard<std::mutex> uplinkLock(uplinkMutex);
            std::lock_guard<std::mutex> downlinkLock(downlinkMutex);

            if (!uplinkQueue.empty() && (downlinkQueue.empty() || uplinkQueue.top().priority >= downlinkQueue.top().priority)) {
                pkt = uplinkQueue.top();
                uplinkQueue.pop();
                direction = "Uplink";
                hasPacket = true;
            } else if (!downlinkQueue.empty()) {
                pkt = downlinkQueue.top();
                downlinkQueue.pop();
                direction = "Downlink";
                hasPacket = true;
            }
        }

        // Step 3: Process the packet (outside any locks)
        if (hasPacket) {
            processPacket(pkt, direction, consumerId);
        } else if (doneProducing) {
            break; // No more packets and production is done
        }
    }

    std::cout << "[Scheduler Consumer " << consumerId << "] Exiting.\n";
}


int main() {
    const int totalPackets = 50;
    unsigned int coreCount = std::thread::hardware_concurrency();
    int schedulerThreads = coreCount > 1 ? coreCount - 1 : 1;

    std::cout << "Detected CPU cores: " << coreCount << "\n";
    std::cout << "Using scheduler threads: " << schedulerThreads << "\n";

    logFile << "PacketID,Direction,Priority,Size,Latency(ms),Retransmissions\n";

    auto startTime = std::chrono::steady_clock::now();

    std::thread producer(packetProducer, totalPackets);

    std::vector<std::thread> schedulers;
    for (int i = 0; i < schedulerThreads; ++i) {
        schedulers.emplace_back(proportionalFairScheduler, i + 1);
    }

    producer.join();
    for (auto& scheduler : schedulers) scheduler.join();

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

    std::cout << "\n=== Simulation Summary ===\n";
    std::cout << "Total packets processed: " << packetsProcessed << "\n";
    std::cout << "Packets lost: " << packetsLost << "\n";
    std::cout << "Packets retransmitted (HARQ): " << packetsRetransmitted << "\n";
    std::cout << "Total simulation time: " << duration << " seconds\n";
    std::cout << "Average processing rate: " << (packetsProcessed / (duration > 0 ? duration : 1)) << " packets/second\n";

    logFile.close();
    return 0;
}
