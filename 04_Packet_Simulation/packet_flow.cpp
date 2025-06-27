#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <unordered_map>

// PDCP: Packet Data Convergence Protocol (header compression, ciphering)
// RLC: Radio Link Control (segmentation, reassembly, retransmission)
// MAC: Medium Access Control (scheduling, priority)

// PDCP Layer: Simulated ciphering / integrity pass-through.
// RLC Layer: Simulated segmentation pass-through.
// MAC Layer: Scheduling + simulated transmission, logging to CSV.

// Packets have a chance to be lost (radio errors).
// Max 3 retransmissions before packet is dropped.
// Uplink and Downlink queues in MAC layer 
// Separate queues simulate realistic radio interface behavior.
// HARQ-style retransmissions at RLC/MAC layer

// Packet structure
struct Packet {
    int id;
    int size;
    int priority;
    std::string direction; // Uplink or Downlink
    int retransmissionCount = 0;
    std::chrono::steady_clock::time_point arrivalTime;
};

// PDCP Layer
class PDCP {
public:
    Packet processPacket(const Packet& pkt) {
        std::cout << "[PDCP] Processing Packet " << pkt.id << "\n";
        return pkt;
    }
};

// RLC Layer with basic HARQ simulation
class RLC {
public:
    Packet processPacket(const Packet& pkt) {
        std::cout << "[RLC] Processing Packet " << pkt.id << "\n";
        Packet processedPkt = pkt;
        processedPkt.retransmissionCount++;
        return processedPkt;
    }
};

// MAC Layer with Uplink and Downlink queues and packet loss simulation
class MAC {
public:
    MAC() : lossProbability(0.1), bufferLimit(10) {}

    void enqueuePacket(const Packet& pkt) {
        std::lock_guard<std::mutex> lock(macMutex);
        auto& queue = (pkt.direction == "Uplink") ? uplinkQueue : downlinkQueue;

        if (queue.size() >= bufferLimit) {
            std::cout << "[MAC] Buffer overflow for " << pkt.direction << " Packet " << pkt.id << ". Dropping.\n";
            return;
        }

        queue.push(pkt);
        packetAvailable.notify_one();
    }

    void processPackets() {
        while (true) {
            std::unique_lock<std::mutex> lock(macMutex);
            packetAvailable.wait(lock, [&] { return !uplinkQueue.empty() || !downlinkQueue.empty() || done; });

            if (uplinkQueue.empty() && downlinkQueue.empty() && done) break;

            Packet pkt;
            if (!uplinkQueue.empty()) {
                pkt = uplinkQueue.front();
                uplinkQueue.pop();
            } else if (!downlinkQueue.empty()) {
                pkt = downlinkQueue.front();
                downlinkQueue.pop();
            } else {
                continue;
            }
            lock.unlock();

            auto now = std::chrono::steady_clock::now();
            auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - pkt.arrivalTime).count();

            // Simulate packet loss
            if (randomLoss() < lossProbability) {
                std::cout << "[MAC] Packet " << pkt.id << " lost during transmission. Retransmitting...\n";
                if (pkt.retransmissionCount < 3) {
                    enqueuePacket(pkt); // Retransmit
                } else {
                    std::cout << "[MAC] Packet " << pkt.id << " dropped after max retransmissions.\n";
                }
                continue;
            }

            std::cout << "[MAC] Transmitting Packet " << pkt.id << " with latency " << latency << "ms\n";

            {
                std::lock_guard<std::mutex> statsLock(statsMutex);
                packetsProcessed++;
                logFile << pkt.id << "," << pkt.priority << "," << pkt.size << "," << latency << "," << pkt.direction << "," << pkt.retransmissionCount << "\n";
            }
        }
    }

    void finish() {
        std::lock_guard<std::mutex> lock(macMutex);
        done = true;
        packetAvailable.notify_all();
    }

private:
    double lossProbability;
    size_t bufferLimit;
    std::queue<Packet> uplinkQueue;
    std::queue<Packet> downlinkQueue;
    std::mutex macMutex;
    std::condition_variable packetAvailable;
    bool done = false;

    double randomLoss() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(0.0, 1.0);
        return dis(gen);
    }
};

// Globals for logging and stats
std::mutex statsMutex;
std::ofstream logFile("packet_log.csv");
int packetsProcessed = 0;

// Simulate packet generator and pipeline
void packetPipeline(int totalPackets, MAC& macLayer) {
    PDCP pdcp;
    RLC rlc;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> sizeDist(64, 1500);
    std::uniform_int_distribution<> priorityDist(1, 5);
    std::uniform_int_distribution<> directionDist(0, 1);

    for (int i = 1; i <= totalPackets; ++i) {
        Packet pkt { i, sizeDist(gen), priorityDist(gen), directionDist(gen) ? "Uplink" : "Downlink", 0, std::chrono::steady_clock::now() };
        std::cout << "[App] Generating Packet " << pkt.id << "\n";

        Packet pdcpPkt = pdcp.processPacket(pkt);
        Packet rlcPkt = rlc.processPacket(pdcpPkt);

        macLayer.enqueuePacket(rlcPkt);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    macLayer.finish();
}

int main() {
    const int totalPackets = 50;

    logFile << "PacketID,Priority,Size,Latency(ms),Direction,Retransmissions\n";

    MAC macLayer;
    std::thread macThread(&MAC::processPackets, &macLayer);

    auto startTime = std::chrono::steady_clock::now();
    packetPipeline(totalPackets, macLayer);
    macThread.join();

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();

    std::cout << "Simulation complete!\n";
    std::cout << "Total packets processed: " << packetsProcessed << "\n";
    std::cout << "Total simulation time: " << duration << " seconds\n";
    std::cout << "Average processing rate: " << (packetsProcessed / (duration > 0 ? duration : 1)) << " packets/second\n";

    logFile.close();

    return 0;
}
