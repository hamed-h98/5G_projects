/*
================================================================================
  5G MAC Scheduler Emulator with HARQ & Proportional Fair Logic (C++)
================================================================================

Overview:
---------
This emulator simulates the behavior of a 5G Medium Access Control (MAC) layer
scheduler under high user traffic with realistic channel conditions. It models:

  • Multiple UEs (users) each with their own packet buffer
  • Proportional Fair (PF) scheduling algorithm to balance throughput and fairness
  • HARQ-like retransmission logic (up to 3 retries per packet)
  • Packet-level queue logging for time-slot-based analysis
  • Output to CSV for real-time visualization with Flask/matplotlib (Python)

Key Components:
---------------
  • Packet         – encapsulates payload, user ID, and retry state
  • User           – maintains buffer and average throughput
  • Scheduler      – applies PF logic and simulates channel success/failure
  • Logging        – outputs queue length and total bytes to 'queue_state.csv'

Usage:
------
  - Add users with `addUser()`
  - Enqueue packets with `enqueuePacket()`
  - Call `simulate(N)` to process N time slots
  - Plot the results using the companion Python Flask app

Author: Hamed Hosseinnejad
Date: May 2025
*/

#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>

struct Packet {
    int id;
    int userId;
    int sizeBytes;
    int retries;
    bool acked;
    std::chrono::steady_clock::time_point arrivalTime;

    Packet(int uid, int sz)
        : id(globalId++), userId(uid), sizeBytes(sz), retries(0), acked(false),
          arrivalTime(std::chrono::steady_clock::now()) {}

    static inline int globalId = 0;
};

struct User {
    int id;
    double avgThroughput = 1e-6;  // Initialize to avoid divide-by-zero
    std::queue<Packet> buffer;
};

class Scheduler {
private:
    std::unordered_map<int, User> users;
    std::default_random_engine gen;
    std::uniform_real_distribution<double> channelSim{0.0, 1.0};

public:
    Scheduler() = default;

    void addUser(int uid) {
        users[uid] = User{uid};
    }

    void enqueuePacket(int uid, int size) {
        users[uid].buffer.emplace(uid, size);
    }

    int selectUserProportionalFair() {
        double bestScore = -1.0;
        int bestUser = -1;
        for (auto& [uid, user] : users) {
            if (!user.buffer.empty()) {
                double instRate = 1.0 + rand() % 10;  // Simulated
                double pfScore = instRate / user.avgThroughput;
                if (pfScore > bestScore) {
                    bestScore = pfScore;
                    bestUser = uid;
                }
            }
        }
        return bestUser;
    }

    void processSlot() {
        int uid = selectUserProportionalFair();
        if (uid == -1) return;

        User& user = users[uid];
        Packet& pkt = user.buffer.front();

        bool success = (channelSim(gen) > 0.2);  // 80% success rate
        if (success) {
            std::cout << "[Scheduler] User " << uid << " successfully sent packet " << pkt.id << "\n";
            user.buffer.pop();

            auto now = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(now - pkt.arrivalTime).count();
            user.avgThroughput = 0.9 * user.avgThroughput + 0.1 * (pkt.sizeBytes / duration);

        } else {
            pkt.retries++;
            if (pkt.retries >= 3) {
                std::cout << "[Scheduler] Dropping packet " << pkt.id << " after 3 retries\n";
                user.buffer.pop();
            } else {
                std::cout << "[Scheduler] NACK for packet " << pkt.id << ", retry " << pkt.retries << "\n";
            }
        }
    }

    void simulate(int slots) {
        std::ofstream log("queue_state.csv");
        log << "slot,user_id,queue_len,total_bytes\n";
    
        for (int t = 0; t < slots; ++t) {
            std::cout << "=== Time Slot " << t << " ===\n";
            processSlot();
            logQueueState(t, log);
        }
    
        log.close();
    }

    void logQueueState(int slot, std::ofstream& logFile) {
        for (const auto& [uid, user] : users) {
            size_t totalBytes = 0;
            std::queue<Packet> tmp = user.buffer;
    
            while (!tmp.empty()) {
                totalBytes += tmp.front().sizeBytes;
                tmp.pop();
            }
    
            logFile << slot << "," << uid << "," << user.buffer.size() << "," << totalBytes << "\n";
        }
    }
};

int main() {
    Scheduler scheduler;
    scheduler.addUser(1);
    scheduler.addUser(2);

    for (int i = 0; i < 5; ++i) scheduler.enqueuePacket(1, 1500);
    for (int i = 0; i < 5; ++i) scheduler.enqueuePacket(2, 1500);

    scheduler.simulate(20);  // Simulate 20 time slots
    return 0;
}
