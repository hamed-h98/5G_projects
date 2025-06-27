import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV log
df = pd.read_csv('packet_log.csv')

# Show basic info
print("Summary of packets:")
print(df.describe())
print("\nFirst few packets:")
print(df.head())

# Set seaborn style
sns.set(style="whitegrid")

# Packet Size Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Size'], kde=True, bins=20)
plt.title('Packet Size Distribution')
plt.xlabel('Packet Size (bytes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("packet_size_distribution.png")
plt.show()

# Packet Priority Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Priority', data=df)
plt.title('Packet Priority Distribution')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("packet_priority_distribution.png")
plt.show()

# Latency Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Latency(ms)'], kde=True, bins=20)
plt.title('Packet Latency Distribution')
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("latency_distribution.png")
plt.show()

# Uplink vs Downlink Latency
plt.figure(figsize=(8, 4))
sns.boxplot(x='Direction', y='Latency(ms)', data=df)
plt.title('Uplink vs Downlink Latency')
plt.ylabel('Latency (ms)')
plt.tight_layout()
plt.savefig("uplink_downlink_latency.png")
plt.show()

# Retransmissions Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='Retransmissions', data=df)
plt.title('Retransmissions Count Distribution')
plt.xlabel('Number of Retransmissions')
plt.ylabel('Packet Count')
plt.tight_layout()
plt.savefig("retransmissions_distribution.png")
plt.show()

# Throughput over time (packets processed)
plt.figure(figsize=(10, 5))
df_sorted = df.sort_values('PacketID')
plt.plot(df_sorted['PacketID'], range(1, len(df_sorted) + 1), marker='o')
plt.title('Throughput over PacketID')
plt.xlabel('Packet ID')
plt.ylabel('Cumulative Packets Processed')
plt.grid(True)
plt.tight_layout()
plt.savefig("throughput_over_time.png")
plt.show()

print("All plots saved as PNG files!")
