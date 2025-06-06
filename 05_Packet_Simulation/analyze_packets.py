import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV log
df = pd.read_csv('packet_log.csv')

# Basic info
print("Summary of packets:")
print(df.describe())
print("\nFirst few packets:")
print(df.head())

# Distribution of Packet Sizes
plt.figure(figsize=(8, 4))
sns.histplot(df['Size'], kde=True, bins=20)
plt.title('Packet Size Distribution')
plt.xlabel('Packet Size (bytes)')
plt.ylabel('Count')
plt.show()

# Distribution of Packet Priorities
plt.figure(figsize=(8, 4))
sns.countplot(x='Priority', data=df)
plt.title('Packet Priority Distribution')
plt.xlabel('Priority')
plt.ylabel('Count')
plt.show()

# Latency Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['Latency(ms)'], kde=True, bins=20)
plt.title('Packet Latency Distribution')
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.show()

# Uplink vs Downlink Latency Comparison
plt.figure(figsize=(8, 4))
sns.boxplot(x='Direction', y='Latency(ms)', data=df)
plt.title('Uplink vs Downlink Latency')
plt.ylabel('Latency (ms)')
plt.show()

# Time Series Latency Plot
plt.figure(figsize=(10, 5))
plt.plot(df['PacketID'], df['Latency(ms)'], marker='o')
plt.title('Latency over Packets Processed')
plt.xlabel('Packet ID')
plt.ylabel('Latency (ms)')
plt.grid(True)
plt.show()
