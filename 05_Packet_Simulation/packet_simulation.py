import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
df = pd.read_csv('packet_log.csv')

# Show first few rows
print(df.head())

# --- Basic Stats ---
total_packets = len(df)
uplink_packets = df[df['Direction'] == 'Uplink']
downlink_packets = df[df['Direction'] == 'Downlink']
average_latency = df['Latency(ms)'].mean()
max_retransmissions = df['Retransmissions'].max()

print(f"\nTotal Packets Processed: {total_packets}")
print(f"Uplink Packets: {len(uplink_packets)}")
print(f"Downlink Packets: {len(downlink_packets)}")
print(f"Average Latency: {average_latency:.2f} ms")
print(f"Max Retransmissions: {max_retransmissions}")

# --- Plot 1: Uplink vs Downlink Packet Count ---
plt.figure(figsize=(6,4))
df['Direction'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Uplink vs Downlink Packet Count')
plt.xlabel('Direction')
plt.ylabel('Number of Packets')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Plot 2: Packet Latency Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(df['Latency(ms)'], bins=20, kde=True, color='purple')
plt.title('Packet Latency Distribution')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Plot 3: Retransmissions Count Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x='Retransmissions', data=df, palette='Set2')
plt.title('Retransmissions Count Distribution')
plt.xlabel('Number of Retransmissions')
plt.ylabel('Packet Count')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# --- Plot 4: Latency by Direction ---
plt.figure(figsize=(8,5))
sns.boxplot(x='Direction', y='Latency(ms)', data=df, palette='Set3')
plt.title('Latency by Direction (Uplink vs Downlink)')
plt.xlabel('Direction')
plt.ylabel('Latency (ms)')
plt.tight_layout()
plt.show()

# --- Plot 5: Priority vs Latency ---
plt.figure(figsize=(8,5))
sns.boxplot(x='Priority', y='Latency(ms)', data=df, palette='coolwarm')
plt.title('Packet Priority vs Latency')
plt.xlabel('Packet Priority')
plt.ylabel('Latency (ms)')
plt.tight_layout()
plt.show()
