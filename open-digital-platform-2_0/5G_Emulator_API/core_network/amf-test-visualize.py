import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Original event data
data = [
    {"name": "POST /amf/ue/{ue_id} http receive", "start_time": 1728252892498409693, "end_time": 1728252892498451522},
    {"name": "POST /amf/ue/{ue_id} http send", "start_time": 1728252892499275899, "end_time": 1728252892499527757},
    {"name": "POST /amf/ue/{ue_id} http send", "start_time": 1728252892499567944, "end_time": 1728252892499656311},
    {"name": "POST /amf/ue/{ue_id}", "start_time": 1728252892497690226, "end_time": 1728252892499663635},
    {"name": "POST /amf/handover http receive", "start_time": 1728252983303056350, "end_time": 1728252983303073863},
    {"name": "ngap_handover_request_ack", "start_time": 1728252983303753685, "end_time": 1728252983314168732},
    {"name": "ngap_handover_request", "start_time": 1728252983303292519, "end_time": 1728252983314187377},
    {"name": "ngap_resource_setup", "start_time": 1728252983314239587, "end_time": 1728252983355134662},
    {"name": "ngap_handover_command", "start_time": 1728252983355214103, "end_time": 1728252983395974041},
    {"name": "POST /amf/handover http send", "start_time": 1728252983396824808, "end_time": 1728252983397342982},
    {"name": "POST /amf/handover http send", "start_time": 1728252983397417664, "end_time": 1728252983397579151},
    {"name": "POST /amf/handover", "start_time": 1728252983302913609, "end_time": 1728252983397591344}
]

# Create DataFrame
df = pd.DataFrame(data)
df['duration'] = df['end_time'] - df['start_time']
df = df.sort_values('start_time').reset_index(drop=True)

# Convert start and end times to relative times in milliseconds
min_time = df['start_time'].min()
df['start_time_rel'] = (df['start_time'] - min_time) / 1e6  # Convert to milliseconds
df['end_time_rel'] = (df['end_time'] - min_time) / 1e6
df['duration_ms'] = df['duration'] / 1e6

# Assign a unique row to each event
df['row'] = np.arange(len(df))

# Define a threshold for maximum allowed gap (in milliseconds)
threshold = 500  # Adjust this value as needed

# Initialize adjusted times
adjusted_start_times = []
adjusted_end_times = []
compressed_regions = []  # To keep track of where gaps were compressed
cumulative_compression = 0  # Total time compressed so far

# Iterate over events to adjust times
for i, row in df.iterrows():
    if i == 0:
        # First event starts at time zero
        adjusted_start_time = 0
    else:
        # Calculate the actual gap from the previous event
        actual_gap = row['start_time_rel'] - df.loc[i - 1, 'end_time_rel']
        if actual_gap > threshold:
            # Compress the gap to the threshold
            cumulative_compression += actual_gap - threshold
            gap = threshold
            # Record the compressed region
            compressed_regions.append({
                'start': adjusted_end_times[-1],
                'end': adjusted_end_times[-1] + gap,
                'actual_gap_start': df.loc[i - 1, 'end_time_rel'],
                'actual_gap_end': row['start_time_rel'],
            })
        else:
            gap = actual_gap
        adjusted_start_time = adjusted_end_times[-1] + gap
    # Adjust start and end times
    adjusted_end_time = adjusted_start_time + row['duration_ms']
    adjusted_start_times.append(adjusted_start_time)
    adjusted_end_times.append(adjusted_end_time)

# Add adjusted times to the dataframe
df['adjusted_start_time'] = adjusted_start_times
df['adjusted_end_time'] = adjusted_end_times

# Plotting with adjusted times
fig, ax = plt.subplots(figsize=(15, 8))

bar_height = 0.8

# Assign colors to each event
colors = plt.cm.tab20(np.linspace(0, 1, len(df)))

# Plot the timeline with adjusted times
for i, row in df.iterrows():
    ax.barh(row['row'], row['adjusted_end_time'] - row['adjusted_start_time'],
            left=row['adjusted_start_time'], height=bar_height,
            align='center', color=colors[i], alpha=0.8)

# Indicate compressed gaps on the time axis
for region in compressed_regions:
    # Draw a dashed line to represent compressed gap
    ax.plot([region['start'], region['end']], [-1, -1], color='black', linestyle='--', linewidth=1)
    # Add a text annotation to indicate the compression
    ax.text((region['start'] + region['end']) / 2, -1.5, 'Gap Compressed',
            ha='center', va='top', fontsize=8, rotation=0)

# Customize the plot
ax.set_title('Event Timeline with Compressed Gaps', fontsize=20, fontweight='bold')
ax.set_xlabel('Adjusted Time (milliseconds)', fontsize=14)
ax.set_ylabel('Events', fontsize=14)

# Set y-axis labels to event names
ax.set_yticks(df['row'])
ax.set_yticklabels(df['name'], fontsize=10)

# Adjust x-axis limits and labels
ax.set_xlim(-100, df['adjusted_end_time'].max() + 100)  # Add some padding
ax.set_xticks(np.linspace(0, df['adjusted_end_time'].max(), 10))
ax.set_xticklabels([f'{x:.0f}' for x in np.linspace(0, df['adjusted_end_time'].max(), 10)])

# Add grid lines for better readability
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Add duration labels inside the bars
for i, row in df.iterrows():
    middle = row['adjusted_start_time'] + (row['adjusted_end_time'] - row['adjusted_start_time']) / 2
    ax.text(middle, row['row'], f'{row["duration_ms"]:.1f} ms',
            ha='center', va='center', fontsize=9, color='white')

# Adjust layout
plt.tight_layout()
plt.savefig("event_timeline_compressed_gaps.png", dpi=300, bbox_inches='tight')
plt.show()
