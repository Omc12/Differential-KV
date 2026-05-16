import matplotlib.pyplot as plt

def plot_latency_waterfall(events, save_path):
    """
    PHASE 9: Kernel Latency Waterfalls
    Visualizes the sequence and duration of kernels in a single token step.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, e in enumerate(events):
        ax.barh(i, e['duration'], left=e['start'], label=e['name'])
    
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels([e['name'] for e in events])
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)")
    ax.set_title("Kernel Execution Waterfall")
    plt.savefig(save_path)
    plt.close()
