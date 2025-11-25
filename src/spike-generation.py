# Explore spike generation with increasing currents
I_values = np.linspace(0, 20, 5)
T = 100
n_steps = int(T / dt)

fig, axes = plt.subplots(len(I_values), 1, figsize=(12, 10), sharex=True)

for idx, I_amp in enumerate(I_values):
    hh_neuron.reset()
    V_trace = []
    t_trace = []
    
    for step in range(n_steps):
        t = step * dt
        hh_neuron.step(dt, I_amp)
        V_trace.append(hh_neuron.V)
        t_trace.append(t)
    
    axes[idx].plot(t_trace, V_trace, linewidth=1.5)
    axes[idx].set_ylabel('V (mV)')
    axes[idx].set_title(f'I = {I_amp:.1f} µA/cm²', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()