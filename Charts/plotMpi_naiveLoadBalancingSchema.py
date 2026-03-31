import matplotlib
matplotlib.use('Agg') # Necessary for headless server environments

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_minimal_mpi_diagram_with_operators():
    # Set up the figure size and axis
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 19) # Slightly wider to accommodate the spacing
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Configuration
    n = 4
    colors = {'A': '#ffadad', 'B': '#9bf6ff', 'C': '#caffbf'}
    start_y = 3
    center_y = start_y + 0.5 # Vertical center of the boxes

    # --- 1. Draw Row Vector A ---
    for j in range(n):
        rect = patches.Rectangle((1 + j, start_y), 1, 1, linewidth=2, edgecolor='black', facecolor=colors['A'])
        ax.add_patch(rect)
        ax.text(1.5 + j, center_y, f'$A_{{0,{j}}}$', ha='center', va='center', fontsize=12)

    # --- OPERATOR: Multiplication ---
    # Centered between A (ends at x=5) and B (starts at x=7.5)
    ax.text(6.25, center_y, r'$\ast$', fontsize=30, ha='center', va='center')

    # --- 2. Draw Matrix B ---
    for i in range(n):
        for j in range(n):
            rect = patches.Rectangle((7.5 + j, (start_y + 1.5) - i), 1, 1, 
                                     linewidth=2, edgecolor='black', facecolor=colors['B'])
            ax.add_patch(rect)
            ax.text(8 + j, (start_y + 2) - i, f'$B_{{{i},{j}}}$', ha='center', va='center', fontsize=11)

    # --- OPERATOR: Equals ---
    # Centered between B (ends at x=11.5) and C (starts at x=14)
    ax.text(12.75, center_y, r'$=$', fontsize=30, ha='center', va='center')

    # --- 3. Draw Row Vector C ---
    for j in range(n):
        rect = patches.Rectangle((14 + j, start_y), 1, 1, linewidth=2, edgecolor='black', facecolor=colors['C'])
        ax.add_patch(rect)
        ax.text(14.5 + j, center_y, f'$C_{{0,{j}}}$', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    output_filename = 'mpi_multiplication_operators.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Done! Image with operators saved as {output_filename}")

if __name__ == "__main__":
    generate_minimal_mpi_diagram_with_operators()