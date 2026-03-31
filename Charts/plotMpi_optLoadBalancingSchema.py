import matplotlib
matplotlib.use('Agg') # Necessary for headless server environments

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_highlighted_summa_diagram():
    # Set up the figure size and axis
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 19) 
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Configuration: using 3x3 blocks to represent local sub-matrices
    n = 3
    grey = '#e0e0e0'
    
    # Matching colors for the dot product elements
    highlight_colors = ['#ffadad', '#9bf6ff', '#bdb2ff'] # Red, Cyan, Purple
    c_highlight = '#caffbf' # Green for the result

    start_y = 2
    center_y = start_y + (n / 2) # Vertical center of the blocks

    # --- 1. Draw Local Panel A ---
    start_x_A = 1.5
    for i in range(n):
        for j in range(n):
            # Color only the first row (i == 0)
            facecolor = highlight_colors[j] if i == 0 else grey
            
            # Red outline for A_{0,1}
            edgecolor = 'red' if (i == 0 and j == 1) else 'black'
            linewidth = 3 if (i == 0 and j == 1) else 2
            
            rect = patches.Rectangle((start_x_A + j, (start_y + n - 1) - i), 1, 1, 
                                     linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, zorder=2 if (i==0 and j==1) else 1)
            ax.add_patch(rect)
            ax.text(start_x_A + j + 0.5, (start_y + n - 0.5) - i, f'$A_{{{i},{j}}}$', ha='center', va='center', fontsize=12, zorder=3)

    # --- OPERATOR: Multiplication ---
    ax.text(6.0, center_y, r'$\ast$', fontsize=30, ha='center', va='center')

    # --- 2. Draw Local Panel B ---
    start_x_B = 7.5
    for i in range(n):
        for j in range(n):
            # Color only the second column (j == 1)
            facecolor = highlight_colors[i] if j == 1 else grey
            
            # Red outline for B_{0,1}
            edgecolor = 'red' if (i == 0 and j == 1) else 'black'
            linewidth = 3 if (i == 0 and j == 1) else 2
            
            rect = patches.Rectangle((start_x_B + j, (start_y + n - 1) - i), 1, 1, 
                                     linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, zorder=2 if (i==0 and j==1) else 1)
            ax.add_patch(rect)
            ax.text(start_x_B + j + 0.5, (start_y + n - 0.5) - i, f'$B_{{{i},{j}}}$', ha='center', va='center', fontsize=12, zorder=3)

    # --- OPERATOR: Accumulation ---
    ax.text(12.25, center_y, r'$=$', fontsize=26, ha='center', va='center')

    # --- 3. Draw Local Block C ---
    start_x_C = 14.0
    for i in range(n):
        for j in range(n):
            # Color only the intersection: Row 0 and Col 1
            facecolor = c_highlight if (i == 0 and j == 1) else grey
            
            # Red outline for C_{0,1}
            edgecolor = 'red' if (i == 0 and j == 1) else 'black'
            linewidth = 3 if (i == 0 and j == 1) else 2
            
            rect = patches.Rectangle((start_x_C + j, (start_y + n - 1) - i), 1, 1, 
                                     linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, zorder=2 if (i==0 and j==1) else 1)
            ax.add_patch(rect)
            ax.text(start_x_C + j + 0.5, (start_y + n - 0.5) - i, f'$C_{{{i},{j}}}$', ha='center', va='center', fontsize=12, zorder=3)

    plt.tight_layout()
    output_filename = 'summa_dot_product_highlight.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Done! Highlighted diagram saved as {output_filename}")

if __name__ == "__main__":
    generate_highlighted_summa_diagram()