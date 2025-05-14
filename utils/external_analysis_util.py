import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_disease_rankings(stats, output_dir):
    """
    Create and save visualizations for disease ranking statistics
    
    Args:
        stats (dict): Dictionary of disease statistics from analyze_disease_rankings
        output_dir (str): Directory to save figures, default is './figures/'
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    data_rows = []
    for position, diseases in stats.items():
        for disease, count in diseases.items():
            data_rows.append({
                'Position': position,
                'Disease': disease,
                'Count': count
            })
    df = pd.DataFrame(data_rows)
    
    plt.figure(figsize=(15, 8))
    heatmap_data = df.pivot_table(
        values='Count', 
        index='Disease',
        columns='Position', 
        fill_value=0
    )
    
    plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Count')
    
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            plt.text(j, i, int(heatmap_data.iloc[i, j]),
                    ha='center', va='center')
    
    plt.title('Disease Distribution Across Positions')
    plt.xlabel('Position')
    plt.ylabel('Disease')
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()  

    plt.figure(figsize=(15, 8))
    pivot_data = df.pivot_table(
        values='Count',
        index='Position',
        columns='Disease',
        fill_value=0
    )
    pivot_data.plot(kind='bar', stacked=True)
    plt.title('Disease Distribution at Each Position')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stacked_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    top_diseases = df.groupby('Disease')['Count'].sum().nlargest(5).index
    plt.figure(figsize=(15, 8))
    for disease in top_diseases:
        disease_data = df[df['Disease'] == disease]
        plt.plot(disease_data['Position'], 
                disease_data['Count'], 
                marker='o', 
                label=disease)
    
    plt.title('Position Distribution of Top 5 Diseases')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'top5_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    total_counts = df.groupby('Disease')['Count'].sum()
    plt.pie(total_counts, 
            labels=total_counts.index,
            autopct='%1.1f%%',
            radius=1,
            wedgeprops=dict(width=0.5))
    plt.title('Overall Disease Distribution')
    plt.savefig(os.path.join(output_dir, 'pie_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All figures have been saved to {output_dir}")

def analyze_disease_rankings(data_list):
    """
    Analyze disease rankings from a list of formatted strings.
    
    Args:
        data_list (list): List of strings, where each string contains rankings like:
            "POTENTIAL DISEASES:\n1. Disease A\n2. Disease B\n...\n10. Disease C"
            
    Returns:
        dict: Dictionary where keys are positions (1-10) and values are 
            dictionaries of disease counts at that position
    """
    # Initialize the statistics dictionary
    stats = {i: {} for i in range(1, 11)}  # Positions 1-10
    
    for case in data_list:
        if not case.strip():  # Skip empty strings
            continue
            
        # Split into lines and process each line
        lines = case.strip().split("\n")
        
        for line in lines:
            line = line.strip()
            if not line or line == "POTENTIAL DISEASES:":  # Skip header and empty lines
                continue
                
            # Extract position and disease name
            try:
                # Split on first period and any following whitespace
                position_str, disease = line.split(".", 1)
                position = int(position_str)
                disease = disease.strip()
                
                if disease:  # Only count if disease name is not empty
                    # Update count for this disease at this position
                    stats[position][disease] = stats[position].get(disease, 0) + 1
            except (ValueError, IndexError):
                continue
    
    return stats

