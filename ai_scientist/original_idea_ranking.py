import json
import csv

def create_simple_ranking(json_file_path, output_csv="idea_scores.csv"):
    """
    Creates a simple two-column CSV with idea_id and combined_score
    
    Args:
        json_file_path: Path to the JSON file with ideas
        output_csv: Path to save the CSV output
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Calculate combined score for each idea
    idea_scores = []
    for item in data:
        # Sum the three scores
        combined_score = item.get('Interestingness', 0) + item.get('Feasibility', 0) + item.get('Novelty', 0)
        
        idea_scores.append({
            'idea_id': item['Name'],
            'combined_score': combined_score
        })
    
    # Sort by combined score (descending)
    idea_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idea_id', 'combined_score'])
        
        for idea in idea_scores:
            writer.writerow([idea['idea_id'], idea['combined_score']])
    
    print(f"Simple ranking saved to {output_csv}")
    return idea_scores

if __name__ == "__main__":
    # Change this to your actual file path
    json_file_path = '/home/huang717/CS673-AI-Scientist/templates/ppo_with_folds/ideas.json'
    output_csv = "original_idea_ranking_by_scores.csv"
    
    create_simple_ranking(json_file_path, output_csv)