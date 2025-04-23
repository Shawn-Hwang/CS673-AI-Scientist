#!/bin/bash

# Define the arrays for temperature and index values
temps=(0.75 2.0)
indices=(1 2 3)

# Nested loop to run through all combinations
for i in "${temps[@]}"; do
  for j in "${indices[@]}"; do
    # First run without --use_personas flag
    echo "Running with temperature=$i and index=$j WITHOUT personas"
    
    # Construct the filename for baseline (no personas)
    baseline_filename="claude_temp_${i}_baseline_ideas${j}.json"
    
    # Run the Python command WITHOUT --use_personas
    python3 main.py --experiment "ppo_with_folds" --num_ideas 8 --skip_lit_review --use_genai --temperature $i --idea_file "$baseline_filename"
    
    echo "Completed run with temperature=$i and index=$j WITHOUT personas"
    echo "------------------------"
    
    # Second run WITH --use_personas flag
    echo "Running with temperature=$i and index=$j WITH personas"
    
    # Construct the filename for persona version
    persona_filename="claude_temp_${i}_persona_ideas${j}.json"
    
    # Run the Python command WITH --use_personas and different filename
    python3 main.py --experiment "ppo_with_folds" --num_ideas 8 --skip_lit_review --use_genai --temperature $i --idea_file "$persona_filename" --use_personas
    
    echo "Completed run with temperature=$i and index=$j WITH personas"
    echo "------------------------"
  done
done

echo "All runs completed successfully"
