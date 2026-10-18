import json
import pandas as pd
import csv

# Load results
with open("evaluation_results.json") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Analysis 1: Overall accuracy by prompt
print("\nAccuracy by Prompt:")
print(df.groupby('prompt_name').agg({
    'exact_match': 'mean',
    'semantic_similarity': 'mean',
    'score': 'mean'
}).round(3))

# Analysis 2: Accuracy by difficulty
print("\n\nAccuracy by Difficulty:")
test_cases_lookup = {}
with open("test_cases.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_cases_lookup[row['id']] = row['difficulty']

buckets = {"easy": [], "medium": [], "hard": []}
for _, row in df.iterrows():
    # Add difficulty from test_cases
    difficulty = test_cases_lookup[row['test_id']]
    buckets[difficulty].append(row['score'])

for difficulty, scores in buckets.items():
    print(difficulty, round(sum(scores) / len(scores), 3))

# Analysis 3: Which tests are hardest?
print("\n\nHardest Tests (by consistency of failure):")
hardest = df.groupby('test_id').agg({
    'score': ['mean', 'std']
}).round(3)
print(hardest.sort_values(('score', 'mean')).head())

# Analysis 4: Recommendations
best_prompt = df.groupby('prompt_name')['score'].mean().idxmax()
best_accuracy = df.groupby('prompt_name')['score'].mean().max()

print(f"\n\n✅ RECOMMENDATION: Use '{best_prompt}' ({best_accuracy:.1%} accuracy)")