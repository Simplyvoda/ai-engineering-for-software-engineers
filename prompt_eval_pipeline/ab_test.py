import json
import math
import random
from openai import OpenAI
from datetime import datetime

from evaluation_system import MODEL, ReviewClassifierEvaluator, load_prompts
from metrics import extract_label

client = OpenAI()

class ABTestRunner:
    def __init__(self, prompt_a, prompt_b, split_ratio=0.5):
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b
        self.split_ratio = split_ratio
        self.results_a = []
        self.results_b = []

    def get_prompt(self):
        """Route to A or B based on split ratio."""
        return self.prompt_a if random.random() < self.split_ratio else self.prompt_b

    def run_classification(self, review_text, expected=None):
        """Classify a review, tracking which prompt was used."""
        prompt = self.get_prompt()
        is_prompt_a = prompt == self.prompt_a

        # Call API
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": review_text}
            ]
        )

        output = response.choices[0].message.content
        label = extract_label(output)

        # Track result
        result = {
            'prompt': 'A' if is_prompt_a else 'B',
            'input': review_text,
            'output': output,
            'label': label,
            'expected': expected,
            # None when there is no known answer to check against
            'correct': None if expected is None else label == expected.lower(),
            'timestamp': datetime.now().isoformat()
        }

        if is_prompt_a:
            self.results_a.append(result)
        else:
            self.results_b.append(result)

        return output

    def _estimate_success(self, results):
        """Share of correct answers among results that have a known answer."""
        graded = [r for r in results if r['correct'] is not None]
        if not graded:
            return None
        return sum(1 for r in graded if r['correct']) / len(graded)

    def get_stats(self):
        """Get A/B test statistics."""
        return {
            'A_samples': len(self.results_a),
            'B_samples': len(self.results_b),
            'A_success_rate': self._estimate_success(self.results_a),
            'B_success_rate': self._estimate_success(self.results_b),
        }

    def get_winner(self, min_samples=30, alpha=0.05):
        """Pick the winner, or None if the difference could be chance."""
        graded_a = [r for r in self.results_a if r['correct'] is not None]
        graded_b = [r for r in self.results_b if r['correct'] is not None]
        n_a, n_b = len(graded_a), len(graded_b)

        if n_a < min_samples or n_b < min_samples:
            return {
                'winner': None,
                'p_value': None,
                'reason': f"Not enough samples yet (A={n_a}, B={n_b}, need {min_samples} each)"
            }

        wins_a = sum(1 for r in graded_a if r['correct'])
        wins_b = sum(1 for r in graded_b if r['correct'])
        rate_a, rate_b = wins_a / n_a, wins_b / n_b

        # Two-proportion z-test
        pooled = (wins_a + wins_b) / (n_a + n_b)
        std_err = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
        if std_err == 0:
            return {'winner': None, 'p_value': 1.0, 'reason': "A and B performed identically"}

        z = (rate_a - rate_b) / std_err
        p_value = math.erfc(abs(z) / math.sqrt(2))

        if p_value >= alpha:
            return {
                'winner': None,
                'p_value': p_value,
                'reason': f"Difference is not significant (p={p_value:.3f})"
            }

        return {
            'winner': 'A' if rate_a > rate_b else 'B',
            'p_value': p_value,
            'reason': f"{rate_a:.1%} vs {rate_b:.1%} (p={p_value:.3f})"
        }

    def save_results(self, output_file):
        """Save every tracked request to JSON."""
        with open(output_file, 'w') as f:
            json.dump(self.results_a + self.results_b, f, indent=2)

def main(name_a="v1_basic", name_b="v2_cot", rounds=8):
    prompts = load_prompts("prompts")
    test_cases = ReviewClassifierEvaluator().load_test_cases("test_cases.csv")

    runner = ABTestRunner(prompts[name_a], prompts[name_b])

    # Stand-in for production traffic: replay the test reviews several times
    print(f"\nA/B test: A={name_a} vs B={name_b} ({rounds * len(test_cases)} requests)...")
    for _ in range(rounds):
        for test in test_cases:
            runner.run_classification(test.input, test.expected)

    stats = runner.get_stats()
    print(f"  A ({name_a}): {stats['A_samples']} samples, {stats['A_success_rate']:.1%} correct")
    print(f"  B ({name_b}): {stats['B_samples']} samples, {stats['B_success_rate']:.1%} correct")

    result = runner.get_winner()
    if result['winner']:
        winner_name = name_a if result['winner'] == 'A' else name_b
        print(f"\n🏆 Winner: {result['winner']} ({winner_name}): {result['reason']}")
    else:
        print(f"\n🤝 No winner: {result['reason']}")

    runner.save_results("ab_test_results.json")
    print("\n📊 Results saved to ab_test_results.json")

if __name__ == "__main__":
    main()
