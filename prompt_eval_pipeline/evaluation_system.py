import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict
from openai import OpenAI

from metrics import extract_label, rouge_1, classification_report

MODEL = "gpt-4o-mini"

client = OpenAI()

@dataclass
class TestCase:
    id: str
    input: str
    expected: str
    category: str
    difficulty: str

@dataclass
class EvaluationResult:
    test_id: str
    prompt_name: str
    input: str
    output: str
    expected: str
    exact_match: float
    semantic_similarity: float
    score: float
    predicted_label: str
    label_match: float
    rouge_1: float
    timestamp: str

def load_prompts(prompts_dir: str) -> Dict[str, str]:
    """Load every prompt in a folder: file name (without .txt) -> prompt text."""
    prompts = {}
    for file_name in sorted(os.listdir(prompts_dir)):
        if file_name.endswith(".txt"):
            with open(os.path.join(prompts_dir, file_name), 'r') as f:
                prompts[file_name[:-4]] = f.read()
    return prompts

def prompt_version(system_prompt: str) -> str:
    """Short fingerprint of a prompt's text; changes whenever the prompt changes."""
    return hashlib.sha256(system_prompt.encode()).hexdigest()[:8]

class ReviewClassifierEvaluator:
    def __init__(self):
        self.results: List[EvaluationResult] = []

    def load_test_cases(self, csv_file: str) -> List[TestCase]:
        """Load test cases from CSV."""
        cases = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cases.append(TestCase(**row))
        return cases

    def evaluate_prompt(
        self,
        prompt_name: str,
        system_prompt: str,
        test_cases: List[TestCase]
    ) -> Dict:
        """Evaluate a single prompt."""
        results = []

        for test in test_cases:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": test.input}
                ],
                temperature=0.3
            )

            output = response.choices[0].message.content.strip()

            # Calculate metrics
            exact_match = 1.0 if output.lower() == test.expected.lower() else 0.0

            # Semantic similarity (simple approach: check for keywords)
            if test.expected.lower() in output.lower():
                semantic_sim = 0.9
            elif any(keyword in output.lower() for keyword in [test.expected[0], test.expected[-3:]]):
                semantic_sim = 0.5
            else:
                semantic_sim = 0.0

            score = (exact_match + semantic_sim) / 2

            # Custom metrics: the label the model settled on, and word overlap
            predicted_label = extract_label(output)
            label_match = 1.0 if predicted_label == test.expected.lower() else 0.0

            result = EvaluationResult(
                test_id=test.id,
                prompt_name=prompt_name,
                input=test.input,
                output=output,
                expected=test.expected,
                exact_match=exact_match,
                semantic_similarity=semantic_sim,
                score=score,
                predicted_label=predicted_label,
                label_match=label_match,
                rouge_1=rouge_1(output, test.expected),
                timestamp=datetime.now().isoformat()
            )

            results.append(result)
            self.results.append(result)

        return self.summarize(prompt_name, results)

    def summarize(self, prompt_name: str, results: List[EvaluationResult]) -> Dict:
        """Summarize one prompt's results."""
        avg_score = sum(r.score for r in results) / len(results)

        return {
            'prompt_name': prompt_name,
            'total_tests': len(results),
            'exact_match_accuracy': sum(1 for r in results if r.exact_match > 0.5) / len(results),
            'label_accuracy': sum(r.label_match for r in results) / len(results),
            'average_rouge_1': sum(r.rouge_1 for r in results) / len(results),
            'average_score': avg_score,
            'per_label': classification_report(
                [r.expected.lower() for r in results],
                [r.predicted_label for r in results]
            ),
            'results': results
        }

    def compare_prompts(self, summaries: List[Dict]):
        """Compare multiple prompts."""
        print("\n" + "="*70)
        print("PROMPT COMPARISON RESULTS")
        print("="*70)
        print(f"{'Prompt':<16} {'Label Acc':<11} {'Exact':<9} {'ROUGE-1':<9} {'Avg Score':<11} {'Status'}")
        print("-"*70)

        sorted_summaries = sorted(
            summaries,
            key=lambda x: x['label_accuracy'],
            reverse=True
        )

        for i, summary in enumerate(sorted_summaries):
            status = "🥇 BEST" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
            print(
                f"{summary['prompt_name']:<16} "
                f"{summary['label_accuracy']:<11.1%} "
                f"{summary['exact_match_accuracy']:<9.1%} "
                f"{summary['average_rouge_1']:<9.3f} "
                f"{summary['average_score']:<11.3f} "
                f"{status}"
            )

        print("-"*70)

    def print_label_report(self, summaries: List[Dict]):
        """Show precision, recall and F1 per sentiment label for each prompt."""
        print("\nPER-LABEL METRICS")
        print(f"{'Prompt':<16} {'Label':<10} {'Precision':<11} {'Recall':<9} {'F1'}")
        print("-"*70)

        for summary in summaries:
            for label, stats in summary['per_label'].items():
                print(
                    f"{summary['prompt_name']:<16} "
                    f"{label:<10} "
                    f"{stats['precision']:<11.2f} "
                    f"{stats['recall']:<9.2f} "
                    f"{stats['f1']:.2f}"
                )

        print("-"*70)

    def save_results(self, output_file: str):
        """Save all results to JSON."""
        data = [asdict(r) for r in self.results]
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def save_run(
        self,
        summaries: List[Dict],
        prompts: Dict[str, str],
        runs_dir: str = "runs",
        run_id: str = None
    ) -> str:
        """Save this run as its own versioned file, so earlier runs are kept."""
        run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            'run_id': run_id,
            'model': MODEL,
            'prompts': {
                name: {'version': prompt_version(text), 'text': text}
                for name, text in prompts.items()
            },
            'summaries': [
                {key: value for key, value in summary.items() if key != 'results'}
                for summary in summaries
            ],
            'results': [asdict(r) for r in self.results]
        }

        os.makedirs(runs_dir, exist_ok=True)
        run_file = os.path.join(runs_dir, f"run_{run_id}.json")
        with open(run_file, 'w') as f:
            json.dump(data, f, indent=2)
        return run_file

    def print_failures(self, prompt_name: str, num=3):
        """Show failing cases for a prompt."""
        failures = [r for r in self.results if r.prompt_name == prompt_name and r.label_match == 0.0]

        if not failures:
            print(f"\n✅ No failures for {prompt_name}!")
            return

        print(f"\n❌ Failures for {prompt_name} ({len(failures)} total):\n")

        for i, failure in enumerate(failures[:num]):
            print(f"Test {i+1}: {failure.test_id}")
            print(f"  Input:    {failure.input[:60]}...")
            print(f"  Expected: {failure.expected}")
            print(f"  Got:      {failure.output}")
            print(f"  Label:    {failure.predicted_label}\n")

def main():
    evaluator = ReviewClassifierEvaluator()
    test_cases = evaluator.load_test_cases("test_cases.csv")

    # Prompts to test live in prompts/, one file per prompt
    prompts = load_prompts("prompts")

    # Evaluate each prompt
    results_summary = []
    for prompt_name, system_prompt in prompts.items():
        print(f"\nEvaluating {prompt_name}...")
        summary = evaluator.evaluate_prompt(prompt_name, system_prompt, test_cases)
        results_summary.append(summary)
        print(f"  ✅ {summary['total_tests']} tests completed")

    # Compare results
    evaluator.compare_prompts(results_summary)
    evaluator.print_label_report(results_summary)

    # Show failures
    for prompt_name in prompts.keys():
        evaluator.print_failures(prompt_name)

    # Save results
    evaluator.save_results("evaluation_results.json")
    print("\n📊 Results saved to evaluation_results.json")

    run_file = evaluator.save_run(results_summary, prompts)
    print(f"🗂️  Run saved to {run_file}")

if __name__ == "__main__":
    main()
