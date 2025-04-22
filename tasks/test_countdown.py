from pathlib import Path
from tokenizer.tokenizer import Tokenizer
from tasks.countdown_task import CountdownTasksDataset, RewardEvaluator

def print_dataset_info(dataset):
    """Print basic information about the dataset."""
    print(f"\nDataset size: {len(dataset)}")
    print("\nFirst example:")
    example = dataset[0]
    print(f"Numbers: {example['nums']}")
    print(f"Target: {example['target']}")
    print(f"Prefix: {example['prefix'][:100]}...")
    print(f"Token IDs length: {len(example['prefix_token_ids'])}")

def test_reward_evaluation():
    """Test the reward evaluation functionality."""
    evaluator = RewardEvaluator()
    
    # Test cases
    test_cases = [
        {
            "response": "<think>Let me think about this...</think>\n<answer>(1 + 2) * 3</answer>",
            "numbers": [1, 2, 3],
            "target": 9,
            "description": "Perfect format and correct answer"
        },
        {
            "response": "<think>Let me think...</think>\n<answer>(1 + 2) * 4</answer>",
            "numbers": [1, 2, 3],
            "target": 9,
            "description": "Perfect format but wrong answer"
        },
        {
            "response": "Just an answer: (1 + 2) * 3",
            "numbers": [1, 2, 3],
            "target": 9,
            "description": "No format tags"
        }
    ]
    
    print("\nTesting reward evaluation:")
    for case in test_cases:
        result = evaluator.evaluate(case["response"], case["numbers"], case["target"])
        print(f"\nCase: {case['description']}")
        print(f"Total reward: {result['reward']}")
        print(f"Format reward: {result['reward_info']['format_reward']}")
        print(f"Answer reward: {result['reward_info']['answer_reward']}")

def main():
    # Initialize tokenizer
    tokenizer = Tokenizer(str(Path("data/Qwen2.5-3B-Instruct") / "tokenizer.json"))
    
    # Create dataset
    data_path = Path("data/Countdown-Tasks-3to4")  # Adjust path as needed
    dataset = CountdownTasksDataset(tokenizer, str(data_path))
    
    # Print dataset information
    print_dataset_info(dataset)
    
    # Test reward evaluation
    test_reward_evaluation()

if __name__ == "__main__":
    main() 