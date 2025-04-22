import time  # Standard library for time-related functions, used here for performance tracking.
from argparse import ArgumentParser  # Standard library for parsing command-line arguments.
from datetime import datetime  # Standard library for getting the current date and time, used for logging directories.
from pathlib import Path  # Standard library for working with file paths in an object-oriented way.

import numpy as np  # Library for numerical operations, especially array manipulation.
import torch  # The main PyTorch library for tensor computations and neural networks.
import yaml  # Library for reading and writing YAML files, used here for configuration.
from torch.utils.data import DataLoader  # PyTorch utility for creating batches of data from a dataset.
from torch.utils.tensorboard.writer import SummaryWriter  # PyTorch utility for logging data to TensorBoard for visualization.

# Import custom modules specific to this project.
from countdown_task import CountdownTasksDataset, reward_function  # Dataset definition and reward calculation logic for the specific task.
from grpo import rollout, update_policy  # Functions for the Generalized Reward Policy Optimization (GRPO) algorithm: generating sequences (rollout) and updating the model (update_policy).
from qwen2_model import Transformer  # The neural network model architecture (likely a Transformer variant).
from tokenizer import Tokenizer  # Utility for converting text to numerical tokens that the model can understand.


# Function to evaluate the model's performance on a separate test dataset.
def evaluate(model, tokenizer, device, dtype, config):
    """
    Evaluates the model's performance on the test set by generating answers and calculating the success rate.

    Args:
        model: The trained Transformer model.
        tokenizer: The tokenizer used to convert text to tokens.
        device: The computing device (CPU or CUDA GPU) to run evaluation on.
        dtype: The data type (e.g., float16, bfloat16) for model computations.
        config: A dictionary containing configuration parameters, including data paths and batch sizes.

    Returns:
        float: The mean success rate on the test dataset.
    """
    # Initialize the test dataset using the CountdownTasksDataset class.
    # This dataset provides the questions for evaluation.
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],  # Path to the dataset file.
        tokenizer=tokenizer,              # Tokenizer for processing text.
        split="test",                     # Specify the 'test' partition of the data.
        test_size=config["data"]["test_size"],  # Proportion of data reserved for testing.
    )
    # Create a random number generator specific to the PyTorch device.
    # Ensures reproducibility if needed, though shuffling is off for evaluation.
    generator = torch.Generator(device=device)
    # Create a DataLoader to efficiently load batches of test data.
    # Shuffling is disabled for evaluation to process data in order.
    # Batch size is halved compared to training, potentially for memory reasons during generation.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,                               # Do not shuffle data for evaluation.
        collate_fn=CountdownTasksDataset.collate_fn, # Custom function to assemble batches.
        generator=generator,                         # Random generator (less relevant here as shuffle=False).
        batch_size=config["training"]["batch_size"] // 2, # Use half the training batch size.
        drop_last=False,                             # Keep the last batch even if it's smaller.
    )
    success = []  # List to store the success status (reward) for each evaluated example.
    # Iterate through batches of data provided by the DataLoader.
    for batch in dataloader:
        # Perform a rollout: Generate model responses (answers) for the questions in the batch.
        # This uses the trained model in inference mode (implicitly, as gradients aren't calculated here).
        episodes = rollout(
            model=model,                          # The model being evaluated.
            tokenizer=tokenizer,                  # Tokenizer for decoding generated tokens.
            batch=batch,                          # The current batch of questions.
            max_gen_len=config["training"]["max_gen_len"] * 2, # Maximum length for generated answers (longer than training).
            num_answer_per_question=1,            # Generate only one answer per question for evaluation.
            reward_function=reward_function,      # Function to calculate the reward (success).
            device=device,                        # Device for computations.
            dtype=dtype,                          # Data type for computations.
        )
        # Collect the success indicator (answer_reward) from the reward information of each episode.
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    # Calculate and return the mean success rate across all test examples.
    return np.mean(success)


# Main function to orchestrate the training process.
def main(config_path: str):
    """
    Loads configuration, initializes model, tokenizer, optimizer, datasets,
    and runs the training loop, including periodic evaluation and checkpointing.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # Load training configuration from the specified YAML file.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) # Parses the YAML file into a Python dictionary.

    # --- Initialization ---

    # Set up model and computation device (CPU or GPU).
    pretrained_model_path = Path(config["model"]["pretrained_model_path"]) # Path to the pre-trained model files.
    device = torch.device(config["model"]["device"]) # Creates a PyTorch device object (e.g., 'cuda:0' or 'cpu').

    # Determine the numerical precision (dtype) for model computations.
    # Using lower precision (like bfloat16 or float16) can speed up training and reduce memory usage on compatible hardware.
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16) # Defaults to bfloat16 if not specified or invalid.

    # Set the default device for all subsequent PyTorch tensor allocations.
    torch.set_default_device(device)
    # Set the random seed for PyTorch's random number generators to ensure reproducibility of results.
    torch.random.manual_seed(config["training"]["random_seed"])

    # Configure batching strategy for training.
    # BATCH_SIZE: Total number of generated sequences (answers) processed in one optimizer step.
    # NUM_QUESTIONS_PER_BATCH: Number of unique questions sampled from the dataset in each step.
    # NUM_ANSWERS_PER_QUESTION: How many different answers the model generates for each question during rollout.
    # The total batch size is the product of NUM_QUESTIONS_PER_BATCH and NUM_ANSWERS_PER_QUESTION.
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    # Initialize TensorBoard SummaryWriter for logging metrics and visualizations.
    # Logs will be saved in a directory named with the current timestamp.
    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S") # Format current time as a string.
    log_dir = Path(config['training']['log_dir']) / current_time # Construct the log directory path.
    log_dir.mkdir(parents=True, exist_ok=True) # Create the log directory if it doesn't exist.
    tb_writer = SummaryWriter(log_dir=str(log_dir)) # Create the writer object.

    # Initialize the tokenizer from the pre-trained model directory.
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    # Load the pre-trained Transformer model and move it to the specified device.
    # `.train()` sets training mode: enables dropout (randomly zeroes units during forward pass)
    # and updates BatchNorm with batch-wise mean/variance (vs. using running stats in `.eval()`).
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    # Set up the optimizer (AdamW).
    # The optimizer adjusts model parameters (weights) based on calculated gradients to minimize the loss function.
    # AdamW is a common optimizer known for good performance, especially with Transformers.
    optimizer = torch.optim.AdamW(
        model.parameters(), # Pass the model's parameters to the optimizer.
        lr=config["training"]["learning_rate"], # Learning rate: controls the step size of parameter updates.
        weight_decay=config["training"]["weight_decay"], # Weight decay: a regularization technique to prevent overfitting.
        betas=config["training"]["betas"], # Coefficients for computing running averages of gradient and its square (AdamW specific).
    )

    # Initialize the training dataset.
    # This dataset provides the questions used for training the model.
    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",                     # Use the 'train' partition of the data.
        test_size=config["data"]["test_size"],
    )
    # Create a random number generator for the DataLoader.
    generator = torch.Generator(device=device)
    # Create a DataLoader for the training dataset.
    # Shuffling is enabled to ensure the model sees data in a random order each epoch.
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,                                # Shuffle the training data each epoch.
        collate_fn=CountdownTasksDataset.collate_fn, # Custom function to assemble batches.
        generator=generator,                         # Random generator for shuffling.
        batch_size=NUM_QUESTIONS_PER_BATCH,          # Load this many unique questions per batch.
    )

    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time() # Record the start time for calculating step duration.
    # Prepare directory for saving model checkpoints.
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True) # Create checkpoint directory if it doesn't exist.

    # Iterate through the training data provided by the DataLoader.
    # `enumerate` adds a step counter, starting from 1.
    for step, batch in enumerate(train_dataloader, start=1):

        # 1. Rollout Phase: Generate sequences (answers) using the current model policy.
        # For each question in the batch, generate multiple candidate answers.
        # The `rollout` function handles the autoregressive generation process.
        episodes = rollout(
            model=model,                          # The current state of the model.
            tokenizer=tokenizer,                  # Tokenizer for decoding.
            batch=batch,                          # Batch of questions from the dataloader.
            max_gen_len=config["training"]["max_gen_len"], # Max length of generated answers.
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION, # Number of answers per question.
            reward_function=reward_function,      # Function to compute reward for each generated answer.
            device=device,                        # Computation device.
            dtype=dtype,                          # Computation data type.
        )

        # Optionally filter out episodes (question-answer pairs) where the model didn't finish generating
        # (e.g., reached max_gen_len without producing an end-of-sequence token).
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]

        # If all episodes were skipped, continue to the next batch.
        if not episodes:
            print(f"\rStep {step}, Skipped batch due to no finished episodes.")
            continue

        # 2. Update Phase: Update the model's policy using the generated episodes and their rewards.
        # The `update_policy` function implements the core RL algorithm (GRPO in this case).
        # It calculates the loss based on the rewards and generated sequences, computes gradients, and updates the model parameters via the optimizer.
        results = update_policy(
            model=model,                         # The model to update.
            optimizer=optimizer,                 # The optimizer to perform the update step.
            episodes=episodes,                   # The generated sequences and associated data (rewards, log probabilities).
            micro_batch_size=config["training"]["micro_batch_size"], # Size for gradient accumulation (if used).
            pad_token_id=tokenizer.pad_token_id, # Padding token ID for handling variable sequence lengths.
            max_grad_norm=config["training"]["max_grad_norm"], # Maximum allowed gradient norm (gradient clipping).
            device=device,                       # Computation device.
            dtype=dtype,                         # Computation data type.
        )

        # Synchronize CUDA operations (if using GPU) to get accurate timing.
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()    # Record the end time for this step.
        duration = end_time - start_time # Calculate the duration of the step.
        start_time = end_time     # Reset start time for the next step.

        # 3. Logging and Evaluation: Calculate and log various metrics.
        # Extract reward information from the episodes generated in this step.
        reward = [episode.reward for episode in episodes] # Overall reward (combination of factors).
        formatted_reward = [episode.reward_info["format_reward"] for episode in episodes] # Reward specifically for correct formatting.
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes] # Reward specifically for correctness of the answer.
        num_finished_episodes = sum(episode.is_finished for episode in episodes) # Count how many generations were complete.

        # Calculate statistics for logging.
        mean_reward = np.mean(reward) if reward else 0.0
        std_reward = np.std(reward) if reward else 0.0
        success_rate = np.mean(answer_reward) if answer_reward else 0.0 # Treat answer_reward as the primary success metric.
        format_reward = np.mean(formatted_reward) if formatted_reward else 0.0
        grad_norm = results.get("grad_norm", 0.0) # Gradient norm (from update_policy results).
        entropy = results.get("entropy", 0.0)     # Policy entropy (measure of randomness/exploration).
        lr = optimizer.param_groups[0]["lr"]    # Current learning rate.
        loss = results.get("loss", 0.0)           # Calculated loss value for the update step.
        # Calculate the average length of the generated answers.
        mean_response_len = np.mean([len(episode.generated_token_ids) for episode in episodes]) if episodes else 0.0

        # Print key metrics to the console. `\r` moves the cursor to the beginning of the line for overwriting.
        print(
            f"\rStep {step}, Loss: {loss:.4f}, Mean Reward: {mean_reward:.2f}, "
            f"Train Success: {success_rate:.2f}, Format Reward: {format_reward:.2f}, "
            f"Grad Norm: {grad_norm:.2f}, Duration: {duration:.2f}s, "
            f"Finished: {num_finished_episodes}/{len(episodes)}, "
            f"Mean Len: {mean_response_len:.1f}, "
            f"Entropy: {entropy:.2f}   ", # Extra spaces to clear previous line
            end='' # Prevent newline to allow overwriting with 
        )

        # Evaluate the model on the test set periodically.
        if step % config["training"]["eval_interval"] == 0:
            print() # Print a newline before evaluation output
            model.eval() # Set the model to evaluation mode (disables dropout, etc.).
            with torch.no_grad(): # Disable gradient calculations during evaluation for efficiency.
                eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            model.train() # Set the model back to training mode.
            print(f"Step {step} Eval Success Rate: {eval_success_rate:.4f}")
            # Log evaluation success rate to TensorBoard.
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        # Log metrics to TensorBoard for visualization.
        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("reward/mean", mean_reward, step)
        tb_writer.add_scalar("reward/std", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("reward/format", format_reward, step)
        tb_writer.add_scalar("gradients/norm", grad_norm, step)
        tb_writer.add_scalar("performance/step_duration_s", duration, step)
        tb_writer.add_scalar("rollout/num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("optimizer/learning_rate", lr, step)
        tb_writer.add_scalar("rollout/mean_response_length", mean_response_len, step)
        tb_writer.add_scalar("policy/entropy", entropy, step)

        # Log the generated text from the first few episodes to TensorBoard.
        # This helps in qualitatively assessing the model's generation capabilities.
        num_texts_to_log = min(len(episodes), 4) # Log up to 4 examples
        for i in range(num_texts_to_log):
            # Use <pre> tags for preformatted text in TensorBoard.
            tb_writer.add_text(f"sample_generations/episode_{i}", f"<pre>{episodes[i].text}</pre>", step)

        # Save a checkpoint of the model's state periodically.
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt" # Checkpoint filename includes the step number.
            # Save the model's parameters (state dictionary).
            torch.save(model.state_dict(), output_file)
            print(f"\rSaved checkpoint to {output_file}") # Print newline before checkpoint message

    # Close the TensorBoard writer after training finishes.
    tb_writer.close()
    print("\nTraining finished.")


# Entry point of the script.
if __name__ == "__main__":
    # Set up argument parsing to accept the configuration file path from the command line.
    parser = ArgumentParser(description="Train a model using GRPO on the Countdown task.")
    # Define the '--config' argument.
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml", # Default configuration file name.
        help="Path to the configuration YAML file.",
    )
    # Parse the command-line arguments provided by the user.
    args = parser.parse_args()
    # Call the main training function with the path to the configuration file.
    main(args.config)
