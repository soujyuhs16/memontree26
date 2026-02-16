# Updated Training Script

# This script has been modified to use the correct TrainingArguments parameters for transformers 4.57.5.

from transformers import TrainingArguments

# Other imports...

# Create TrainingArguments
training_args = TrainingArguments(
    # Other parameters...
    eval_strategy='epoch',  # updated parameter name
    save_strategy='epoch',  # ensure it remains as 'epoch'
)