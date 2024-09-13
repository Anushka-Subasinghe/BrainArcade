import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define parameters
n_samples = 600

# Initialize lists to store data
data = []

for _ in range(n_samples):
    # Generate stronger correlations for registration parameters
    ageLessThanFive = np.random.choice([0, 1], p=[0.8, 0.2])  # Lower chance of being age < 5
    familyHistory = np.random.choice([0, 1], p=[0.6, 0.4])  # Family history has moderate impact
    birthWeight = np.random.choice([0, 1], p=[0.85, 0.15])  # Lower probability of birth weight issue
    attentionIssue = np.random.choice([0, 1], p=[0.7, 0.3])  # Attention issues are less common
    socialIssue = np.random.choice([0, 1], p=[0.75, 0.25])  # Social issues less frequent
    gender = np.random.choice([0, 1], p=[0.55, 0.45])  # Slightly more males than females

    # Generate average reaction time and average score with stronger bounds
    avg_reaction_time = np.random.uniform(0.5, 2.0)  # Narrowed reaction time range for realism
    avg_score = np.random.uniform(60, 100)  # Increased minimum score for higher performance

    # Introduce more deterministic correlation patterns for registration quality
    reg_quality = (ageLessThanFive * 0.2 + familyHistory * 0.5 +
                   birthWeight * 0.3 + attentionIssue * 0.4 + socialIssue * 0.3) / 1.7

    # Refine real-time performance impact using avg_score and reaction time with weight adjustments
    rt_quality = (avg_score / 100) - (avg_reaction_time / 2.0)  # Normalize score and reaction time

    # Determine category with stronger thresholds and controlled randomness
    if reg_quality > 0.7 and rt_quality > 0.7:
        category = 1  # Excellent
    elif reg_quality > 0.7 and rt_quality < 0.5:
        category = 2  # Good Registration but Bad Real-Time
    elif reg_quality < 0.5 and rt_quality > 0.7:
        category = 3  # Average Registration but Good Real-Time
    elif reg_quality < 0.5 and rt_quality < 0.5:
        category = 4  # Bad Registration and Bad Real-Time
    else:
        category = 5  # Rarely occurring: Bad Registration but Good Real-Time

    # Ensure category 5 remains rare
    if category == 5 and random.random() > 0.05:  # Reduce occurrence to 5% chance for category 5
        category = random.choices([1, 2, 3, 4], weights=[40, 30, 20, 10], k=1)[0]

    # Append data to the list
    data.append([ageLessThanFive, familyHistory, birthWeight, attentionIssue, socialIssue,
                 gender, avg_reaction_time, avg_score, category])

# Create DataFrame
columns = [
    'ageLessThanFive', 'familyHistory', 'birthWeight', 'attentionIssue',
    'socialIssue', 'gender', 'avgReactionTime', 'avgScore', 'category'
]
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
df.to_csv('dataset.csv', index=False)
