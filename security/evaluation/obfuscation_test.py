import re
import random
import string
import unicodedata
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

# Define toxic words and patterns
TOXIC_WORDS = [
    "hate", "kill", "stupid", "idiot", "moron", "retard", "fuck", "shit",
    "asshole", "bitch", "bastard", "cunt", "whore", "slut", "nigger"
]

# Define synonym mappings for word substitution
SYNONYMS = {
    'hate': ['despise', 'loathe', 'detest'],
    'kill': ['murder', 'eliminate', 'destroy'],
    'stupid': ['idiotic', 'foolish', 'dumb'],
    'fuck': ['screw', 'damn', 'hell'],
    'shit': ['crap', 'garbage', 'trash'],
    'die': ['perish', 'expire', 'succumb'],
    'cancer': ['disease', 'illness', 'malady']
}

# Define obfuscation patterns
OBFUSCATION_PATTERNS = {
    'leetspeak': {
        'pattern': r'[a-z]',
        'replacement': lambda m: random.choice({
            'a': ['4', '@'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['5', '$'],
            't': ['7']
        }.get(m.group(0).lower(), [m.group(0)]))
    },
    'space_injection': {
        'pattern': r'(.)',
        'replacement': lambda m: m.group(1) + ' '
    },
    'char_repetition': {
        'pattern': r'(.)',
        'replacement': lambda m: m.group(1) * random.randint(2, 3)
    },
    'special_chars': {
        'pattern': r'(.)',
        'replacement': lambda m: m.group(1) + random.choice(['*', '#', '@', '$', '%'])
    },
    'case_variation': {
        'pattern': r'[a-zA-Z]',
        'replacement': lambda m: m.group(0).upper() if random.random() > 0.5 else m.group(0).lower()
    },
    'unicode': {
        'pattern': r'(.)',
        'replacement': lambda m: unicodedata.normalize('NFKC', m.group(1))
    },
    'word_substitution': {
        'pattern': r'\b\w+\b',
        'replacement': lambda m: random.choice(SYNONYMS.get(m.group(0).lower(), [m.group(0)]))
    },
    'character_swapping': {
        'pattern': r'(\w)(\w)',
        'replacement': lambda m: m.group(2) + m.group(1)
    },
    'emoji_injection': {
        'pattern': r'(.)',
        'replacement': lambda m: m.group(1) + random.choice(['😡', '🤬', '💩', '👎', '🤢'])
    },
    'zero_width': {
        'pattern': r'(.)',
        'replacement': lambda m: m.group(1) + '\u200B'  # Zero-width space
    }
}

def apply_obfuscation(text: str, pattern_type: str) -> str:
    """Apply specific obfuscation pattern to text."""
    if pattern_type not in OBFUSCATION_PATTERNS:
        return text
        
    pattern = OBFUSCATION_PATTERNS[pattern_type]
    try:
        return re.sub(pattern['pattern'], pattern['replacement'], text)
    except Exception as e:
        print(f"Error applying {pattern_type} pattern: {str(e)}")
        return text

def generate_obfuscated_text(text: str, pattern_types: List[str] = None) -> Dict[str, str]:
    """Generate obfuscated versions of text using specified patterns."""
    if pattern_types is None:
        pattern_types = list(OBFUSCATION_PATTERNS.keys())
        
    obfuscated = {'original': text}
    for pattern in pattern_types:
        try:
            obfuscated[pattern] = apply_obfuscation(text, pattern)
        except Exception as e:
            print(f"Error generating {pattern} version: {str(e)}")
            obfuscated[pattern] = text
            
    return obfuscated

def create_test_set(num_samples: int = 500) -> pd.DataFrame:
    """Create a test set with various obfuscation patterns."""
    test_data = []
    
    # Define toxic examples for each class with more diversity
    class_examples = {
        'toxic': [
            "This is a toxic comment",
            "You are toxic and harmful",
            "What a toxic thing to say",
            "Your behavior is toxic",
            "This environment is toxic",
            "You're spreading toxicity",
            "Your words are poisonous",
            "This is toxic behavior"
        ],
        'severe_toxic': [
            "You should die in a fire",
            "I hope you get cancer",
            "You deserve to be tortured",
            "I wish you were dead",
            "You should be eliminated",
            "You're better off dead",
            "The world would be better without you",
            "You're a waste of oxygen"
        ],
        'obscene': [
            "You are a fucking idiot",
            "What a piece of shit",
            "Go fuck yourself",
            "You're a worthless piece of garbage",
            "You're a disgusting human being",
            "You're a fucking moron",
            "What a fucking waste",
            "You're a piece of trash"
        ],
        'threat': [
            "I will kill you",
            "You better watch your back",
            "I'm coming for you",
            "You're going to regret this",
            "I'll make you pay for this",
            "You're dead meat",
            "I'll destroy you",
            "You'll pay for this"
        ],
        'insult': [
            "You are stupid",
            "What a moron",
            "You are worthless",
            "You're an absolute idiot",
            "You're completely incompetent",
            "You're a fool",
            "You're brainless",
            "You're clueless"
        ],
        'identity_hate': [
            "I hate all [group] people",
            "[group] people are trash",
            "All [group] should die",
            "[group] people are inferior",
            "[group] people don't belong here",
            "[group] are subhuman",
            "[group] are worthless",
            "[group] should be eliminated"
        ]
    }
    
    # Generate balanced samples for each class
    samples_per_class = num_samples // (len(class_examples) + 1)  # +1 for clean examples
    
    for class_name, examples in class_examples.items():
        for _ in range(samples_per_class):
            # Select a base example
            base_text = random.choice(examples)
            
            # Generate multiple obfuscated versions
            for _ in range(3):  # Generate 3 obfuscated versions per text
                # Randomly select 2-3 obfuscation patterns
                num_patterns = random.randint(2, 3)
                selected_patterns = random.sample(list(OBFUSCATION_PATTERNS.keys()), num_patterns)
                
                # Apply patterns sequentially
                obf_text = base_text
                for pattern in selected_patterns:
                    obf_text = apply_obfuscation(obf_text, pattern)
                
                # Add metadata
                test_data.append({
                    'text': obf_text,
                    'original_text': base_text,
                    'pattern': '+'.join(selected_patterns),
                    'class': class_name,
                    'is_toxic': 1,
                    'confidence': random.uniform(0.7, 1.0),  # Add confidence score
                    'obfuscation_level': len(selected_patterns)  # Track obfuscation complexity
                })
    
    # Add clean (non-toxic) examples with more variety
    clean_examples = [
        "This is a normal comment",
        "I like this post",
        "Great work!",
        "Thanks for sharing",
        "Interesting perspective",
        "I appreciate your input",
        "This is helpful information",
        "Good point!",
        "I agree with you",
        "Well said!",
        "Thanks for the insight",
        "This is valuable",
        "I learned something new",
        "Great perspective",
        "Well explained"
    ]
    
    for _ in range(samples_per_class):
        base_text = random.choice(clean_examples)
        test_data.append({
            'text': base_text,
            'original_text': base_text,
            'pattern': 'clean',
            'class': 'clean',
            'is_toxic': 0,
            'confidence': random.uniform(0.8, 1.0),  # Higher confidence for clean examples
            'obfuscation_level': 0  # No obfuscation for clean examples
        })
    
    return pd.DataFrame(test_data)

def save_test_set(df: pd.DataFrame, output_path: str = "security/obfuscation_test_set.csv"):
    """Save test set to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved test set to {output_path}")

def main():
    # Create and save test set
    test_df = create_test_set()
    save_test_set(test_df)
    
    # Print sample of test set
    print("\nSample of test set:")
    print(test_df.head())
    
    # Print statistics
    print("\nTest set statistics:")
    print(f"Total samples: {len(test_df)}")
    print(f"Toxic samples: {test_df['is_toxic'].sum()}")
    print("\nPattern distribution:")
    print(test_df['pattern'].value_counts())

if __name__ == "__main__":
    main()
