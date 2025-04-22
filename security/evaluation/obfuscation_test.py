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

def create_test_set(num_samples: int = 100) -> pd.DataFrame:
    """Create a test set with various obfuscation patterns."""
    test_data = []
    
    # Generate clean samples
    clean_samples = [
        "This is a normal comment",
        "I like this post",
        "Great work!",
        "Thanks for sharing",
        "Interesting perspective"
    ]
    
    # Generate toxic samples
    toxic_samples = [
        f"This is a {word} comment" for word in TOXIC_WORDS
    ]
    
    # Combine and shuffle samples
    all_samples = clean_samples + toxic_samples
    random.shuffle(all_samples)
    
    # Generate obfuscated versions
    for text in all_samples[:num_samples]:
        try:
            is_toxic = any(word in text.lower() for word in TOXIC_WORDS)
            obfuscated = generate_obfuscated_text(text)
            
            for pattern, obf_text in obfuscated.items():
                test_data.append({
                    'text': obf_text,
                    'original_text': text,
                    'pattern': pattern,
                    'is_toxic': is_toxic
                })
        except Exception as e:
            print(f"Error processing text '{text}': {str(e)}")
            continue
    
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
