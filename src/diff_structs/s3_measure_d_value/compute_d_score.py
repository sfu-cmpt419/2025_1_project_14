"""
compute_d_score.py

Computes final 'D' value from the # of dermoscopic structures found.
"""

def compute_d_score(num_structures_present, multiplier=0.5):
    """
    D = multiplier * num_structures_present
    Typically multiplier=0.5 in the standard ABCD formula.
    """
    return multiplier * num_structures_present

# Example usage:
if __name__ == "__main__":
    d_val = compute_d_score(num_structures_present=3, multiplier=0.5)
    print(f"[compute_d_score] D value = {d_val}")
