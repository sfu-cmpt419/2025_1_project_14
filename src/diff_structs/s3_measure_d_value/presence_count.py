"""
presence_count.py

Sums how many dermoscopic structures are present.
"""

def presence_count(structure_flags):
    """
    Given a dict or list of booleans, returns how many are True.
    
    structure_flags example:
      {'globules': True, 'streaks': False, 'pigment_network': True}
    """
    if isinstance(structure_flags, dict):
        return sum(structure_flags.values())
    elif isinstance(structure_flags, list):
        return sum(structure_flags)
    else:
        raise ValueError("[presence_count] structure_flags must be dict or list of bools.")

# Example usage:
if __name__ == "__main__":
    flags = {'globules': True, 'streaks': True, 'pigment_network': False}
    count = presence_count(flags)
    print(f"[presence_count] # of structures present: {count}")
