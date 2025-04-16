#!/usr/bin/env python
"""
Setup script to create directory structure for the DelE Prediction project.
"""

import os

def create_directories():
    """Create the necessary directories for the project."""
    
    # Define directories to create
    directories = [
        'data',
        'results',
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
            
    # Create a blank placeholder in the data directory
    with open(os.path.join('data', '.gitkeep'), 'w') as f:
        pass
        
    # Create a blank placeholder in the results directory
    with open(os.path.join('results', '.gitkeep'), 'w') as f:
        pass
        
    print("\nDirectory structure setup complete!")
    print("\nNext steps:")
    print("1. Place your datasets in the 'data' directory")
    print("2. Update file paths in the main scripts if needed")
    print("3. Run the main scripts: python PIML_combined_delE.py and python PIML_f.py")

if __name__ == "__main__":
    create_directories() 