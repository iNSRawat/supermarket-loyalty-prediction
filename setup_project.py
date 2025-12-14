"""
Project Setup Script
====================
This script creates the complete directory structure for the
Supermarket Loyalty Prediction project.

Author: iNSRawat
"""

import os
import sys


def create_directory_structure():
    """Create all necessary directories for the project."""
    
    directories = [
        'data',
        'notebooks',
        'src',
        'results',
        'results/figures',
        'results/models'
    ]
    
    print("📁 Creating project directory structure...\n")
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created: {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}/: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Project structure created successfully!")
    print("=" * 60)


def create_placeholder_files():
    """Create placeholder files in appropriate directories."""
    
    print("\n📄 Creating placeholder files...\n")
    
    # Create __init__.py in src/
    with open('src/__init__.py', 'w') as f:
        f.write('"""Supermarket Loyalty Prediction Package"""\n')
    print("✅ Created: src/__init__.py")
    
    # Create .gitkeep files to preserve empty directories
    gitkeep_dirs = ['data', 'results/figures', 'results/models']
    for directory in gitkeep_dirs:
        gitkeep_path = os.path.join(directory, '.gitkeep')
        with open(gitkeep_path, 'w') as f:
            f.write('')
        print(f"✅ Created: {gitkeep_path}")
    
    # Create a sample model_performance.csv
    with open('results/model_performance.csv', 'w') as f:
        f.write('Model,R2,RMSE,MAE,MAPE\n')
        f.write('# Model comparison results will be saved here\n')
    print("✅ Created: results/model_performance.csv")


def display_project_tree():
    """Display the project structure as a tree."""
    
    print("\n" + "=" * 60)
    print("PROJECT STRUCTURE")
    print("=" * 60)
    print("""
supermarket-loyalty-prediction/
│
├── data/
│   └── .gitkeep
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── features.py
│   └── models.py
│
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   ├── models/
│   │   └── .gitkeep
│   └── model_performance.csv
│
├── .gitignore
├── requirements.txt
├── README.md
├── LICENSE
└── setup_project.py
    """)
    print("=" * 60)


def display_next_steps():
    """Display instructions for next steps."""
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. 📊 Add your data:
   - Place loyalty.csv in the data/ directory

2. 🐍 Set up Python environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt

3. 📓 Create Jupyter notebooks:
   - Run the provided code in notebooks/
   - Start with 01_data_cleaning.ipynb

4. 🚀 Start your analysis:
   jupyter notebook

5. 📤 Push to GitHub:
   git add .
   git commit -m "Initial project setup"
   git push origin main

Happy Data Science! 🎉
    """)
    print("=" * 60)


def main():
    """Main function to set up the project."""
    
    print("\n" + "=" * 60)
    print("SUPERMARKET LOYALTY PREDICTION PROJECT SETUP")
    print("=" * 60)
    print("\nThis script will create the complete project structure.\n")
    
    # Create directories
    create_directory_structure()
    
    # Create placeholder files
    create_placeholder_files()
    
    # Display structure
    display_project_tree()
    
    # Display next steps
    display_next_steps()
    
    print("\n✅ Setup complete! Your project is ready.\n")


if __name__ == "__main__":
    main()
