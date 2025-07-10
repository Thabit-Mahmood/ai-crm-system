#!/usr/bin/env python3
"""
INNOVATION NAMING UPDATE SCRIPT
Replaces all "Adaptive Confidence-Based Model Selection" and "Sentiment Trajectory Pattern Mining" references with proper descriptive names
"""

import os
import re
import glob
from typing import Dict, List

class InnovationNameUpdater:
    """Updates innovation names throughout the codebase"""
    
    def __init__(self):
        self.replacements = {
            # Adaptive Confidence-Based Model Selection replacements
            'Adaptive Confidence-Based Model Selection': 'Adaptive Confidence-Based Model Selection',
            'Adaptive Confidence-Based Model Selection': 'Adaptive Confidence-Based Model Selection',
            'ADAPTIVE CONFIDENCE-BASED MODEL SELECTION': 'ADAPTIVE CONFIDENCE-BASED MODEL SELECTION',
            'AdaptiveConfidenceModelSelection': 'AdaptiveConfidenceModelSelection',
            'adaptive_confidence_model_selection': 'adaptive_confidence_model_selection',
            
            # Sentiment Trajectory Pattern Mining replacements
            'Sentiment Trajectory Pattern Mining': 'Sentiment Trajectory Pattern Mining',
            'Sentiment Trajectory Pattern Mining': 'Sentiment Trajectory Pattern Mining',
            'SENTIMENT TRAJECTORY PATTERN MINING': 'SENTIMENT TRAJECTORY PATTERN MINING',
            'SentimentTrajectoryPatternMining': 'SentimentTrajectoryPatternMining',
            'sentiment_trajectory_pattern_mining': 'sentiment_trajectory_pattern_mining',
        }
        
        self.files_updated = []
        self.total_replacements = 0
    
    def update_all_files(self):
        """Update all files in the project"""
        print("🔄 Starting Innovation Name Updates...")
        
        # Get all Python files
        python_files = []
        for root, dirs, files in os.walk('.'):
            # Skip certain directories
            if any(skip in root for skip in ['.git', '__pycache__', 'node_modules', '.pytest_cache']):
                continue
            
            for file in files:
                if file.endswith(('.py', '.md', '.txt', '.json', '.js', '.jsx', '.ts', '.tsx')):
                    python_files.append(os.path.join(root, file))
        
        print(f"Found {len(python_files)} files to process")
        
        for file_path in python_files:
            self.update_file(file_path)
        
        print(f"\n✅ Innovation naming update completed!")
        print(f"📊 Files updated: {len(self.files_updated)}")
        print(f"📊 Total replacements: {self.total_replacements}")
        
        return {
            'files_updated': self.files_updated,
            'total_replacements': self.total_replacements
        }
    
    def update_file(self, file_path: str):
        """Update a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            file_replacements = 0
            
            # Apply all replacements
            for old_name, new_name in self.replacements.items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(old_name) + r'\b'
                new_content, count = re.subn(pattern, new_name, content)
                content = new_content
                file_replacements += count
            
            # If changes were made, write back to file
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_updated.append(file_path)
                self.total_replacements += file_replacements
                print(f"✅ Updated {file_path} ({file_replacements} replacements)")
        
        except Exception as e:
            print(f"❌ Error updating {file_path}: {e}")

def main():
    """Main function"""
    updater = InnovationNameUpdater()
    results = updater.update_all_files()
    
    print("\n" + "="*80)
    print("INNOVATION NAMING UPDATE SUMMARY")
    print("="*80)
    print(f"Files Updated: {len(results['files_updated'])}")
    print(f"Total Replacements: {results['total_replacements']}")
    
    if results['files_updated']:
        print("\nUpdated Files:")
        for file_path in results['files_updated']:
            print(f"  - {file_path}")

if __name__ == "__main__":
    main() 