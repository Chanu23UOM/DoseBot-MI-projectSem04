import fitz
import sys
import re
sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r"d:\Logee Sir Project\ML\DoseBot-MI-projectSem04\DoseBotV2_Training.ipynb - Colab.pdf")

# Collect ALL text
full_text = ""
for page in doc:
    full_text += page.get_text()

# Extract epoch lines
for line in full_text.split('\n'):
    line = line.strip()
    if 'Epoch' in line and ('val_accuracy' in line or 'improved' in line or 'reducing' in line or '/60' in line):
        if 'val_accuracy' in line and 'val_loss' in line:
            # Parse the final metrics line
            print(line[:150])
        elif 'improved' in line or 'reducing' in line:
            print(line[:150])

print("\n\n--- FINAL RESULTS ---")
for line in full_text.split('\n'):
    if 'Test Accuracy' in line or 'Test Loss' in line or 'Final Test Accuracy' in line:
        print(line.strip())
    if 'macro avg' in line or 'weighted avg' in line:
        print(line.strip())
