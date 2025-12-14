import sys
from transformers import AutoProcessor, AutoTokenizer, AutoFeatureExtractor

model_name = "microsoft/VibeVoice-Realtime-0.5B"
print(f"Attempting to load processor for: {model_name}")

try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("Success: AutoProcessor loaded.")
except Exception as e:
    print(f"Failure: AutoProcessor failed with: {e}")

print("\nAttempting to load Tokenizer separately...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Success: AutoTokenizer loaded.")
except Exception as e:
    print(f"Failure: AutoTokenizer failed with: {e}")

print("\nAttempting to load FeatureExtractor separately...")
try:
    fe = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    print("Success: AutoFeatureExtractor loaded.")
except Exception as e:
    print(f"Failure: AutoFeatureExtractor failed with: {e}")
