"""
System Test Script (Windows Compatible)

This script tests all major components of the ASL Recognition System.
Run this to verify everything is working correctly.

Usage:
    python test_system_simple.py
"""

import sys
from pathlib import Path
import numpy as np

print("="*70)
print("ASL RECOGNITION SYSTEM - COMPONENT TEST")
print("="*70)

# Test 1: Configuration
print("\n[TEST 1] Testing Configuration...")
try:
    from src.utils.config import (
        MediaPipeConfig,
        DataCollectionConfig,
        FeatureConfig,
        create_directories,
        print_configuration
    )
    create_directories()
    print("[OK] Configuration module working")
    print(f"     - Static classes: {len(DataCollectionConfig.STATIC_CLASSES)}")
    print(f"     - Dynamic classes: {len(DataCollectionConfig.DYNAMIC_CLASSES)}")
    print(f"     - Feature dimension: {FeatureConfig.FEATURE_DIM}")
except Exception as e:
    print(f"[ERROR] Configuration error: {e}")
    sys.exit(1)

# Test 2: Hand Detector
print("\n[TEST 2] Testing Hand Detector...")
try:
    from src.preprocessing.hand_detector import HandDetector
    import cv2

    detector = HandDetector()
    print("[OK] Hand detector initialized")
    print(f"     - Max hands: {MediaPipeConfig.MAX_NUM_HANDS}")
    print(f"     - Detection confidence: {MediaPipeConfig.MIN_DETECTION_CONFIDENCE}")

    # Test with dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    success, results = detector.detect_hands(dummy_image)
    print(f"     - Detector callable: [OK]")

    detector.close()
except Exception as e:
    print(f"[ERROR] Hand detector error: {e}")
    sys.exit(1)

# Test 3: Feature Extractor
print("\n[TEST 3] Testing Feature Extractor...")
try:
    from src.preprocessing.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    print("[OK] Feature extractor initialized")
    print(f"     - Normalization: {extractor.normalization_method}")
    print(f"     - Feature dimension: {extractor.get_feature_dimension()}")

    # Test with dummy landmarks
    dummy_landmarks = np.random.rand(21, 3).astype(np.float32)
    features = extractor.extract_features(dummy_landmarks)
    print(f"     - Feature extraction: [OK]")
    print(f"     - Output shape: {features.shape}")
    print(f"     - Output range: [{features.min():.3f}, {features.max():.3f}]")
except Exception as e:
    print(f"[ERROR] Feature extractor error: {e}")
    sys.exit(1)

# Test 4: Data Collection Scripts
print("\n[TEST 4] Checking Data Collection Scripts...")
try:
    from src.data_collection.collect_static import StaticGestureCollector
    from src.data_collection.collect_dynamic import DynamicGestureCollector

    print("[OK] Static collection script importable")
    print("[OK] Dynamic collection script importable")
except Exception as e:
    print(f"[ERROR] Data collection error: {e}")
    sys.exit(1)

# Test 5: Directory Structure
print("\n[TEST 5] Checking Directory Structure...")
try:
    from src.utils.config import (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        LABELS_DIR
    )

    dirs_to_check = [
        ("Data", DATA_DIR),
        ("Raw data", RAW_DATA_DIR),
        ("Processed data", PROCESSED_DATA_DIR),
        ("Models", MODELS_DIR),
        ("Labels", LABELS_DIR)
    ]

    for name, path in dirs_to_check:
        if path.exists():
            print(f"     [OK] {name}: {path}")
        else:
            print(f"     [WARNING] {name} missing: {path}")
except Exception as e:
    print(f"[ERROR] Directory structure error: {e}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("\n[SUCCESS] All core components working!")
print("\nNext Steps:")
print("1. Run hand detector test:")
print("   python src/preprocessing/hand_detector.py")
print("\n2. Collect your dataset:")
print("   python src/data_collection/collect_static.py")
print("   python src/data_collection/collect_dynamic.py")
print("\n3. After data collection, proceed to Step 3 (Model Training)")
print("\n" + "="*70)
