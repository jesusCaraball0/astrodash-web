#!/usr/bin/env python
"""Check what model and params the backend is actually using."""

import os
import sys
from pathlib import Path

# Add prod_backend to path so "app" can be imported
PROD_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROD_BACKEND))

from app.config.settings import get_settings
from app.infrastructure.ml.classifiers.dash_classifier import DashClassifier
import torch
import pickle

def main():
    print("="*70)
    print("BACKEND MODEL CONFIGURATION CHECK")
    print("="*70)
    
    settings = get_settings()
    
    # Paths in settings are relative to prod_backend (where uvicorn runs)
    model_path = (PROD_BACKEND / settings.dash_model_path).resolve()
    params_path = (PROD_BACKEND / settings.dash_training_params_path).resolve()
    
    print("\n📁 Settings paths (as in config):")
    print(f"  dash_model_path: {settings.dash_model_path}")
    print(f"  dash_training_params_path: {settings.dash_training_params_path}")
    print(f"\n📁 Resolved paths (relative to prod_backend):")
    print(f"  Model:  {model_path}")
    print(f"  Params: {params_path}")
    print(f"  w0: {settings.w0}, w1: {settings.w1}, nw: {settings.nw}")
    
    print(f"\n📂 File existence:")
    print(f"  Model file exists: {model_path.exists()}")
    if model_path.exists():
        print(f"    Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Params file exists: {params_path.exists()}")
    
    # Load and check model
    if model_path.exists():
        print(f"\n🤖 Model file info:")
        try:
            state_dict = torch.load(str(model_path), map_location='cpu', weights_only=False)
            print(f"  Keys: {len(state_dict.keys())}")
            if 'output.weight' in state_dict:
                n_classes = state_dict['output.weight'].shape[0]
                print(f"  Output classes (from output.weight): {n_classes}")
            elif 'classifier.3.weight' in state_dict:
                n_classes = state_dict['classifier.3.weight'].shape[0]
                print(f"  Output classes (from classifier.3.weight): {n_classes}")
            else:
                print(f"  ⚠️  Could not determine number of classes")
                print(f"  Available keys: {list(state_dict.keys())[:10]}...")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
    
    # Load and check params
    if params_path.exists():
        print(f"\n📋 Training params info:")
        try:
            with open(params_path, 'rb') as f:
                params = pickle.load(f, encoding='latin1')
            print(f"  Keys: {sorted(params.keys())}")
            print(f"  w0: {params.get('w0')}, w1: {params.get('w1')}, nw: {params.get('nw')}")
            print(f"  nTypes: {params.get('nTypes')}")
            print(f"  typeList: {params.get('typeList', [])}")
            min_age = params.get('minAge', -20)
            max_age = params.get('maxAge', 50)
            age_bin_size = params.get('ageBinSize', 4)
            num_age_bins = int((max_age - min_age) / age_bin_size) + 1
            total_classes = params.get('nTypes', 17) * num_age_bins
            print(f"  Expected total classes: {total_classes} ({params.get('nTypes')} types × {num_age_bins} age bins)")
        except Exception as e:
            print(f"  ❌ Error loading params: {e}")
    
    # Try to instantiate classifier (from prod_backend cwd so relative paths work)
    print(f"\n🔧 Testing DashClassifier instantiation:")
    orig_cwd = os.getcwd()
    try:
        os.chdir(PROD_BACKEND)
        classifier = DashClassifier(config=settings)
        print(f"  ✅ Classifier loaded successfully")
        print(f"  Model path used: {classifier.model_path}")
        print(f"  Type names count: {len(classifier.type_names_list)}")
        if classifier.type_names_list:
            print(f"  First 3 type names: {classifier.type_names_list[:3]}")
            print(f"  Last 3 type names: {classifier.type_names_list[-3:]}")
        if classifier.model is not None:
            # Get actual output size
            with torch.no_grad():
                dummy_input = torch.zeros(1, classifier.nw)
                output = classifier.model(dummy_input)
                print(f"  Model output shape: {output.shape}")
                print(f"  Model output classes: {output.shape[1]}")
        else:
            print(f"  ⚠️  Model is None!")
    except Exception as e:
        print(f"  ❌ Error instantiating classifier: {e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(orig_cwd)
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
