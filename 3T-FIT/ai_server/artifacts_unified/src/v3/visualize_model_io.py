"""
visualize_model_io.py
Script để visualize input và output của V3 model

Hiển thị:
- Input features shape và values
- Model predictions (1RM, suitability, readiness)
- Raw input → Processed input → Output transformation
- Sample data visualization

Author: Claude Code Assistant
Date: 2025-11-25
"""

import os, json, numpy as np, pandas as pd, torch
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Import model classes
try:
    from train_v3_enhanced import (
        V3EnhancedModel, MOOD_MAPPING, FATIGUE_MAPPING, EFFORT_MAPPING,
        calculate_readiness_factor, decode_1rm_to_workout, WORKOUT_GOAL_MAPPING,
        map_sepa_to_numeric
    )
except ImportError:
    print("Error: Could not import from train_v3_enhanced.py")
    exit(1)

class ModelIOVisualizer:
    """Visualize input and output of V3 model"""

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model artifacts"""
        try:
            # Load model
            model_path = self.artifacts_dir / "best_v3.pt"
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            self.model = V3EnhancedModel(
                input_dim=checkpoint['input_dim'],
                **checkpoint.get('model_config', {})
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Load preprocessor
            preprocessor_path = self.artifacts_dir / "preprocessor_v3.joblib"
            self.preprocessor = joblib.load(preprocessor_path)

            # Load metadata
            metadata_path = self.artifacts_dir / "meta_v3.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            self.feature_columns = self.metadata['dataset_info']['feature_columns']

            print("✅ Artifacts loaded successfully")
            print(f"   - Model input dim: {checkpoint['input_dim']}")
            print(f"   - Feature columns: {self.feature_columns}")

        except Exception as e:
            print(f"❌ Error loading artifacts: {e}")
            raise

    def visualize_single_sample(self, sample_data: Dict):
        """Visualize single sample input/output"""
        print("="*80)
        print("🔍 SINGLE SAMPLE VISUALIZATION")
        print("="*80)

        # Convert dict to DataFrame
        df_sample = pd.DataFrame([sample_data])

        print("\n[1] RAW INPUT DATA:")
        print("-" * 40)
        for key, value in sample_data.items():
            print(f"{key:20}: {value}")

        # Process SePA if needed
        if 'mood' in sample_data and isinstance(sample_data['mood'], str):
            df_sample['mood_numeric'] = map_sepa_to_numeric(sample_data['mood'], MOOD_MAPPING)
        elif 'mood_numeric' not in sample_data:
            df_sample['mood_numeric'] = 3  # Default

        if 'fatigue' in sample_data and isinstance(sample_data['fatigue'], str):
            df_sample['fatigue_numeric'] = map_sepa_to_numeric(sample_data['fatigue'], FATIGUE_MAPPING)
        elif 'fatigue_numeric' not in sample_data:
            df_sample['fatigue_numeric'] = 3  # Default

        if 'effort' in sample_data and isinstance(sample_data['effort'], str):
            df_sample['effort_numeric'] = map_sepa_to_numeric(sample_data['effort'], EFFORT_MAPPING)
        elif 'effort_numeric' not in sample_data:
            df_sample['effort_numeric'] = 3  # Default

        # Calculate readiness factor
        readiness_factor = calculate_readiness_factor(
            df_sample['mood_numeric'].iloc[0],
            df_sample['fatigue_numeric'].iloc[0],
            df_sample['effort_numeric'].iloc[0]
        )
        df_sample['readiness_factor'] = readiness_factor

        print(f"\n[2] PROCESSED SEPA VALUES:")
        print("-" * 40)
        print(f"{'mood_numeric':20}: {df_sample['mood_numeric'].iloc[0]} ({sample_data.get('mood', 'N/A')})")
        print(f"{'fatigue_numeric':20}: {df_sample['fatigue_numeric'].iloc[0]} ({sample_data.get('fatigue', 'N/A')})")
        print(f"{'effort_numeric':20}: {df_sample['effort_numeric'].iloc[0]} ({sample_data.get('effort', 'N/A')})")
        print(f"{'readiness_factor':20}: {readiness_factor:.3f}")

        # Extract features for model
        available_features = [col for col in self.feature_columns if col in df_sample.columns]
        df_features = df_sample[available_features].copy()

        print(f"\n[3] MODEL INPUT FEATURES:")
        print("-" * 40)
        for col in available_features:
            value = df_features[col].iloc[0]
            print(f"{col:20}: {value}")

        # Apply preprocessing
        processed_input = self.preprocessor.transform(df_features)
        if hasattr(processed_input, 'toarray'):
            processed_input = processed_input.toarray()

        print(f"\n[4] PROCESSED INPUT ARRAY:")
        print("-" * 40)
        print(f"Shape: {processed_input.shape}")
        print(f"Values: {processed_input[0][:10]}...")  # Show first 10 values

        # Get feature names after preprocessing
        feature_names = self.preprocessor.get_feature_names_out()
        print(f"\nFeature names after preprocessing: {list(feature_names)}")

        # Make prediction
        with torch.no_grad():
            input_tensor = torch.from_numpy(processed_input.astype(np.float32))
            pred_1rm, pred_suitability, pred_readiness = self.model(input_tensor)

        print(f"\n[5] MODEL OUTPUTS (RAW):")
        print("-" * 40)
        print(f"1RM tensor:         {pred_1rm}")
        print(f"Suitability tensor: {pred_suitability}")
        print(f"Readiness tensor:   {pred_readiness}")

        # Convert to final values
        final_1rm = pred_1rm.item()
        final_suitability = pred_suitability.item()
        final_readiness = pred_readiness.item()

        print(f"\n[6] FINAL OUTPUT VALUES:")
        print("-" * 40)
        print(f"{'Predicted 1RM':20}: {final_1rm:.2f} kg")
        print(f"{'Suitability Score':20}: {final_suitability:.3f}")
        print(f"{'Readiness Factor':20}: {final_readiness:.3f}")

        # Generate workout recommendations
        print(f"\n[7] WORKOUT RECOMMENDATIONS:")
        print("-" * 40)

        goals = ['strength', 'hypertrophy', 'endurance']
        for goal in goals:
            workout = decode_1rm_to_workout(final_1rm, goal, final_readiness)
            print(f"\n{goal.upper()}:")
            print(f"  Target 1RM: {workout['predicted_1rm']:.1f} kg")
            print(f"  Adjusted 1RM: {workout['adjusted_1rm']:.1f} kg (readiness: {workout['readiness_factor']})")
            print(f"  Weight: {workout['training_weight_kg']['recommended']:.1f} kg ({workout['training_weight_kg']['min']:.1f}-{workout['training_weight_kg']['max']:.1f})")
            print(f"  Reps: {workout['reps']['recommended']} ({workout['reps']['min']}-{workout['reps']['max']})")
            print(f"  Sets: {workout['sets']['recommended']} ({workout['sets']['min']}-{workout['sets']['max']})")
            print(f"  Rest: {workout['rest_minutes']['recommended']:.1f} min")

        return {
            'raw_input': sample_data,
            'processed_features': df_features.iloc[0].to_dict(),
            'processed_array': processed_input[0],
            'predictions': {
                '1rm': final_1rm,
                'suitability': final_suitability,
                'readiness': final_readiness
            },
            'feature_names': list(feature_names)
        }

    def visualize_dataset_samples(self, data_file: str, num_samples: int = 5):
        """Visualize multiple samples from dataset"""
        print(f"\n{'='*80}")
        print(f"📊 DATASET SAMPLES VISUALIZATION: {data_file}")
        print(f"{'='*80}")

        # Load dataset
        df = pd.read_excel(data_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Select samples
        samples = df.head(num_samples).to_dict('records')

        all_results = []

        for i, sample in enumerate(samples):
            print(f"\n{'='*60}")
            print(f"SAMPLE {i+1}/{num_samples}")
            print(f"{'='*60}")

            # Prepare sample with required columns
            processed_sample = {}

            # Basic features
            basic_features = ['age', 'weight_kg', 'height_m', 'bmi', 'experience_level',
                            'workout_frequency', 'resting_heartrate', 'gender']
            for feature in basic_features:
                if feature in sample:
                    processed_sample[feature] = sample[feature]

            # Handle estimated_1rm (target)
            if 'estimated_1rm' in sample:
                processed_sample['true_1rm'] = sample['estimated_1rm']

            # Handle SePA
            sepa_mapping = {
                'mood': ('mood', MOOD_MAPPING),
                'fatigue': ('fatigue', FATIGUE_MAPPING),
                'effort': ('effort', EFFORT_MAPPING)
            }

            for sepa_col, (target_col, mapping) in sepa_mapping.items():
                if sepa_col in sample:
                    if isinstance(sample[sepa_col], str):
                        processed_sample[sepa_col] = sample[sepa_col]
                    else:
                        # If already numeric, convert to string for processing
                        processed_sample[sepa_col] = str(int(sample[sepa_col]))
                        # Also add numeric version
                        processed_sample[f'{sepa_col}_numeric'] = sample[sepa_col]

            # Visualize this sample
            result = self.visualize_single_sample(processed_sample)
            result['sample_index'] = i
            all_results.append(result)

        # Summary comparison
        print(f"\n{'='*80}")
        print("📋 SUMMARY COMPARISON")
        print(f"{'='*80}")

        print(f"{'Sample':<10} {'True 1RM':<12} {'Pred 1RM':<12} {'Error':<12} {'Suitability':<12} {'Readiness':<12}")
        print("-" * 80)

        for result in all_results:
            sample_idx = result['sample_index']
            true_1rm = result['raw_input'].get('true_1rm', 'N/A')
            pred_1rm = result['predictions']['1rm']
            suitability = result['predictions']['suitability']
            readiness = result['predictions']['readiness']

            if true_1rm != 'N/A':
                error = abs(pred_1rm - true_1rm)
                error_str = f"{error:.1f}"
            else:
                error_str = "N/A"

            print(f"{sample_idx+1:<10} {true_1rm:<12} {pred_1rm:<12.1f} {error_str:<12} {suitability:<12.3f} {readiness:<12.3f}")

        return all_results

    def visualize_preprocessing_pipeline(self, sample_data: Dict):
        """Detailed visualization of preprocessing pipeline"""
        print("\n" + "="*80)
        print("🔧 PREPROCESSING PIPELINE VISUALIZATION")
        print("="*80)

        # Create DataFrame
        df_sample = pd.DataFrame([sample_data])

        # Show original data
        print("\n[1] ORIGINAL DATA:")
        print(df_sample.to_string())

        # Process SePA columns
        for col, mapping in [('mood', MOOD_MAPPING), ('fatigue', FATIGUE_MAPPING), ('effort', EFFORT_MAPPING)]:
            if col in df_sample.columns:
                original_values = df_sample[col].copy()
                if df_sample[col].dtype == 'object':
                    df_sample[f'{col}_numeric'] = df_sample[col].apply(lambda x: map_sepa_to_numeric(x, mapping))
                else:
                    df_sample[f'{col}_numeric'] = df_sample[col].astype(float)
                print(f"\n[2] {col.upper()} PROCESSING:")
                print(f"   Original: {original_values.tolist()}")
                print(f"   Numeric:  {df_sample[f'{col}_numeric'].tolist()}")

        # Feature selection
        feature_cols = [col for col in self.feature_columns if col in df_sample.columns]
        df_features = df_sample[feature_cols].copy()

        print(f"\n[3] FEATURE SELECTION:")
        print(f"   Selected features: {feature_cols}")
        print(df_features.to_string())

        # Preprocessing step by step
        print(f"\n[4] PREPROCESSING STEPS:")

        # Handle missing values
        print("   [4a] Handling missing values...")
        df_imputed = df_features.copy()
        for col in df_features.columns:
            if df_features[col].isnull().any():
                median_val = df_features[col].median()
                df_imputed[col] = df_features[col].fillna(median_val)
                print(f"       {col}: filled {df_features[col].isnull().sum()} missing values with {median_val}")

        # Handle categorical variables
        print("   [4b] Handling categorical variables...")
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df_imputed.select_dtypes(include=['number']).columns.tolist()

        if categorical_cols:
            print(f"       Categorical: {categorical_cols}")
        if numeric_cols:
            print(f"       Numeric: {numeric_cols}")

        # Apply preprocessing
        processed = self.preprocessor.transform(df_features)
        if hasattr(processed, 'toarray'):
            processed = processed.toarray()

        print(f"\n[5] FINAL PROCESSED ARRAY:")
        print(f"   Shape: {processed.shape}")
        print(f"   Type: {type(processed)}")
        print(f"   Min/Max values: [{processed.min():.3f}, {processed.max():.3f}]")

        # Show feature importance (if available)
        feature_names = self.preprocessor.get_feature_names_out()
        print(f"\n[6] FEATURE NAMES AFTER PREPROCESSING:")
        for i, name in enumerate(feature_names):
            print(f"   {i:3d}: {name} = {processed[0, i]:.4f}")

        return processed, feature_names


def main():
    """Main function to demonstrate model IO"""
    artifacts_dir = "D:/dacn_omnimer_health/3T-FIT/ai_server/artifacts_unified/v3"

    # Initialize visualizer
    visualizer = ModelIOVisualizer(artifacts_dir)

    # Example 1: Single sample with raw data
    print("\n" + "="*100)
    print("EXAMPLE 1: SINGLE SAMPLE WITH RAW USER INPUT")
    print("="*100)

    sample_user = {
        'age': 28,
        'weight_kg': 75.5,
        'height_m': 1.78,
        'bmi': 23.8,
        'experience_level': 2,
        'workout_frequency': 4,
        'resting_heartrate': 65,
        'gender': 'male',
        'mood': 'good',
        'fatigue': 'low',
        'effort': 'medium'
    }

    visualizer.visualize_single_sample(sample_user)

    # Example 2: Detailed preprocessing pipeline
    print("\n" + "="*100)
    print("EXAMPLE 2: DETAILED PREPROCESSING PIPELINE")
    print("="*100)

    visualizer.visualize_preprocessing_pipeline(sample_user)

    # Example 3: Dataset samples (if available)
    data_dir = Path(artifacts_dir).parent / "src" / "v3" / "data"
    test_file = data_dir / "test_dataset.xlsx"

    if test_file.exists():
        print("\n" + "="*100)
        print("EXAMPLE 3: DATASET SAMPLES")
        print("="*100)
        visualizer.visualize_dataset_samples(str(test_file), num_samples=3)
    else:
        print(f"\nTest dataset not found at {test_file}")

    # Example 4: Different user profiles
    print("\n" + "="*100)
    print("EXAMPLE 4: DIFFERENT USER PROFILES")
    print("="*100)

    profiles = [
        {
            'name': 'Beginner Male',
            'data': {
                'age': 22, 'weight_kg': 70, 'height_m': 1.75, 'bmi': 22.9,
                'experience_level': 1, 'workout_frequency': 2, 'resting_heartrate': 70,
                'gender': 'male', 'mood': 'neutral', 'fatigue': 'medium', 'effort': 'low'
            }
        },
        {
            'name': 'Advanced Female',
            'data': {
                'age': 35, 'weight_kg': 60, 'height_m': 1.65, 'bmi': 22.0,
                'experience_level': 3, 'workout_frequency': 5, 'resting_heartrate': 60,
                'gender': 'female', 'mood': 'very good', 'fatigue': 'low', 'effort': 'high'
            }
        }
    ]

    for profile in profiles:
        print(f"\n{'='*60}")
        print(f"PROFILE: {profile['name']}")
        print(f"{'='*60}")
        result = visualizer.visualize_single_sample(profile['data'])

        # Quick summary
        print(f"\nQuick Summary:")
        print(f"  1RM Prediction: {result['predictions']['1rm']:.1f} kg")
        print(f"  Strength Workout: {decode_1rm_to_workout(result['predictions']['1rm'], 'strength', result['predictions']['readiness'])['training_weight_kg']['recommended']:.1f} kg")

    print("\n" + "="*100)
    print("✅ VISUALIZATION COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()