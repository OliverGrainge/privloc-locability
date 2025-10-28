"""
Enhanced Concept Analysis with Prompt Ensembling

This script analyzes the relationship between visual concepts and geolocation accuracy
using SigLip embeddings. Key improvements include:

1. PROMPT ENSEMBLING: Uses multiple diverse prompt templates per concept to reduce
   variance and improve robustness. Instead of single "a photo of X" prompts,
   now uses 8-16 diverse templates like "a satellite photo of X", "aerial view of X",
   "landscape with X", etc.

2. EMBEDDING AVERAGING: Averages normalized text embeddings across all prompts
   for each concept before computing similarities, providing more stable results.

3. CONFIGURABLE ENSEMBLING: Number of prompts per concept can be configured
   via num_prompts_per_concept parameter in config files.

Usage: python analyse.py <config_path>
"""

from dotenv import load_dotenv 
load_dotenv()

import torch
import numpy as np
from data.datasets.predictiongeodataset import PredictionGeoDataset
from torch.utils.data import DataLoader
import sys
from utils import load_config_yaml
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def enhance_concept_prompts_ensemble(concepts, num_prompts=12):
    """
    Enhanced concept prompts using prompt ensembling to improve SigLip text-image matching.
    Uses multiple diverse templates to reduce variance and improve robustness.
    
    Args:
        concepts: List of basic concept strings
        num_prompts: Number of prompt templates to use per concept (default: 12)
    
    Returns:
        List of enhanced concept prompts (flattened list with multiple prompts per concept)
    """
    # Diverse prompt templates for better robustness
    prompt_templates = [
        "a photo of {}",
        "a satellite photo of {}",
        "an image of {}",
        "a landscape with {}",
        "a picture of {}",
        "a view of {}",
        "a scene with {}",
        "a photograph of {}",
        "aerial view of {}",
        "overhead view of {}",
        "ground level view of {}",
        "close-up of {}",
        "wide shot of {}",
        "detailed view of {}",
        "natural scene with {}",
        "urban scene with {}"
    ]
    
    # Select subset of templates if num_prompts is less than total templates
    if num_prompts < len(prompt_templates):
        # Use diverse selection from templates
        step = len(prompt_templates) // num_prompts
        selected_templates = prompt_templates[::step][:num_prompts]
    else:
        selected_templates = prompt_templates
    
    enhanced_concepts = []
    
    for concept in concepts:
        concept_lower = concept.lower()
        
        # Handle multi-word concepts (like "blue sky", "red car", etc.)
        if ' ' in concept_lower:
            # For multi-word concepts, use them as-is in templates
            for template in selected_templates:
                enhanced_concepts.append(template.format(concept))
        # Handle single-word concepts
        else:
            # Use appropriate article based on starting letter
            if concept_lower[0] in 'aeiou':
                concept_with_article = f"an {concept}"
            else:
                concept_with_article = f"a {concept}"
            
            for template in selected_templates:
                enhanced_concepts.append(template.format(concept_with_article))
    
    return enhanced_concepts


def enhance_concept_prompts(concepts):
    """
    Legacy function for backward compatibility.
    Now uses prompt ensembling by default.
    
    Args:
        concepts: List of basic concept strings
    
    Returns:
        List of enhanced concept prompts
    """
    return enhance_concept_prompts_ensemble(concepts, num_prompts=12)


def load_siglip_model(device):
    """
    Load complete SigLip model for image-text similarity
    """
    from transformers import AutoModel, AutoProcessor
    
    model_name = "google/siglip-so400m-patch14-384"
    
    # Load processor and full model
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    return model, processor


def compute_text_embeddings_ensemble(texts, model, processor, device, num_prompts_per_concept=12):
    """
    Pre-compute text embeddings for concept strings using prompt ensembling.
    Averages embeddings from multiple prompts per concept for better robustness.
    
    Args:
        texts: List of text strings (e.g., ["vegetation", "building", "blue sky", "road"])
        model: SigLip model
        processor: SigLip processor
        device: torch device
        num_prompts_per_concept: Number of prompts per concept (should match enhance_concept_prompts_ensemble)
    
    Returns:
        text_embeds_norm: Normalized averaged text embeddings of shape [num_concepts, embed_dim]
    """
    # Generate ensemble prompts for each concept
    ensemble_prompts = enhance_concept_prompts_ensemble(texts, num_prompts=num_prompts_per_concept)
    
    print(f"Generated {len(ensemble_prompts)} ensemble prompts for {len(texts)} concepts")
    print(f"Sample prompts: {ensemble_prompts[:3]}...")
    
    # Process all ensemble prompts
    text_inputs = processor(
        text=ensemble_prompts,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    # Get text embeddings for all prompts
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)  # [num_prompts_total, embed_dim]
        
        # Normalize embeddings (important for cosine similarity)
        text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Reshape to [num_concepts, num_prompts_per_concept, embed_dim]
        text_embeds_norm = text_embeds_norm.view(len(texts), num_prompts_per_concept, -1)
        
        # Average embeddings across prompts for each concept
        text_embeds_avg = text_embeds_norm.mean(dim=1)  # [num_concepts, embed_dim]
        
        # Re-normalize the averaged embeddings
        text_embeds_avg_norm = text_embeds_avg / text_embeds_avg.norm(dim=-1, keepdim=True)
    
    print(f"Computed averaged embeddings: {text_embeds_avg_norm.shape}")
    return text_embeds_avg_norm


def compute_text_embeddings(texts, model, processor, device):
    """
    Legacy function for backward compatibility.
    Now uses prompt ensembling by default.
    
    Args:
        texts: List of text strings (e.g., ["vegetation", "building", "blue sky", "road"])
        model: SigLip model
        processor: SigLip processor
        device: torch device
    
    Returns:
        text_embeds_norm: Normalized text embeddings of shape [num_texts, embed_dim]
    """
    return compute_text_embeddings_ensemble(texts, model, processor, device, num_prompts_per_concept=12)


def compute_similarities_batch(image_embeds, text_embeds_norm, device):
    """
    Compute similarity between image embeddings and pre-computed text embeddings
    
    Args:
        image_embeds: Tensor of shape [batch_size, embed_dim] - pre-computed SigLip vision embeddings
        text_embeds_norm: Normalized text embeddings of shape [num_texts, embed_dim]
        device: torch device
    
    Returns:
        similarity_matrix: shape [batch_size, num_texts] with similarity scores
    """
    # Move image embeddings to device if needed
    image_embeds = image_embeds.to(device)
    
    with torch.no_grad():
        # Normalize image embeddings
        image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Compute similarity (cosine similarity via dot product)
        similarity = image_embeds_norm @ text_embeds_norm.T  # [batch_size, num_texts]
    
    return similarity


def compute_concept_similarities(dataset_path, concepts, original_concepts=None, device='cuda', batch_size=32, output_path=None, num_prompts_per_concept=12):
    """
    Compute similarity scores between all images in dataset and given concepts using prompt ensembling
    
    Args:
        dataset_path: Path to PredictionGeoDataset
        concepts: List of enhanced text concepts to compare against (should be from ensemble function)
        original_concepts: List of original concept names for output column names
        device: Device to run computation on
        batch_size: Batch size for processing
        output_path: Optional path to save results as parquet file
        num_prompts_per_concept: Number of prompts per concept (for ensemble averaging)
    
    Returns:
        DataFrame with columns: idx, true_lat, true_lon, pred_lat, pred_lon, 
                                similarity_concept_0, similarity_concept_1, ..., similarity_concept_n
    """
    # Load model
    print("Loading SigLip model...")
    model, processor = load_siglip_model(device)
    
    # Extract original concepts from the ensemble concepts
    if original_concepts is None:
        # If no original concepts provided, extract them from the ensemble concepts
        # This assumes concepts were generated using enhance_concept_prompts_ensemble
        original_concepts = []
        for i in range(0, len(concepts), num_prompts_per_concept):
            # Take the first prompt and extract the concept name
            first_prompt = concepts[i]
            # Extract concept from "a photo of a/an X" format
            if "a photo of a " in first_prompt:
                concept = first_prompt.replace("a photo of a ", "")
            elif "a photo of an " in first_prompt:
                concept = first_prompt.replace("a photo of an ", "")
            else:
                # Fallback: use the prompt as-is
                concept = first_prompt
            original_concepts.append(concept)
    
    # Pre-compute text embeddings using ensemble approach
    print(f"Pre-computing text embeddings using prompt ensembling...")
    print(f"Original concepts: {original_concepts}")
    text_embeds_norm = compute_text_embeddings_ensemble(original_concepts, model, processor, device, num_prompts_per_concept)
    print(f"Text embeddings computed: {text_embeds_norm.shape}")
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    dataset = PredictionGeoDataset(dataset_path)
    
    # Check if embeddings are available
    if not dataset.has_embeddings:
        raise ValueError("Dataset does not contain pre-computed embeddings!")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Store results
    results = {
        'idx': [],
        'true_lat': [],
        'true_lon': [],
        'pred_lat': [],
        'pred_lon': []
    }
    
    # Use original concept names for column names if provided, otherwise use enhanced concepts
    concept_names = original_concepts if original_concepts is not None else concepts
    
    # Add columns for each concept
    for concept_name in concept_names:
        results[f'similarity_{concept_name}'] = []
    
    # Stream through batches and compute similarities
    print("\nStreaming through dataset and computing similarities...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        # Get pre-computed image embeddings from batch
        image_embeds = batch['embedding']  # [batch_size, embed_dim]
        
        # Compute similarities using pre-computed text embeddings
        similarities = compute_similarities_batch(
            image_embeds, text_embeds_norm, device
        )  # [batch_size, num_concepts]
        
        # Store metadata
        results['idx'].extend(batch['idx'].cpu().numpy().tolist())
        results['true_lat'].extend(batch['true_lat'].cpu().numpy().tolist())
        results['true_lon'].extend(batch['true_lon'].cpu().numpy().tolist())
        results['pred_lat'].extend(batch['pred_lat'].cpu().numpy().tolist())
        results['pred_lon'].extend(batch['pred_lon'].cpu().numpy().tolist())
        
        # Store similarity scores for each concept
        similarities_np = similarities.cpu().numpy()
        for i, concept_name in enumerate(concept_names):
            results[f'similarity_{concept_name}'].extend(similarities_np[:, i].tolist())
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving results to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Results saved!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    for concept_name in concept_names:
        col_name = f'similarity_{concept_name}'
        print(f"\n{concept_name}:")
        print(f"  Mean: {df[col_name].mean():.4f}")
        print(f"  Std:  {df[col_name].std():.4f}")
        print(f"  Min:  {df[col_name].min():.4f}")
        print(f"  Max:  {df[col_name].max():.4f}")
    
    return df


def compute_concept_localization_correlations(df):
    """
    Compute correlations between concept similarities and negative log-transformed geodetic errors.
    High correlation now means HIGH localization ability (low error).
    
    Args:
        df: DataFrame with columns like 'similarity_vegetation', 'similarity_building', etc.
             and coordinate columns 'true_lat', 'true_lon', 'pred_lat', 'pred_lon'
    
    Returns:
        DataFrame with correlation results
    """
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance between two points in kilometers."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # Earth radius in km
    
    # Calculate haversine distance errors
    errors = haversine_distance(
        df['true_lat'], df['true_lon'], 
        df['pred_lat'], df['pred_lon']
    )
    
    # Negative log transform: -log(1 + error)
    # Now high values = low error = high localization ability
    neg_log_errors = -np.log1p(errors)  # -log(1 + error)
    
    # Find concept columns
    concept_cols = [col for col in df.columns if col.startswith('similarity_')]
    concepts = [col.replace('similarity_', '') for col in concept_cols]
    
    print(f"\nComputing correlations for {len(concepts)} concepts...")
    print(f"Error statistics:")
    print(f"  Mean error: {errors.mean():.2f} km")
    print(f"  Median error: {np.median(errors):.2f} km")
    print(f"  Std error: {errors.std():.2f} km")
    print(f"  Min error: {errors.min():.2f} km")
    print(f"  Max error: {errors.max():.2f} km")
    
    # Compute correlations
    results = []
    for concept, col in zip(concepts, concept_cols):
        similarities = df[col]
        
        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = pearsonr(similarities, neg_log_errors)
        
        # Spearman correlation (monotonic relationship)
        spearman_r, spearman_p = spearmanr(similarities, neg_log_errors)
        
        results.append({
            'concept': concept,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'pearson_significant': pearson_p < 0.05,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'spearman_significant': spearman_p < 0.05,
            'mean_similarity': similarities.mean(),
            'std_similarity': similarities.std(),
            'mean_error_km': errors.mean(),
            'mean_neg_log_error': neg_log_errors.mean(),
            'n_samples': len(df)
        })
    
    return pd.DataFrame(results)


def plot_correlation_bars(correlation_df, output_path):
    """
    Create bar chart visualization of correlation results.
    
    Args:
        correlation_df: DataFrame with correlation results
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Sort by Pearson correlation for better visualization
    df_sorted = correlation_df.sort_values('pearson_r', ascending=True)
    
    # Plot Pearson correlations
    colors = ['red' if not sig else 'green' if r > 0 else 'blue' 
              for r, sig in zip(df_sorted['pearson_r'], df_sorted['pearson_significant'])]
    
    bars1 = ax1.barh(df_sorted['concept'], df_sorted['pearson_r'], color=colors, alpha=0.8)
    ax1.set_xlabel('Pearson Correlation')
    ax1.set_title('Pearson Correlation')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation values
    for i, (bar, r) in enumerate(zip(bars1, df_sorted['pearson_r'])):
        ax1.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.01, 
                bar.get_y() + bar.get_height()/2, f'{r:.3f}', 
                ha='left' if bar.get_width() > 0 else 'right', va='center', fontsize=10)
    
    # Plot Spearman correlations
    bars2 = ax2.barh(df_sorted['concept'], df_sorted['spearman_r'], color=colors, alpha=0.8)
    ax2.set_xlabel('Spearman Correlation')
    ax2.set_title('Spearman Correlation')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Add correlation values
    for i, (bar, r) in enumerate(zip(bars2, df_sorted['spearman_r'])):
        ax2.text(bar.get_width() + 0.01 if bar.get_width() > 0 else bar.get_width() - 0.01, 
                bar.get_y() + bar.get_height()/2, f'{r:.3f}', 
                ha='left' if bar.get_width() > 0 else 'right', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation bar chart saved to: {output_path}")


def plot_scatter_plots(correlation_df, similarities_df, output_path):
    """
    Create scatter plot visualization showing concept similarity vs localization ability.
    
    Args:
        correlation_df: DataFrame with correlation results
        similarities_df: DataFrame with similarity scores and errors
        output_path: Path to save the plot
    """
    # Calculate neg_log_errors for scatter plots
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c
    
    errors = haversine_distance(
        similarities_df['true_lat'], similarities_df['true_lon'], 
        similarities_df['pred_lat'], similarities_df['pred_lon']
    )
    neg_log_errors = -np.log1p(errors)
    
    # Sort by Pearson correlation and get all concepts
    df_sorted = correlation_df.sort_values('pearson_r', ascending=False)
    all_concepts = df_sorted['concept'].tolist()
    
    # Calculate grid dimensions dynamically
    n_concepts = len(all_concepts)
    n_cols = min(4, n_concepts)  # Max 4 columns for readability
    n_rows = (n_concepts + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots with dynamic grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_concepts == 1:
        axes = [axes]  # Ensure axes is always a list
    elif n_rows == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    for i, concept in enumerate(all_concepts):
        ax = axes[i]
        col_name = f'similarity_{concept}'
        
        # Create scatter plot
        scatter = ax.scatter(similarities_df[col_name], neg_log_errors, 
                           alpha=0.6, s=20, c=errors, cmap='viridis_r')
        
        # Add trend line
        z = np.polyfit(similarities_df[col_name], neg_log_errors, 1)
        p = np.poly1d(z)
        ax.plot(similarities_df[col_name], p(similarities_df[col_name]), 
               "r--", alpha=0.8, linewidth=2)
        
        # Get correlation info
        corr_info = correlation_df[correlation_df['concept'] == concept].iloc[0]
        
        ax.set_xlabel(f'{concept.title()} Similarity')
        ax.set_ylabel('-log(1 + Error)')
        ax.set_title(f'{concept.title()}: r={corr_info["pearson_r"]:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for error
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Error (km)')
    
    # Hide unused subplots
    for i in range(n_concepts, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Scatter plots saved to: {output_path}")


def main(): 
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_path>")
        sys.exit(1)
    
    cfg_path = sys.argv[1]
    config = load_config_yaml(cfg_path)
    
    # Extract parameters
    dataset_path = config.parameters.dataset_path
    raw_concepts = list(config.parameters.concepts)
    
    # Get number of prompts per concept from config (default: 12)
    num_prompts_per_concept = getattr(config.parameters, 'num_prompts_per_concept', 12)
    
    # Enhance concept prompts using prompt ensembling for better SigLip matching
    concepts = enhance_concept_prompts_ensemble(raw_concepts, num_prompts=num_prompts_per_concept)
    print(f"Original concepts: {raw_concepts}")
    print(f"Using prompt ensembling with {num_prompts_per_concept} prompts per concept")
    print(f"Total enhanced prompts: {len(concepts)}")
    print(f"Sample enhanced prompts: {concepts[:6]}...")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results directory structure
    dataset_name = os.path.basename(dataset_path)
    results_dir = os.path.join("results", "analyse", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create output paths (use original concept names for cleaner file names)
    similarities_path = os.path.join(results_dir, f"concept_similarities_{dataset_name}.parquet")
    correlations_path = os.path.join(results_dir, f"concept_correlations_{dataset_name}.csv")
    bars_plot_path = os.path.join(results_dir, f"correlation_bars_{dataset_name}.png")
    scatter_plot_path = os.path.join(results_dir, f"correlation_scatter_{dataset_name}.png")
    
    # Compute similarities using prompt ensembling
    df = compute_concept_similarities(
        dataset_path=dataset_path,
        concepts=concepts,
        original_concepts=raw_concepts,
        device=device,
        batch_size=32,
        output_path=similarities_path,
        num_prompts_per_concept=num_prompts_per_concept
    )
    
    print(f"\nProcessed {len(df)} images successfully!")
    print(f"Similarities saved to: {similarities_path}")
    
    # Compute correlations with localization ability
    print("\n" + "="*60)
    print("COMPUTING CONCEPT-LOCALIZATION CORRELATIONS")
    print("="*60)
    print("High correlation = High localization ability (low error)")
    print("Low correlation = Low localization ability (high error)")
    
    correlation_df = compute_concept_localization_correlations(df)
    
    # Save correlation results
    correlation_df.to_csv(correlations_path, index=False)
    print(f"\nCorrelation results saved to: {correlations_path}")
    
    # Print correlation summary
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    print("Concepts ranked by Pearson correlation with localization ability:")
    print("(Higher correlation = better localization ability)")
    
    sorted_corr = correlation_df.sort_values('pearson_r', ascending=False)
    for _, row in sorted_corr.iterrows():
        significance = "***" if row['pearson_p'] < 0.001 else "**" if row['pearson_p'] < 0.01 else "*" if row['pearson_p'] < 0.05 else ""
        print(f"{row['concept']:12} | Pearson: {row['pearson_r']:6.3f} {significance:3} | Spearman: {row['spearman_r']:6.3f} | Mean similarity: {row['mean_similarity']:6.3f}")
    
    # Create visualizations
    print(f"\nCreating correlation bar chart...")
    plot_correlation_bars(correlation_df, bars_plot_path)
    
    print(f"\nCreating scatter plots...")
    plot_scatter_plots(correlation_df, df, scatter_plot_path)
    
    print(f"\nAll results saved to: {results_dir}/")
    print(f"  - Similarities: {similarities_path}")
    print(f"  - Correlations: {correlations_path}")
    print(f"  - Bar chart: {bars_plot_path}")
    print(f"  - Scatter plots: {scatter_plot_path}")


if __name__ == "__main__":
    main()