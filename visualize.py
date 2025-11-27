import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from .gradcam import GradCAM

def visualize_enhanced_results(history, evaluation_results, output_dir, model, test_loader, device):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Training Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_mse'], label='Training MSE')
    axes[0, 1].plot(history['val_mse'], label='Validation MSE')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)

    all_losses = history['train_loss'] + history['val_loss']
    axes[1, 1].hist(all_losses, bins=30, alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Zero')
    axes[1, 1].set_xlabel('Loss Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Loss Value Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

    target_names = ['E1', 'E2', 'E3', 'G12', 'G13', 'G23', 'v12', 'v13', 'v23']

    csv_data = []
    for i in range(len(evaluation_results['model_names'])):
        row_data = {
            'model_name': evaluation_results['model_names'][i]
        }
        for j, name in enumerate(target_names):
            row_data[f'{name}_true'] = evaluation_results['targets'][i, j]
            row_data[f'{name}_pred'] = evaluation_results['predictions'][i, j]
            row_data[f'{name}_uncertainty'] = evaluation_results['uncertainties'][i, j]
        csv_data.append(row_data)

    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join(output_dir, 'predictions_with_uncertainty.csv'), index=False)
    print(f"Saved predictions to {os.path.join(output_dir, 'predictions_with_uncertainty.csv')}")
 
    all_uncertainties = evaluation_results['uncertainties'].flatten()
    uncertainty_min = all_uncertainties.min()
    uncertainty_max = all_uncertainties.max()
 
    print(f"Uncertainty range: [{uncertainty_min:.6f}, {uncertainty_max:.6f}]")

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    scatter_plots = []
 
    for i, name in enumerate(target_names):
        ax = axes[i]
     
        preds = evaluation_results['predictions'][:, i]
        targets = evaluation_results['targets'][:, i]
        uncertainties = evaluation_results['uncertainties'][:, i]
     
        uncertainties_norm = (uncertainties - uncertainty_min) / (uncertainty_max - uncertainty_min + 1e-8)
     
        scatter = ax.scatter(targets, preds, c=uncertainties_norm,
                           cmap='viridis', alpha=0.6, vmin=0, vmax=1)
        scatter_plots.append(scatter)

        ax.set_aspect('equal', adjustable='box')

        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())

        plot_range = [min_val - 0.1 * abs(max_val - min_val),
                      max_val + 0.1 * abs(max_val - min_val)]
        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
     
        ax.plot(plot_range, plot_range, 'r--', lw=2)
     
        ax.fill_between(plot_range,
                       [v*0.9 for v in plot_range],
                       [v*1.1 for v in plot_range],
                       alpha=0.2, color='gray', label='±10% error')
     
        ax.set_xlabel(f'True {name}', fontsize=12)
        ax.set_ylabel(f'Predicted {name}', fontsize=12)
        ax.set_title(f'{name} - R² = {evaluation_results["r2_scores"][name]:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3)
     

    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02]) # [left, bottom, width, height]
    cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized Uncertainty', fontsize=14)

    cbar_ticks = cbar.get_ticks()
    cbar_labels = [f'{uncertainty_min + t * (uncertainty_max - uncertainty_min):.3f}'
                   for t in cbar_ticks]
    cbar.set_ticklabels(cbar_labels)
 
    plt.savefig(os.path.join(output_dir, 'predictions_with_uncertainty.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
 
    # R² scores
    axes[0].bar(target_names, [evaluation_results['r2_scores'][name] for name in target_names])
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² Scores by Property')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    for i, v in enumerate([evaluation_results['r2_scores'][name] for name in target_names]):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
 
    # RMSE
    axes[1].bar(target_names, [evaluation_results['rmse_scores'][name] for name in target_names])
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE by Property')
    axes[1].grid(True, alpha=0.3)
 
    # MAPE
    axes[2].bar(target_names, [evaluation_results['mape_scores'][name] for name in target_names])
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('Mean Absolute Percentage Error by Property')
    axes[2].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
    plt.close()

    attention_weights = evaluation_results['attention_weights']
    avg_attention = np.mean(attention_weights, axis=0)
 
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attention, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Attention Weight')
    plt.xticks([0, 1, 2], ['Image', 'Structural', 'Material'])
    plt.yticks([0, 1, 2], ['Image', 'Structural', 'Material'])
    plt.title('Cross-Modal Attention Weights')
 
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{avg_attention[i, j]:.3f}',
                    ha='center', va='center', color='white' if avg_attention[i, j] > 0.5 else 'black')
 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'))
    plt.close()

    v23_idx = 8
    v23_preds = evaluation_results['predictions'][:, v23_idx]
    v23_targets = evaluation_results['targets'][:, v23_idx]
 
    plt.figure(figsize=(12, 5))
 
    plt.subplot(1, 2, 1)
    plt.hist(v23_targets, bins=30, alpha=0.5, label='True v23', color='blue')
    plt.hist(v23_preds, bins=30, alpha=0.5, label='Predicted v23', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='0.5 threshold')
    plt.xlabel('v23 value')
    plt.ylabel('Frequency')
    plt.title('Distribution of v23 values')
    plt.legend()
    plt.grid(True, alpha=0.3)
 
    plt.subplot(1, 2, 2)
    errors = v23_preds - v23_targets
    plt.hist(errors, bins=30, alpha=0.7, color='green')
    plt.xlabel('Prediction Error (Pred - True)')
    plt.ylabel('Frequency')
    plt.title('v23 Prediction Error Distribution')
    plt.axvline(x=0, color='red', linestyle='--', label='Zero error')
    plt.legend()
    plt.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v23_analysis.png'))
    plt.close()

    gradcam_dir = os.path.join(output_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    target_layer = model.image_encoder.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)

    num_samples = 5
    counter = 0
    for batch in test_loader:
        images = batch['image'].to(device)
        structural = batch['structural'].to(device)
        material_props = batch['material_props'].to(device)
        img_paths = batch['img_path']

        for i in range(images.size(0)):
            if counter >= num_samples:
                break

            single_image = images[i:i+1]
            single_structural = structural[i:i+1]
            single_material = material_props[i:i+1]
            original_img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
            original_img = cv2.resize(original_img, (512, 512))
            original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels

            heatmap = gradcam(single_image, single_structural, single_material)

            heatmap = cv2.resize(heatmap, (512, 512))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

            save_path = os.path.join(gradcam_dir, f'gradcam_{counter}.png')
            cv2.imwrite(save_path, superimposed_img)

            counter += 1

        if counter >= num_samples:
            break

    print(f"Generated Grad-CAM visualizations for {num_samples} samples in {gradcam_dir}")