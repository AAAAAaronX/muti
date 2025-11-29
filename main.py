import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from .dataset import MaterialPropertiesDataset
from .models import ImprovedMultiModalFusionNetwork
from .train import train_model
from .evaluate import evaluate_model, generate_performance_report
from .visualize import visualize_enhanced_results

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def main():
    csv_path = ''
    img_dir = ''
    output_dir =''
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])

    print("Loading dataset...")
    dataset = MaterialPropertiesDataset(csv_path, img_dir, transform=transform, augment=False)

    structural_dim = len(dataset.fiber_shapes) # 5
    material_dim = len(dataset.material_columns) # 12
    num_outputs = 9 
 
    print(f"\nDataset size: {len(dataset)}")
    print(f"Fiber shapes: {dataset.fiber_shapes}")
    print(f"Structural features dimension: {structural_dim} ({len(dataset.fiber_shapes)} shape encodings)")
    print(f"Material properties: {material_dim} dimensions")
    print(f"Target properties: {dataset.target_columns}")

    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.1,
        random_state=42
    )
 
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.2,
        random_state=42
    )

    train_dataset = MaterialPropertiesDataset(csv_path, img_dir, transform=transform, augment=True)
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=4)
 
    print(f"\nDataset splits:")
    print(f"Training: {len(train_indices)} samples")
    print(f"Validation: {len(val_indices)} samples")
    print(f"Testing: {len(test_indices)} samples")

    print(f"\nModel architecture:")
    print(f"Structural features: {structural_dim}")
    print(f"Material features: {material_dim}")
    print(f"Output dimensions: {num_outputs}")
 
    model = ImprovedMultiModalFusionNetwork(structural_dim, material_dim, num_outputs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

    print("\nStarting training...")
    trained_model, history = train_model(model, train_loader, val_loader, epochs=100, early_stopping_patience=15)
 
    print("\nEvaluating model...")
    scalers = dataset.get_scalers()
    evaluation_results = evaluate_model(trained_model, test_loader, scalers['target'])

    print("\nGenerating visualizations...")
    visualize_enhanced_results(history, evaluation_results, output_dir, trained_model, test_loader)

    print("\nSaving model...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'model_config': {
            'structural_dim': structural_dim,
            'material_dim': material_dim,
            'num_outputs': num_outputs
        },
        'scalers': scalers,
        'material_columns': dataset.material_columns,
        'target_columns': dataset.target_columns,
        'evaluation_results': evaluation_results,
        'fiber_shapes': dataset.fiber_shapes
    }, os.path.join(output_dir, 'improved_model.pth'))

    generate_performance_report(evaluation_results, output_dir)
 
    print("\nTraining completed successfully!")
    print(f"Results saved to: {output_dir}")
 
    return trained_model, dataset, evaluation_results

if __name__ == "__main__":
    trained_model, dataset, evaluation_results = main()