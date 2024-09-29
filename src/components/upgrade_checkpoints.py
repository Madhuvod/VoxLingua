import torch
from pytorch_lightning.utilities.migration import migrate_checkpoint
from omegaconf import ListConfig, ContainerMetadata
import torch.serialization

# Add ListConfig and ContainerMetadata to the safe globals
torch.serialization.add_safe_globals([ListConfig, ContainerMetadata])

checkpoint_path = "../../../../../.cache/torch/pyannote/models--pyannote--segmentation/snapshots/c4c8ceafcbb3a7a280c2d357aee9fbc9b0be7f9b/pytorch_model.bin"

try:
    # Load the checkpoint on CPU
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)

    # Migrate the checkpoint
    migrated_checkpoint = migrate_checkpoint(checkpoint)

    # Save the migrated checkpoint
    torch.save(migrated_checkpoint, checkpoint_path + ".migrated", weights_only=True)

    print(f"Migrated checkpoint saved to {checkpoint_path}.migrated")

except ModuleNotFoundError as e:
    print(f"Error: Missing module - {e}")
    print("Please install the missing module and try again.")
    print("You can install it using: pip install <module_name>")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Attempting to load without weights_only...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        migrated_checkpoint = migrate_checkpoint(checkpoint)
        torch.save(migrated_checkpoint, checkpoint_path + ".migrated")
        print(f"Migrated checkpoint saved to {checkpoint_path}.migrated")
    except Exception as e2:
        print(f"Failed to load and migrate checkpoint: {e2}")
        print("Please check the checkpoint file and ensure all dependencies are installed.")