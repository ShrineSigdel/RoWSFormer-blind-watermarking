import torch
from tqdm import tqdm


def train_epoch(encoder, decoder, noise_layer, dataloader, criterion, 
                optimizer, device, epoch):
    """Train for one epoch."""
    encoder.train()
    decoder.train()
    
    total_losses = {'total': 0, 'image': 0, 'watermark': 0, 'constraint': 0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
    for batch_idx, (images, watermarks) in enumerate(pbar):
        images = images.to(device)
        watermarks = watermarks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        watermarked = encoder(images, watermarks)
        attacked = noise_layer(watermarked, images)
        extracted = decoder(attacked)
        
        # Compute loss
        loss, loss_dict = criterion(images, watermarked, watermarks, extracted)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'img': f"{loss_dict['image']:.4f}",
            'wm': f"{loss_dict['watermark']:.4f}"
        })
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


def train(encoder, decoder, noise_layer, train_loader,
          criterion, optimizer, device, num_epochs=10):
    """Simplified training loop."""
    
    for epoch in range(num_epochs):
        train_losses = train_epoch(
            encoder, decoder, noise_layer, train_loader,
            criterion, optimizer, device, epoch
        )
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Total Loss: {train_losses['total']:.4f}")
        print(f"  Image Loss: {train_losses['image']:.4f}")
        print(f"  Watermark Loss: {train_losses['watermark']:.4f}")
        print("-" * 40)
    
    print("\n✓ Training completed!")