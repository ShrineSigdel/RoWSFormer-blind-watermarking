 # window_partition, window_reverse

def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.

    Args:
        x: (B, C, H, W)
        window_size: Window size M
    Returns:
        windows: (B*num_windows, C, M, M)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous() # (B, num_windows_H, num_windows_W, C, M, M)

    # -1 is a cheesy syntax for saying "infer this dimension" 
    windows = windows.view(-1, C, window_size, window_size) # (B*num_windows, C, M, M)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.

    Args:
        windows: (B*num_windows, C, M, M)
        window_size: Window size M
        H, W: Original height and width
    Returns:
        x: (B, C, H, W)
    """
    B_w, C, M, M = windows.shape
    B = B_w // ((H // window_size) * (W // window_size))
    x = windows.view(B, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x
