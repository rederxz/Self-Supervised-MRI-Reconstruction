import torch
import torch.fft


# def fftshift(x, axes=None):
#     """
#     Similar to np.fft.fftshift but applies to PyTorch Tensors
#     """
#     assert torch.is_tensor(x) is True
#     if axes is None:
#         axes = tuple(range(x.ndim()))
#         shift = [dim // 2 for dim in x.shape]
#     elif isinstance(axes, int):
#         shift = x.shape[axes] // 2
#     else:
#         shift = [x.shape[axis] // 2 for axis in axes]
#     return torch.roll(x, shift, axes)


# def ifftshift(x, axes=None):
#     """
#     Similar to np.fft.ifftshift but applies to PyTorch Tensors
#     """
#     assert torch.is_tensor(x) is True
#     if axes is None:
#         axes = tuple(range(x.ndim()))
#         shift = [-(dim // 2) for dim in x.shape]
#     elif isinstance(axes, int):
#         shift = -(x.shape[axes] // 2)
#     else:
#         shift = [-(x.shape[axis] // 2) for axis in axes]
#     return torch.roll(x, shift, axes)


def fft2(x):
    assert len(x.shape) == 4
    x = torch.fft.fft2(x, norm='ortho')
    return x

def fft2_tensor(data):
    """input is a [bs, 2, x, y] shape tensor"""
    assert data.shape[-3] == 2
    data = torch.view_as_complex(data.permute(0, 2, 3, 1).contiguous())
    data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data).permute(0, 3, 1, 2).contiguous()
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data

def ifft2(x):
    assert len(x.shape) == 4
    x = torch.fft.ifft2(x, norm='ortho')
    return x

def ifft2_tensor(data):
    """input is a [bs, 2, x, y] shape tensor"""
    assert data.shape[-3] == 2
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.view_as_complex(data.permute(0, 2, 3, 1).contiguous())
    data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
    data = torch.view_as_real(data).permute(0, 3, 1, 2).contiguous()
    return data

def fftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, (x.shape[-2]//2, x.shape[-1]//2), dims=(-2, -1))
    return x

def ifftshift2(x):
    assert len(x.shape) == 4
    x = torch.roll(x, ((x.shape[-2]+1)//2, (x.shape[-1]+1)//2), dims=(-2, -1))
    return x


# def fft2_channel_last(data):
#     assert data.shape[-1] == 2
#     data = torch.view_as_complex(data)
#     data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
#     data = torch.view_as_real(data)
#     data = fftshift(data, axes=(-3, -2))
#     return data


# def ifft2_channel_last(data):
#     assert data.shape[-1] == 2
#     data = ifftshift(data, axes=(-3, -2))
#     data = torch.view_as_complex(data)
#     data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
#     data = torch.view_as_real(data)
#     return data


def rfft2(data):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data)
    return data


def rifft2(data):
    assert data.shape[-1] == 2
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1)
    return data


def rA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    return data


def rAt(data, mask):
    assert data.shape[-1] == 2
    data = ifft2(data * mask)
    data = data[..., 0].unsqueeze(-1)
    return data


def rAtA(data, mask):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data) * mask
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1)
    return data
