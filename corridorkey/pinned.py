import torch


class HostBuffer:
    """Page-locked host memory of an exact size, outside PyTorch's pinned
    cache (which rounds blocks up to powers of two and never returns them)."""

    def __init__(self):
        self._storage = None
        self._registered = False

    def view(self, shape, dtype) -> torch.Tensor:
        numel = 1
        for size in shape:
            numel *= int(size)
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        if self._storage is None or self._storage.numel() < nbytes:
            self.release()
            self._storage = torch.empty(nbytes, dtype=torch.uint8)
            err = torch.cuda.cudart().cudaHostRegister(self._storage.data_ptr(), nbytes, 0)
            self._registered = int(err) == 0
        return self._storage[:nbytes].view(dtype).view(tuple(shape))

    def release(self) -> None:
        if self._storage is not None and self._registered:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(self._storage.data_ptr())
        self._storage = None
        self._registered = False
