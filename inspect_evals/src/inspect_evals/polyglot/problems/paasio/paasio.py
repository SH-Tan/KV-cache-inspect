<<<<<<< Updated upstream
=======
import io

class MeteredFile(io.BufferedRandom):
    """Implement using a subclassing model."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def read(self, size=-1):
        if size < 0:
            size = self._total_size
        total_bytes = 0
        chunk_size = self._chunk_size
        for _ in range(size // chunk_size + 1):
            chunk = self._file.read(chunk_size)
            if chunk:
                total_bytes += len(chunk)
        return total_bytes

    @property
    def read_bytes(self):
        return self.read()

    @property
    def read_ops(self):
        return self.read_ops

class MeteredSocket(io.BufferedRandom):
    """Implement using a delegation model."""

    def __init__(self, socket):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def recv(self, bufsize, flags=0):
        pass

    @property
    def recv_bytes(self):
        pass

    @property
    def recv_ops(self):
        return self.read_ops

    def send(self, data, flags=0):
        pass

    @property
    def send_bytes(self):
        pass

    @property
    def send_ops(self):
        return self.read_ops + self.read_ops

    @property
    def send_ops(self):
        return self.read_ops + self.read_ops
>>>>>>> Stashed changes
