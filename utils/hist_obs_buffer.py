import torch

class ObsHistBuffer:
    def __init__(self, batch_size, vector_dim=16, buffer_size=5, device='cpu'):
        """
        Initialize the buffer.
        :param batch_size: Number of samples in a batch.
        :param vector_dim: Dimension of each vector (e.g., 16).
        :param buffer_size: Number of recent vectors to store (e.g., 5).
        :param device: The device where the tensors will be stored (e.g., 'cpu' or 'cuda').
        """
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.buffer_size = buffer_size
        self.device = device
        # Initialize buffer with zeros on the specified device
        self.buffer = torch.zeros((batch_size, buffer_size, vector_dim), device=self.device)

    def update(self, new_vectors):
        """
        Update the buffer with new vectors.
        :param new_vectors: Tensor of shape (batch_size, vector_dim).
        """
        assert new_vectors.shape == (self.batch_size, self.vector_dim), \
            "New vectors must have shape (batch_size, vector_dim)"
        
        # Detach the buffer to avoid gradient tracking
        self.buffer = self.buffer.detach()
        
        # Shift the existing buffer to the left (discard oldest vectors)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        
        # Replace the last slot with the new vectors (ensure no in-place ops)
        self.buffer = self.buffer.clone()
        self.buffer[:, -1, :] = new_vectors.clone().detach()

    def get_concatenated(self):
        """
        Get the concatenated matrix of recent vectors.
        :return: Tensor of shape (batch_size, buffer_size * vector_dim).
        """
        # Detach the buffer before returning to prevent tracking
        return self.buffer.detach().view(self.batch_size, -1)


if __name__ == '__main__':
    # Example usage
    batch_size = 4
    vector_dim = 16
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    buffer = ObsHistBuffer(batch_size, vector_dim, device=device)

    # Simulate 6 steps of input
    for step in range(6):
        new_vectors = torch.rand((batch_size, vector_dim), device=device)  # Generate random vectors
        buffer.update(new_vectors)
        concatenated = buffer.get_concatenated()
        print(f"Step {step + 1}: Concatenated Shape = {concatenated.shape}, Device = {concatenated.device}")
        print(concatenated)
