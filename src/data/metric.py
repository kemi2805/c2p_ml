import torch
import numpy as np


class metric:

    # Input has to be tensors

    def __init__(self, g, beta, alp):
        assert g.shape[0] == g.shape[1] == 3, "Metric tensor must be square"
        
        # If the input metric is a NumPy array, keep it as such
        if isinstance(g, np.ndarray):
            self.g = g
            self.sqrtg = np.sqrt(np.linalg.det(g))
            self.invg = np.linalg.inv(g)
        # If the input is a tensor, make sure it's a 3x3 tensor
        elif isinstance(g, torch.Tensor):
            self.g = g
            self.sqrtg = torch.sqrt(torch.det(g))
            self.invg = torch.inverse(g)
        else:
            raise ValueError("The metric 'g' must be either a numpy array or a torch tensor.")
        self.beta  = beta
        self.alp   = alp

    def raise_index(self, covec):
        """Raise the index of a single covariant vector or a batch."""
        if covec.ndimension() == 1:
            return self.Raise(covec)
        elif covec.ndimension() == 2:
            return self.RaiseBatched(covec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")

    def lower_index(self, vec):
        """Lower the index of a single contravariant vector or a batch."""
        if vec.ndimension() == 1:
            return self.Lower(vec)
        elif vec.ndimension() == 2:
            return self.LowerBatched(vec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")

    def square_norm_upper(self, vec):
        """Compute square norm of a single contravariant vector or a batch."""
        if vec.ndimension() == 1:
            return self.squareU(vec)
        elif vec.ndimension() == 2:
            return self.squareUbatched(vec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")

    def square_norm_lower(self, covec):
        """Compute square norm of a single covariant vector or a batch."""
        if covec.ndimension() == 1:
            return self.squareL(covec)
        elif covec.ndimension() == 2:
            return self.squareLbatched(covec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")
        
    def Raise(self, covec):
        """Raise the index of a single covariant vector."""
        if isinstance(covec, torch.Tensor):
            return torch.matmul(self.invg, covec)
        else:
            return np.dot(self.invg, covec)


    def RaiseBatched(self, covecs):
        """Raise the index of a batch of covariant vectors."""
        if isinstance(covecs, torch.Tensor):
            return torch.einsum('ij,bj->bi', self.invg, covecs)
        else:
            return np.einsum('ij,bj->bi',self.invg,covecs)

    def Lower(self, vec):
        """Lower the index of a single contravariant vector."""
        if isinstance(vec, torch.Tensor):
            return torch.matmul(self.g, vec)
        else:
            return np.dot(self.g, vec)

    def LowerBatched(self, vecs):
        """Lower the index of a batch of contravariant vectors."""
        if isinstance(vecs, torch.Tensor):
            return torch.einsum('ij,bj->bi', self.g, vecs)
        else:
            return np.einsum('ij,bj->bi',self.g,vecs)


    def squareU(self, vec):
        """Compute the squared norm of a contravariant vector."""
        if isinstance(vec, torch.Tensor):
            return torch.matmul(vec, torch.matmul(self.g, vec))
        else:
            return np.dot(vec, np.dot(self.g, vec))

    def squareUbatched(self, vecs):
        """Compute the squared norm of a batch of contravariant vectors."""
        if isinstance(vecs, torch.Tensor):
            return torch.einsum('bi,ij,bj->b', vecs, self.g, vecs)
        else:
            return np.einsum('bi,ij,bj->b', vecs, self.g, vecs)
    
    def squareL(self, covec):
        """Compute the squared norm of a covariant vector."""
        if isinstance(covec, torch.Tensor):
            return torch.matmul(covec, torch.matmul(self.invg, covec))
        else:
            return np.dot(covec, np.dot(self.invg, covec))
            
    def squareLbatched(self, covecs):
        """Compute the squared norm of a batch of covariant vectors."""
        if isinstance(covecs, torch.Tensor):
            return torch.einsum('bi,ij,bj->b', covecs, self.invg, covecs)
        else:
            return np.einsum('bi,ij,bj->b', covecs, self.invg, covecs)



if __name__ == "__main__":
    g = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    beta = torch.tensor([0., 0., 0.])
    alp = torch.tensor(1.)
    metric_instance = metric(g, beta, alp)
    vec = torch.tensor([1., 2., 3.])
    covec = torch.tensor([1., 2., 3.])
    print("Raise", metric_instance.Raise(covec))
    print("RaiseBatched", metric_instance.RaiseBatched(torch.stack((covec, covec))))
    print("Lower", metric_instance.Lower(vec))
    print("LowerBatched", metric_instance.LowerBatched(torch.stack((vec, vec))))
    print("squareU", metric_instance.squareU(vec))
    print("squareUbatched", metric_instance.squareUbatched(torch.stack((vec, vec))))
    print("squareL",metric_instance.squareL(covec))
    print("squareLbatched",metric_instance.squareLbatched(torch.stack((covec, covec))))
    
    print("------------------------")
    
    # Example with numpy array
    g = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    beta = np.array([0., 0., 0.])
    alp = 1.
    
    # Use a different instance variable name for the NumPy case
    metric_instance_np = metric(g, beta, alp)  # Rename for clarity
    
    vec = np.array([1., 2., 3.])
    covec = np.array([1., 2., 3.])
    
    print("Raise", metric_instance_np.Raise(covec))
    print("RaiseBatched", metric_instance_np.RaiseBatched(np.stack((covec, covec), axis=0)))
    print("Lower", metric_instance_np.Lower(vec))
    print("LowerBatched", metric_instance_np.LowerBatched(np.stack((vec, vec), axis=0)))
    print("squareU(vec)", metric_instance_np.squareU(vec))
    print("squareUbatched", metric_instance_np.squareUbatched(np.stack((vec, vec), axis=0)))
    print("squareL",metric_instance_np.squareL(covec))
    print("squareLbatched", metric_instance_np.squareLbatched(np.stack((covec, covec), axis=0)))

    print("alle tests geschafft")
