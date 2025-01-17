import numpy as np

class metric:

    def __init__(self, g, beta, alp):
        assert g.shape[0] == g.shape[1] == 3, "Metric tensor must be square"
        self.sqrtg = np.sqrt(np.linalg.det(g))
        self.invg  = np.linalg.inv(g)
        self.g     = g
        self.beta  = beta
        self.alp   = alp

    def raise_index(self,covec):
        """Raise the index of a single covariant vector or a batch."""
        if covec.ndim == 1:
            return self.Raise(covec)
        elif covec.ndim == 2:
            return self.RaiseBatched(covec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")

    def lower_index(self,vec):
        """Lower the index of a single contravariant vector or a batch."""
        if vec.ndim == 1:
            return self.Lower(vec)
        elif vec.ndim == 2:
            return self.LowerBatched(vec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")

    def square_norm_upper(self,vec):
        """Compute square norm of a single contravariant vector or a batch."""
        if vec.ndim == 1:
            return self.squareU(vec)
        elif vec.ndim == 2:
            return self.squareUbatched(vec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")

    def square_norm_lower(self,covec):
        """Compute square norm of a single covariant vector or a batch."""
        if covec.ndim == 1:
            return self.squareL(covec)
        elif covec.ndim == 2:
            return self.squareLbatched(covec)
        else:
            raise ValueError("Input must be either one (single mode) or two (batched mode) dimensional")
        
    def Raise(self, covec):
        """Raise the index of a single covariant vector."""
        return np.dot(self.invg, covec)

    def RaiseBatched(self,covecs):
        """Raise the index of a batch of covariant vectors."""
        return np.einsum('ij,bj->bi',self.invg,covecs)

    def Lower(self, vec):
        """Lower the index of a single contravariant vector."""
        return np.dot(self.g, vec)

    def LowerBatched(self,vecs):
        """Lower the index of a batch of contravariant vectors."""
        return np.einsum('ij,bj->bi',self.g,vecs)

    def squareU(self,vec):
        """Compute the squared norm of a contravariant vector."""
        return np.dot(vec,np.dot(self.g,vec))

    def squareUbatched(self,vecs):
        """Compute the squared norm of a batch of contravariant vectors."""
        return np.einsum('bi,ij,bj->b', vecs, self.g, vecs)
    
    def squareL(self,covec):
        """Compute the squared norm of a covariant vector."""
        return np.dot(covec,np.dot(self.invg,covec))

    def squareLbatched(self,covecs):
        """Compute the squared norm of a batch of covariant vectors."""
        return np.einsum('bi,ij,bj->b', covecs, self.invg, covecs)

    
