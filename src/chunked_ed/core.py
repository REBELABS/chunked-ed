"""
Core functions for chunked Energy Distance computation and
permutation-based p-value estimation for 1D and 2D samples.
"""
import numpy as np
import time
from tqdm import tqdm

#Def statement to chunk the pairwise comparison
def chunked_diff_sum(a,b,block=1000):
    """
    Compute the sum of pairwise distances in chunks to reduce memory usage. 

    Parameters
    ---------- 
    a : array-like
        First sample distribution.
    b : array-like
        Second sample distribution.
    block : int, optional,  
        Block size for chunked computation. Default is 1000.
    
    Returns
    ------- 
    float
        Sum of pairwise distances between elements of a and b.
        
    Notes
   ------
   - For 1D inputs, absolute differences are used.
   - For 2D inputs, Euclidean distances are used.
    """
    
    a = np.asarray(a,dtype=np.float64)
    b = np.asarray(b,dtype=np.float64)
    
    if block <= 0:
        raise ValueError("block must be a positive integer.")

    if len(a) == 0 or len(b) == 0:
        raise ValueError("Input arrays must not be empty.")
        
    total = 0.0
    
    try:
        # 1D case: absoute difference
        if a.ndim == 1 and b.ndim == 1:
            for i in range(0, len(a), block):
                x_block = a[i:i+block]
                for j in range(0, len(b), block):
                    y_block = b[j:j+block]
                    #Absolute difference of column vector - row vector
                    abs_diff = np.abs(x_block[:,None]-y_block[None,:])
                    total +=abs_diff.sum()
        
        # 2D case: Euclidean distance
        elif a.ndim == 2 and b.ndim == 2:
            if a.shape[1] != b.shape[1]:
                raise ValueError("For 2D inputs, both arrays must have the same number of columns.")
            
            for i in range(0, len(a), block):
                x_block = a[i:i+block]
                for j in range(0, len(b), block):
                    y_block = b[j:j+block]
                    #Absolute difference of column vector - row vector
                    diff = (x_block[:,None,:]-y_block[None,:,:])
                    eu_dist = np.linalg.norm(diff,axis=2)#Euclidean distance
                    total +=eu_dist.sum()

        else:
            raise ValueError("Inputs must both be 1D or both be 2D.") 
        
        #Answer       
        return total
        
    except Exception as e:
        raise RuntimeError(f"chunked_diff_sum failed: {e}")


#Def statement for energy distance
def energy_distance(a,b,block=1000):
    """
    Compute the energy distance between two samples.

    Parameters
    ----------
    a : array-like
        First sample distribution.
    b : array-like
        Second sample distribution.
    block : int, optional
        Block size for chunked computation. Default is 1000.

    Returns
    -------
    float
        Observed energy distance between a and b.
    """
    
    a = np.asarray(a,dtype=np.float64)
    b = np.asarray(b,dtype=np.float64)
    
    n, m = len(a), len(b)
    
    if n == 0 or m == 0:
        raise ValueError("Input arrays must not be empty.")
    
    xy = chunked_diff_sum(a, b, block)
    xx = chunked_diff_sum(a, a, block)
    yy = chunked_diff_sum(b, b, block)
    
    ed_real = 2 * (xy/(n*m)) - (xx/(n*n)) - (yy/(m*m))
    return ed_real


##Estimate P-value and Energy Distance for null
def ed_p_value(a,b,block=1000,n_perm=1000, full=True, seed=None): #n_perm is number of permutation
    """
    Estimate the energy distance and permutation-based p-value.

    Parameters
    ----------
    a : array-like
        First sample distribution.
    b : array-like
        Second sample distribution.
    block : int, optional
        Block size for energy distance calculation. Default is 1000.
    n_perm : int, optional
        Number of permutations for estimating p-value. Default is 1000.
    full : bool, optional
        If True, print a summary of results. Default is True.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    ed_real : float
        Observed energy distance between a and b.
    p_val : float
        Permutation-based p-value.
    perm_ed : list
        Energy distances from permuted samples.
    elapsed : float
        Time taken for the full computation in seconds.
    """
    
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    if n_perm <= 0:
        raise ValueError("n_perm must be a positive integer.")
        
    #Seed
    if seed is not None:
        np.random.seed(seed)
        
    startp= time.time()
    
    #Energy distance of real values
    ed_real = energy_distance(a,b,block)
    
    #Merge the data
    merged_data = np.concatenate([a,b])
        
    #Empty lists to hold the permuted energy distance values
    perm_ed = []
    
    for o in tqdm(range(n_perm),desc="Permutations"):   
        #Shuffle
        sh_data = merged_data.copy()
        np.random.shuffle(sh_data)
        
        #Partition the fake data
        a_perm = sh_data[:len(a)]
        b_perm = sh_data[len(a):]
        
        #Energy distance of fake data
        ed_sh = energy_distance(a_perm, b_perm,block)
        perm_ed.append(ed_sh)
   
    #Estimate the P_value
    perm_ed = np.array(perm_ed, dtype=np.float64)
    p_val = (np.sum(perm_ed>= ed_real) + 1)/(n_perm + 1)
    
    elapsed = time.time()-startp
    
    if full:
        print(f"ED: {ed_real:.5f} | P-value: {p_val:.5f} | Time: {elapsed:.3f}s")
    
    return ed_real, p_val, perm_ed, elapsed

