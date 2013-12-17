def estimate_hyperparameters(x, ii, jj, kk, return_grid=True):
    def likelihood(x, p, a, b):
        print "Calculating p={} a={} b={}".format(p, a, b)
        p_stars = np.zeros((len(x),len(x)))
        
        def A(i, t):
            return a + sum(x[i:t+1])
        
        def B(i, t):
            return (1 / b + t - i + 1) ** -1
        
        def P(i, t):
            if i > t:
                raise ValueError('P(i, t) undefined for i > t.')
            if i == t:
                # same as p_stars[i, t] / p_stars[t:t+1, t].sum()
                return 1.0
            else:
                return p_stars[i, t] / p_stars[i:t+1, t].sum()

        def pi(i, j):
            return B(i, j) ** A(i, j) / Gamma(A(i, j))

        def pi_ratio(i, j):
            '''
            pi(i, t-1) / pi(i, t)
            '''               
            A_i_jminus1 = A(i, j-1)
            r1 = np.exp( A_i_jminus1 * np.log(B(i, j-1)) -
                         A(i, j)     * np.log(B(i, j  )) )
            #Not sure where this cutoff should be
            if A_i_jminus1 < 60:
                r2 = Gamma(A(i, j)) / Gamma(A_i_jminus1)
            else:
                # from Stirling's approximation
                r2 = A_i_jminus1 ** x[j]
            
            if np.isinf(r1) or np.isinf(r2) or np.isnan(r1) or np.isnan(r2):
                raise ValueError('r1: {}\nr2: {}'.format(r1, r2))
            return r1 * r2
        
        def p_star(i, t):
            if i > t:
                raise ValueError('p_star(i, t) undefined for i > t.')
            if i == t:
                pi_00 = b ** -a / Gamma(a)
                val = p * pi_00 / pi(t, t)
            else:
                #same as val = p * (1 - p) * P(i, t-1) * pi(i, t-1) / pi(i, t)
                val = (p * (1 - p) * P(i, t-1) * pi_ratio(i, t))
            if np.isnan(val) or np.isinf(val):
            #    print "(i, t): ", (i, t)
            #    print "p   :", p
            #    print "P   :", P(i, t-1)
            #    print "pi_ratio:", pr
            #    print 'p*[i:t+1, t]:', p_stars[0:t+1, t]
            #    print 'p*[i, :t]:', p_stars[i, :t]
            #    print 'pi[i, :t]:', [pi(i, t_) for t_ in range(t+1)]
            #    print A(i, t)
                raise ValueError('p*({}, {}) value "{}" out of range.'.format(i, j, val))
            
            return val
        
        for t in range(len(x)):
            for i in range(t + 1):
                val = p_star(i, t)
                p_stars[i, t] = val
        
        # same as np.product(p_stars.sum(1))
        return np.sum(np.log(p_stars.sum(1))), p_stars
    
    # Use a grid search. This can and should be sped up with Numba(pro) and/or parallel
    grid = np.zeros((len(ii), len(jj), len(kk)))   
    p_star_grid = {}
    for (i_idx, i) in enumerate(ii):
        for (j_idx, j) in enumerate(jj):
            for (k_idx, k) in enumerate(kk):
                p = 2 ** i / float(len(x))
                a = 0.5 * j
                b = 0.1 + 0.2 * k
                lik, p_stars = likelihood(x, p, a, b)
                grid[i_idx,j_idx,k_idx] = lik
                p_star_grid[i_idx,j_idx,k_idx] = p_stars
                    
    indices = np.unravel_index(grid.argmax(), grid.shape)
    p = 2 ** ii[indices[0]] / float(len(x))
    a = 0.5 * jj[indices[1]]
    b = 0.1 + 0.2 * kk[indices[2]]

    return {'est':{'p':p, 'a':a, 'b':b}, 
            'indices':indices, 
            'grid': grid,
            'p_stars': p_star_grid[indices],
            }

def estimate_thetas(x, **kwargs):
    p = kwargs['est']['p']
    p_stars = kwargs['p_stars']
    a = kwargs['est']['a']
    b = kwargs['est']['b']
    g_stars = np.zeros((len(x), len(x), len(x)))
    q_stars = np.zeros((len(x), len(x)))
    
    def A(i, t):
        return a + sum(x[i:t+1])
    
    def B(i, t):
        return (1 / b + t - i + 1) ** -1
    
    def P(i, t):
        if i > t:
            raise ValueError('P(i, t) undefined for i > t.')
        if i == t:
            # same as p_stars[i, t] / p_stars[t:t+1, t].sum()
            return 1.0
        else:
            return p_stars[i, t] / p_stars[i:t+1, t].sum()
            
    #@memoize
    def bigP(t):
        return p + g_stars[:t+1, t:, t].sum()
    
    def G(i, j, t):
        return g_stars[i, j, t] / bigP(t)
        
    def Q(j, t):
        if j < t:
            raise ValueError('Q(j, t) undefined for j < t. j={}, t={}'.format(j, t))
        if j == t:
            return 1.0
        else:
            return q_stars[j, t] / q_stars[t:j+1, t].sum()
            
    def pi(i, j):
        return B(i, j) ** A(i, j) / Gamma(A(i, j))
        
    def pi_ratio(i, j):
        '''
        pi(i, t) / pi(i, t+1)
        '''               
        A_i_jplus1 = A(i, j+1)
        r1 = np.exp(A(i, j)     * np.log(B(i, j  )) - 
                    A_i_jplus1  * np.log(B(i, j + 1)))
        #Not sure where this cutoff should be
        if A_i_jplus1 < 62:
            r2 =  Gamma(A_i_jplus1) / Gamma(A(i, j))
        else:
            # from Stirling's approximation
            r2 = A(i, j) ** x[j+1]
        
        if np.isinf(r1) or np.isinf(r2) or np.isnan(r1) or np.isnan(r2):
            raise ValueError('r1: {}\nr2: {}'.format(r1, r2))
        return r1 * r2
        
    def big_pi_ratio(i, j, t):
        def _log(i,t):
            return A(i, t) + np.log(B(i, t))
        def minmax(*args):
            return min(*args), max(*args)
        r1 = np.exp(_log(i, t) + _log(t+1, j) - _log(i, j) + a * np.log(b))
        if abs(A(i, t)) > abs(A(t+1, j)):
            r22 = Gamma(a) / Gamma(A(i, t))
            _min, _max = minmax(A(i, j), A(t+1, j))
            r21 = _min ** (A(i,j) - A(t+1, j))
        else:
            r22 = Gamma(a) / Gamma(A(t+1, j))
            _min, _max = minmax(A(i, j), A(i, t))
            r21 = _min ** (A(i,j) - A(i, t))
        r = r1 * r22 * r21
        if np.isnan(r) or np.isinf(r):
            raise ValueError('big_pi_ratio({}, {}, {}) = {}'.format(i, j, t, r))
        return r
            
    def q_star(j, t):
        if j < t:
            raise ValueError('q_star(j, t) undefined for j < t.')
        if j == t:
            pi_00 = b ** -a / Gamma(a)
            val = p * pi_00 / pi(t, t)
        else:
            #same as val = p * (1 - p) * P(i, t-1) * pi(i, t-1) / pi(i, t)
            val = (1 - p) * Q(j, t+1) * pi_ratio(i, t)

        if np.isnan(val) or np.isinf(val):
        #    print "(i, t): ", (i, t)
        #    print "p   :", p
        #    print "P   :", P(i, t-1)
        #    print "pi_ratio:", pr
        #    print 'p*[i:t+1, t]:', p_stars[0:t+1, t]
        #    print 'p*[i, :t]:', p_stars[i, :t]
        #    print 'pi[i, :t]:', [pi(i, t_) for t_ in range(t+1)]
        #    print A(i, t)
            raise ValueError('p*({}, {}) value "{}" out of range.'.format(i, j, val))
        return val
    
    for j in range(len(x)):
        for t in range(j + 1)[::-1]:
            if j == t:
                q_stars[j, t] = q_star(j, t)
    
    for j in range(len(x)):
        for t in range(j + 1)[::-1]:
            for i in range(t + 1)[::-1]:
                print i, j, t
                if i == t:
                    val = p * p_stars[i, t]
                else:
                    val = ((1-p) * p * P(i, t) * Q(j, t + 1) * 
                           big_pi_ratio(i, j, t))
                g_stars[i, j, t] = val
                
    posterior = np.zeros(len(x))
    for t in range(len(x)):
        _sum = 0
        for i in range(t+1):
            for j in range(t, len(x)):
                _sum += G(i, j, t) * A(i, j) * B(i, j)
        posterior[t] = _sum
    return posterior
