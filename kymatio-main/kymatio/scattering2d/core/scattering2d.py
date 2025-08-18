import math

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array', downsample=True, dilation_optimization=True):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    stack = backend.stack

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)

    U_0_c = rfft(U_r)

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi['levels'][0])
    if downsample:
        U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    S_0 = irfft(U_1_c)
    S_0 = unpad(S_0) if downsample else S_0

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])
        if downsample and j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi['levels'][j1])
        if downsample:
            S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))

        S_1_r = irfft(S_1_c)
        S_1_r = unpad(S_1_r) if downsample else S_1_r

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'n': (n1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1])
            if downsample:
                U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi['levels'][j2])
            if downsample:
                S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2))

            S_2_r = irfft(S_2_c)
            S_2_r = unpad(S_2_r) if downsample else S_2_r

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'n': (n1, n2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = stack([x['coef'] for x in out_S])

    return out_S

from kymatio.scattering2d.backend import numpy_backend

# def invertibleScattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
#         out_type='array'):
#     subsample_fourier = backend.subsample_fourier
#     modulus = backend.modulus
#     rfft = backend.rfft
#     ifft = backend.ifft
#     irfft = backend.irfft    
#     cdgmm = backend.cdgmm
#     stack = backend.stack
#     fft = backend.fft
#     custom_relu_split = backend.custom_relu_split
#     pad_cmplx = backend.pad_cmplx
#     unpad_cmplx = backend.unpad_cmplx
#     stack1 = backend.stack1

#     # Define lists for output.
#     out_S_0, out_S_1, out_S_2 = [], [], []

#     U_r = pad_cmplx(pad,x)

#     U_0_c = fft(U_r)#F(x)

#     # First low pass filter
#     U_1_c = cdgmm(U_0_c, phi['levels'][0])#<F(x), F(father)>
#     U_1_c = subsample_fourier(U_1_c, k=2 ** J)

#     #changed from ifft to irfft for the output to be real valued
#     S_0 = irfft(U_1_c)
#     #S_0 = ifft(U_1_c)# F^-1(<F(x), F(father)>)  = x * father
#     S_0 = unpad_cmplx(unpad,S_0)

#     out_S_0.append({'coef': S_0,
#                     'j': (),
#                     'n': (),
#                     'theta': ()})

#     for n1 in range(len(psi)):
#         j1 = psi[n1]['j']
#         theta1 = psi[n1]['theta']

#         U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])#< F(x) , F(mother_n1) >
#         if j1 > 0:
#             U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
#         U_1_c = ifft(U_1_c)# x * mother_n1
#         positive_real_U1, positive_imag_U1, neg_real_U1, neg_imag_U1 = custom_relu_split(U_1_c)
        
#         U_1 = [positive_real_U1, positive_imag_U1, neg_real_U1, neg_imag_U1]
#         U_1 = [fft(signal) for signal in U_1]

        

#         # Second low pass filter
#         S_1 = [cdgmm(signal, phi['levels'][j1]) for signal in U_1]
#         S_1 = [subsample_fourier(signal, k=2 ** (J - j1)) for signal in S_1]
#         #changed from ifft to irfft for the output to be real valued
#         S_1 = [irfft(signal) for signal in S_1]
#         #S_1 = [ifft(signal) for signal in S_1]
#         S_1 = [unpad_cmplx(unpad,signal) for signal in S_1]


#         for signal in S_1:
#             out_S_1.append({'coef': signal,
#                             'j': (j1,),
#                             'n': (n1,),
#                             'theta': (theta1,)})

#         if max_order < 2:
#             continue
#         for U_1_c in U_1: #U_1_c = F(pos_real(x*mother))
#             for n2 in range(len(psi)):
#                 j2 = psi[n2]['j']
#                 theta2 = psi[n2]['theta']

#                 if j2 <= j1:
#                     continue
        
#                 U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1]) # < F(pos_real(x*mother_1)) , F(mother_2) >
#                 U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))  
#                 U_2_c = ifft(U_2_c) # pos_real(x*mother_1) * mother_2


#                 positive_real_U2, positive_imag_U2, neg_real_U2, neg_imag_U2 = custom_relu_split(U_2_c)
#                 U_2 = [positive_real_U2, positive_imag_U2, neg_real_U2, neg_imag_U2]
#                 U_2 = [fft(signal) for signal in U_2]


#                 # Third low pass filter
#                 S_2 = [cdgmm(signal, phi['levels'][j2]) for signal in U_2]
#                 S_2 = [subsample_fourier(signal, k=2 ** (J - j2)) for signal in S_2]

#                 #changed from ifft to irfft for the output to be real valued
#                 S_2 = [irfft(signal) for signal in S_2]
#                 #S_2 = [ifft(signal) for signal in S_2]
#                 S_2 = [unpad_cmplx(unpad,signal) for signal in S_2]


#                 for signal in S_2:
#                     out_S_2.append({'coef': signal,
#                                     'j': (j1, j2),
#                                     'n': (n1, n2),
#                                     'theta': (theta1, theta2)})

#     out_S = []
#     out_S.extend(out_S_0)
#     out_S.extend(out_S_1)
#     out_S.extend(out_S_2)

#     if out_type == 'array':
#         out_S = stack1([x['coef'] for x in out_S])

#     return out_S





def invertibleScattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array', downsample=True, dilation_optimization=True):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    stack = backend.stack
    fft = backend.fft
    custom_relu_split = backend.custom_relu_split
    pad_cmplx = backend.pad_cmplx
    unpad_cmplx = backend.unpad_cmplx
    stack1 = backend.stack1

    # Define lists for output.
    out_S = []

    U_r = pad_cmplx(pad,x)

    U_0_c = fft(U_r)#F(x)

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi['levels'][0])#<F(x), F(father)>
    if downsample:
        U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    #changed from ifft to irfft for the output to be real valued
    S_0 = irfft(U_1_c)
    #S_0 = ifft(U_1_c)# F^-1(<F(x), F(father)>)  = x * father
    S_0 = unpad_cmplx(unpad,S_0) if downsample else S_0

    out_S.append({'coef': S_0.contiguous(),
                    'j': -1,
                    'n': -1,
                    'theta': -1, 
                    'depth' : 0,
                    'path' : (),
                    'split' : 'no_split'})
    

    recursiveInvertibleScattering2d(U_0_c, pad, unpad, backend, J, L, phi, psi, max_order, 1 ,None, () ,out_type, out_S, downsample, dilation_optimization)
    # print("out_s.len : %d \n" %len(out_S) )
    if out_type == 'array':
        out_S = stack1([x['coef'] for x in out_S])
        # print("out_s.shape : {} \n".format(out_S.shape))

    return out_S



def recursiveInvertibleScattering2d(U_0_c, pad, unpad, backend, J, L, phi, psi, max_order, level, last_n, path,
        out_type, out_S, downsample=True, dilation_optimization=True):
    
    if level > max_order:
        return

    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    stack = backend.stack
    fft = backend.fft
    custom_relu_split = backend.custom_relu_split
    pad_cmplx = backend.pad_cmplx
    unpad_cmplx = backend.unpad_cmplx
    stack1 = backend.stack1 
    

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        if last_n == None:
            last_j = 0
        else:
            last_j = psi[last_n]['j']

        if j1 <= last_j and last_n != None and dilation_optimization:
            continue


        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][last_j])#< F(x) , F(mother_n1) >
        if downsample:
            U_1_c = subsample_fourier(U_1_c, k=2 ** (j1 - last_j))
        U_1_c = ifft(U_1_c)# x * mother_n1
        positive_real_U1, positive_imag_U1, neg_real_U1, neg_imag_U1 = custom_relu_split(U_1_c)
        
        U_1 = {}
        U_1['re_pos'] = fft(positive_real_U1)
        U_1['im_pos'] = fft(positive_imag_U1)
        U_1['re_neg'] = fft(neg_real_U1)
        U_1['im_neg'] = fft(neg_imag_U1)


        

        # low pass filter
        S_1 = {k: cdgmm(v, phi['levels'][j1]) for k,v in U_1.items()}
        if downsample:
            S_1 = {k: subsample_fourier(v, k=2 ** (J - j1)) for k,v in S_1.items()}
        #changed from ifft to irfft for the output to be real valued
        S_1 = {k: irfft(v) for k,v in S_1.items()}
        S_1 = {k: unpad_cmplx(unpad,v) if downsample else v for k,v in S_1.items() }

        split_order = {'re_pos': 0, 'im_pos': 1, 're_neg': 2, 'im_neg': 3, 'no_split': 4}
        for key,signal in S_1.items():
            new_path = path + (n1,split_order[key])
            out_S.append({'coef': signal.contiguous(),
                            'j': j1,
                            'n': n1,
                            'theta': theta1, 
                            'depth' : level,
                            'path' : new_path,
                            'split' : key})
        
        for key,U_1_c in U_1.items():
            new_path = path + (n1,split_order[key])
            recursiveInvertibleScattering2d(U_1_c, pad, unpad, backend, J, L, phi, psi, max_order, level+1, n1, new_path, out_type, out_S, downsample, dilation_optimization)

def num_of_scattering_coefficients(max_depth, J, L, dilation_optimization=True):
    total_sum = 0
    for q in range(0, max_depth + 1):
        if dilation_optimization:
            combination = math.comb(J, q)
        else:
            combination = J ** q
        term = combination * (4 ** q) * (L ** q)
        total_sum += term
    return total_sum

def num_of_scattering_coefficients_in_layer(layer_depth, J, L, dilation_optimization=True):
    if dilation_optimization:
        return math.comb(J, layer_depth) * (4 ** layer_depth) * (L ** layer_depth)
    else:
        return (J ** layer_depth) * (4 ** layer_depth) * (L ** layer_depth)

def num_of_childrens(coeff, J , L, dilation_optimization=True):
    """
    Returns the number of direct childrens under the node corresponding to the coefficient.
    every path in the scattering transform is of the form:
    (f*mother_(j_1,theta_1))_i1)*(mother_(j_2,theta_2))_i2)*...*(mother_(j_k,theta_k))_ik
    wherre 0<= j_1 < j_2 < ... < j_k < J are the scales and theta_m are angels in [0, L-1].
    and i_m represent the split type of the coefficient at that scale (i.e. re_pos, im_pos, re_neg, im_neg, no_split),
    """
    if dilation_optimization and coeff['depth'] > 0:
        return L*len(range(coeff['j']+1, J))
    else:
        return L*J
    

def sort_coefficients(coefficients):
    """
    Sort the coefficients by depth, then by n, then by split in the order:
    're_pos', 'im_pos', 're_neg', 'im_neg', 'no_split'.
    """
    split_order = {'re_pos': 0, 'im_pos': 1, 're_neg': 2, 'im_neg': 3, 'no_split': 4}
    return sorted(
        coefficients,
        key=lambda x: (
            x['depth']
            #reversed(x['path']),
            #split_order.get(x['split'], 5)  # 5 for any unexpected value
        )
    )



def build_zero_last_layer(coefficients, J, L, max_order, phi, psi,
                        backend, out_type='array', dilation_optimization=True):
    """
    Build the intermediate nodes computed in the last layer of the scattering
        transform as all zero.
    Parameters
    ----------
    coefficients : list of dict
        Coefficients from the scattering transform.
    J : int
        Logscale of the scattering.
    L : int
        Number of angles used for the wavelet transform.
    max_order : int 
        The depth of the scattering transform.
    image_shape : tuple
        Shape of the original image.
    phi : dict
        Low-pass filter.
    psi : list of dict
        Wavelet filters.
    backend : str
        Backend to use for the computation.
    out_type : str
        The format of the output of a scattering transform. If set to
        `'list'`, then the output is a list containing each individual
        scattering path with meta information. Otherwise, if set to
        `'array'`, the output is a large array containing the
        concatenation of all scattering coefficients. Defaults to
        `'array'`.  
    """

    out_S = []
    S_0 = backend.zero_coeff(coefficients[-1]['coef'].shape)
    num_of_coeffs_in_last_layer = num_of_scattering_coefficients_in_layer(max_order, J, L, dilation_optimization=dilation_optimization)
    num_of_coeffs = len(coefficients)
    coefficients_in_last_layer = coefficients[num_of_coeffs - num_of_coeffs_in_last_layer : num_of_coeffs]

    for coeff in coefficients_in_last_layer:
        if coeff['depth'] != max_order:
            raise RuntimeError('Wrong calculation of number of coeffs in the last layer')
        
        for n in range(len(psi)):
            j = psi[n]['j']
            theta = psi[n]['theta']

            if dilation_optimization and coeff['j'] >= j:#scale j must go down in each layer
                continue
            
            out_S.append({'coef': S_0,
                'j': j,
                'n': n,
                'theta': theta,
                'depth' : max_order+1,
                'split' : coeff['split']}) 
            
        
    
    return out_S

def RefacrorCoefficients(coefficients, J, L, max_order, phi, psi,
                        backend, out_type='array', last_layer=None, dilation_optimization=True):
    """
    Refactor the coefficients to match the shape of the wavelets (phi and psi).
    Parameters
    ----------
    coefficients : list of dict
        Coefficients from the scattering transform.
    J : int
        Logscale of the scattering.
    L : int
        Number of angles used for the wavelet transform.
    max_order : int 
        The depth of the scattering transform.
    phi : dict
        Low-pass filter.
    psi : list of dict
        Wavelet filters.
    backend : str
        Backend to use for the computation.
    out_type : str
        The format of the output of a scattering transform. If set to
        `'list'`, then the output is a list containing each individual
        scattering path with meta information. Otherwise, if set to
        `'array'`, the output is a large array containing the
        concatenation of all scattering coefficients. Defaults to
        `'array'`.
    last_layer : list of dict
        The intermediate nodes computed in the last layer of the scattering
        transform. i.e olnly convolution with mother filters (U_1_c).
    dilation_optimization : bool
        If True, the coefficients are refactored to match the dilation optimization.
    Returns
    -------
    coefficients : list of dict
        Refactored coefficients from the scattering transform.
    """
    from_real_to_complex = backend.from_real_to_complex
    #sort coefficiants for efficincy
    coefficients = sort_coefficients(coefficients)
    for coeff in coefficients:
        coeff['coef'] = from_real_to_complex(coeff['coef'])
    # sorted_coefficients_by_path = coefficients.copy()
    # sorted_coefficients_by_path.sort(key=lambda x: (x['depth'],x['path']))
    # for i in range(len(sorted_coefficients_by_path)):
    #     if sorted_coefficients_by_path[i]['path'] != coefficients[i]['path']:
    #         raise RuntimeError('Sorting by path failed, the paths are not equal')
    
    print("sorted coefficients by path: \n")
    for coeff in coefficients:
            print("depth: ", coeff['depth'], " path: ", coeff['path']," j: ",coeff['j'], " theta: ",coeff['theta'], " split: ", coeff['split'])    

    return coefficients
    # print("--------------------------------------------------")
    # print("coefficients refactoring: \n")
    # # Refactor the coefficients to match the shape of the wavelets (phi and psi).
    # refactored_coefficients = []
    # for coeff in coefficients:
    #     #for debugging purposes
    #     print("coeff  shape before refactoring: ", coeff['coef'].shape)
        



def InverseScattering2D(coefficients, J, L, max_order, phi, psi,
                        backend, out_type='array', last_layer=None, dilation_optimization=True):
    """
    Inverse scattering function to reconstruct the image from coefficients.
    Parameters
    ----------
    coefficients : list of dict
        Coefficients from the scattering transform.
    J : int
        Logscale of the scattering.
    L : int
        Number of angles used for the wavelet transform.
    max_order : int 
        The depth of the scattering transform.
    image_shape : tuple
        Shape of the original image.
    phi : dict
        Low-pass filter.
    psi : list of dict
        Wavelet filters.
    backend : str
        Backend to use for the computation.
    out_type : str
        The format of the output of a scattering transform. If set to
        `'list'`, then the output is a list containing each individual
        scattering path with meta information. Otherwise, if set to
        `'array'`, the output is a large array containing the
        concatenation of all scattering coefficients. Defaults to
        `'array'`.  
    last_layer : int    
        The intermediate nodes computed in the last layer of the scattering
        transform. i.e olnly convolution with mother filters (U_1_c).
        If None, zero is used.
    """
    #TODO -- implement inversion in this edge case.
    # one possible solution is to take the second to last coefficients and compute from them.
    if max_order >= J:
        raise RuntimeError('max_order must be less than J-1, max_order = %d, J-1 = %d' % (max_order, J-1))
    

    coefficients = RefacrorCoefficients(coefficients, J, L, max_order, phi, psi,
                        backend, out_type, last_layer, dilation_optimization)

    
    if last_layer == None:
        last_layer = build_zero_last_layer(coefficients, J, L, max_order, phi, psi, backend, out_type='array', dilation_optimization=dilation_optimization) 

    return RecursiveInverseScattering2D(coefficients, J, L, max_order, phi, psi,
                                    backend, last_layer=last_layer, dilation_optimization=dilation_optimization)



def RecursiveInverseScattering2D(coefficients, J, L, max_order, phi, psi,
                        backend, last_layer, dilation_optimization=True):
    
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft    
    cdgmm = backend.cdgmm
    stack = backend.stack
    fft = backend.fft
    pad_cmplx = backend.pad_cmplx
    unpad_cmplx = backend.unpad_cmplx
    stack1 = backend.stack1
    custom_relu_unsplit = backend.custom_relu_unsplit
    conjugate_transpose = backend.conjugate_transpose
    zero_coeff = backend.zero_coeff

    
    #Convolve each of the last layer nodes with the corresponding congugated and transposed mother filter
    #i.e: if the node y was of the form y=x*mother_n1 then computing y*mother_n1^H where H is the conjugate transpose
    #using F^-1(<F(y), F(mother_n1^H)>)
    #----------------------------------
    #doing the same for the last coefficients
    #i.e : if the coefficiant c was of the form c=x*father_n1 then computing c*mother_n1^H where H is the conjugate transpose
    #using F^-1(<F(c), F(father_n1^H)>)
    #----------------------------------

    num_coeffs_in_last_layer = num_of_scattering_coefficients_in_layer(max_order, J, L, dilation_optimization=dilation_optimization)    
    num_coeffs = len(coefficients)
    last_coeffs = coefficients[num_coeffs - num_coeffs_in_last_layer : num_coeffs]


##--------------  multiply the last layer nodes with the mother filters ----------------
    reconstructed_nodes_splited = []
    i = 0
    for coeff in last_coeffs:
        reconstructed_node = ifft(cdgmm(fft(coeff['coef']), (phi['levels'][coeff['j']])))#F^-1(<F(c), F(father^H)>)
        
        num_of_coeff_childrens = num_of_childrens(coeff,J,L, dilation_optimization)
        corresponding_intermediate_nodes = last_layer[i:i+num_of_coeff_childrens]
        i += num_of_coeff_childrens

        for (node, j) in zip(corresponding_intermediate_nodes, range(len(psi))):
            j1 = psi[j]['j']
            theta1 = psi[j]['theta']
            reconstructed_node += ifft(cdgmm(fft(node['coef']), (psi[j]['levels'][j1])))#F^-1(<F(c), F(mother^H)>)
        
        reconstructed_nodes_splited.append({'coef': reconstructed_node,
                                    'j': coeff['j'],
                                    'n': coeff['n'],
                                    'theta': coeff['theta'],
                                    'depth' : coeff['depth'],
                                    'split' : coeff['split']})
##-----------------  apply relu unsplit to recollect every 4 nodes ----------------
    if max_order == 0:
        return reconstructed_nodes_splited[0]['coef']
    
    reconstructed_nodes = []    
    for i in range(0,len(reconstructed_nodes_splited),4):
        unified_node = custom_relu_unsplit(reconstructed_nodes_splited[i]['coef'][...,0], reconstructed_nodes_splited[i+1]['coef'][...,1], reconstructed_nodes_splited[i+2]['coef'][...,0], reconstructed_nodes_splited[i+3]['coef'][...,1])
        reconstructed_nodes.append({'coef': unified_node,
                                    'j': reconstructed_nodes_splited[i]['j'],
                                    'n': reconstructed_nodes_splited[i]['n'],
                                    'theta': reconstructed_nodes_splited[i]['theta'],
                                    'depth' : reconstructed_nodes_splited[i]['depth'],
                                    'split' : 'no_split'})
    
    return RecursiveInverseScattering2D(coefficients[0:num_coeffs - num_coeffs_in_last_layer], J, L, max_order-1, phi, psi,
                        backend, last_layer=reconstructed_nodes, dilation_optimization=dilation_optimization)


    








__all__ = ['scattering2d']




#x
#layer 0
#output: x * father
# 
#layer 1
# x * mother
# output: pos_re(x * mother) * father,  pos_im(x * mother) * father , ...
#
# layer 2
# pos_re(x * mother) * mother , pos_im(x * mother) * mother
# output:
