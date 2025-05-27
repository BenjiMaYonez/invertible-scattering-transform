import math

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array', downsample=True):
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
    S_0 = unpad(S_0)

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
        S_1_r = unpad(S_1_r)

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
            S_2_r = unpad(S_2_r)

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
        out_type='array', downsample=True):
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
    S_0 = unpad_cmplx(unpad,S_0)

    out_S.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': (), 
                    'depth' : 0,
                    'split' : 'no_split'})
    

    recursiveInvertibleScattering2d(U_0_c, pad, unpad, backend, J, L, phi, psi, max_order, 1 ,None, out_type, out_S, downsample)
    # print("out_s.len : %d \n" %len(out_S) )
    if out_type == 'array':
        out_S = stack1([x['coef'] for x in out_S])
        # print("out_s.shape : {} \n".format(out_S.shape))

    return out_S



def recursiveInvertibleScattering2d(U_0_c, pad, unpad, backend, J, L, phi, psi, max_order, level, last_n,
        out_type, out_S, downsample=True):
    
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

        if j1 <= last_j and last_n != None:
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


        

        # Second low pass filter
        S_1 = {k: cdgmm(v, phi['levels'][j1]) for k,v in U_1.items()}
        if downsample:
            S_1 = {k: subsample_fourier(v, k=2 ** (J - j1)) for k,v in S_1.items()}
        #changed from ifft to irfft for the output to be real valued
        S_1 = {k: irfft(v) for k,v in S_1.items()}
        S_1 = {k: unpad_cmplx(unpad,v) for k,v in S_1.items()}


        for key,signal in S_1.items():
            out_S.append({'coef': signal,
                            'j': (j1,),
                            'n': (n1,),
                            'theta': (theta1,), 
                            'depth' : level,
                            'split' : key})
        
        for U_1_c in U_1.values():
            recursiveInvertibleScattering2d(U_1_c, pad, unpad, backend, J, L, phi, psi, max_order, level+1, n1, out_type, out_S, downsample)

def num_of_scattering_coefficients(max_depth, J, L):
    total_sum = 0
    for q in range(0, max_depth + 1):
        combination = math.comb(J, q)
        term = combination * (4 ** q) * (L ** q)
        total_sum += term
    return total_sum

def num_of_scattering_coefficients_in_layer(layer_depth, J, L):
        return math.comb(J, layer_depth) * (4 ** layer_depth) * (L ** layer_depth)

def sort_coefficients(coefficients):
    """
    Sort the coefficients by depth, then by n, then by split in the order:
    're_pos', 'im_pos', 're_neg', 'im_neg', 'no_split'.
    """
    split_order = {'re_pos': 0, 'im_pos': 1, 're_neg': 2, 'im_neg': 3, 'no_split': 4}
    return sorted(
        coefficients,
        key=lambda x: (
            x['depth'],
            x['n'],
            split_order.get(x['split'], 5)  # 5 for any unexpected value
        )
    )



def build_zero_last_layer(coefficients, J, L, max_order, phi, psi,
                        backend, out_type='array'):
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
    num_of_coeffs_in_last_layer = num_of_scattering_coefficients_in_layer(max_order, J, L)
    num_of_coeffs = len(coefficients)

    for i in range(num_of_coeffs - num_of_coeffs_in_last_layer, num_of_coeffs, num_of_coeffs):
        if coefficients[i]['depth'] != max_order:
            raise RuntimeError('Wrong calculation of number of coeffs in the last layer')
        
        for n in range(len(psi)):
            j = psi[n]['j']
            theta = psi[n]['theta']
        
            out_S.append({'coef': S_0,
                        'j': j,
                        'n': n,
                        'theta': theta,
                        'depth' : max_order+1,
                        'split' : coefficients[i]['split']}) 
    
    return out_S


def InverseScattering2D(coefficients, J, L, max_order, phi, psi,
                        backend, out_type='array', last_layer=None):
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
    #sort coefficiants for efficincy
    coefficients = sort_coefficients(coefficients)

    if last_layer == None:
     last_layer = build_zero_last_layer(coefficients, J, L, max_order, phi, psi, backend, out_type='array') 

    RecursiveInverseScattering2D(coefficients, J, L, max_order, phi, psi,
                                    backend, last_layer=last_layer)



def RecursiveInverseScattering2D(coefficients, J, L, max_order, phi, psi,
                        backend, last_layer):
    
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

    if max_order == 0:
        return coefficients[0]['coef']

    
    #Convolve each of the last layer nodes with the corresponding congugated and transposed mother filter
    #i.e: if the node y was of the form y=x*mother_n1 then computing y*mother_n1^H where H is the conjugate transpose
    #using F^-1(<F(y), F(mother_n1^H)>)
    #----------------------------------
    #doing the same for the last coefficients
    #i.e : if the coefficiant c was of the form c=x*father_n1 then computing c*mother_n1^H where H is the conjugate transpose
    #using F^-1(<F(c), F(father_n1^H)>)
    #----------------------------------

    num_coeffs_in_last_layer = num_of_scattering_coefficients_in_layer(max_order, J, L)
    num_coeffs = len(coefficients)
    last_coeffs = coefficients[num_coeffs - num_coeffs_in_last_layer : num_coeffs]

    reconstructed_nodes_splited = []
    for (coeff,i) in zip(last_coeffs, range(0,num_coeffs_in_last_layer,len(psi))):# i is the index of the first node in the last layer that corresponds to coeff
        reconstructed_node = ifft(cdgmm(fft(coeff['coef']), conjugate_transpose(phi['levels'][coeff['j']])))#F^-1(<F(c), F(father^H)>)
        
        corresponding_intermediate_nodes = last_layer[i:i+len(psi)]#TODO:  i must jump by len(psi)
        for (node, j) in zip(corresponding_intermediate_nodes, range(len(psi))):
            j1 = psi[j]['j']
            theta1 = psi[j]['theta']
            reconstructed_node += ifft(cdgmm(fft(node['coef']), conjugate_transpose(psi[j]['levels'][j1])))#F^-1(<F(c), F(mother^H)>)
        
        reconstructed_nodes_splited.append({'coef': reconstructed_node,
                                    'j': coeff['j'],
                                    'n': coeff['n'],
                                    'theta': coeff['theta'],
                                    'depth' : coeff['depth'],
                                    'split' : coeff['split']})
        
    reconstructed_nodes = []    
    for i in range(0,len(reconstructed_nodes_splited),4):
        unified_node = custom_relu_unsplit(reconstructed_nodes_splited[i]['coef'], reconstructed_nodes_splited[i+1]['coef'], reconstructed_nodes_splited[i+2]['coef'], reconstructed_nodes_splited[i+3]['coef'])
        reconstructed_nodes.append({'coef': unified_node,
                                    'j': reconstructed_nodes_splited[i]['j'],
                                    'n': reconstructed_nodes_splited[i]['n'],
                                    'theta': reconstructed_nodes_splited[i]['theta'],
                                    'depth' : reconstructed_nodes_splited[i]['depth'],
                                    'split' : 'no_split'})
    
    return RecursiveInverseScattering2D(coefficients[0:num_coeffs - num_coeffs_in_last_layer], J, L, max_order-1, phi, psi,
                        backend, last_layer=reconstructed_nodes)


    








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
