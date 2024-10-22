def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array'):
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
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)
        U_1_c = modulus(U_1_c)
        U_1_c = rfft(U_1_c)

        # Second low pass filter
        S_1_c = cdgmm(U_1_c, phi['levels'][j1])
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
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))
            U_2_c = ifft(U_2_c)
            U_2_c = modulus(U_2_c)
            U_2_c = rfft(U_2_c)

            # Third low pass filter
            S_2_c = cdgmm(U_2_c, phi['levels'][j2])
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

def invertibleScattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array'):
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
    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad_cmplx(pad,x)

    U_0_c = fft(U_r)#F(x)

    # First low pass filter
    U_1_c = cdgmm(U_0_c, phi['levels'][0])#<F(x), F(father)>
    U_1_c = subsample_fourier(U_1_c, k=2 ** J)

    #S_0 = irfft(U_1_c)
    S_0 = ifft(U_1_c)# F^-1(<F(x), F(father)>)  = x * father
    S_0 = unpad_cmplx(unpad,S_0)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1]['levels'][0])#< F(x) , F(mother_n1) >
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)# x * mother_n1
        positive_real_U1, positive_imag_U1, neg_real_U1, neg_imag_U1 = custom_relu_split(U_1_c)
        
        U_1 = [positive_real_U1, positive_imag_U1, neg_real_U1, neg_imag_U1]
        U_1 = [fft(signal) for signal in U_1]

        

        # Second low pass filter
        S_1 = [cdgmm(signal, phi['levels'][j1]) for signal in U_1]
        S_1 = [subsample_fourier(signal, k=2 ** (J - j1)) for signal in S_1]
        S_1 = [ifft(signal) for signal in S_1]
        S_1 = [unpad_cmplx(unpad,signal) for signal in S_1]


        for signal in S_1:
            out_S_1.append({'coef': signal,
                            'j': (j1,),
                            'n': (n1,),
                            'theta': (theta1,)})

        if max_order < 2:
            continue
        for U_1_c in U_1: #U_1_c = F(pos_real(x*mother))
            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                theta2 = psi[n2]['theta']

                if j2 <= j1:
                    continue
        
                U_2_c = cdgmm(U_1_c, psi[n2]['levels'][j1]) # < F(pos_real(x*mother_1)) , F(mother_2) >
                U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1))  
                U_2_c = ifft(U_2_c) # pos_real(x*mother_1) * mother_2


                positive_real_U2, positive_imag_U2, neg_real_U2, neg_imag_U2 = custom_relu_split(U_2_c)
                U_2 = [positive_real_U2, positive_imag_U2, neg_real_U2, neg_imag_U2]
                U_2 = [fft(signal) for signal in U_2]


                # Third low pass filter
                S_2 = [cdgmm(signal, phi['levels'][j2]) for signal in U_2]
                S_2 = [subsample_fourier(signal, k=2 ** (J - j2)) for signal in S_2]
                S_2 = [ifft(signal) for signal in S_2]
                S_2 = [unpad_cmplx(unpad,signal) for siganl in S_2]


                for signal in S_2:
                    out_S_2.append({'coef': signal,
                                    'j': (j1, j2),
                                    'n': (n1, n2),
                                    'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        out_S = stack1([x['coef'] for x in out_S])

    return out_S




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