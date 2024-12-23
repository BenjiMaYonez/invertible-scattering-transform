import torch
from torch.nn import ReflectionPad2d
from collections import namedtuple
from packaging import version

from ...backend.torch_backend import TorchBackend


if version.parse(torch.__version__) >= version.parse('1.8'):
    _fft = lambda x: torch.view_as_real(torch.fft.fft2(torch.view_as_complex(x)))
    _ifft = lambda x: torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(x)))
    _irfft = lambda x: torch.fft.ifft2(torch.view_as_complex(x)).real[..., None]
else:
    _fft = lambda x: torch.fft(x, 2, normalized=False)
    _ifft = lambda x: torch.ifft(x, 2, normalized=False)
    _irfft = lambda x: torch.irfft(x, 2, normalized=False, onesided=False)[..., None]


class Pad(object):
    def __init__(self, pad_size, input_size):
        """Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.

            Parameters
            ----------
            pad_size : list of 4 integers
                Size of padding to apply [top, bottom, left, right].
            input_size : list of 2 integers
                size of the original signal [height, width].
        """
        self.pad_size = pad_size
        self.input_size = input_size

        self.build()

    def build(self):
        """Builds the padding module.

            Attributes
            ----------
            padding_module : ReflectionPad2d
                Pads the input tensor using the reflection of the input
                boundary.

        """
        pad_size_tmp = list(self.pad_size)

        # This handles the case where the padding is equal to the image size
        if pad_size_tmp[0] == self.input_size[0]:
            pad_size_tmp[0] -= 1
            pad_size_tmp[1] -= 1
        if pad_size_tmp[2] == self.input_size[1]:
            pad_size_tmp[2] -= 1
            pad_size_tmp[3] -= 1
        # Pytorch expects its padding as [left, right, top, bottom]
        self.padding_module = ReflectionPad2d([pad_size_tmp[2], pad_size_tmp[3],
                                               pad_size_tmp[0], pad_size_tmp[1]])

    def __call__(self, x):
        """Applies padding and maps to complex.

            Parameters
            ----------
            x : tensor
                Real tensor input to be padded and sent to complex domain.

            Returns
            -------
            output : tensor
                Complex torch tensor that has been padded.

        """
        batch_shape = x.shape[:-2]
        signal_shape = x.shape[-2:]
        x = x.reshape((-1, 1) + signal_shape)
        x = self.padding_module(x)

        # Note: PyTorch is not effective to pad signals of size N-1 with N
        # elements, thus we had to add this fix.
        if self.pad_size[0] == self.input_size[0]:
            x = torch.cat([x[:, :, 1, :].unsqueeze(2), x, x[:, :, x.shape[2] - 2, :].unsqueeze(2)], 2)
        if self.pad_size[2] == self.input_size[1]:
            x = torch.cat([x[:, :, :, 1].unsqueeze(3), x, x[:, :, :, x.shape[3] - 2].unsqueeze(3)], 3)

        output = x.reshape(batch_shape + x.shape[-2:] + (1,))
        return output


class TorchBackend2D(TorchBackend):
    Pad = Pad

    @classmethod
    def subsample_fourier(cls, x, k):
        """Subsampling of a 2D image performed in the Fourier domain

            Subsampling in the spatial domain amounts to periodization
            in the Fourier domain, hence the formula.

            Parameters
            ----------
            x : tensor
                Input tensor with at least 5 dimensions, the last being the real
                and imaginary parts.
            k : int
                Integer such that x is subsampled by k along the spatial variables.

            Returns
            -------
            out : tensor
                Tensor such that its Fourier transform is the Fourier
                transform of a subsampled version of x, i.e. in
                F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k].

        """
        cls.contiguous_check(x)
        cls.complex_check(x)

        batch_shape = x.shape[:-3]
        signal_shape = x.shape[-3:]
        x = x.view((-1,) + signal_shape)
        y = x.view(-1,
                       k, x.shape[1] // k,
                       k, x.shape[2] // k,
                       2)

        out = y.mean(3, keepdim=False).mean(1, keepdim=False)
        out = out.reshape(batch_shape + out.shape[-3:])

        return out


    #BINYAMIN - START CHNAGE
    @staticmethod
    def _is_real(x):
        return x.shape[-1] == 1

    @classmethod
    def fft(cls, x):
        cls.contiguous_check(x)
        # if cls._is_real(x):
        #     return cls.rfft(x)

        return _fft(x)
    #BINYAMIN - END CHNAGE

    # we cast to complex here then fft rather than use torch.rfft as torch.rfft is
    # inefficent.
    @classmethod
    def rfft(cls, x):
        cls.contiguous_check(x)
        cls.real_check(x)

        x_r = torch.zeros((x.shape[:-1] + (2,)), dtype=x.dtype, layout=x.layout, device=x.device)
        x_r[..., 0] = x[..., 0]

        return _fft(x_r)

    @classmethod
    def irfft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return _irfft(x)

    @classmethod
    def ifft(cls, x):
        cls.contiguous_check(x)
        cls.complex_check(x)

        return _ifft(x)
    
    #BINYAMIN - START CHNAGE
    @classmethod
    def pad_cmplx(cls, pad, x):
        if torch.is_complex(x):
            real_part = x.real
            imag_part = x.imag
            padded_real = pad(real_part)
            padded_imag = pad(imag_part)
            return torch.view_as_real(torch.complex(padded_real, padded_imag))
        
        if x.ndim == 4 and x.shape[-1] == 2: #x.shape = [n1,n2,...,nk,2]  last dimansion is real,cmplx
            real_part = x[...,0]
            imag_part = x[...,1]
            padded_real = pad(real_part)
            padded_imag = pad(imag_part)
            return torch.stack((real_part,imag_part), dim=-1)
        
        #x.shape = [n1,n2,...,nk] - only real
        result = pad(x)
        imaginary_part = torch.zeros_like(result)
        result = torch.cat((result, imaginary_part), dim=-1)
        return result

    @classmethod
    def unpad_cmplx(cls, unpad, x):
        if torch.is_complex(x):
            real_part = x.real
            imag_part = x.imag
            unpadded_real = unpad(real_part)
            unpadded_imag = unpad(imag_part)
            return torch.view_as_real(torch.complex(unpadded_real, unpadded_imag))
        
        if x.ndim == 4 and x.shape[-1] == 2:
            real_part = x[...,0]
            imag_part = x[...,1]
            padded_real = unpad(real_part)
            padded_imag = unpad(imag_part)
            return torch.stack((real_part,imag_part), dim=-1)
        
        result = unpad(x)
        #-----------------------
        # now x is real values,
        # so : x.shape = [batch_size, num_coefficients, width, length, 1]
        # no need to add zeros to create imag part 
        # want the output of scattering to be real
        #----------------------------

        #imaginary_part = torch.zeros_like(result)
        #result = torch.stack((result, imaginary_part), dim=-1)

        return result

    @classmethod
    def custom_relu_split(cls ,input):
        real_part = input[...,0]
        imag_part = input[...,1]

        # Apply ReLU to real and imaginary parts, and their negatives
        relu_real = torch.relu(real_part)
        relu_imag = torch.relu(imag_part)
        relu_neg_real = torch.relu(-real_part)
        relu_neg_imag = torch.relu(-imag_part)

        seperated_signals = [relu_real, relu_imag, relu_neg_real, relu_neg_imag]
        result = []
        for signal_real_part in seperated_signals:
            signal_imag_part = torch.zeros_like(signal_real_part)
            signal = torch.stack((signal_real_part, signal_imag_part), dim=-1)
            result.append(signal)

        return tuple(result)
    #BINYAMIN - END CHNAGE


    @staticmethod
    def unpad(in_):
        """Unpads input.

            Slices the input tensor at indices between 1:-1.

            Parameters
            ----------
            in_ : tensor
                Input tensor.

            Returns
            -------
            in_[..., 1:-1, 1:-1] : tensor
                Output tensor.  Unpadded input.

        """
        #BINYAMIN - START CHANGE
        #new version
        #in_ = in_[..., 1:-1, 1:-1]
        
        #old version
        in_ = in_[..., 1:-1, 1:-1, :]
        in_ = in_.reshape(in_.shape[:-1])
        #BINYAMIN - END CHANGE
        return in_

    @staticmethod
    def stack(arrays):
        return TorchBackend.stack(arrays, -3)
        
    #BINYAMIN - START CHNAGE
    @staticmethod
    def stack1(arrays):
        return TorchBackend.stack(arrays, 1)
    #BINYAMIN - END CHNAGE

    

     


backend = TorchBackend2D
