def sum_scattering_coeffs(coeffs):
    """
    Sums all scattering coefficients (assumed to be arrays of the same shape).
    coeffs: output of scattering2d with out_type='list' or 'array'
    Returns: summed array
    """
    if isinstance(coeffs, list):
        # list of dicts with 'coef' key
        arrs = [c['coef'] for c in coeffs]
    else:
        arrs = coeffs
    return sum(arrs)