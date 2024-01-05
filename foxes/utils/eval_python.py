

def eval_str(s, indicator="%"):
    """
    Evaluates strings that start with the
    indicator as python commands
    
    Parameters
    ----------
    s: list, dict or object
        The source to by analyzed
    indicator: str
        The indicator that trigger python evaluation
        
    Returns
    -------
    out: list, dict or object
        The same structure, but all python
        strings evaluated

    """
    if isinstance(s, str):
        L = len(indicator)
        if len(s) > L and s[:L] == indicator:
            return eval(s[L:])
    elif isinstance(s, list):
        return [eval_str(a, indicator) for a in s]
    elif isinstance(s, tuple):
        return tuple(eval_str(list(s), indicator))
    elif isinstance(s, dict):
        return {k: eval_str(a, indicator) for k, a in s.items()}
    return s
