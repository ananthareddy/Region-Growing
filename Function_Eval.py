from obj_fun import obj_fun

def feval(fname,soln,inp,tar):
    if fname == 'obj_fun':
        out = obj_fun(soln,inp,tar)
    else:
        out = None
    return out


