import types
from loguru import logger

def wrap(obj, name, funcname, newfunc, isobj=True):
    # --- init ---
    wrapper_chain_name = f'_{funcname}_wrapper_chain'
    if not hasattr(obj, wrapper_chain_name):
        obj.__setattr__(wrapper_chain_name, [])

    # --- check ---
    wrapper_chain = obj.__getattribute__(wrapper_chain_name)
    for wrapper in wrapper_chain:
        if wrapper['name'] == name:
            logger.error(f'already has the {name} in {func_chain_name}, please check')
            raise Exception()

    if not isinstance(newfunc, types.MethodType) and isobj:
        newfunc = types.MethodType(newfunc, obj)

    # --- update ---
    new_wrapper = {
        "name": name,
        "funcname": funcname,
        "origin": obj.__getattribute__(funcname) if hasattr(obj, funcname) else None,
        "new": newfunc
    }
    wrapper_chain.insert(0, new_wrapper)

    obj.__setattr__(funcname, newfunc)

    return obj

def obtain_origin(obj, name, funcname):
    wrapper_chain_name = f'_{funcname}_wrapper_chain'
    if not hasattr(obj, wrapper_chain_name):
        logger.error(f'the obj has no wrapper')
        raise Exception
    
    wrapper_chain = obj.__getattribute__(wrapper_chain_name)

    for wi, wrapper in enumerate(wrapper_chain):
        if wrapper['name'] == name:
            return wrapper["origin"]
    else:
        logger.error(f'the obj has no wrapper named {name}')
        raise Exception

def unwrap(obj, name, funcname):
    wrapper_chain_name = f'_{funcname}_wrapper_chain'
    if not hasattr(obj, wrapper_chain_name):
        logger.warning(f'the obj has no wrapper')
        return obj
    
    wrapper_chain = obj.__getattribute__(wrapper_chain_name)

    for wi, wrapper in enumerate(wrapper_chain):
        if wrapper['name'] == name:
            if wi == 0:
                if wrapper['origin'] is not None:
                    obj.__setattr__(funcname, wrapper['origin'])
                else:
                    obj.__delattr__(funcname)
            else:
                wrapper[wi-1]["origin"] = wrapper["origin"]
            del wrapper_chain[wi]
            break
    else:
        logger.warning(f'the obj has no wrapper named {name}')

    return obj
