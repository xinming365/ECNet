
global global_dict
global_dict={} # # Loading data with ef and eg labels at default


def set_value(name, value):
    global_dict[name]=value


def get_value(name, defValue=None):
    try:

        return global_dict[name]
    except KeyError:
        return defValue

def get_keys():
    if '_global_dict' in globals().keys():
        return global_dict.keys()
    else:
        return None

if __name__=='__main__':
    print(locals())