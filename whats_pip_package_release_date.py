import requests
import pprint

# TODO
PKG_LIST = [('numpy', '1.15.0'), 
            ('torch', '0.4.0'), 
            ('torchvision', '0.2.0'), 
            ('scipy', '1.0.0'), 
            ('scikit-image', '0.14.1')]


for pkg_name, pkg_version in PKG_LIST:
    print(f"************ {pkg_name}=={pkg_version} ************")
    res = requests.get(f'https://pypi.org/pypi/{pkg_name}/json')
    d = res.json()
    
    try:
        # All information 
        # pprint.pprint(d['releases'][pkg_version][0])
        print(f"Upload time = {d['releases'][pkg_version][0]['upload_time']}")
    except Exception as e:
        # print(e) 
        print("Can't find specified version. Here's released version availiable.")
        print([i for i in d['releases']])
