import yaml
from pathlib import Path

def read_resource(res_yaml):
    """
    Reads a WindIO energy resource

    Parameters
    ----------
    res_yaml : str
        Path to the yaml file

    Returns
    -------


    """
    res_yaml = Path(res_yaml)

    with open(res_yaml, 'r') as file:
        res = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    print(res)
    quit()

def read_site(site_yaml):
    """
    Reads a WindIO site

    Parameters
    ----------
    site_yaml : str
        Path to the yaml file

    Returns
    -------


    """
    site_yaml = Path(site_yaml)

    with open(site_yaml, 'r') as file:
        site = yaml.load(file, Loader=yaml.loader.BaseLoader)

    res_yaml = site["energy_resource"]
    if res_yaml[0] == ".":
        res_yaml = site_yaml.parent / res_yaml
    resource = read_resource(res_yaml)

def read_case(case_yaml):
    """
    Reads a WindIO case

    Parameters
    ----------
    case_yaml : str
        Path to the yaml file

    Returns
    -------


    """
    case_yaml = Path(case_yaml)

    with open(case_yaml, 'r') as file:
        case = yaml.load(file, Loader=yaml.loader.BaseLoader)
    
    site_yaml = case["site"]
    if site_yaml[0] == ".":
        site_yaml = case_yaml.parent / site_yaml
    site = read_site(site_yaml)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("case_yaml", help="The case yaml file")
    args = parser.parse_args()

    read_case(args.case_yaml)