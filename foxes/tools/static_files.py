import pkg_resources

""" TODO
# Could be any dot-separated package/module name or a "Requirement"
resource_package = __name__
resource_path = '/'.join(('templates', 'temp_file'))  # Do not use os.path.join()
template = pkg_resources.resource_string(resource_package, resource_path)
# or for a file-like stream:
template = pkg_resources.resource_stream(resource_package, resource_path)
"""