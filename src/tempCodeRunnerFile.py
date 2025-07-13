    """Determine the language of a file based on its extension"""
        if filename.endswith('.java'):
            return 'java'
        elif filename.endswith(('.cpp', '.cc', '.cxx', '.h', '.hpp')):
            return 'cpp'
        return 'unknown'