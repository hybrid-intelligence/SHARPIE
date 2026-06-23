def log(msg: str, level: int = 1, verbosity: int = 1):
    if verbosity >= level:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', 'replace').decode('ascii'))