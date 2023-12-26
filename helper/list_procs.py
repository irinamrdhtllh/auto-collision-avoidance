import psutil


def proc_names():
    return dict([(proc.pid, proc.name()) for proc in psutil.process_iter()])


def proc_cmdlines():
    cmdlines = {}
    for proc in psutil.process_iter():
        try:
            cmdlines[proc.pid] = proc.cmdline()
        except psutil.AccessDenied:
            cmdlines[proc.pid] = None
    return cmdlines


def to_regex(regex):
    if not hasattr(regex, "search"):
        import re

        regex = re.compile(regex)
    return regex


def search_procs_by_name(regex):
    pid_names = {}
    regex = to_regex(regex)
    for pid, name in proc_names().items():
        if regex.search(name):
            pid_names[pid] = name
    return pid_names


def search_procs_by_cmdline(regex):
    pid_cmdlines = {}
    regex = to_regex(regex)
    for pid, cmdline in proc_cmdlines().items():
        if cmdline is not None:
            for part in cmdline:
                if regex.search(part):
                    pid_cmdlines[pid] = cmdline
                    break
    return pid_cmdlines
