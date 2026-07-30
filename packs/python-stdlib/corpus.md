# python-stdlib corpus — generated from CPython 3.13.5

Every signature below came from `inspect.signature()` on the
running interpreter, not from documentation or memory — correct
by construction for this Python version, and regenerated rather
than edited. Docstring summaries are CPython's own.

Recipes (task → idiomatic snippet) live in recipes.md and are
authored, not generated: ground truth can be introspected;
idiom cannot.

## pathlib.UnsupportedOperation: an exception that is raised when an unsupported operation is called on a path object

An exception that is raised when an unsupported operation is called on a path object.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## pathlib.PurePath: base class for manipulating paths without i/o

Base class for manipulating paths without I/O. PurePath represents a filesystem path and offers operations which don't imply any actual filesystem I/O. Depending on your system, instantiating a PurePath will return either a PurePosixPath or a PureWindowsPath object. You can also instantiate either of these classes directly, regardless of your system.

Key methods are as_posix, as_uri, full_match, is_absolute, is_relative_to, is_reserved, joinpath, match, relative_to, with_name, with_segments, with_stem, with_suffix.

```python
as_posix(self)
as_uri(self)
full_match(self, pattern, *, case_sensitive=None)
is_absolute(self)
is_relative_to(self, other, /, *_deprecated)
is_reserved(self)
joinpath(self, *pathsegments)
match(self, path_pattern, *, case_sensitive=None)
relative_to(self, other, /, *_deprecated, walk_up=False)
with_name(self, name)
with_segments(self, *pathsegments)
with_stem(self, stem)
with_suffix(self, suffix)
```

## pathlib.PurePosixPath: purepath subclass for non-windows systems

PurePath subclass for non-Windows systems. On a POSIX system, instantiating a PurePath should return this object. However, you can also instantiate it directly on any system.

Key methods are as_posix, as_uri, full_match, is_absolute, is_relative_to, is_reserved, joinpath, match.

```python
as_posix(self)
as_uri(self)
full_match(self, pattern, *, case_sensitive=None)
is_absolute(self)
is_relative_to(self, other, /, *_deprecated)
is_reserved(self)
joinpath(self, *pathsegments)
match(self, path_pattern, *, case_sensitive=None)
```

## pathlib.PureWindowsPath: purepath subclass for windows systems

PurePath subclass for Windows systems. On a Windows system, instantiating a PurePath should return this object. However, you can also instantiate it directly on any system.

Key methods are as_posix, as_uri, full_match, is_absolute, is_relative_to, is_reserved, joinpath, match.

```python
as_posix(self)
as_uri(self)
full_match(self, pattern, *, case_sensitive=None)
is_absolute(self)
is_relative_to(self, other, /, *_deprecated)
is_reserved(self)
joinpath(self, *pathsegments)
match(self, path_pattern, *, case_sensitive=None)
```

## pathlib.Path: purepath subclass that can make system calls

PurePath subclass that can make system calls. Path represents a filesystem path but unlike PurePath, also offers methods to do system calls on path objects. Depending on your system, instantiating a Path will return either a PosixPath or a WindowsPath object. You can also instantiate a PosixPath or WindowsPath directly, but cannot instantiate a WindowsPath on a POSIX system or vice versa.

Key methods are absolute, as_posix, as_uri, chmod, cwd, exists, expanduser, from_uri, full_match, glob, group, hardlink_to, home, is_absolute, is_block_device, is_char_device, is_dir, is_fifo.

```python
absolute(self)
as_posix(self)
as_uri(self)
chmod(self, mode, *, follow_symlinks=True)
cwd()
exists(self, *, follow_symlinks=True)
expanduser(self)
from_uri(uri)
full_match(self, pattern, *, case_sensitive=None)
glob(self, pattern, *, case_sensitive=None, recurse_symlinks=False)
group(self, *, follow_symlinks=True)
hardlink_to(self, target)
home()
is_absolute(self)
is_block_device(self)
is_char_device(self)
is_dir(self, *, follow_symlinks=True)
is_fifo(self)
```

## pathlib.PosixPath: path subclass for non-windows systems

Path subclass for non-Windows systems. On a POSIX system, instantiating a Path should return this object.

Key methods are absolute, as_posix, as_uri, chmod, cwd, exists, expanduser, from_uri.

```python
absolute(self)
as_posix(self)
as_uri(self)
chmod(self, mode, *, follow_symlinks=True)
cwd()
exists(self, *, follow_symlinks=True)
expanduser(self)
from_uri(uri)
```

## pathlib.WindowsPath: path subclass for windows systems

Path subclass for Windows systems. On a Windows system, instantiating a Path should return this object.

Key methods are absolute, as_posix, as_uri, chmod, cwd, exists, expanduser, from_uri.

```python
absolute(self)
as_posix(self)
as_uri(self)
chmod(self, mode, *, follow_symlinks=True)
cwd()
exists(self, *, follow_symlinks=True)
expanduser(self)
from_uri(uri)
```

## shutil.copyfileobj: copy data from file-like object fsrc to file-like object fdst

copy data from file-like object fsrc to file-like object fdst.

```python
shutil.copyfileobj(fsrc, fdst, length=0)
```

## shutil.copyfile: copy data from src to dst in the most efficient way possible

Copy data from src to dst in the most efficient way possible. If follow_symlinks is not set and src is a symbolic link, a new symlink will be created instead of copying the file it points to.

```python
shutil.copyfile(src, dst, *, follow_symlinks=True)
```

## shutil.copymode: copy mode bits from src to dst

Copy mode bits from src to dst. If follow_symlinks is not set, symlinks aren't followed if and only if both `src` and `dst` are symlinks.

```python
shutil.copymode(src, dst, *, follow_symlinks=True)
```

## shutil.copystat: copy file metadata copy the permission bits, last access time, last modification time, and flags from `src` to `dst`

Copy file metadata Copy the permission bits, last access time, last modification time, and flags from `src` to `dst`. On Linux, copystat() also copies the "extended attributes" where possible.

```python
shutil.copystat(src, dst, *, follow_symlinks=True)
```

## shutil.copy: copy data and mode bits ("cp src dst")

Copy data and mode bits ("cp src dst"). Return the file's destination.

```python
shutil.copy(src, dst, *, follow_symlinks=True)
```

## shutil.copy2: copy data and metadata

Copy data and metadata. Return the file's destination.

```python
shutil.copy2(src, dst, *, follow_symlinks=True)
```

## shutil.copytree: recursively copy a directory tree and return the destination directory

Recursively copy a directory tree and return the destination directory. If exception(s) occur, an Error is raised with a list of reasons.

```python
shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=<function copy2 at 0x0000027F967284A0>, ignore_dangling_symlinks=False, dirs_exist_ok=False)
```

## shutil.move: recursively move a file or directory to another location

Recursively move a file or directory to another location. This is similar to the Unix "mv" command.

```python
shutil.move(src, dst, copy_function=<function copy2 at 0x0000027F967284A0>)
```

## shutil.rmtree: recursively delete a directory tree

Recursively delete a directory tree. If dir_fd is not None, it should be a file descriptor open to a directory; path will then be relative to that directory.

**Hazard:** onerror is deprecated and only remains for backwards compatibility.

```python
shutil.rmtree(path, ignore_errors=False, onerror=None, *, onexc=None, dir_fd=None)
```

## shutil.Error: base class for i/o related errors

Base class for I/O related errors.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## shutil.SpecialFileError: raised when trying to do a kind of operation (e

Raised when trying to do a kind of operation (e.g. copying) which is not supported on a special file (e.g. a named pipe).

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## shutil.ExecError: raised when a command could not be executed

Raised when a command could not be executed.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## shutil.make_archive: create an archive file (eg

Create an archive file (eg. zip or tar).

```python
shutil.make_archive(base_name, format, root_dir=None, base_dir=None, verbose=0, dry_run=0, owner=None, group=None, logger=None)
```

## shutil.get_archive_formats: returns a list of supported formats for archiving and unarchiving

Returns a list of supported formats for archiving and unarchiving. Each element of the returned sequence is a tuple (name, description).

```python
shutil.get_archive_formats()
```

## shutil.register_archive_format: registers an archive format

Registers an archive format. name is the name of the format.

```python
shutil.register_archive_format(name, function, extra_args=None, description='')
```

## shutil.unregister_archive_format


```python
shutil.unregister_archive_format(name)
```

## shutil.get_unpack_formats: returns a list of supported formats for unpacking

Returns a list of supported formats for unpacking. Each element of the returned sequence is a tuple (name, extensions, description).

```python
shutil.get_unpack_formats()
```

## shutil.register_unpack_format: registers an unpack format

Registers an unpack format. `name` is the name of the format.

```python
shutil.register_unpack_format(name, extensions, function, extra_args=None, description='')
```

## shutil.unregister_unpack_format: removes the pack format from the registry

Removes the pack format from the registry.

```python
shutil.unregister_unpack_format(name)
```

## shutil.unpack_archive: unpack an archive

Unpack an archive. `filename` is the name of the archive.

```python
shutil.unpack_archive(filename, extract_dir=None, format=None, *, filter=None)
```

## shutil.ignore_patterns: function that can be used as copytree() ignore parameter

Function that can be used as copytree() ignore parameter. Patterns is a sequence of glob-style patterns that are used to exclude files.

```python
shutil.ignore_patterns(*patterns)
```

## shutil.chown: change owner user and group of the given path

Change owner user and group of the given path. user and group can be the uid/gid or the user/group names, and in that case, they are converted to their respective uid/gid.

```python
shutil.chown(path, user=None, group=None, *, dir_fd=None, follow_symlinks=True)
```

## shutil.which: given a command, mode, and a path string, return the path which conforms to the given mode on the path, or none if there is no such file

Given a command, mode, and a PATH string, return the path which conforms to the given mode on the PATH, or None if there is no such file. `mode` defaults to os.F_OK | os.X_OK.

```python
shutil.which(cmd, mode=1, path=None)
```

## shutil.get_terminal_size: get the size of the terminal window

Get the size of the terminal window. For each of the two dimensions, the environment variable, COLUMNS and LINES respectively, is checked.

```python
shutil.get_terminal_size(fallback=(80, 24))
```

## shutil.SameFileError: raised when source and destination are the same file

Raised when source and destination are the same file.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## shutil.disk_usage: return disk usage statistics about the given path

Return disk usage statistics about the given path. Returned values is a named tuple with attributes 'total', 'used' and 'free', which are the amount of total, used and free space, in bytes.

```python
shutil.disk_usage(path)
```

## subprocess.Popen: execute a child program in a new process

Execute a child program in a new process. For a complete description of the arguments see the Python documentation. Arguments: args: A string, or a sequence of program arguments. bufsize: supplied as the buffering argument to the open() function when creating the stdin/stdout/stderr pipe file objects executable: A replacement program to execute.

```python
subprocess.Popen(args, bufsize=-1, executable=None, stdin=None, stdout=None, stderr=None, preexec_fn=None, close_fds=True, shell=False, cwd=None, env=None, universal_newlines=None, startupinfo=None, creationflags=0, restore_signals=True, start_new_session=False, pass_fds=(), *, user=None, group=None, extra_groups=None, encoding=None, errors=None, text=None, umask=-1, pipesize=-1, process_group=None)
```

Key methods are communicate, kill, poll, send_signal.

```python
communicate(self, input=None, timeout=None)
kill(self)
poll(self)
send_signal(self, sig)
```

## subprocess.call: run command with arguments

Run command with arguments. Wait for command to complete or for timeout seconds, then return the returncode attribute.

```python
subprocess.call(*popenargs, timeout=None, **kwargs)
```

## subprocess.check_call: run command with arguments

Run command with arguments. Wait for command to complete.

```python
subprocess.check_call(*popenargs, **kwargs)
```

## subprocess.getstatusoutput: return (exitcode, output) of executing cmd in a shell

Return (exitcode, output) of executing cmd in a shell. Execute the string 'cmd' in a shell with 'check_output' and return a 2-tuple (status, output).

```python
subprocess.getstatusoutput(cmd, *, encoding=None, errors=None)
```

## subprocess.getoutput: return output (stdout or stderr) of executing cmd in a shell

Return output (stdout or stderr) of executing cmd in a shell. Like getstatusoutput(), except the exit status is ignored and the return value is a string containing the command's output.

```python
subprocess.getoutput(cmd, *, encoding=None, errors=None)
```

## subprocess.check_output: run command with arguments and return its output

Run command with arguments and return its output. If the exit code was non-zero it raises a CalledProcessError.

```python
subprocess.check_output(*popenargs, timeout=None, **kwargs)
```

## subprocess.run: run command with arguments and return a completedprocess instance

Run command with arguments and return a CompletedProcess instance. The returned instance will have attributes args, returncode, stdout and stderr.

```python
subprocess.run(*popenargs, input=None, capture_output=False, timeout=None, check=False, **kwargs)
```

## subprocess.CalledProcessError: raised when run() is called with check=true and the process returns a non-zero exit status

Raised when run() is called with check=True and the process returns a non-zero exit status. Attributes: cmd, returncode, stdout, stderr, output.

```python
subprocess.CalledProcessError(returncode, cmd, output=None, stderr=None)
```

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## subprocess.SubprocessError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## subprocess.TimeoutExpired: this exception is raised when the timeout expires while waiting for a child process

This exception is raised when the timeout expires while waiting for a child process. Attributes: cmd, output, stdout, stderr, timeout.

```python
subprocess.TimeoutExpired(cmd, timeout, output=None, stderr=None)
```

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## subprocess.CompletedProcess: a process that has finished running

A process that has finished running. This is returned by run(). Attributes: args: The list or str args passed to run(). returncode: The exit code of the process, negative for signals.

```python
subprocess.CompletedProcess(args, returncode, stdout=None, stderr=None)
```

Key methods are check_returncode.

```python
check_returncode(self)
```

## subprocess.STARTUPINFO


```python
subprocess.STARTUPINFO(*, dwFlags=0, hStdInput=None, hStdOutput=None, hStdError=None, wShowWindow=0, lpAttributeList=None)
```

Key methods are copy.

```python
copy(self)
```

## os.path.normcase: normalize case of pathname

Normalize case of pathname. Makes all characters lowercase and all slashes into backslashes.

```python
os.path.normcase(s)
```

## os.path.isabs: test whether a path is absolute

Test whether a path is absolute.

```python
os.path.isabs(s)
```

## os.path.join


```python
os.path.join(path, *paths)
```

## os.path.splitdrive: split a pathname into drive/unc sharepoint and relative path specifiers

Split a pathname into drive/UNC sharepoint and relative path specifiers. Returns a 2-tuple (drive_or_unc, path); either part may be empty.

```python
os.path.splitdrive(p)
```

## os.path.split: split a pathname

Split a pathname. Return tuple (head, tail) where tail is everything after the final slash.

```python
os.path.split(p)
```

## os.path.splitext: split the extension from a pathname

Split the extension from a pathname. Extension is everything from the last dot to the end, ignoring leading dots.

```python
os.path.splitext(p)
```

## os.path.basename: returns the final component of a pathname

Returns the final component of a pathname.

```python
os.path.basename(p)
```

## os.path.dirname: returns the directory component of a pathname

Returns the directory component of a pathname.

```python
os.path.dirname(p)
```

## os.path.ismount: test whether a path is a mount point (a drive root, the root of a share, or a mounted volume)

Test whether a path is a mount point (a drive root, the root of a share, or a mounted volume).

```python
os.path.ismount(path)
```

## os.path.isreserved: return true if the pathname is reserved by the system

Return true if the pathname is reserved by the system.

```python
os.path.isreserved(path)
```

## os.path.expanduser: expand ~ and ~user constructs

Expand ~ and ~user constructs. If user or $HOME is unknown, do nothing.

```python
os.path.expanduser(path)
```

## os.path.expandvars: expand shell variables of the forms $var, ${var} and %var%

Expand shell variables of the forms $var, ${var} and %var%. Unknown variables are left unchanged.

```python
os.path.expandvars(path)
```

## os.path.abspath: return the absolute version of a path

Return the absolute version of a path.

```python
os.path.abspath(path)
```

## os.path.realpath


```python
os.path.realpath(path, *, strict=False)
```

## os.path.relpath: return a relative version of a path

Return a relative version of a path.

```python
os.path.relpath(path, start=None)
```

## os.path.commonpath: given an iterable of path names, returns the longest common sub-path

Given an iterable of path names, returns the longest common sub-path.

```python
os.path.commonpath(paths)
```

## os.path.isdevdrive: determines whether the specified path is on a windows dev drive

Determines whether the specified path is on a Windows Dev Drive.

```python
os.path.isdevdrive(path)
```

## tempfile.NamedTemporaryFile: create and return a temporary file

Create and return a temporary file. Arguments: 'prefix', 'suffix', 'dir' -- as for mkstemp.

```python
tempfile.NamedTemporaryFile(mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, delete=True, *, errors=None, delete_on_close=True)
```

## tempfile.TemporaryFile: create and return a temporary file

Create and return a temporary file. Arguments: 'prefix', 'suffix', 'dir' -- as for mkstemp.

```python
tempfile.TemporaryFile(mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, delete=True, *, errors=None, delete_on_close=True)
```

## tempfile.SpooledTemporaryFile: temporary file wrapper, specialized to switch from bytesio or stringio to a real file when it exceeds a certain size or when a fileno is needed

Temporary file wrapper, specialized to switch from BytesIO or StringIO to a real file when it exceeds a certain size or when a fileno is needed.

```python
tempfile.SpooledTemporaryFile(max_size=0, mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, *, errors=None)
```

Key methods are close, detach, fileno, flush, isatty, read, read1, readable, readinto, readinto1, readline, readlines, rollover.

```python
close(self)
detach(self)
fileno(self)
flush(self)
isatty(self)
read(self, *args)
read1(self, *args)
readable(self)
readinto(self, b)
readinto1(self, b)
readline(self, *args)
readlines(self, *args)
rollover(self)
```

## tempfile.TemporaryDirectory: create and return a temporary directory

Create and return a temporary directory. This has the same behavior as mkdtemp but can be used as a context manager. For example: with TemporaryDirectory() as tmpdir: ... Upon exiting the context, the directory and everything contained in it are removed (unless delete=False is passed or an exception is raised during cleanup and ignore_cleanup_errors is not True).

```python
tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False, *, delete=True)
```

Key methods are cleanup.

```python
cleanup(self)
```

## tempfile.mkstemp: user-callable function to create and return a unique temporary file

User-callable function to create and return a unique temporary file. The return value is a pair (fd, name) where fd is the file descriptor returned by os.open, and name is the filename.

```python
tempfile.mkstemp(suffix=None, prefix=None, dir=None, text=False)
```

## tempfile.mkdtemp: user-callable function to create and return a unique temporary directory

User-callable function to create and return a unique temporary directory. The return value is the pathname of the directory.

```python
tempfile.mkdtemp(suffix=None, prefix=None, dir=None)
```

## tempfile.mktemp: user-callable function to return a unique temporary file name

User-callable function to return a unique temporary file name. The file is not created.

**Hazard:** THIS FUNCTION IS UNSAFE AND SHOULD NOT BE USED.

```python
tempfile.mktemp(suffix='', prefix='tmp', dir=None)
```

## tempfile.gettempprefix: the default prefix for temporary directories as string

The default prefix for temporary directories as string.

```python
tempfile.gettempprefix()
```

## tempfile.gettempdir: returns tempfile

Returns tempfile.tempdir as str.

```python
tempfile.gettempdir()
```

## tempfile.gettempprefixb: the default prefix for temporary directories as bytes

The default prefix for temporary directories as bytes.

```python
tempfile.gettempprefixb()
```

## tempfile.gettempdirb: returns tempfile

Returns tempfile.tempdir as bytes.

```python
tempfile.gettempdirb()
```

## glob.glob: return a list of paths matching a pathname pattern

Return a list of paths matching a pathname pattern. The pattern may contain simple shell-style wildcards a la fnmatch.

```python
glob.glob(pathname, *, root_dir=None, dir_fd=None, recursive=False, include_hidden=False)
```

## glob.iglob: return an iterator which yields the paths matching a pathname pattern

Return an iterator which yields the paths matching a pathname pattern. The pattern may contain simple shell-style wildcards a la fnmatch.

```python
glob.iglob(pathname, *, root_dir=None, dir_fd=None, recursive=False, include_hidden=False)
```

## glob.escape: escape all special characters

Escape all special characters.

```python
glob.escape(pathname)
```

## glob.translate: translate a pathname with shell wildcards to a regular expression

Translate a pathname with shell wildcards to a regular expression. If `recursive` is true, the pattern segment '**' will match any number of path segments.

```python
glob.translate(pat, *, recursive=False, include_hidden=False, seps=None)
```

## json.dump: serialize ``obj`` as a json formatted stream to ``fp`` (a ``

Serialize ``obj`` as a JSON formatted stream to ``fp`` (a ``.write()``-supporting file-like object). If ``skipkeys`` is true then ``dict`` keys that are not basic types (``str``, ``int``, ``float``, ``bool``, ``None``) will be skipped instead of raising a ``TypeError``.

```python
json.dump(obj, fp, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
```

## json.dumps: serialize ``obj`` to a json formatted ``str``

Serialize ``obj`` to a JSON formatted ``str``. If ``skipkeys`` is true then ``dict`` keys that are not basic types (``str``, ``int``, ``float``, ``bool``, ``None``) will be skipped instead of raising a ``TypeError``.

```python
json.dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=None, sort_keys=False, **kw)
```

## json.load: deserialize ``fp`` (a ``

Deserialize ``fp`` (a ``.read()``-supporting file-like object containing a JSON document) to a Python object. ``object_hook`` is an optional function that will be called with the result of any object literal decode (a ``dict``).

```python
json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
```

## json.loads: deserialize ``s`` (a ``str``, ``bytes`` or ``bytearray`` instance containing a json document) to a python object

Deserialize ``s`` (a ``str``, ``bytes`` or ``bytearray`` instance containing a JSON document) to a Python object. ``object_hook`` is an optional function that will be called with the result of any object literal decode (a ``dict``).

```python
json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
```

## json.JSONDecoder: simple json <https://json

Simple JSON <https://json.org> decoder Performs the following translations in decoding by default: +---------------+-------------------+ | JSON          | Python            | +===============+===================+ | object        | dict              | +---------------+-------------------+ | array         | list              | +---------------+-------------------+ | string        | str               | +---------------+-------------------+ | number (int)  | int               | +---------------+-------------------+ | number (real) | float             | +---------------+-------------------+ | true          | True              | +---------------+-------------------+ | false         | False             | +---------------+-------------------+ | null          | None              | +---------------+-------------------+ It also understands ``NaN``, ``Infinity``, and ``-Infinity`` as their corresponding ``float`` values, which is outside the JSON spec.

```python
json.JSONDecoder(*, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, strict=True, object_pairs_hook=None)
```

Key methods are decode, raw_decode.

```python
decode(self, s, _w=<built-in method match of re.Pattern object at 0x0000027F96715700>)
raw_decode(self, s, idx=0)
```

## json.JSONDecodeError: subclass of valueerror with the following additional properties: msg: the unformatted error message doc: the json document being parsed pos: the start index of doc where parsing failed lineno: the line corresponding to pos colno: the column corresponding to pos

Subclass of ValueError with the following additional properties: msg: The unformatted error message doc: The JSON document being parsed pos: The start index of doc where parsing failed lineno: The line corresponding to pos colno: The column corresponding to pos.

```python
json.JSONDecodeError(msg, doc, pos)
```

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## json.JSONEncoder: extensible json <https://json

Extensible JSON <https://json.org> encoder for Python data structures. Supports the following objects and types by default: +-------------------+---------------+ | Python            | JSON          | +===================+===============+ | dict              | object        | +-------------------+---------------+ | list, tuple       | array         | +-------------------+---------------+ | str               | string        | +-------------------+---------------+ | int, float        | number        | +-------------------+---------------+ | True              | true          | +-------------------+---------------+ | False             | false         | +-------------------+---------------+ | None              | null          | +-------------------+---------------+ To extend this to recognize other objects, subclass and implement a ``.default()`` method with another method that returns a serializable object for ``o`` if possible, otherwise it should call the superclass implementation (to raise ``TypeError``).

```python
json.JSONEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)
```

Key methods are default, encode, iterencode.

```python
default(self, o)
encode(self, o)
iterencode(self, o, _one_shot=False)
```

## csv.Dialect: describe a csv dialect

Describe a CSV dialect. This must be subclassed (see csv.excel). Valid attributes are: delimiter, quotechar, escapechar, doublequote, skipinitialspace, lineterminator, quoting.

```python
csv.Dialect()
```

## csv.excel: describe the usual properties of excel-generated csv files

Describe the usual properties of Excel-generated CSV files.

```python
csv.excel()
```

## csv.excel_tab: describe the usual properties of excel-generated tab-delimited files

Describe the usual properties of Excel-generated TAB-delimited files.

```python
csv.excel_tab()
```

## csv.Sniffer: "sniffs" the format of a csv file (i

"Sniffs" the format of a CSV file (i.e. delimiter, quotechar) Returns a Dialect object.

```python
csv.Sniffer()
```

Key methods are has_header, sniff.

```python
has_header(self, sample)
sniff(self, sample, delimiters=None)
```

## csv.DictReader


```python
csv.DictReader(f, fieldnames=None, restkey=None, restval=None, dialect='excel', *args, **kwds)
```

## csv.DictWriter


```python
csv.DictWriter(f, fieldnames, restval='', extrasaction='raise', dialect='excel', *args, **kwds)
```

Key methods are writeheader, writerow, writerows.

```python
writeheader(self)
writerow(self, rowdict)
writerows(self, rowdicts)
```

## csv.unix_dialect: describe the usual properties of unix-generated csv files

Describe the usual properties of Unix-generated CSV files.

```python
csv.unix_dialect()
```

## re.match: try to apply the pattern at the start of the string, returning a match object, or none if no match was found

Try to apply the pattern at the start of the string, returning a Match object, or None if no match was found.

```python
re.match(pattern, string, flags=0)
```

## re.fullmatch: try to apply the pattern to all of the string, returning a match object, or none if no match was found

Try to apply the pattern to all of the string, returning a Match object, or None if no match was found.

```python
re.fullmatch(pattern, string, flags=0)
```

## re.search: scan through string looking for a match to the pattern, returning a match object, or none if no match was found

Scan through string looking for a match to the pattern, returning a Match object, or None if no match was found.

```python
re.search(pattern, string, flags=0)
```

## re.sub: return the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in string by the replacement repl

Return the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in string by the replacement repl. repl can be either a string or a callable; if a string, backslash escapes in it are processed.

```python
re.sub(pattern, repl, string, count=0, flags=0)
```

## re.subn: return a 2-tuple containing (new_string, number)

Return a 2-tuple containing (new_string, number). new_string is the string obtained by replacing the leftmost non-overlapping occurrences of the pattern in the source string by the replacement repl.

```python
re.subn(pattern, repl, string, count=0, flags=0)
```

## re.split: split the source string by the occurrences of the pattern, returning a list containing the resulting substrings

Split the source string by the occurrences of the pattern, returning a list containing the resulting substrings. If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting list.

```python
re.split(pattern, string, maxsplit=0, flags=0)
```

## re.findall: return a list of all non-overlapping matches in the string

Return a list of all non-overlapping matches in the string. If one or more capturing groups are present in the pattern, return a list of groups; this will be a list of tuples if the pattern has more than one group.

```python
re.findall(pattern, string, flags=0)
```

## re.finditer: return an iterator over all non-overlapping matches in the string

Return an iterator over all non-overlapping matches in the string. For each match, the iterator returns a Match object.

```python
re.finditer(pattern, string, flags=0)
```

## re.compile: compile a regular expression pattern, returning a pattern object

Compile a regular expression pattern, returning a Pattern object.

```python
re.compile(pattern, flags=0)
```

## re.purge: clear the regular expression caches

Clear the regular expression caches.

```python
re.purge()
```

## re.escape: escape special characters in a string

Escape special characters in a string.

```python
re.escape(pattern)
```

## re.error: exception raised for invalid regular expressions

Exception raised for invalid regular expressions. Attributes: msg: The unformatted error message pattern: The regular expression pattern pos: The index in the pattern where compilation failed (may be None) lineno: The line corresponding to pos (may be None) colno: The column corresponding to pos (may be None).

```python
re.error(msg, pattern=None, pos=None)
```

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## re.Pattern: compiled regular expression object

Compiled regular expression object.

```python
re.Pattern()
```

Key methods are findall, finditer, fullmatch, match.

```python
findall(self, /, string, pos=0, endpos=9223372036854775807)
finditer(self, /, string, pos=0, endpos=9223372036854775807)
fullmatch(self, /, string, pos=0, endpos=9223372036854775807)
match(self, /, string, pos=0, endpos=9223372036854775807)
```

## re.Match: the result of re

The result of re.match() and re.search(). Match objects always have a boolean value of True.

```python
re.Match()
```

Key methods are end, expand, groupdict, groups.

```python
end(self, group=0, /)
expand(self, /, template)
groupdict(self, /, default=None)
groups(self, /, default=None)
```

## re.RegexFlag: an enumeration

An enumeration.

```python
re.RegexFlag(*values)
```

Key methods are as_integer_ratio, bit_count, bit_length, conjugate.

```python
as_integer_ratio(self, /)
bit_count(self, /)
bit_length(self, /)
conjugate(self, /)
```

## re.PatternError: exception raised for invalid regular expressions

Exception raised for invalid regular expressions. Attributes: msg: The unformatted error message pattern: The regular expression pattern pos: The index in the pattern where compilation failed (may be None) lineno: The line corresponding to pos (may be None) colno: The column corresponding to pos (may be None).

```python
re.PatternError(msg, pattern=None, pos=None)
```

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## datetime.date


```python
datetime.date(year, month, day) -
```

Key methods are ctime, fromtimestamp, isocalendar, isoformat.

```python
ctime(self, /)
fromtimestamp(timestamp, /)
isocalendar(self, /)
isoformat(self, /)
```

## datetime.datetime: the year, month and day arguments are required

The year, month and day arguments are required. tzinfo may be None, or an instance of a tzinfo subclass. The remaining arguments may be ints.

```python
datetime.datetime(year, month, day[, hour[, minute[, second[, microsecond[,tzinfo]]]]])
```

Key methods are ctime, date, dst, isocalendar, isoweekday, now, time, timestamp.

```python
ctime(self, /)
date(self, /)
dst(self, /)
isocalendar(self, /)
isoweekday(self, /)
now(tz=None)
time(self, /)
timestamp(self, /)
```

## datetime.time: all arguments are optional

All arguments are optional. tzinfo may be None, or an instance of a tzinfo subclass. The remaining arguments may be ints.

```python
datetime.time([hour[, minute[, second[, microsecond[, tzinfo]]]]]) -
```

Key methods are dst, tzname, utcoffset.

```python
dst(self, /)
tzname(self, /)
utcoffset(self, /)
```

## datetime.timedelta: difference between two datetime values

Difference between two datetime values. timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0) All arguments are optional and default to 0. Arguments may be integers or floats, and may be positive or negative.

Key methods are total_seconds.

```python
total_seconds(self, /)
```

## datetime.timezone: fixed offset from utc implementation of tzinfo

Fixed offset from UTC implementation of tzinfo.

Key methods are dst, fromutc, tzname, utcoffset.

```python
dst(self, object, /)
fromutc(self, object, /)
tzname(self, object, /)
utcoffset(self, object, /)
```

## datetime.tzinfo: abstract base class for time zone info objects

Abstract base class for time zone info objects.

Key methods are dst, fromutc, tzname, utcoffset.

```python
dst(self, object, /)
fromutc(self, object, /)
tzname(self, object, /)
utcoffset(self, object, /)
```

## time.asctime: convert a time tuple to a string, e

Convert a time tuple to a string, e.g. 'Sat Jun 06 16:26:11 1998'.

```python
time.asctime([tuple])
```

## time.ctime: convert a time in seconds since the epoch to a string in local time

Convert a time in seconds since the Epoch to a string in local time. This is equivalent to asctime(localtime(seconds)).

```python
time.ctime(seconds)
```

## time.get_clock_info: get information of the specified clock

Get information of the specified clock.

```python
time.get_clock_info(name: str)
```

## time.gmtime: tm_sec, tm_wday, tm_yday, tm_isdst) convert seconds since the epoch to a time tuple expressing utc (a

tm_sec, tm_wday, tm_yday, tm_isdst) Convert seconds since the Epoch to a time tuple expressing UTC (a.k.a. GMT).

```python
time.gmtime([seconds])
```

## time.localtime: tm_sec,tm_wday,tm_yday,tm_isdst) convert seconds since the epoch to a time tuple expressing local time

tm_sec,tm_wday,tm_yday,tm_isdst) Convert seconds since the Epoch to a time tuple expressing local time. When 'seconds' is not passed in, convert the current time instead.

```python
time.localtime([seconds])
```

## time.mktime: convert a time tuple in local time to seconds since the epoch

Convert a time tuple in local time to seconds since the Epoch. Note that mktime(gmtime(0)) will not generally return zero for most time zones; instead the returned value will either be equal to that of the timezone or altzone attributes on the time module.

```python
time.mktime(tuple)
```

## time.monotonic: monotonic clock, cannot go backward

Monotonic clock, cannot go backward.

```python
time.monotonic()
```

## time.monotonic_ns: monotonic clock, cannot go backward, as nanoseconds

Monotonic clock, cannot go backward, as nanoseconds.

```python
time.monotonic_ns()
```

## time.perf_counter: performance counter for benchmarking

Performance counter for benchmarking.

```python
time.perf_counter()
```

## time.perf_counter_ns: performance counter for benchmarking as nanoseconds

Performance counter for benchmarking as nanoseconds.

```python
time.perf_counter_ns()
```

## time.process_time: process time for profiling: sum of the kernel and user-space cpu time

Process time for profiling: sum of the kernel and user-space CPU time.

```python
time.process_time()
```

## time.process_time_ns: process_time() -> int process time for profiling as nanoseconds: sum of the kernel and user-space cpu time

process_time() -> int Process time for profiling as nanoseconds: sum of the kernel and user-space CPU time.

```python
time.process_time_ns()
```

## time.sleep: delay execution for a given number of seconds

Delay execution for a given number of seconds. The argument may be a floating-point number for subsecond precision.

```python
time.sleep(seconds)
```

## time.strftime: convert a time tuple to a string according to a format specification

Convert a time tuple to a string according to a format specification. See the library reference manual for formatting codes.

```python
time.strftime(format[, tuple])
```

## time.strptime: parse a string to a time tuple according to a format specification

Parse a string to a time tuple according to a format specification. See the library reference manual for formatting codes (same as strftime()).

```python
time.strptime(string, format)
```

## time.struct_time: the time value as returned by gmtime(), localtime(), and strptime(), and accepted by asctime(), mktime() and strftime()

The time value as returned by gmtime(), localtime(), and strptime(), and accepted by asctime(), mktime() and strftime(). May be considered as a sequence of 9 integers. Note that several fields' values are not the same as those defined by the C language standard for struct tm. For example, the value of the field tm_year is the actual year, not year - 1900.

```python
time.struct_time(iterable=(), /)
```

Key methods are count, index.

```python
count(self, value, /)
index(self, value, start=0, stop=9223372036854775807, /)
```

## time.thread_time: thread time for profiling: sum of the kernel and user-space cpu time

Thread time for profiling: sum of the kernel and user-space CPU time.

```python
time.thread_time()
```

## time.thread_time_ns: thread_time() -> int thread time for profiling as nanoseconds: sum of the kernel and user-space cpu time

thread_time() -> int Thread time for profiling as nanoseconds: sum of the kernel and user-space CPU time.

```python
time.thread_time_ns()
```

## time.time: return the current time in seconds since the epoch

Return the current time in seconds since the Epoch. Fractions of a second may be present if the system clock provides them.

```python
time.time()
```

## time.time_ns: return the current time in nanoseconds since the epoch

Return the current time in nanoseconds since the Epoch.

```python
time.time_ns()
```

## collections.ChainMap: a chainmap groups multiple dicts (or other mappings) together to create a single, updateable view

A ChainMap groups multiple dicts (or other mappings) together to create a single, updateable view. The underlying mappings are stored in a list. That list is public and can be accessed or updated using the *maps* attribute. There is no other state.

```python
collections.ChainMap(*maps)
```

Key methods are clear, copy, fromkeys, get, items, keys, new_child, pop, popitem, setdefault, update, values.

```python
clear(self)
copy(self)
fromkeys(iterable, value=None, /)
get(self, key, default=None)
items(self)
keys(self)
new_child(self, m=None, **kwargs)
pop(self, key, *args)
popitem(self)
setdefault(self, key, default=None)
update(self, other=(), /, **kwds)
values(self)
```

## collections.Counter: dict subclass for counting hashable items

Dict subclass for counting hashable items. Sometimes called a bag or multiset. Elements are stored as dictionary keys and their counts are stored as dictionary values. >>> c = Counter('abcdeabcdabcaba')  # count elements from a string >>> c.most_common(3)                # three most common elements [('a', 5), ('b', 4), ('c', 3)] >>> sorted(c)                       # list all unique elements ['a', 'b', 'c', 'd', 'e'] >>> ''.join(sorted(c.elements()))   # list elements with repetitions 'aaaaabbbbcccdde' >>> sum(c.values())                 # total of all counts 15 >>> c['a']                          # count of letter 'a' 5 >>> for elem in 'shazam':           # update counts from an iterable .

```python
collections.Counter(iterable=None, /, **kwds)
```

Key methods are clear, copy, elements, fromkeys, get, items, keys, most_common, popitem, setdefault, subtract, total, update, values.

```python
clear(self, /)
copy(self)
elements(self)
fromkeys(iterable, v=None)
get(self, key, default=None, /)
items(self, /)
keys(self, /)
most_common(self, n=None)
popitem(self, /)
setdefault(self, key, default=None, /)
subtract(self, iterable=None, /, **kwds)
total(self)
update(self, iterable=None, /, **kwds)
values(self, /)
```

## collections.OrderedDict: dictionary that remembers insertion order

Dictionary that remembers insertion order.

Key methods are clear, copy, fromkeys, get, items.

```python
clear(self, /)
copy(self, /)
fromkeys(iterable, value=None)
get(self, key, default=None, /)
items(self, /)
```

## collections.UserDict: a mutablemapping is a generic container for associating key/value pairs

A MutableMapping is a generic container for associating key/value pairs. This class provides concrete generic implementations of all methods except for __getitem__, __setitem__, __delitem__, __iter__, and __len__.

```python
collections.UserDict(dict=None, /, **kwargs)
```

Key methods are clear, copy, fromkeys, get, items, keys, pop, popitem, setdefault, update, values.

```python
clear(self)
copy(self)
fromkeys(iterable, value=None)
get(self, key, default=None)
items(self)
keys(self)
pop(self, key, default=<object object at 0x0000027F95F301D0>)
popitem(self)
setdefault(self, key, default=None)
update(self, other=(), /, **kwds)
values(self)
```

## collections.UserList: a more or less complete user-defined wrapper around list objects

A more or less complete user-defined wrapper around list objects.

```python
collections.UserList(initlist=None)
```

Key methods are append, clear, copy, count, extend, index, insert, pop.

```python
append(self, item)
clear(self)
copy(self)
count(self, item)
extend(self, other)
index(self, item, *args)
insert(self, i, item)
pop(self, i=-1)
```

## collections.UserString: all the operations on a read-only sequence

All the operations on a read-only sequence. Concrete subclasses must override __new__ or __init__, __getitem__, and __len__.

```python
collections.UserString(seq)
```

Key methods are capitalize, casefold, center, count, encode.

```python
capitalize(self)
casefold(self)
center(self, width, *args)
count(self, sub, start=0, end=9223372036854775807)
encode(self, encoding='utf-8', errors='strict')
```

## collections.defaultdict: the default factory is called without arguments to produce a new value when a key is not present, in __getitem__ only

The default factory is called without arguments to produce a new value when a key is not present, in __getitem__ only. A defaultdict compares equal to a dict with the same items. All remaining arguments are treated the same as if they were passed to the dict constructor, including keyword arguments.

```python
collections.defaultdict(default_factory=None, /, [...]) -
```

Key methods are clear, copy, fromkeys, get, items, keys, popitem, setdefault, values.

```python
clear(self, /)
copy(self, /)
fromkeys(iterable, value=None, /)
get(self, key, default=None, /)
items(self, /)
keys(self, /)
popitem(self, /)
setdefault(self, key, default=None, /)
values(self, /)
```

## collections.deque: a list-like sequence optimized for data accesses near its endpoints

A list-like sequence optimized for data accesses near its endpoints.

Key methods are append, appendleft, clear, copy, count, extend, extendleft.

```python
append(self, item, /)
appendleft(self, item, /)
clear(self, /)
copy(self, /)
count(self, value, /)
extend(self, iterable, /)
extendleft(self, iterable, /)
```

## collections.namedtuple: returns a new subclass of tuple with named fields

Returns a new subclass of tuple with named fields. >>> Point = namedtuple('Point', ['x', 'y']) >>> Point.__doc__                   # docstring for the new class 'Point(x, y)' >>> p = Point(11, y=22)             # instantiate with positional args or keywords >>> p[0] + p[1]                     # indexable like a plain tuple 33 >>> x, y = p                        # unpack like a regular tuple >>> x, y (11, 22) >>> p.x + p.y                       # fields also accessible by name 33 >>> d = p._asdict()                 # convert to a dictionary >>> d['x'] 11 >>> Point(**d)                      # convert from a dictionary Point(x=11, y=22) >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields Point(x=100, y=22).

```python
collections.namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)
```

## itertools.accumulate: return series of accumulated sums (or other binary function results)

Return series of accumulated sums (or other binary function results).

```python
itertools.accumulate(iterable, func=None, *, initial=None)
```

## itertools.batched: batch data into tuples of length n

Batch data into tuples of length n. The last batch may be shorter than n. Loops over the input iterable and accumulates data into tuples up to size n. The input is consumed lazily, just enough to fill a batch.

```python
itertools.batched(iterable, n, *, strict=False)
```

## itertools.chain: return a chain object whose 

Return a chain object whose .__next__() method returns elements from the first iterable until it is exhausted, then elements from the next iterable, until all of the iterables are exhausted.

```python
itertools.chain(*iterables)
```

Key methods are from_iterable.

```python
from_iterable(iterable, /)
```

## itertools.combinations: return successive r-length combinations of elements in the iterable

Return successive r-length combinations of elements in the iterable. combinations(range(4), 3) --> (0,1,2), (0,1,3), (0,2,3), (1,2,3).

```python
itertools.combinations(iterable, r)
```

## itertools.combinations_with_replacement: return successive r-length combinations of elements in the iterable allowing individual elements to have successive repeats

Return successive r-length combinations of elements in the iterable allowing individual elements to have successive repeats. combinations_with_replacement('ABC', 2) --> ('A','A'), ('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C').

```python
itertools.combinations_with_replacement(iterable, r)
```

## itertools.compress: return data elements corresponding to true selector elements

Return data elements corresponding to true selector elements. Forms a shorter iterator from selected data elements using the selectors to choose the data elements.

```python
itertools.compress(data, selectors)
```

## itertools.count: return a count object whose 

Return a count object whose .__next__() method returns consecutive values. Equivalent to: def count(firstval=0, step=1): x = firstval while 1: yield x x += step.

```python
itertools.count(start=0, step=1)
```

## itertools.cycle: return elements from the iterable until it is exhausted

Return elements from the iterable until it is exhausted. Then repeat the sequence indefinitely.

```python
itertools.cycle(iterable, /)
```

## itertools.dropwhile: drop items from the iterable while predicate(item) is true

Drop items from the iterable while predicate(item) is true. Afterwards, return every element until the iterable is exhausted.

```python
itertools.dropwhile(predicate, iterable, /)
```

## itertools.filterfalse: return those items of iterable for which function(item) is false

Return those items of iterable for which function(item) is false. If function is None, return the items that are false.

```python
itertools.filterfalse(function, iterable, /)
```

## itertools.groupby: make an iterator that returns consecutive keys and groups from the iterable iterable elements to divide into groups according to the key function

make an iterator that returns consecutive keys and groups from the iterable iterable Elements to divide into groups according to the key function. key A function for computing the group category for each element. If the key function is not specified or is None, the element itself is used for grouping.

```python
itertools.groupby(iterable, key=None)
```

## itertools.islice: islice(iterable, start, stop[, step]) --> islice object return an iterator whose next() method returns selected values from an iterable

islice(iterable, start, stop[, step]) --> islice object Return an iterator whose next() method returns selected values from an iterable. If start is specified, will skip all preceding elements; otherwise, start defaults to zero. Step defaults to one. If specified as another value, step determines how many values are skipped between successive calls.

```python
itertools.islice(iterable, stop) -
```

## itertools.pairwise: return an iterator of overlapping pairs taken from the input iterator

Return an iterator of overlapping pairs taken from the input iterator. s -> (s0,s1), (s1,s2), (s2, s3), .

```python
itertools.pairwise(iterable, /)
```

## itertools.permutations: return successive r-length permutations of elements in the iterable

Return successive r-length permutations of elements in the iterable. permutations(range(3), 2) --> (0,1), (0,2), (1,0), (1,2), (2,0), (2,1).

```python
itertools.permutations(iterable, r=None)
```

## itertools.product: cartesian product of input iterables

Cartesian product of input iterables. Equivalent to nested for-loops. For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B). The leftmost iterators are in the outermost for-loop, so the output tuples cycle in a manner similar to an odometer (with the rightmost element changing on every iteration).

```python
itertools.product(*iterables, repeat=1)
```

## itertools.repeat: for the specified number of times

for the specified number of times. If not specified, returns the object endlessly.

```python
itertools.repeat(object [,times])
```

## itertools.starmap: return an iterator whose values are returned from the function evaluated with an argument tuple taken from the given sequence

Return an iterator whose values are returned from the function evaluated with an argument tuple taken from the given sequence.

```python
itertools.starmap(function, iterable, /)
```

## itertools.takewhile: return successive entries from an iterable as long as the predicate evaluates to true for each entry

Return successive entries from an iterable as long as the predicate evaluates to true for each entry.

```python
itertools.takewhile(predicate, iterable, /)
```

## itertools.tee: returns a tuple of n independent iterators

Returns a tuple of n independent iterators.

```python
itertools.tee(iterable, n=2, /)
```

## itertools.zip_longest: return a zip_longest object whose 

Return a zip_longest object whose .__next__() method returns a tuple where the i-th element comes from the i-th iterable argument. The .__next__() method continues until the longest iterable in the argument sequence is exhausted and then it raises StopIteration. When the shorter iterables are exhausted, the fillvalue is substituted in their place. The fillvalue defaults to None or can be specified by a keyword argument.

```python
itertools.zip_longest(*iterables, fillvalue=None)
```

## functools.update_wrapper: update a wrapper function to look like the wrapped function wrapper is the function to be updated wrapped is the original function assigned is a tuple naming the attributes assigned directly from the wrapped function to the wrapper function (defaults to functools

Update a wrapper function to look like the wrapped function wrapper is the function to be updated wrapped is the original function assigned is a tuple naming the attributes assigned directly from the wrapped function to the wrapper function (defaults to functools.WRAPPER_ASSIGNMENTS) updated is a tuple naming the attributes of the wrapper that are updated with the corresponding attribute from the wrapped function (defaults to functools.WRAPPER_UPDATES).

```python
functools.update_wrapper(wrapper, wrapped, assigned=('__module__', '__name__', '__qualname__', '__doc__', '__annotations__', '__type_params__'), updated=('__dict__',))
```

## functools.wraps: decorator factory to apply update_wrapper() to a wrapper function returns a decorator that invokes update_wrapper() with the decorated function as the wrapper argument and the arguments to wraps() as the remaining arguments

Decorator factory to apply update_wrapper() to a wrapper function Returns a decorator that invokes update_wrapper() with the decorated function as the wrapper argument and the arguments to wraps() as the remaining arguments. Default arguments are as for update_wrapper().

```python
functools.wraps(wrapped, assigned=('__module__', '__name__', '__qualname__', '__doc__', '__annotations__', '__type_params__'), updated=('__dict__',))
```

## functools.total_ordering: class decorator that fills in missing ordering methods

Class decorator that fills in missing ordering methods.

```python
functools.total_ordering(cls)
```

## functools.cache: simple lightweight unbounded cache

Simple lightweight unbounded cache. Sometimes called "memoize".

```python
functools.cache(user_function, /)
```

## functools.lru_cache: least-recently-used cache decorator

Least-recently-used cache decorator. If *maxsize* is set to None, the LRU features are disabled and the cache can grow without bound.

```python
functools.lru_cache(maxsize=128, typed=False)
```

## functools.partial: create a new function with partial application of the given arguments and keywords

Create a new function with partial application of the given arguments and keywords.

```python
functools.partial(func, /, *args, **keywords)
```

## functools.partialmethod: method descriptor with partial application of the given arguments and keywords

Method descriptor with partial application of the given arguments and keywords. Supports wrapping existing descriptors and handles non-descriptor callables as instance methods.

```python
functools.partialmethod(func, /, *args, **keywords)
```

## functools.singledispatch: single-dispatch generic function decorator

Single-dispatch generic function decorator. Transforms a function into a generic function, which can have different behaviours depending upon the type of its first argument.

```python
functools.singledispatch(func)
```

## functools.singledispatchmethod: single-dispatch generic method descriptor

Single-dispatch generic method descriptor. Supports wrapping existing descriptors and handles non-descriptor callables as instance methods.

```python
functools.singledispatchmethod(func)
```

Key methods are register.

```python
register(self, cls, method=None)
```

## functools.cached_property


```python
functools.cached_property(func)
```

## dataclasses.dataclass: add dunder methods based on the fields defined in the class

Add dunder methods based on the fields defined in the class. Examines PEP 526 __annotations__ to determine fields.

**Hazard:** If unsafe_hash is true, a __hash__() method is added.

```python
dataclasses.dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)
```

## dataclasses.field: return an object to identify dataclass fields

Return an object to identify dataclass fields. default is the default value of the field.

```python
dataclasses.field(*, default=<dataclasses._MISSING_TYPE object at 0x0000027F967F0050>, default_factory=<dataclasses._MISSING_TYPE object at 0x0000027F967F0050>, init=True, repr=True, hash=None, compare=True, metadata=None, kw_only=<dataclasses._MISSING_TYPE object at 0x0000027F967F0050>)
```

## dataclasses.Field


```python
dataclasses.Field(default, default_factory, init, repr, hash, compare, metadata, kw_only)
```

## dataclasses.FrozenInstanceError: attribute not found

Attribute not found.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## dataclasses.InitVar


```python
dataclasses.InitVar(type)
```

## dataclasses.fields: return a tuple describing the fields of this dataclass

Return a tuple describing the fields of this dataclass. Accepts a dataclass or an instance of one.

```python
dataclasses.fields(class_or_instance)
```

## dataclasses.asdict: return the fields of a dataclass instance as a new dictionary mapping field names to field values

Return the fields of a dataclass instance as a new dictionary mapping field names to field values. Example usage:: @dataclass class C: x: int y: int c = C(1, 2) assert asdict(c) == {'x': 1, 'y': 2} If given, 'dict_factory' will be used instead of built-in dict.

```python
dataclasses.asdict(obj, *, dict_factory=<class 'dict'>)
```

## dataclasses.astuple: return the fields of a dataclass instance as a new tuple of field values

Return the fields of a dataclass instance as a new tuple of field values. Example usage:: @dataclass class C: x: int y: int c = C(1, 2) assert astuple(c) == (1, 2) If given, 'tuple_factory' will be used instead of built-in tuple.

```python
dataclasses.astuple(obj, *, tuple_factory=<class 'tuple'>)
```

## dataclasses.make_dataclass: return a new dynamically created dataclass

Return a new dynamically created dataclass. The dataclass name will be 'cls_name'.

**Hazard:** The parameters init, repr, eq, order, unsafe_hash, frozen, match_args, kw_only, slots, and weakref_slot are passed to dataclass().

```python
dataclasses.make_dataclass(cls_name, fields, *, bases=(), namespace=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False, module=None)
```

## dataclasses.replace: return a new object replacing specified fields with new values

Return a new object replacing specified fields with new values. This is especially useful for frozen classes.

```python
dataclasses.replace(obj, /, **changes)
```

## dataclasses.is_dataclass: returns true if obj is a dataclass or an instance of a dataclass

Returns True if obj is a dataclass or an instance of a dataclass.

```python
dataclasses.is_dataclass(obj)
```

## argparse.ArgumentParser: object for parsing command line strings into python objects

Object for parsing command line strings into Python objects. Keyword Arguments: - prog -- The name of the program (default: ``os.path.basename(sys.argv[0])``) - usage -- A usage message (default: auto-generated from arguments) - description -- A description of what the program does - epilog -- Text following the argument descriptions - parents -- Parsers whose arguments should be copied into this one - formatter_class -- HelpFormatter class for printing help messages - prefix_chars -- Characters that prefix optional arguments - fromfile_prefix_chars -- Characters that prefix files containing additional arguments - argument_default -- The default value for all arguments - conflict_handler -- String indicating how to handle conflicts - add_help -- Add a -h/-help option - allow_abbrev -- Allow long options to be abbreviated unambiguously - exit_on_error -- Determines whether or not ArgumentParser exits with error info when an error occurs.

```python
argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=<class 'argparse.HelpFormatter'>, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
```

Key methods are add_argument, add_argument_group, add_mutually_exclusive_group, add_subparsers, convert_arg_line_to_args, error, exit, format_help, format_usage, get_default, parse_args, parse_intermixed_args, parse_known_args, parse_known_intermixed_args, print_help, print_usage, register, set_defaults.

```python
add_argument(self, *args, **kwargs)
add_argument_group(self, *args, **kwargs)
add_mutually_exclusive_group(self, **kwargs)
add_subparsers(self, **kwargs)
convert_arg_line_to_args(self, arg_line)
error(self, message)
exit(self, status=0, message=None)
format_help(self)
format_usage(self)
get_default(self, dest)
parse_args(self, args=None, namespace=None)
parse_intermixed_args(self, args=None, namespace=None)
parse_known_args(self, args=None, namespace=None)
parse_known_intermixed_args(self, args=None, namespace=None)
print_help(self, file=None)
print_usage(self, file=None)
register(self, registry_name, value, object)
set_defaults(self, **kwargs)
```

## argparse.ArgumentError: an error from creating or using an argument (optional or positional)

An error from creating or using an argument (optional or positional). The string value of this exception is the message, augmented with information about the argument that caused it.

```python
argparse.ArgumentError(argument, message)
```

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## argparse.ArgumentTypeError: an error from trying to convert a command line string to a type

An error from trying to convert a command line string to a type.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## argparse.BooleanOptionalAction: information about how to convert command line strings to python objects

Information about how to convert command line strings to Python objects. Action objects are used by an ArgumentParser to represent the information needed to parse a single argument from one or more strings from the command line. The keyword arguments to the Action constructor are also all attributes of Action instances. Keyword Arguments: - option_strings -- A list of command-line option strings which should be associated with this action.

```python
argparse.BooleanOptionalAction(option_strings, dest, default=None, type=<object object at 0x0000027F95F305F0>, choices=<object object at 0x0000027F95F305F0>, required=False, help=None, metavar=<object object at 0x0000027F95F305F0>, deprecated=False)
```

Key methods are format_usage.

```python
format_usage(self)
```

## argparse.FileType: factory for creating file object types instances of filetype are typically passed as type= arguments to the argumentparser add_argument() method

Factory for creating file object types Instances of FileType are typically passed as type= arguments to the ArgumentParser add_argument() method. Keyword Arguments: - mode -- A string indicating how the file is to be opened. Accepts the same values as the builtin open() function. - bufsize -- The file's desired buffer size.

```python
argparse.FileType(mode='r', bufsize=-1, encoding=None, errors=None)
```

## argparse.HelpFormatter: formatter for generating usage messages and argument help strings

Formatter for generating usage messages and argument help strings. Only the name of this class is considered a public API. All the methods provided by the class are considered an implementation detail.

```python
argparse.HelpFormatter(prog, indent_increment=2, max_help_position=24, width=None)
```

Key methods are add_argument, add_arguments, add_text, add_usage, end_section, format_help, start_section.

```python
add_argument(self, action)
add_arguments(self, actions)
add_text(self, text)
add_usage(self, usage, actions, groups, prefix=None)
end_section(self)
format_help(self)
start_section(self, heading)
```

## argparse.ArgumentDefaultsHelpFormatter: help message formatter which adds default values to argument help

Help message formatter which adds default values to argument help. Only the name of this class is considered a public API. All the methods provided by the class are considered an implementation detail.

```python
argparse.ArgumentDefaultsHelpFormatter(prog, indent_increment=2, max_help_position=24, width=None)
```

Key methods are add_argument, add_arguments, add_text, add_usage, end_section, format_help, start_section.

```python
add_argument(self, action)
add_arguments(self, actions)
add_text(self, text)
add_usage(self, usage, actions, groups, prefix=None)
end_section(self)
format_help(self)
start_section(self, heading)
```

## argparse.RawDescriptionHelpFormatter: help message formatter which retains any formatting in descriptions

Help message formatter which retains any formatting in descriptions. Only the name of this class is considered a public API. All the methods provided by the class are considered an implementation detail.

```python
argparse.RawDescriptionHelpFormatter(prog, indent_increment=2, max_help_position=24, width=None)
```

Key methods are add_argument, add_arguments, add_text, add_usage, end_section, format_help, start_section.

```python
add_argument(self, action)
add_arguments(self, actions)
add_text(self, text)
add_usage(self, usage, actions, groups, prefix=None)
end_section(self)
format_help(self)
start_section(self, heading)
```

## argparse.RawTextHelpFormatter: help message formatter which retains formatting of all help text

Help message formatter which retains formatting of all help text. Only the name of this class is considered a public API. All the methods provided by the class are considered an implementation detail.

```python
argparse.RawTextHelpFormatter(prog, indent_increment=2, max_help_position=24, width=None)
```

Key methods are add_argument, add_arguments, add_text, add_usage, end_section, format_help, start_section.

```python
add_argument(self, action)
add_arguments(self, actions)
add_text(self, text)
add_usage(self, usage, actions, groups, prefix=None)
end_section(self)
format_help(self)
start_section(self, heading)
```

## argparse.MetavarTypeHelpFormatter: help message formatter which uses the argument 'type' as the default metavar value (instead of the argument 'dest') only the name of this class is considered a public api

Help message formatter which uses the argument 'type' as the default metavar value (instead of the argument 'dest') Only the name of this class is considered a public API. All the methods provided by the class are considered an implementation detail.

```python
argparse.MetavarTypeHelpFormatter(prog, indent_increment=2, max_help_position=24, width=None)
```

Key methods are add_argument, add_arguments, add_text, add_usage, end_section, format_help, start_section.

```python
add_argument(self, action)
add_arguments(self, actions)
add_text(self, text)
add_usage(self, usage, actions, groups, prefix=None)
end_section(self)
format_help(self)
start_section(self, heading)
```

## argparse.Namespace: simple object for storing attributes

Simple object for storing attributes. Implements equality by attribute names and values, and provides a simple string representation.

```python
argparse.Namespace(**kwargs)
```

## argparse.Action: information about how to convert command line strings to python objects

Information about how to convert command line strings to Python objects. Action objects are used by an ArgumentParser to represent the information needed to parse a single argument from one or more strings from the command line. The keyword arguments to the Action constructor are also all attributes of Action instances. Keyword Arguments: - option_strings -- A list of command-line option strings which should be associated with this action.

```python
argparse.Action(option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, deprecated=False)
```

Key methods are format_usage.

```python
format_usage(self)
```

## urllib.request.Request


```python
urllib.request.Request(url, data=None, headers={}, origin_req_host=None, unverifiable=False, method=None)
```

Key methods are add_header, add_unredirected_header, get_full_url, get_header.

```python
add_header(self, key, val)
add_unredirected_header(self, key, val)
get_full_url(self)
get_header(self, header_name, default=None)
```

## urllib.request.OpenerDirector


```python
urllib.request.OpenerDirector()
```

Key methods are add_handler, close, error, open.

```python
add_handler(self, handler)
close(self)
error(self, proto, *args)
open(self, fullurl, data=None, timeout=<object object at 0x0000027F95F30A60>)
```

## urllib.request.BaseHandler


```python
urllib.request.BaseHandler()
```

Key methods are add_parent, close.

```python
add_parent(self, parent)
close(self)
```

## urllib.request.HTTPDefaultErrorHandler


```python
urllib.request.HTTPDefaultErrorHandler()
```

Key methods are add_parent, close, http_error_default.

```python
add_parent(self, parent)
close(self)
http_error_default(self, req, fp, code, msg, hdrs)
```

## urllib.request.HTTPRedirectHandler


```python
urllib.request.HTTPRedirectHandler()
```

Key methods are add_parent, close, http_error_301, http_error_302.

```python
add_parent(self, parent)
close(self)
http_error_301(self, req, fp, code, msg, headers)
http_error_302(self, req, fp, code, msg, headers)
```

## urllib.request.HTTPCookieProcessor


```python
urllib.request.HTTPCookieProcessor(cookiejar=None)
```

Key methods are add_parent, close, http_request, http_response.

```python
add_parent(self, parent)
close(self)
http_request(self, request)
http_response(self, request, response)
```

## urllib.request.ProxyHandler


```python
urllib.request.ProxyHandler(proxies=None)
```

Key methods are add_parent, close, proxy_open.

```python
add_parent(self, parent)
close(self)
proxy_open(self, req, proxy, type)
```

## urllib.request.HTTPPasswordMgr


```python
urllib.request.HTTPPasswordMgr()
```

Key methods are add_password, find_user_password, is_suburi, reduce_uri.

```python
add_password(self, realm, uri, user, passwd)
find_user_password(self, realm, authuri)
is_suburi(self, base, test)
reduce_uri(self, uri, default_port=True)
```

## urllib.request.HTTPPasswordMgrWithDefaultRealm


```python
urllib.request.HTTPPasswordMgrWithDefaultRealm()
```

Key methods are add_password, find_user_password, is_suburi, reduce_uri.

```python
add_password(self, realm, uri, user, passwd)
find_user_password(self, realm, authuri)
is_suburi(self, base, test)
reduce_uri(self, uri, default_port=True)
```

## urllib.request.HTTPPasswordMgrWithPriorAuth


```python
urllib.request.HTTPPasswordMgrWithPriorAuth()
```

Key methods are add_password, find_user_password, is_authenticated, is_suburi.

```python
add_password(self, realm, uri, user, passwd, is_authenticated=False)
find_user_password(self, realm, authuri)
is_authenticated(self, authuri)
is_suburi(self, base, test)
```

## urllib.request.AbstractBasicAuthHandler


```python
urllib.request.AbstractBasicAuthHandler(password_mgr=None)
```

Key methods are http_error_auth_reqed, http_request, http_response, https_request.

```python
http_error_auth_reqed(self, authreq, host, req, headers)
http_request(self, req)
http_response(self, req, response)
https_request(self, req)
```

## urllib.request.HTTPBasicAuthHandler


```python
urllib.request.HTTPBasicAuthHandler(password_mgr=None)
```

Key methods are add_parent, close, http_error_401, http_error_auth_reqed.

```python
add_parent(self, parent)
close(self)
http_error_401(self, req, fp, code, msg, headers)
http_error_auth_reqed(self, authreq, host, req, headers)
```

## urllib.request.ProxyBasicAuthHandler


```python
urllib.request.ProxyBasicAuthHandler(password_mgr=None)
```

Key methods are add_parent, close, http_error_407, http_error_auth_reqed.

```python
add_parent(self, parent)
close(self)
http_error_407(self, req, fp, code, msg, headers)
http_error_auth_reqed(self, authreq, host, req, headers)
```

## urllib.request.AbstractDigestAuthHandler


```python
urllib.request.AbstractDigestAuthHandler(passwd=None)
```

Key methods are get_algorithm_impls, get_authorization, get_cnonce, get_entity_digest.

```python
get_algorithm_impls(self, algorithm)
get_authorization(self, req, chal)
get_cnonce(self, nonce)
get_entity_digest(self, data, chal)
```

## urllib.request.HTTPDigestAuthHandler: an authentication protocol defined by rfc 2069 digest authentication improves on basic authentication because it does not transmit passwords in the clear

An authentication protocol defined by RFC 2069 Digest authentication improves on basic authentication because it does not transmit passwords in the clear.

```python
urllib.request.HTTPDigestAuthHandler(passwd=None)
```

Key methods are add_parent, close, get_algorithm_impls, get_authorization, get_cnonce, get_entity_digest, http_error_401, http_error_auth_reqed, reset_retry_count.

```python
add_parent(self, parent)
close(self)
get_algorithm_impls(self, algorithm)
get_authorization(self, req, chal)
get_cnonce(self, nonce)
get_entity_digest(self, data, chal)
http_error_401(self, req, fp, code, msg, headers)
http_error_auth_reqed(self, auth_header, host, req, headers)
reset_retry_count(self)
```

## urllib.request.ProxyDigestAuthHandler


```python
urllib.request.ProxyDigestAuthHandler(passwd=None)
```

Key methods are add_parent, close, get_algorithm_impls, get_authorization.

```python
add_parent(self, parent)
close(self)
get_algorithm_impls(self, algorithm)
get_authorization(self, req, chal)
```

## urllib.request.HTTPHandler


```python
urllib.request.HTTPHandler(debuglevel=None)
```

Key methods are add_parent, close, do_open, do_request_.

```python
add_parent(self, parent)
close(self)
do_open(self, http_class, req, **http_conn_args)
do_request_(self, request)
```

## urllib.request.FileHandler


```python
urllib.request.FileHandler()
```

Key methods are add_parent, close, file_open, get_names.

```python
add_parent(self, parent)
close(self)
file_open(self, req)
get_names(self)
```

## urllib.request.FTPHandler


```python
urllib.request.FTPHandler()
```

Key methods are add_parent, close, connect_ftp, ftp_open.

```python
add_parent(self, parent)
close(self)
connect_ftp(self, user, passwd, host, port, dirs, timeout)
ftp_open(self, req)
```

## urllib.request.CacheFTPHandler


```python
urllib.request.CacheFTPHandler()
```

Key methods are add_parent, check_cache, clear_cache, close.

```python
add_parent(self, parent)
check_cache(self)
clear_cache(self)
close(self)
```

## urllib.request.DataHandler


```python
urllib.request.DataHandler()
```

Key methods are add_parent, close, data_open.

```python
add_parent(self, parent)
close(self)
data_open(self, req)
```

## urllib.request.UnknownHandler


```python
urllib.request.UnknownHandler()
```

Key methods are add_parent, close, unknown_open.

```python
add_parent(self, parent)
close(self)
unknown_open(self, req)
```

## urllib.request.HTTPErrorProcessor: process http error responses

Process HTTP error responses.

```python
urllib.request.HTTPErrorProcessor()
```

Key methods are add_parent, close, http_response, https_response.

```python
add_parent(self, parent)
close(self)
http_response(self, request, response)
https_response(self, request, response)
```

## urllib.request.urlopen: open the url url, which can be either a string or a request object

Open the URL url, which can be either a string or a Request object. *data* must be an object specifying additional data to be sent to the server, or None if no such data is needed.

```python
urllib.request.urlopen(url, data=None, timeout=<object object at 0x0000027F95F30A60>, *, context=None)
```

## urllib.request.install_opener


```python
urllib.request.install_opener(opener)
```

## urllib.request.build_opener: create an opener object from a list of handlers

Create an opener object from a list of handlers. The opener will use several default handlers, including support for HTTP, FTP and when applicable HTTPS.

```python
urllib.request.build_opener(*handlers)
```

## urllib.request.getproxies: return a dictionary of scheme -> proxy server url mappings

Return a dictionary of scheme -> proxy server URL mappings. Returns settings gathered from the environment, if specified, or the registry.

```python
urllib.request.getproxies()
```

## urllib.request.urlretrieve: retrieve a url into a temporary location on disk

Retrieve a URL into a temporary location on disk. Requires a URL argument.

```python
urllib.request.urlretrieve(url, filename=None, reporthook=None, data=None)
```

## urllib.request.urlcleanup: clean up temporary files from urlretrieve calls

Clean up temporary files from urlretrieve calls.

```python
urllib.request.urlcleanup()
```

## urllib.request.URLopener: class to open urls

Class to open URLs. This is a class rather than just a subroutine because we may need more than one set of global protocol-specific options. Note -- this is a base class for those who don't want the automatic handling of errors type 302 (relocated) and 401 (authorization needed).

```python
urllib.request.URLopener(proxies=None, **x509)
```

Key methods are addheader, cleanup, close, http_error, http_error_default, open, open_data, open_file, open_ftp, open_http.

```python
addheader(self, *args)
cleanup(self)
close(self)
http_error(self, url, fp, errcode, errmsg, headers, data=None)
http_error_default(self, url, fp, errcode, errmsg, headers)
open(self, fullurl, data=None)
open_data(self, url, data=None)
open_file(self, url)
open_ftp(self, url)
open_http(self, url, data=None)
```

## urllib.request.FancyURLopener: derived class with handlers for errors we can handle (perhaps)

Derived class with handlers for errors we can handle (perhaps).

Key methods are addheader, cleanup, close, get_user_passwd.

```python
addheader(self, *args)
cleanup(self)
close(self)
get_user_passwd(self, host, realm, clear_cache=0)
```

## urllib.request.HTTPSHandler


```python
urllib.request.HTTPSHandler(debuglevel=None, context=None, check_hostname=None)
```

Key methods are add_parent, close, do_open, do_request_.

```python
add_parent(self, parent)
close(self)
do_open(self, http_class, req, **http_conn_args)
do_request_(self, request)
```

## urllib.parse.urlparse: parse a url into 6 components: <scheme>://<netloc>/<path>;<params>?<query>#<fragment> the result is a named 6-tuple with fields corresponding to the above

Parse a URL into 6 components: <scheme>://<netloc>/<path>;<params>?<query>#<fragment> The result is a named 6-tuple with fields corresponding to the above. It is either a ParseResult or ParseResultBytes object, depending on the type of the url parameter.

```python
urllib.parse.urlparse(url, scheme='', allow_fragments=True)
```

## urllib.parse.urlunparse: put a parsed url back together again

Put a parsed URL back together again. This may result in a slightly different, but equivalent URL, if the URL that was parsed originally had redundant delimiters, e.g.

```python
urllib.parse.urlunparse(components)
```

## urllib.parse.urljoin: join a base url and a possibly relative url to form an absolute interpretation of the latter

Join a base URL and a possibly relative URL to form an absolute interpretation of the latter.

```python
urllib.parse.urljoin(base, url, allow_fragments=True)
```

## urllib.parse.urldefrag: removes any existing fragment from url

Removes any existing fragment from URL. Returns a tuple of the defragmented URL and the fragment.

```python
urllib.parse.urldefrag(url)
```

## urllib.parse.urlsplit: parse a url into 5 components: <scheme>://<netloc>/<path>?<query>#<fragment> the result is a named 5-tuple with fields corresponding to the above

Parse a URL into 5 components: <scheme>://<netloc>/<path>?<query>#<fragment> The result is a named 5-tuple with fields corresponding to the above. It is either a SplitResult or SplitResultBytes object, depending on the type of the url parameter.

```python
urllib.parse.urlsplit(url, scheme='', allow_fragments=True)
```

## urllib.parse.urlunsplit: combine the elements of a tuple as returned by urlsplit() into a complete url as a string

Combine the elements of a tuple as returned by urlsplit() into a complete URL as a string. The data argument can be any five-item iterable.

```python
urllib.parse.urlunsplit(components)
```

## urllib.parse.urlencode: encode a dict or sequence of two-element tuples into a url query string

Encode a dict or sequence of two-element tuples into a URL query string. If any values in the query arg are sequences and doseq is true, each sequence element is converted to a separate parameter.

```python
urllib.parse.urlencode(query, doseq=False, safe='', encoding=None, errors=None, quote_via=<function quote_plus at 0x0000027F96CC7920>)
```

## urllib.parse.parse_qs: parse a query given as a string argument

Parse a query given as a string argument. Arguments: qs: percent-encoded query string to be parsed keep_blank_values: flag indicating whether blank values in percent-encoded queries should be treated as blank strings.

```python
urllib.parse.parse_qs(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace', max_num_fields=None, separator='&')
```

## urllib.parse.parse_qsl: parse a query given as a string argument

Parse a query given as a string argument. Arguments: qs: percent-encoded query string to be parsed keep_blank_values: flag indicating whether blank values in percent-encoded queries should be treated as blank strings.

```python
urllib.parse.parse_qsl(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace', max_num_fields=None, separator='&')
```

## urllib.parse.quote: each part of a url, e

Each part of a URL, e.g. the path info, the query, etc., has a different set of reserved characters that must be quoted.

```python
urllib.parse.quote(string, safe='/', encoding=None, errors=None)
```

## urllib.parse.quote_plus: like quote(), but also replace ' ' with '+', as required for quoting html form values

Like quote(), but also replace ' ' with '+', as required for quoting HTML form values. Plus signs in the original string are escaped unless they are included in safe.

```python
urllib.parse.quote_plus(string, safe='', encoding=None, errors=None)
```

## urllib.parse.quote_from_bytes: like quote(), but accepts a bytes object rather than a str, and does not perform string-to-bytes encoding

Like quote(), but accepts a bytes object rather than a str, and does not perform string-to-bytes encoding. It always returns an ASCII string.

```python
urllib.parse.quote_from_bytes(bs, safe='/')
```

## urllib.parse.unquote: replace %xx escapes by their single-character equivalent

Replace %xx escapes by their single-character equivalent. The optional encoding and errors parameters specify how to decode percent-encoded sequences into Unicode characters, as accepted by the bytes.decode() method.

```python
urllib.parse.unquote(string, encoding='utf-8', errors='replace')
```

## urllib.parse.unquote_plus: like unquote(), but also replace plus signs by spaces, as required for unquoting html form values

Like unquote(), but also replace plus signs by spaces, as required for unquoting HTML form values. unquote_plus('%7e/abc+def') -> '~/abc def'.

```python
urllib.parse.unquote_plus(string, encoding='utf-8', errors='replace')
```

## urllib.parse.unquote_to_bytes


```python
urllib.parse.unquote_to_bytes(string)
```

## urllib.parse.DefragResult: a 2-tuple that contains the url without fragment identifier and the fragment identifier as a separate argument

A 2-tuple that contains the url without fragment identifier and the fragment identifier as a separate argument.

```python
urllib.parse.DefragResult(url, fragment)
```

Key methods are count, encode, geturl, index.

```python
count(self, value, /)
encode(self, encoding='ascii', errors='strict')
geturl(self)
index(self, value, start=0, stop=9223372036854775807, /)
```

## urllib.parse.ParseResult: a 6-tuple that contains components of a parsed url

A 6-tuple that contains components of a parsed URL.

```python
urllib.parse.ParseResult(scheme, netloc, path, params, query, fragment)
```

Key methods are count, encode, geturl, index.

```python
count(self, value, /)
encode(self, encoding='ascii', errors='strict')
geturl(self)
index(self, value, start=0, stop=9223372036854775807, /)
```

## urllib.parse.SplitResult: a 5-tuple that contains the different components of a url

A 5-tuple that contains the different components of a URL. Similar to ParseResult, but does not split params.

```python
urllib.parse.SplitResult(scheme, netloc, path, query, fragment)
```

Key methods are count, encode, geturl, index.

```python
count(self, value, /)
encode(self, encoding='ascii', errors='strict')
geturl(self)
index(self, value, start=0, stop=9223372036854775807, /)
```

## urllib.parse.DefragResultBytes: defragresult(url, fragment) a 2-tuple that contains the url without fragment identifier and the fragment identifier as a separate argument

DefragResult(url, fragment) A 2-tuple that contains the url without fragment identifier and the fragment identifier as a separate argument.

```python
urllib.parse.DefragResultBytes(url, fragment)
```

Key methods are count, decode, geturl, index.

```python
count(self, value, /)
decode(self, encoding='ascii', errors='strict')
geturl(self)
index(self, value, start=0, stop=9223372036854775807, /)
```

## urllib.parse.ParseResultBytes: parseresult(scheme, netloc, path, params, query, fragment) a 6-tuple that contains components of a parsed url

ParseResult(scheme, netloc, path, params, query, fragment) A 6-tuple that contains components of a parsed URL.

```python
urllib.parse.ParseResultBytes(scheme, netloc, path, params, query, fragment)
```

Key methods are count, decode, geturl, index.

```python
count(self, value, /)
decode(self, encoding='ascii', errors='strict')
geturl(self)
index(self, value, start=0, stop=9223372036854775807, /)
```

## urllib.parse.SplitResultBytes: splitresult(scheme, netloc, path, query, fragment) a 5-tuple that contains the different components of a url

SplitResult(scheme, netloc, path, query, fragment) A 5-tuple that contains the different components of a URL. Similar to ParseResult, but does not split params.

```python
urllib.parse.SplitResultBytes(scheme, netloc, path, query, fragment)
```

Key methods are count, decode, geturl, index.

```python
count(self, value, /)
decode(self, encoding='ascii', errors='strict')
geturl(self)
index(self, value, start=0, stop=9223372036854775807, /)
```

## textwrap.TextWrapper: object for wrapping/filling text

Object for wrapping/filling text. The public interface consists of the wrap() and fill() methods; the other methods are just there for subclasses to override in order to tweak the default behaviour. If you want to completely replace the main wrapping algorithm, you'll probably have to override _wrap_chunks(). Several instance attributes control various aspects of wrapping: width (default: 70) the maximum width of wrapped lines (unless break_long_words is false) initial_indent (default: "") string that will be prepended to the first line of wrapped output.

```python
textwrap.TextWrapper(width=70, initial_indent='', subsequent_indent='', expand_tabs=True, replace_whitespace=True, fix_sentence_endings=False, break_long_words=True, drop_whitespace=True, break_on_hyphens=True, tabsize=8, *, max_lines=None, placeholder=' [...]')
```

Key methods are fill, wrap.

```python
fill(self, text)
wrap(self, text)
```

## textwrap.wrap: wrap a single paragraph of text, returning a list of wrapped lines

Wrap a single paragraph of text, returning a list of wrapped lines. Reformat the single paragraph in 'text' so it fits in lines of no more than 'width' columns, and return a list of wrapped lines.

```python
textwrap.wrap(text, width=70, **kwargs)
```

## textwrap.fill: fill a single paragraph of text, returning a new string

Fill a single paragraph of text, returning a new string. Reformat the single paragraph in 'text' to fit in lines of no more than 'width' columns, and return a new string containing the entire wrapped paragraph.

```python
textwrap.fill(text, width=70, **kwargs)
```

## textwrap.dedent: remove any common leading whitespace from every line in `text`

Remove any common leading whitespace from every line in `text`. This can be used to make triple-quoted strings line up with the left edge of the display, while still presenting them in the source code in indented form.

```python
textwrap.dedent(text)
```

## textwrap.indent: adds 'prefix' to the beginning of selected lines in 'text'

Adds 'prefix' to the beginning of selected lines in 'text'. If 'predicate' is provided, 'prefix' will only be added to the lines where 'predicate(line)' is True.

```python
textwrap.indent(text, prefix, predicate=None)
```

## textwrap.shorten: collapse and truncate the given text to fit in the given width

Collapse and truncate the given text to fit in the given width. The text first has its whitespace collapsed.

```python
textwrap.shorten(text, width, **kwargs)
```

## secrets.randbelow: return a random int in the range [0, n)

Return a random int in the range [0, n).

```python
secrets.randbelow(exclusive_upper_bound)
```

## secrets.token_bytes: return a random byte string containing *nbytes* bytes

Return a random byte string containing *nbytes* bytes. If *nbytes* is ``None`` or not supplied, a reasonable default is used.

```python
secrets.token_bytes(nbytes=None)
```

## secrets.token_hex: return a random text string, in hexadecimal

Return a random text string, in hexadecimal. The string has *nbytes* random bytes, each byte converted to two hex digits.

```python
secrets.token_hex(nbytes=None)
```

## secrets.token_urlsafe: return a random url-safe text string, in base64 encoding

Return a random URL-safe text string, in Base64 encoding. The string has *nbytes* random bytes.

```python
secrets.token_urlsafe(nbytes=None)
```

## hashlib.new: optionally initialized with data (which must be a bytes-like object)

optionally initialized with data (which must be a bytes-like object).

```python
hashlib.new(name, *args, **kwargs)
```

## hashlib.file_digest: hash the contents of a file-like object

Hash the contents of a file-like object. Returns a digest object.

```python
hashlib.file_digest(fileobj, digest, /, *, _bufsize=262144)
```

## base64.encode: encode a file; input and output are binary files

Encode a file; input and output are binary files.

```python
base64.encode(input, output)
```

## base64.decode: decode a file; input and output are binary files

Decode a file; input and output are binary files.

```python
base64.decode(input, output)
```

## base64.encodebytes: encode a bytestring into a bytes object containing multiple lines of base-64 data

Encode a bytestring into a bytes object containing multiple lines of base-64 data.

```python
base64.encodebytes(s)
```

## base64.decodebytes: decode a bytestring of base-64 data into a bytes object

Decode a bytestring of base-64 data into a bytes object.

```python
base64.decodebytes(s)
```

## base64.b64encode: encode the bytes-like object s using base64 and return a bytes object

Encode the bytes-like object s using Base64 and return a bytes object. Optional altchars should be a byte string of length 2 which specifies an alternative alphabet for the '+' and '/' characters.

```python
base64.b64encode(s, altchars=None)
```

## base64.b64decode: decode the base64 encoded bytes-like object or ascii string s

Decode the Base64 encoded bytes-like object or ASCII string s. Optional altchars must be a bytes-like object or ASCII string of length 2 which specifies the alternative alphabet used instead of the '+' and '/' characters.

```python
base64.b64decode(s, altchars=None, validate=False)
```

## base64.b32encode: encode the bytes-like objects using base32 and return a bytes object

Encode the bytes-like objects using base32 and return a bytes object.

```python
base64.b32encode(s)
```

## base64.b32decode: decode the base32 encoded bytes-like object or ascii string s

Decode the base32 encoded bytes-like object or ASCII string s. Optional casefold is a flag specifying whether a lowercase alphabet is acceptable as input.

```python
base64.b32decode(s, casefold=False, map01=None)
```

## base64.b32hexencode: encode the bytes-like objects using base32hex and return a bytes object

Encode the bytes-like objects using base32hex and return a bytes object.

```python
base64.b32hexencode(s)
```

## base64.b32hexdecode: decode the base32hex encoded bytes-like object or ascii string s

Decode the base32hex encoded bytes-like object or ASCII string s. Optional casefold is a flag specifying whether a lowercase alphabet is acceptable as input.

```python
base64.b32hexdecode(s, casefold=False)
```

## base64.b16encode: encode the bytes-like object s using base16 and return a bytes object

Encode the bytes-like object s using Base16 and return a bytes object.

```python
base64.b16encode(s)
```

## base64.b16decode: decode the base16 encoded bytes-like object or ascii string s

Decode the Base16 encoded bytes-like object or ASCII string s. Optional casefold is a flag specifying whether a lowercase alphabet is acceptable as input.

```python
base64.b16decode(s, casefold=False)
```

## base64.b85encode: encode bytes-like object b in base85 format and return a bytes object

Encode bytes-like object b in base85 format and return a bytes object. If pad is true, the input is padded with b'\0' so its length is a multiple of 4 bytes before encoding.

```python
base64.b85encode(b, pad=False)
```

## base64.b85decode: decode the base85-encoded bytes-like object or ascii string b the result is returned as a bytes object

Decode the base85-encoded bytes-like object or ASCII string b The result is returned as a bytes object.

```python
base64.b85decode(b)
```

## base64.a85encode: encode bytes-like object b using ascii85 and return a bytes object

Encode bytes-like object b using Ascii85 and return a bytes object. foldspaces is an optional flag that uses the special short sequence 'y' instead of 4 consecutive spaces (ASCII 0x20) as supported by 'btoa'.

```python
base64.a85encode(b, *, foldspaces=False, wrapcol=0, pad=False, adobe=False)
```

## base64.a85decode: decode the ascii85 encoded bytes-like object or ascii string b

Decode the Ascii85 encoded bytes-like object or ASCII string b. foldspaces is a flag that specifies whether the 'y' short sequence should be accepted as shorthand for 4 consecutive spaces (ASCII 0x20).

```python
base64.a85decode(b, *, foldspaces=False, adobe=False, ignorechars=b' \t\n\r\x0b')
```

## base64.z85encode: encode bytes-like object b in z85 format and return a bytes object

Encode bytes-like object b in z85 format and return a bytes object.

```python
base64.z85encode(s)
```

## base64.z85decode: decode the z85-encoded bytes-like object or ascii string b the result is returned as a bytes object

Decode the z85-encoded bytes-like object or ASCII string b The result is returned as a bytes object.

```python
base64.z85decode(s)
```

## base64.standard_b64encode: encode bytes-like object s using the standard base64 alphabet

Encode bytes-like object s using the standard Base64 alphabet. The result is returned as a bytes object.

```python
base64.standard_b64encode(s)
```

## base64.standard_b64decode: decode bytes encoded with the standard base64 alphabet

Decode bytes encoded with the standard Base64 alphabet. Argument s is a bytes-like object or ASCII string to decode.

```python
base64.standard_b64decode(s)
```

## base64.urlsafe_b64encode: encode bytes using the url- and filesystem-safe base64 alphabet

Encode bytes using the URL- and filesystem-safe Base64 alphabet. Argument s is a bytes-like object to encode.

```python
base64.urlsafe_b64encode(s)
```

## base64.urlsafe_b64decode: decode bytes using the url- and filesystem-safe base64 alphabet

Decode bytes using the URL- and filesystem-safe Base64 alphabet. Argument s is a bytes-like object or ASCII string to decode.

```python
base64.urlsafe_b64decode(s)
```

## sqlite3.Blob


```python
sqlite3.Blob()
```

Key methods are close, read, seek, tell.

```python
close(self, /)
read(self, length=-1, /)
seek(self, offset, origin=0, /)
tell(self, /)
```

## sqlite3.Connection: sqlite database connection object

SQLite database connection object.

Key methods are backup, blobopen, close, commit.

```python
backup(self, /, target, *, pages=-1, progress=None, name='main', sleep=0.25)
blobopen(self, table, column, row, /, *, readonly=False, name='main')
close(self, /)
commit(self, /)
```

## sqlite3.Cursor: sqlite database cursor class

SQLite database cursor class.

Key methods are close, execute, executemany, executescript.

```python
close(self, /)
execute(self, sql, parameters=(), /)
executemany(self, sql, seq_of_parameters, /)
executescript(self, sql_script, /)
```

## sqlite3.DataError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.DatabaseError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.DateFromTicks


```python
sqlite3.DateFromTicks(ticks)
```

## sqlite3.Error: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.IntegrityError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.InterfaceError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.InternalError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.NotSupportedError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.OperationalError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.PrepareProtocol: pep 246 style object adaption protocol type

PEP 246 style object adaption protocol type.

## sqlite3.ProgrammingError: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## sqlite3.TimeFromTicks


```python
sqlite3.TimeFromTicks(ticks)
```

## sqlite3.TimestampFromTicks


```python
sqlite3.TimestampFromTicks(ticks)
```

## sqlite3.Warning: common base class for all non-exit exceptions

Common base class for all non-exit exceptions.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## logging.BufferingFormatter: a formatter suitable for formatting a number of records

A formatter suitable for formatting a number of records.

```python
logging.BufferingFormatter(linefmt=None)
```

Key methods are format, formatFooter, formatHeader.

```python
format(self, records)
formatFooter(self, records)
formatHeader(self, records)
```

## logging.FileHandler: a handler class which writes formatted logging records to disk files

A handler class which writes formatted logging records to disk files.

```python
logging.FileHandler(filename, mode='a', encoding=None, delay=False, errors=None)
```

Key methods are acquire, addFilter, close, createLock, emit.

```python
acquire(self)
addFilter(self, filter)
close(self)
createLock(self)
emit(self, record)
```

## logging.Filter: filter instances are used to perform arbitrary filtering of logrecords

Filter instances are used to perform arbitrary filtering of LogRecords. Loggers and Handlers can optionally use Filter instances to filter records as desired. The base filter class only allows events which are below a certain point in the logger hierarchy. For example, a filter initialized with "A.B" will allow events logged by loggers "A.B", "A.B.C", "A.B.C.D", "A.B.D" etc.

```python
logging.Filter(name='')
```

Key methods are filter.

```python
filter(self, record)
```

## logging.Formatter: formatter instances are used to convert a logrecord to text

Formatter instances are used to convert a LogRecord to text. Formatters need to know how a LogRecord is constructed. They are responsible for converting a LogRecord to (usually) a string which can be interpreted by either a human or an external system. The base Formatter allows a formatting string to be specified.

```python
logging.Formatter(fmt=None, datefmt=None, style='%', validate=True, *, defaults=None)
```

Key methods are format, formatException, formatMessage, formatStack, formatTime, usesTime.

```python
format(self, record)
formatException(self, ei)
formatMessage(self, record)
formatStack(self, stack_info)
formatTime(self, record, datefmt=None)
usesTime(self)
```

## logging.Handler: handler instances dispatch logging events to specific destinations

Handler instances dispatch logging events to specific destinations. The base handler class. Acts as a placeholder which defines the Handler interface. Handlers can optionally use Formatter instances to format records as desired.

```python
logging.Handler(level=0)
```

Key methods are acquire, addFilter, close, createLock, emit, filter, flush, format, get_name, handle, handleError, release, removeFilter, setFormatter, setLevel, set_name.

```python
acquire(self)
addFilter(self, filter)
close(self)
createLock(self)
emit(self, record)
filter(self, record)
flush(self)
format(self, record)
get_name(self)
handle(self, record)
handleError(self, record)
release(self)
removeFilter(self, filter)
setFormatter(self, fmt)
setLevel(self, level)
set_name(self, name)
```

## logging.LogRecord: a logrecord instance represents an event being logged

A LogRecord instance represents an event being logged. LogRecord instances are created every time something is logged. They contain all the information pertinent to the event being logged. The main information passed in is in msg and args, which are combined using str(msg) % args to create the message field of the record.

```python
logging.LogRecord(name, level, pathname, lineno, msg, args, exc_info, func=None, sinfo=None, **kwargs)
```

Key methods are getMessage.

```python
getMessage(self)
```

## logging.Logger: instances of the logger class represent a single logging channel

Instances of the Logger class represent a single logging channel. A "logging channel" indicates an area of an application. Exactly how an "area" is defined is up to the application developer. Since an application can have any number of areas, logging channels are identified by a unique string.

```python
logging.Logger(name, level=0)
```

Key methods are addFilter, addHandler, callHandlers, critical, debug, error, exception, fatal, filter, findCaller, getChild.

```python
addFilter(self, filter)
addHandler(self, hdlr)
callHandlers(self, record)
critical(self, msg, *args, **kwargs)
debug(self, msg, *args, **kwargs)
error(self, msg, *args, **kwargs)
exception(self, msg, *args, exc_info=True, **kwargs)
fatal(self, msg, *args, **kwargs)
filter(self, record)
findCaller(self, stack_info=False, stacklevel=1)
getChild(self, suffix)
```

## logging.LoggerAdapter: an adapter for loggers which makes it easier to specify contextual information in logging output

An adapter for loggers which makes it easier to specify contextual information in logging output.

```python
logging.LoggerAdapter(logger, extra=None, merge_extra=False)
```

Key methods are critical, debug, error, exception, getEffectiveLevel.

```python
critical(self, msg, *args, **kwargs)
debug(self, msg, *args, **kwargs)
error(self, msg, *args, **kwargs)
exception(self, msg, *args, exc_info=True, **kwargs)
getEffectiveLevel(self)
```

## logging.NullHandler: this handler does nothing

This handler does nothing. It's intended to be used to avoid the "No handlers could be found for logger XXX" one-off warning. This is important for library code, which may contain code to log events. If a user of the library does not configure logging, the one-off warning might be produced; to avoid this, the library developer simply needs to instantiate a NullHandler and add it to the top-level logger of the library module or package.

```python
logging.NullHandler(level=0)
```

Key methods are acquire, addFilter, close, createLock, emit, filter, flush, format, get_name, handle, handleError, release, removeFilter, setFormatter, setLevel, set_name.

```python
acquire(self)
addFilter(self, filter)
close(self)
createLock(self)
emit(self, record)
filter(self, record)
flush(self)
format(self, record)
get_name(self)
handle(self, record)
handleError(self, record)
release(self)
removeFilter(self, filter)
setFormatter(self, fmt)
setLevel(self, level)
set_name(self, name)
```

## logging.StreamHandler: a handler class which writes logging records, appropriately formatted, to a stream

A handler class which writes logging records, appropriately formatted, to a stream. Note that this class does not close the stream, as sys.stdout or sys.stderr may be used.

```python
logging.StreamHandler(stream=None)
```

Key methods are acquire, addFilter, close, createLock, emit, filter, flush, format, get_name, handle, handleError, release, removeFilter, setFormatter.

```python
acquire(self)
addFilter(self, filter)
close(self)
createLock(self)
emit(self, record)
filter(self, record)
flush(self)
format(self, record)
get_name(self)
handle(self, record)
handleError(self, record)
release(self)
removeFilter(self, filter)
setFormatter(self, fmt)
```

## logging.addLevelName: associate 'levelname' with 'level'

Associate 'levelName' with 'level'. This is used when converting levels to text during message formatting.

```python
logging.addLevelName(level, levelName)
```

## logging.basicConfig: do basic configuration for the logging system

Do basic configuration for the logging system. This function does nothing if the root logger already has handlers configured, unless the keyword argument *force* is set to ``True``.

```python
logging.basicConfig(**kwargs)
```

## logging.captureWarnings: if capture is true, redirect all warnings to the logging package

If capture is true, redirect all warnings to the logging package. If capture is False, ensure that warnings are not redirected to logging but to their original destinations.

```python
logging.captureWarnings(capture)
```

## logging.critical: log a message with severity 'critical' on the root logger

Log a message with severity 'CRITICAL' on the root logger. If the logger has no handlers, call basicConfig() to add a console handler with a pre-defined format.

```python
logging.critical(msg, *args, **kwargs)
```

## logging.debug: log a message with severity 'debug' on the root logger

Log a message with severity 'DEBUG' on the root logger. If the logger has no handlers, call basicConfig() to add a console handler with a pre-defined format.

```python
logging.debug(msg, *args, **kwargs)
```

## logging.disable: disable all logging calls of severity 'level' and below

Disable all logging calls of severity 'level' and below.

```python
logging.disable(level=50)
```

## logging.error: log a message with severity 'error' on the root logger

Log a message with severity 'ERROR' on the root logger. If the logger has no handlers, call basicConfig() to add a console handler with a pre-defined format.

```python
logging.error(msg, *args, **kwargs)
```

## logging.exception: log a message with severity 'error' on the root logger, with exception information

Log a message with severity 'ERROR' on the root logger, with exception information. If the logger has no handlers, basicConfig() is called to add a console handler with a pre-defined format.

```python
logging.exception(msg, *args, exc_info=True, **kwargs)
```

## logging.fatal: don't use this function, use critical() instead

Don't use this function, use critical() instead.

```python
logging.fatal(msg, *args, **kwargs)
```

## logging.getLevelName: return the textual or numeric representation of logging level 'level'

Return the textual or numeric representation of logging level 'level'. If the level is one of the predefined levels (CRITICAL, ERROR, WARNING, INFO, DEBUG) then you get the corresponding string.

```python
logging.getLevelName(level)
```

## logging.getLogger: return a logger with the specified name, creating it if necessary

Return a logger with the specified name, creating it if necessary. If no name is specified, return the root logger.

```python
logging.getLogger(name=None)
```

## logging.getLoggerClass: return the class to be used when instantiating a logger

Return the class to be used when instantiating a logger.

```python
logging.getLoggerClass()
```

## logging.info: log a message with severity 'info' on the root logger

Log a message with severity 'INFO' on the root logger. If the logger has no handlers, call basicConfig() to add a console handler with a pre-defined format.

```python
logging.info(msg, *args, **kwargs)
```

## logging.log: log 'msg % args' with the integer severity 'level' on the root logger

Log 'msg % args' with the integer severity 'level' on the root logger. If the logger has no handlers, call basicConfig() to add a console handler with a pre-defined format.

```python
logging.log(level, msg, *args, **kwargs)
```

## logging.makeLogRecord: make a logrecord whose attributes are defined by the specified dictionary, this function is useful for converting a logging event received over a socket connection (which is sent as a dictionary) into a logrecord instance

Make a LogRecord whose attributes are defined by the specified dictionary, This function is useful for converting a logging event received over a socket connection (which is sent as a dictionary) into a LogRecord instance.

```python
logging.makeLogRecord(dict)
```

## logging.setLoggerClass: set the class to be used when instantiating a logger

Set the class to be used when instantiating a logger. The class should define __init__() such that only a name argument is required, and the __init__() should call Logger.__init__().

```python
logging.setLoggerClass(klass)
```

## logging.shutdown: perform any cleanup actions in the logging system (e

Perform any cleanup actions in the logging system (e.g. flushing buffers).

```python
logging.shutdown(handlerList=[<weakref at 0x0000027F96FFAF20; to 'logging._StderrHandler' at 0x0000027F9700C590>])
```

## logging.warn


```python
logging.warn(msg, *args, **kwargs)
```

## logging.warning: log a message with severity 'warning' on the root logger

Log a message with severity 'WARNING' on the root logger. If the logger has no handlers, call basicConfig() to add a console handler with a pre-defined format.

```python
logging.warning(msg, *args, **kwargs)
```

## logging.getLogRecordFactory: return the factory to be used when instantiating a log record

Return the factory to be used when instantiating a log record.

```python
logging.getLogRecordFactory()
```

## logging.setLogRecordFactory: set the factory to be used when instantiating a log record

Set the factory to be used when instantiating a log record. :param factory: A callable which will be called to instantiate a log record.

```python
logging.setLogRecordFactory(factory)
```

## logging.getLevelNamesMapping


```python
logging.getLevelNamesMapping()
```

## logging.getHandlerByName: get a handler with the specified *name*, or none if there isn't one with that name

Get a handler with the specified *name*, or None if there isn't one with that name.

```python
logging.getHandlerByName(name)
```

## logging.getHandlerNames: return all known handler names as an immutable set

Return all known handler names as an immutable set.

```python
logging.getHandlerNames()
```

## unittest.mock.Mock: create a new `mock` object

Create a new `Mock` object. `Mock` takes several optional arguments that specify the behaviour of the Mock object: * `spec`: This can be either a list of strings or an existing object (a class or instance) that acts as the specification for the mock object. If you pass in an object then a list of strings is formed by calling dir on the object (excluding unsupported magic attributes and methods). Accessing any attribute not in this list will raise an `AttributeError`.

**Hazard:** * `unsafe`: By default, accessing any attribute whose name starts with *assert*, *assret*, *asert*, *aseert*, or *assrt* raises an AttributeError.

```python
unittest.mock.Mock(spec=None, side_effect=None, return_value=sentinel.DEFAULT, wraps=None, name=None, spec_set=None, parent=None, _spec_state=None, _new_name='', _new_parent=None, **kwargs)
```

Key methods are assert_any_call, assert_called, assert_called_once, assert_called_once_with, assert_called_with, assert_has_calls, assert_not_called, attach_mock, configure_mock, mock_add_spec, reset_mock.

```python
assert_any_call(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
assert_called_with(self, /, *args, **kwargs)
assert_has_calls(self, calls, any_order=False)
assert_not_called(self)
attach_mock(self, mock, attribute)
configure_mock(self, /, **kwargs)
mock_add_spec(self, spec, spec_set=False)
reset_mock(self, visited=None, *, return_value: bool = False, side_effect: bool = False)
```

## unittest.mock.MagicMock: magicmock is a subclass of mock with default implementations of most of the magic methods

MagicMock is a subclass of Mock with default implementations of most of the magic methods. You can use MagicMock without having to configure the magic methods yourself. If you use the `spec` or `spec_set` arguments then *only* magic methods that exist in the spec will be created. Attributes and the return value of a `MagicMock` will also be `MagicMocks`.

```python
unittest.mock.MagicMock(*args, **kw)
```

Key methods are assert_any_call, assert_called, assert_called_once, assert_called_once_with, assert_called_with, assert_has_calls, assert_not_called, attach_mock, configure_mock, mock_add_spec, reset_mock.

```python
assert_any_call(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
assert_called_with(self, /, *args, **kwargs)
assert_has_calls(self, calls, any_order=False)
assert_not_called(self)
attach_mock(self, mock, attribute)
configure_mock(self, /, **kwargs)
mock_add_spec(self, spec, spec_set=False)
reset_mock(self, /, *args, return_value: bool = False, **kwargs)
```

## unittest.mock.patch: `patch` acts as a function decorator, class decorator or a context manager

`patch` acts as a function decorator, class decorator or a context manager. Inside the body of the function or with statement, the `target` is patched with a `new` object.

**Hazard:** Pass the argument `unsafe` with the value True to disable that check.

```python
unittest.mock.patch(target, new=sentinel.DEFAULT, spec=None, create=False, spec_set=None, autospec=None, new_callable=None, *, unsafe=False, **kwargs)
```

## unittest.mock.create_autospec: create a mock object using another object as a spec

Create a mock object using another object as a spec. Attributes on the mock will use the corresponding attribute on the `spec` object as their spec.

**Hazard:** Pass the argument `unsafe` with the value True to disable that check.

```python
unittest.mock.create_autospec(spec, spec_set=False, instance=False, _parent=None, _name=None, *, unsafe=False, **kwargs)
```

## unittest.mock.AsyncMock: enhance :class:`mock` with features allowing to mock an async function

Enhance :class:`Mock` with features allowing to mock an async function. The :class:`AsyncMock` object will behave so the object is recognized as an async function, and the result of a call is an awaitable: >>> mock = AsyncMock() >>> iscoroutinefunction(mock) True >>> inspect.isawaitable(mock()) True The result of ``mock()`` is an async function which will have the outcome of ``side_effect`` or ``return_value``: - if ``side_effect`` is a function, the async function will return the result of that function, - if ``side_effect`` is an exception, the async function will raise the exception, - if ``side_effect`` is an iterable, the async function will return the next value of the iterable, however, if the sequence of result is exhausted, ``StopIteration`` is raised immediately, - if ``side_effect`` is not defined, the async function will return the value defined by ``return_value``, hence, by default, the async function returns a new :class:`AsyncMock` object. If the outcome of ``side_effect`` or ``return_value`` is an async function, the mock async function obtained when the mock object is called will be this async function itself (and not an async function returning an async function). The test author can also specify a wrapped object with ``wraps``.

Key methods are assert_any_await, assert_any_call, assert_awaited, assert_awaited_once, assert_awaited_once_with, assert_awaited_with, assert_called, assert_called_once, assert_called_once_with, assert_called_with, assert_has_awaits, assert_has_calls, assert_not_awaited, assert_not_called, attach_mock, configure_mock, mock_add_spec, reset_mock.

```python
assert_any_await(self, /, *args, **kwargs)
assert_any_call(self, /, *args, **kwargs)
assert_awaited(self)
assert_awaited_once(self)
assert_awaited_once_with(self, /, *args, **kwargs)
assert_awaited_with(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
assert_called_with(self, /, *args, **kwargs)
assert_has_awaits(self, calls, any_order=False)
assert_has_calls(self, calls, any_order=False)
assert_not_awaited(self)
assert_not_called(self)
attach_mock(self, mock, attribute)
configure_mock(self, /, **kwargs)
mock_add_spec(self, spec, spec_set=False)
reset_mock(self, /, *args, **kwargs)
```

## unittest.mock.ThreadingMock: a mock that can be used to wait until on calls happening in a different thread

A mock that can be used to wait until on calls happening in a different thread. The constructor can take a `timeout` argument which controls the timeout in seconds for all `wait` calls of the mock. You can change the default timeout of all instances via the `ThreadingMock.DEFAULT_TIMEOUT` attribute. If no timeout is set, it will block undefinetively.

```python
unittest.mock.ThreadingMock(*args, timeout=sentinel.TIMEOUT_UNSET, **kwargs)
```

Key methods are assert_any_call, assert_called, assert_called_once, assert_called_once_with, assert_called_with, assert_has_calls, assert_not_called, attach_mock, configure_mock, mock_add_spec, reset_mock.

```python
assert_any_call(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
assert_called_with(self, /, *args, **kwargs)
assert_has_calls(self, calls, any_order=False)
assert_not_called(self)
attach_mock(self, mock, attribute)
configure_mock(self, /, **kwargs)
mock_add_spec(self, spec, spec_set=False)
reset_mock(self, /, *args, **kwargs)
```

## unittest.mock.NonCallableMock: a non-callable version of `mock`

A non-callable version of `Mock`.

```python
unittest.mock.NonCallableMock(spec=None, wraps=None, name=None, spec_set=None, parent=None, _spec_state=None, _new_name='', _new_parent=None, _spec_as_instance=False, _eat_self=None, unsafe=False, **kwargs)
```

Key methods are assert_any_call, assert_called, assert_called_once, assert_called_once_with.

```python
assert_any_call(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
```

## unittest.mock.NonCallableMagicMock: a version of `magicmock` that isn't callable

A version of `MagicMock` that isn't callable.

```python
unittest.mock.NonCallableMagicMock(*args, **kw)
```

Key methods are assert_any_call, assert_called, assert_called_once, assert_called_once_with.

```python
assert_any_call(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
```

## unittest.mock.mock_open: a helper function to create a mock to replace the use of `open`

A helper function to create a mock to replace the use of `open`. It works for `open` called directly or used as a context manager.

```python
unittest.mock.mock_open(mock=None, read_data='')
```

## unittest.mock.PropertyMock: a mock intended to be used as a property, or other descriptor, on a class

A mock intended to be used as a property, or other descriptor, on a class. `PropertyMock` provides `__get__` and `__set__` methods so you can specify a return value when it is fetched. Fetching a `PropertyMock` instance from an object calls the mock, with no args. Setting it calls the mock with the value being set.

```python
unittest.mock.PropertyMock(spec=None, side_effect=None, return_value=sentinel.DEFAULT, wraps=None, name=None, spec_set=None, parent=None, _spec_state=None, _new_name='', _new_parent=None, **kwargs)
```

Key methods are assert_any_call, assert_called, assert_called_once, assert_called_once_with, assert_called_with, assert_has_calls, assert_not_called.

```python
assert_any_call(self, /, *args, **kwargs)
assert_called(self)
assert_called_once(self)
assert_called_once_with(self, /, *args, **kwargs)
assert_called_with(self, /, *args, **kwargs)
assert_has_calls(self, calls, any_order=False)
assert_not_called(self)
```

## unittest.mock.seal: disable the automatic generation of child mocks

Disable the automatic generation of child mocks. Given an input Mock, seals it to ensure no further mocks will be generated when accessing an attribute that was not already defined.

```python
unittest.mock.seal(mock)
```

## concurrent.futures.CancelledError: the future was cancelled

The Future was cancelled.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## concurrent.futures.InvalidStateError: the operation is not allowed in this state

The operation is not allowed in this state.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## concurrent.futures.BrokenExecutor: raised when a executor has become non-functional after a severe failure

Raised when a executor has become non-functional after a severe failure.

Key methods are add_note, with_traceback.

```python
add_note(self, object, /)
with_traceback(self, object, /)
```

## concurrent.futures.Future: represents the result of an asynchronous computation

Represents the result of an asynchronous computation.

```python
concurrent.futures.Future()
```

Key methods are add_done_callback, cancel, cancelled, done, exception, result.

```python
add_done_callback(self, fn)
cancel(self)
cancelled(self)
done(self)
exception(self, timeout=None)
result(self, timeout=None)
```

## concurrent.futures.Executor: this is an abstract base class for concrete asynchronous executors

This is an abstract base class for concrete asynchronous executors.

```python
concurrent.futures.Executor()
```

Key methods are map, shutdown, submit.

```python
map(self, fn, *iterables, timeout=None, chunksize=1)
shutdown(self, wait=True, *, cancel_futures=False)
submit(self, fn, /, *args, **kwargs)
```

## concurrent.futures.wait: wait for the futures in the given sequence to complete

Wait for the futures in the given sequence to complete. Args: fs: The sequence of Futures (possibly created by different Executors) to wait upon.

```python
concurrent.futures.wait(fs, timeout=None, return_when='ALL_COMPLETED')
```

## concurrent.futures.as_completed: an iterator over the given futures that yields each as it completes

An iterator over the given futures that yields each as it completes. Args: fs: The sequence of Futures (possibly created by different Executors) to iterate over.

```python
concurrent.futures.as_completed(fs, timeout=None)
```

## concurrent.futures.ProcessPoolExecutor: this is an abstract base class for concrete asynchronous executors

This is an abstract base class for concrete asynchronous executors.

```python
concurrent.futures.ProcessPoolExecutor(max_workers=None, mp_context=None, initializer=None, initargs=(), *, max_tasks_per_child=None)
```

Key methods are map, shutdown, submit.

```python
map(self, fn, *iterables, timeout=None, chunksize=1)
shutdown(self, wait=True, *, cancel_futures=False)
submit(self, fn, /, *args, **kwargs)
```

## concurrent.futures.ThreadPoolExecutor: this is an abstract base class for concrete asynchronous executors

This is an abstract base class for concrete asynchronous executors.

```python
concurrent.futures.ThreadPoolExecutor(max_workers=None, thread_name_prefix='', initializer=None, initargs=())
```

Key methods are map, shutdown, submit.

```python
map(self, fn, *iterables, timeout=None, chunksize=1)
shutdown(self, wait=True, *, cancel_futures=False)
submit(self, fn, /, *args, **kwargs)
```

# Recipes — authored, not generated

Each recipe names a task in the words someone doing it would use, then
shows the shortest stdlib-only way that is correct on Python 3.13.

## Reading and writing a text file with pathlib

The whole-file read and write are one call each on `Path` — no `open()`
needed, and encoding should always be explicit.

```python
from pathlib import Path
text = Path("notes.txt").read_text(encoding="utf-8")
Path("out.txt").write_text(text, encoding="utf-8")
```

`read_bytes()` / `write_bytes()` are the binary pair. For appending or
line-by-line streaming, fall back to `Path.open("a", encoding="utf-8")`.

## Running a shell command and capturing its output

`subprocess.run` with `capture_output=True, text=True` returns stdout and
stderr as strings on a `CompletedProcess`. Pass the command as a list —
never build a shell string from variables.

```python
import subprocess
r = subprocess.run(["git", "status", "--short"],
                   capture_output=True, text=True, check=True)
print(r.stdout)
```

`check=True` raises `CalledProcessError` on nonzero exit; without it you
must test `r.returncode` yourself. There is no `subprocess.run(...).output`
attribute — it is `stdout`.

## Walking a directory tree and matching file names

`Path.rglob` is the recursive glob; it yields `Path` objects, not strings.

```python
from pathlib import Path
for p in Path("src").rglob("*.py"):
    if p.is_file():
        print(p, p.stat().st_size)
```

For a flat, non-recursive match use `Path.glob("*.py")`. `os.walk` still
exists but returns string triples; prefer `rglob` in new code.

## Reading and writing JSON with a file

`json.load` / `json.dump` take file objects; `json.loads` / `json.dumps`
take strings — the trailing `s` means *string*.

```python
import json
from pathlib import Path
data = json.loads(Path("cfg.json").read_text(encoding="utf-8"))
Path("cfg.json").write_text(
    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
```

`json.dumps` has no `pretty=True` parameter — indentation is `indent=2`.
Keys are always strings after a round-trip; integer dict keys do not
survive.

## Parsing and formatting dates and times

`datetime.strptime` parses, `strftime` formats, and the format codes are
the same in both directions. Timezone-aware "now" is
`datetime.now(timezone.utc)` — `utcnow()` is deprecated and naive.

```python
from datetime import datetime, timezone
d = datetime.strptime("2026-07-30 14:00", "%Y-%m-%d %H:%M")
stamp = datetime.now(timezone.utc).isoformat()
back = datetime.fromisoformat(stamp)
```

`fromisoformat` accepts the `Z` suffix since 3.11. There is no
`datetime.parse` and no `strptime` on `date` objects' instances.

## Making an HTTP GET request without third-party packages

`urllib.request.urlopen` is the stdlib route; the response is bytes and
must be decoded. There is no `requests` in the standard library.

```python
from urllib.request import urlopen, Request
req = Request("https://api.example.com/items",
              headers={"Accept": "application/json"})
with urlopen(req, timeout=10) as resp:
    body = resp.read().decode("utf-8")
```

Always pass `timeout` — the default is no timeout at all. For anything
beyond a simple GET/POST, recommend installing `httpx` or `requests`
rather than fighting `urllib`.

## Building a URL query string safely

`urllib.parse.urlencode` handles quoting and multi-value keys;
never concatenate user input into a URL by hand.

```python
from urllib.parse import urlencode, urlparse, parse_qs
qs = urlencode({"q": "black & white", "page": 2})   # q=black+%26+white&page=2
parts = urlparse("https://x.dev/search?" + qs)
params = parse_qs(parts.query)                       # values are LISTS
```

`parse_qs` returns `{"q": ["black & white"]}` — every value is a list.

## Extracting groups from text with a regular expression

`re.search` finds the first match anywhere; `re.match` only matches at the
start of the string — using `match` where `search` is meant is the classic
silent bug. A no-match returns `None`, so guard before `.group()`.

```python
import re
m = re.search(r"v(\d+)\.(\d+)\.(\d+)", "release v0.11.2 shipped")
if m:
    major, minor, patch = m.groups()
```

`re.findall` returns strings (or tuples when the pattern has groups);
`re.finditer` returns match objects. Compile with `re.compile` only when
the pattern is reused in a loop.

## Copying, moving and deleting files and directories

`shutil.copy2` preserves metadata, plain `copy` does not; `shutil.move`
works across filesystems; directory trees need `copytree`/`rmtree` —
`Path.unlink` and `os.remove` only delete single files.

```python
import shutil
from pathlib import Path
shutil.copy2("a.db", "a.db.bak")
shutil.copytree("site", "site_backup", dirs_exist_ok=True)
shutil.rmtree("build")            # directory tree, no confirmation
Path("stale.log").unlink(missing_ok=True)
```

`copytree` raises if the target exists unless `dirs_exist_ok=True` (3.8+).

## Creating a temporary file or directory that cleans itself up

The context-manager forms of `tempfile` remove what they created.
On Windows a `NamedTemporaryFile` cannot be reopened while open — use a
`TemporaryDirectory` and create files inside it instead.

```python
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as td:
    scratch = Path(td) / "work.json"
    scratch.write_text("{}", encoding="utf-8")
# gone here
```

## Counting, grouping and deduplicating with collections

`Counter` counts hashables, `defaultdict(list)` groups, and `dict` itself
deduplicates while preserving order.

```python
from collections import Counter, defaultdict
words = ["red", "blue", "red", "green", "red"]
Counter(words).most_common(2)        # [("red", 3), ("blue", 1)]
by_len = defaultdict(list)
for w in words:
    by_len[len(w)].append(w)
unique = list(dict.fromkeys(words))  # order-preserving dedupe
```

`most_common()` with no argument returns everything, sorted.

## Batching, pairing and flattening iterables with itertools

`itertools.batched` (3.12+) chunks an iterable; `pairwise` gives sliding
pairs; `chain.from_iterable` flattens one level.

```python
from itertools import batched, pairwise, chain
list(batched("ABCDEFG", 3))          # [('A','B','C'), ('D','E','F'), ('G',)]
list(pairwise([1, 4, 9]))            # [(1, 4), (4, 9)]
list(chain.from_iterable([[1, 2], [3]]))   # [1, 2, 3]
```

Before 3.12 there is no `batched` — write the two-line grouper instead of
inventing an import for it.

## Caching a pure function's results

`functools.lru_cache` memoises by argument; `functools.cache` (3.9+) is
the unbounded spelling. Arguments must be hashable — passing a list or
dict raises `TypeError` at call time.

```python
from functools import lru_cache
@lru_cache(maxsize=256)
def price(sku: str) -> float:
    ...
price.cache_clear()      # reset between tests
```

## Defining a small data-carrying class

`dataclasses.dataclass` writes `__init__`, `__repr__` and `__eq__`.
Mutable defaults must use `field(default_factory=...)` — a bare `[]`
default is rejected at class-creation time.

```python
from dataclasses import dataclass, field
@dataclass
class Job:
    name: str
    retries: int = 3
    tags: list[str] = field(default_factory=list)
```

`frozen=True` makes instances hashable and immutable; `slots=True` (3.10+)
saves memory on many instances.

## Hashing a file or string with hashlib

`hashlib.sha256` wants bytes, not str; `file_digest` (3.11+) streams a
file without loading it.

```python
import hashlib
digest = hashlib.sha256("hello".encode("utf-8")).hexdigest()
with open("wheel.whl", "rb") as f:
    fd = hashlib.file_digest(f, "sha256").hexdigest()
```

Use `secrets.token_hex(16)` for tokens — never a hash of `random()`.

## Reading and writing CSV with headers

`csv.DictReader` maps rows to dicts using the header row; always open CSV
files with `newline=""` or Windows doubles every line ending.

```python
import csv
with open("rows.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
with open("out.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["name", "qty"])
    w.writeheader()
    w.writerows(rows)
```

## Setting up logging instead of print

One `basicConfig` call at the entry point; modules take
`logging.getLogger(__name__)` and never configure anything themselves.

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)
log.info("mounted %s in %.1fms", pack, ms)
```

Pass lazy `%s` arguments, not f-strings, so skipped levels cost nothing.
`log.exception("...")` inside an `except` block records the traceback.

## Running functions in parallel threads with a pool

`ThreadPoolExecutor.map` keeps input order; `as_completed` yields in
finish order. Threads suit I/O-bound work; CPU-bound work needs
`ProcessPoolExecutor`.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, u): u for u in urls}
    for fut in as_completed(futs):
        url, body = futs[fut], fut.result()   # .result() re-raises errors
```

A future swallows its exception until `.result()` is called — iterate the
results or the errors vanish silently.

## Using sqlite3 with parameters and rows as dicts

Placeholders are `?`, never string formatting; `row_factory = sqlite3.Row`
gives name access; `with conn` commits or rolls back a transaction.

```python
import sqlite3
conn = sqlite3.connect("app.db")
conn.row_factory = sqlite3.Row
with conn:
    conn.execute("INSERT INTO jobs(name, qty) VALUES(?, ?)", ("build", 3))
row = conn.execute("SELECT * FROM jobs WHERE name=?", ("build",)).fetchone()
print(row["qty"])
```

`executemany` takes a sequence of parameter tuples for bulk inserts.

## Command-line arguments with argparse

Flags with `--` are optional; bare names are positional; `action=
"store_true"` makes a boolean switch. `parse_args()` exits with a usage
message on bad input — that is intended behaviour, not a crash.

```python
import argparse
ap = argparse.ArgumentParser(description="Sync packs")
ap.add_argument("target")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--limit", type=int, default=10)
args = ap.parse_args()
```

Attribute names swap dashes for underscores: `--dry-run` → `args.dry_run`.

## Replacing a function with a mock in a test

Patch where the name is *looked up*, not where it is defined: if
`app.sync` does `from client import fetch`, the patch target is
`app.sync.fetch`.

```python
from unittest.mock import patch
with patch("app.sync.fetch", return_value={"ok": True}) as m:
    run_sync()
    m.assert_called_once_with("packs")
```

`side_effect=Exception("boom")` makes the mock raise;
`side_effect=[a, b]` returns successive values across calls.

## Retrying an operation with backoff, stdlib only

There is no `retrying` or `tenacity` in the standard library. The honest
stdlib version is a loop; sleep grows exponentially and re-raises on the
final attempt.

```python
import time
for attempt in range(4):
    try:
        result = flaky()
        break
    except TimeoutError:
        if attempt == 3:
            raise
        time.sleep(2 ** attempt)      # 1s, 2s, 4s
```

## Wrapping and dedenting multi-line text

`textwrap.dedent` strips the common leading whitespace from triple-quoted
strings; `shorten` truncates on word boundaries with a placeholder.

```python
import textwrap
sql = textwrap.dedent("""\
    SELECT name, qty
    FROM jobs
    WHERE qty > ?""")
title = textwrap.shorten(long_title, width=60, placeholder="…")
```

## Base64 for binary data in JSON or URLs

`b64encode` takes and returns bytes — decode to str for JSON. URL-safe
variants swap `+/` for `-_`.

```python
import base64
payload = base64.b64encode(b"\x00\x01binary").decode("ascii")
raw = base64.b64decode(payload)
token = base64.urlsafe_b64encode(raw).rstrip(b"=")
```

Padding matters on decode: add `==` back if you stripped it, or pass
`validate=False` and accept the risk.
