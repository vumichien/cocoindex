---
title: Sources
description: CocoIndex Built-in Sources
---

# CocoIndex Built-in Sources

## LocalFile

The `LocalFile` source imports files from a local file system.

The spec takes the following fields:
*   `path` (type: `str`, required): full path of the root directory to import files from
*   `binary` (type: `bool`, default: `False`): whether reading files as binary (instead of text)

The output is a table with the following sub fields:
*   `filename` (key, type: `str`): the filename of the file, including the path, relative to the root directory, e.g. `"dir1/file1.md"`
*   `content` (type: `str` if `binary` is `False`, otherwise `bytes`): the content of the file
