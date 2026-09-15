#!/bin/bash
printf '%s\n' 'Cache cleanup is disabled: directory names do not prove project ownership.' 'No files were changed. Use PYTHONDONTWRITEBYTECODE=1 to prevent new bytecode caches.' >&2
exit 2
