#! /bin/bash
find . -type d -name "__pycache__" | xargs rm -rf
echo "__pycache__ directories cleaned!"
