#!/bin/bash
# Function to delete all files in a directory
file_remove() {
  echo "Deleting all files in: $1"
  rm -f "$1"/*
}
# Alias without a space (e.g., "file_remove")
alias file_remove=file_remove
