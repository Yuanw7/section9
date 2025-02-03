#!/bin/bash

# Validate input
if [ $# -ne 1 ] || ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -ge 100000 ]; then
    echo "Error: Provide a valid decimal number < 100000"
    exit 1
fi

decimal=$1

# Hard-coded hex mapping (Table 1)
hex_map=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "A" "B" "C" "D" "E" "F")

# Binary conversion with 4-bit grouping (Table 1)
dec_to_bin() {
    local dec=$1 bin=""
    [ $dec -eq 0 ] && { echo "0000"; return; }
    while ((dec > 0)); do
        bin="$((dec % 2))$bin"
        dec=$((dec / 2))
    done
    # Pad to 4-bit groups
    while (( ${#bin} % 4 != 0 )); do bin="0$bin"; done
    echo "$bin"
}

# Hexadecimal conversion via division (Table 2)
dec_to_hex() {
    local dec=$1 hex="" quotient remainder
    while ((dec > 0)); do
        quotient=$((dec / 16))
        remainder=$((dec % 16))
        hex="${hex_map[remainder]}$hex"
        dec=$quotient
    done
    [ -z "$hex" ] && hex="0"
    echo "$hex"
}

# Generate results
binary=$(dec_to_bin "$decimal")
hexadecimal=$(dec_to_hex "$decimal")

# Save output
echo "Decimal: $decimal" > "convertion result.txt"
echo "Binary: $binary" >> "convertion result.txt"
echo "Hexadecimal: $hexadecimal" >> "convertion result.txt"
echo "Results saved to convertion result.txt"
