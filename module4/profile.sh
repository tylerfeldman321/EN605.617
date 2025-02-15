#!/bin/bash

for i in {8..30}
do
  output=$(./host_memory $i)  # Run the command and capture the output
  N=$(echo "$output" | grep -oP 'N=\K\d+')  # Extract N value
  time=$(echo "$output" | grep -oP 'Time elapsed GPU = \K\d+')  # Extract time in microseconds
  echo "For n exponent $i: N = $N, Time elapsed GPU = ${time} microseconds"
done