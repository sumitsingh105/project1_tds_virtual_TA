#!/bin/bash
# File: run_indexing_batches.sh

ranges=(0 1000 2000 3000 4000 5000)

for ((i = 0; i < ${#ranges[@]} - 1; i++)); do
  start=${ranges[$i]}
  end=${ranges[$i + 1]}
  echo "🚀 Running: python3 typesense_indexer_range.py $start $end"
  python3 typesense_indexer_range.py $start $end
  echo "✅ Completed: $start to $end"
done
