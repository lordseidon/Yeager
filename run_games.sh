#!/bin/bash

MAX_GAMES=200
COUNTER=0

while true; do
  # Count current running games
  RUNNING=$(pgrep -fc "./chess")

  # Launch new games if below limit
  while [ "$RUNNING" -lt "$MAX_GAMES" ]; do
    COUNTER=$((COUNTER + 1))
    ./chess > "game_${COUNTER}.log" 2>&1 &
    echo "Started game $COUNTER (running: $RUNNING)"
    RUNNING=$((RUNNING + 1))
    sleep 0.1  # small delay to avoid race conditions
  done

  # Check every few seconds
  sleep 5
done
