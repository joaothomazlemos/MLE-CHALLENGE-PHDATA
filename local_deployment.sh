#!/bin/bash
# Sequentially restart all api replicas to achieve a rolling update

for container in $(docker ps --filter "name=api" --format "{{.Names}}"); do
    echo "Restarting $container..."
    docker restart "$container"
    # wait a few seconds for health check
    sleep 11
done

echo "All API replicas restarted sequentially."
