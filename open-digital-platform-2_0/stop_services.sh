#!/bin/bash
echo "🛑 Stopping 5G Core Network Services..."

if [ -f .service_pids ]; then
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Stopping process $pid"
            kill $pid
        fi
    done < .service_pids
    rm -f .service_pids
    echo "✅ All services stopped"
else
    echo "No service PIDs found"
fi
