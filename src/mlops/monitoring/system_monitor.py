"""
System Monitoring for EcoVision AI

Monitors system performance, model metrics, and health status.
"""

import asyncio
import psutil
import time
from datetime import datetime
from typing import Dict, List
import json

from loguru import logger


class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, config: Dict):
        """Initialize system monitor."""
        self.config = config
        self.metrics_history = []
    
    async def check_system_health(self) -> Dict:
        """Check overall system health."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "issues": []
        }
        
        # Check for issues
        if health_status["cpu_usage"] > 90:
            health_status["issues"].append("High CPU usage")
            health_status["status"] = "warning"
        
        if health_status["memory_usage"] > 85:
            health_status["issues"].append("High memory usage")
            health_status["status"] = "warning"
        
        if health_status["disk_usage"] > 90:
            health_status["issues"].append("High disk usage")
            health_status["status"] = "critical"
        
        return health_status
    
    async def log_analysis_results(self, results: Dict):
        """Log analysis results for monitoring."""
        logger.info(f"Analysis completed for {results.get('region', 'unknown')}")
        
        # Store metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "region": results.get("region"),
            "analysis_type": results.get("analysis_type"),
            "success": "error" not in results
        }
        
        self.metrics_history.append(metrics)