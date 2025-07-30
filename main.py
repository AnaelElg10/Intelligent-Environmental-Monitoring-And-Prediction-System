#!/usr/bin/env python3
"""
EcoVision AI: Intelligent Environmental Monitoring & Prediction System
Main entry point for the application.

This module orchestrates the entire AI pipeline including:
- Satellite image analysis using Vision Transformers
- Climate forecasting with Temporal Fusion Networks
- Conservation optimization using Reinforcement Learning
- Real-time monitoring and alerting
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.vision.satellite_analyzer import SatelliteAnalyzer
from src.forecasting.climate_predictor import ClimatePredictor
from src.rl.conservation_optimizer import ConservationOptimizer
from src.data.ingestion.data_collector import DataCollector
from src.mlops.monitoring.system_monitor import SystemMonitor
from src.utils.config_manager import ConfigManager
from src.utils.logger_setup import setup_logging


class EcoVisionAI:
    """
    Main EcoVision AI system orchestrator.
    
    Coordinates all AI components and provides unified interface for
    environmental monitoring and prediction tasks.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the EcoVision AI system."""
        self.config = ConfigManager(config_path)
        setup_logging(self.config.get("logging", {}))
        
        # Initialize core components
        self.satellite_analyzer = SatelliteAnalyzer(self.config.get("vision", {}))
        self.climate_predictor = ClimatePredictor(self.config.get("forecasting", {}))
        self.conservation_optimizer = ConservationOptimizer(self.config.get("rl", {}))
        self.data_collector = DataCollector(self.config.get("data", {}))
        self.system_monitor = SystemMonitor(self.config.get("monitoring", {}))
        
        logger.info("EcoVision AI system initialized successfully")
    
    async def analyze_region(self, region: str, analysis_type: str = "comprehensive") -> Dict:
        """
        Perform comprehensive environmental analysis for a specific region.
        
        Args:
            region: Geographic region identifier
            analysis_type: Type of analysis ('deforestation', 'climate', 'comprehensive')
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting {analysis_type} analysis for region: {region}")
        
        results = {
            "region": region,
            "analysis_type": analysis_type,
            "timestamp": self.data_collector.get_current_timestamp()
        }
        
        try:
            # Collect satellite data
            satellite_data = await self.data_collector.get_satellite_data(region)
            results["data_collection"] = {"satellite": "success"}
            
            # Analyze satellite imagery
            if analysis_type in ["deforestation", "comprehensive"]:
                vision_results = await self.satellite_analyzer.analyze_deforestation(
                    satellite_data["images"]
                )
                results["deforestation_analysis"] = vision_results
            
            # Climate prediction
            if analysis_type in ["climate", "comprehensive"]:
                climate_data = await self.data_collector.get_climate_data(region)
                forecast_results = await self.climate_predictor.predict_weather(
                    climate_data, days=7
                )
                results["climate_forecast"] = forecast_results
            
            # Conservation optimization
            if analysis_type == "comprehensive":
                optimization_results = await self.conservation_optimizer.optimize_strategy(
                    region, results
                )
                results["conservation_strategy"] = optimization_results
            
            # Generate alerts if necessary
            alerts = self._generate_alerts(results)
            if alerts:
                results["alerts"] = alerts
                await self._send_alerts(alerts)
            
            logger.info(f"Analysis completed successfully for {region}")
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed for {region}: {str(e)}")
            results["error"] = str(e)
            return results
    
    async def real_time_monitoring(self, regions: List[str], interval: int = 3600):
        """
        Start real-time monitoring for multiple regions.
        
        Args:
            regions: List of regions to monitor
            interval: Monitoring interval in seconds (default: 1 hour)
        """
        logger.info(f"Starting real-time monitoring for {len(regions)} regions")
        
        while True:
            try:
                # Monitor each region
                for region in regions:
                    results = await self.analyze_region(region, "comprehensive")
                    await self.system_monitor.log_analysis_results(results)
                
                # System health check
                health_status = await self.system_monitor.check_system_health()
                logger.info(f"System health: {health_status['status']}")
                
                # Wait for next cycle
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def _generate_alerts(self, results: Dict) -> List[Dict]:
        """Generate alerts based on analysis results."""
        alerts = []
        
        # Deforestation alerts
        if "deforestation_analysis" in results:
            deforestation_rate = results["deforestation_analysis"].get("change_rate", 0)
            if deforestation_rate > self.config.get("alerts.deforestation_threshold", 0.05):
                alerts.append({
                    "type": "deforestation",
                    "severity": "high" if deforestation_rate > 0.1 else "medium",
                    "message": f"Deforestation rate of {deforestation_rate:.2%} detected",
                    "region": results["region"]
                })
        
        # Climate alerts
        if "climate_forecast" in results:
            extreme_weather = results["climate_forecast"].get("extreme_weather_risk", 0)
            if extreme_weather > 0.7:
                alerts.append({
                    "type": "extreme_weather",
                    "severity": "high",
                    "message": f"High risk of extreme weather events",
                    "region": results["region"]
                })
        
        return alerts
    
    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts to configured channels."""
        for alert in alerts:
            logger.warning(f"ALERT: {alert['message']} in {alert['region']}")
            # TODO: Implement email, SMS, webhook notifications


async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="EcoVision AI: Environmental Monitoring & Prediction System"
    )
    parser.add_argument(
        "--mode", 
        choices=["analysis", "monitoring", "training", "api"],
        default="analysis",
        help="Operation mode"
    )
    parser.add_argument(
        "--region",
        default="amazon",
        help="Region to analyze (for analysis mode)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--analysis-type",
        choices=["deforestation", "climate", "comprehensive"],
        default="comprehensive",
        help="Type of analysis to perform"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    eco_vision = EcoVisionAI(args.config)
    
    try:
        if args.mode == "analysis":
            # Single region analysis
            results = await eco_vision.analyze_region(args.region, args.analysis_type)
            print("\n" + "="*60)
            print(f"ECOVISION AI ANALYSIS RESULTS - {args.region.upper()}")
            print("="*60)
            
            if "deforestation_analysis" in results:
                print(f"\n🌳 DEFORESTATION ANALYSIS:")
                deforestation = results["deforestation_analysis"]
                print(f"   Change Rate: {deforestation.get('change_rate', 0):.2%}")
                print(f"   Forest Cover: {deforestation.get('forest_cover', 0):.1f}%")
                print(f"   Risk Level: {deforestation.get('risk_level', 'Unknown')}")
            
            if "climate_forecast" in results:
                print(f"\n🌤️  CLIMATE FORECAST:")
                climate = results["climate_forecast"]
                print(f"   Temperature Trend: {climate.get('temperature_trend', 'Unknown')}")
                print(f"   Precipitation: {climate.get('precipitation_forecast', 'Unknown')}")
                print(f"   Extreme Weather Risk: {climate.get('extreme_weather_risk', 0):.1%}")
            
            if "conservation_strategy" in results:
                print(f"\n🎯 CONSERVATION RECOMMENDATIONS:")
                strategy = results["conservation_strategy"]
                for i, rec in enumerate(strategy.get('recommendations', [])[:3], 1):
                    print(f"   {i}. {rec}")
            
            if "alerts" in results:
                print(f"\n⚠️  ALERTS:")
                for alert in results["alerts"]:
                    print(f"   {alert['severity'].upper()}: {alert['message']}")
            
            print("\n" + "="*60)
            
        elif args.mode == "monitoring":
            # Real-time monitoring
            regions = ["amazon", "congo_basin", "boreal_forest", "great_barrier_reef"]
            await eco_vision.real_time_monitoring(regions)
            
        elif args.mode == "training":
            # Model training mode
            print("Starting model training pipeline...")
            await eco_vision.satellite_analyzer.train_models()
            await eco_vision.climate_predictor.train_models()
            await eco_vision.conservation_optimizer.train_agent()
            
        elif args.mode == "api":
            # API server mode
            from api.server import create_app
            app = create_app(eco_vision)
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())