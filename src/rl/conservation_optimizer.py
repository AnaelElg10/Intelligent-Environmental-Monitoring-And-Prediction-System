"""
Conservation Optimization Module for EcoVision AI

Advanced reinforcement learning system for optimizing conservation strategies
and resource allocation to maximize environmental protection outcomes.

Key Features:
- Deep Q-Network (DQN) for conservation strategy optimization
- Multi-objective optimization with Pareto efficiency
- Real-time adaptation to environmental changes
- Policy evaluation and continuous improvement
- Resource allocation optimization across multiple conservation areas
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
import random
from datetime import datetime
import json
from pathlib import Path

from loguru import logger
from sklearn.preprocessing import StandardScaler
import gym
from gym import spaces

from .agents.dqn_agent import DQNAgent
from .agents.policy_gradient_agent import PolicyGradientAgent
from .environments.conservation_env import ConservationEnvironment
from .optimization.multi_objective_optimizer import MultiObjectiveOptimizer
from ..utils.reward_shaping import RewardShaper
from ..utils.experience_replay import PrioritizedExperienceReplay


# Experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ConservationOptimizer:
    """
    Advanced reinforcement learning system for conservation optimization.
    
    Uses Deep Q-Networks and policy optimization to learn optimal conservation
    strategies, resource allocation, and intervention timing for maximum
    environmental protection impact.
    """
    
    def __init__(self, config: Dict):
        """Initialize the conservation optimizer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # RL Environment
        self.env = ConservationEnvironment(config.get("environment", {}))
        
        # RL Agents
        self.dqn_agent = None
        self.policy_agent = None
        
        # Multi-objective optimization
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        
        # Experience replay and utilities
        self.memory = PrioritizedExperienceReplay(config.get("memory_size", 100000))
        self.reward_shaper = RewardShaper()
        
        # Training parameters
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.epsilon = config.get("epsilon", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        
        # Performance tracking
        self.training_history = []
        self.optimization_results = []
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"ConservationOptimizer initialized on {self.device}")
    
    def _initialize_agents(self):
        """Initialize RL agents."""
        try:
            # State and action space dimensions
            state_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
            
            # DQN Agent for discrete action optimization
            self.dqn_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.learning_rate,
                device=self.device
            )
            
            # Policy Gradient Agent for continuous action spaces
            self.policy_agent = PolicyGradientAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.learning_rate,
                device=self.device
            )
            
            # Load pre-trained weights if available
            self._load_pretrained_agents()
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            self._initialize_dummy_agents()
    
    def _load_pretrained_agents(self):
        """Load pre-trained agent weights."""
        try:
            # Load DQN weights
            dqn_path = "models/dqn_conservation_agent.pth"
            if Path(dqn_path).exists():
                self.dqn_agent.load_state_dict(
                    torch.load(dqn_path, map_location=self.device)
                )
                logger.info("Loaded DQN agent weights")
            
            # Load policy agent weights
            policy_path = "models/policy_conservation_agent.pth"
            if Path(policy_path).exists():
                self.policy_agent.load_state_dict(
                    torch.load(policy_path, map_location=self.device)
                )
                logger.info("Loaded policy agent weights")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained agents: {str(e)}")
    
    def _initialize_dummy_agents(self):
        """Initialize dummy agents for demonstration."""
        logger.warning("Initializing dummy agents for demonstration")
        
        class DummyAgent:
            def __init__(self, action_dim):
                self.action_dim = action_dim
            
            def select_action(self, state, epsilon=0.0):
                return np.random.randint(0, self.action_dim)
            
            def update(self, *args, **kwargs):
                pass
        
        self.dqn_agent = DummyAgent(self.env.action_space.n)
        self.policy_agent = DummyAgent(self.env.action_space.n)
    
    async def optimize_strategy(self, region: str, environmental_data: Dict) -> Dict:
        """
        Optimize conservation strategy for a specific region.
        
        Args:
            region: Geographic region identifier
            environmental_data: Current environmental conditions and analysis
            
        Returns:
            Dictionary containing optimized conservation strategy
        """
        logger.info(f"Optimizing conservation strategy for {region}")
        
        try:
            # Prepare environment state
            state = self._prepare_environment_state(region, environmental_data)
            
            # Generate action recommendations using different approaches
            strategies = {}
            
            # DQN-based strategy
            dqn_strategy = await self._optimize_with_dqn(state)
            strategies["dqn"] = dqn_strategy
            
            # Policy gradient strategy
            policy_strategy = await self._optimize_with_policy(state)
            strategies["policy"] = policy_strategy
            
            # Multi-objective optimization
            pareto_strategy = await self._optimize_multi_objective(state, environmental_data)
            strategies["pareto"] = pareto_strategy
            
            # Combine strategies into final recommendations
            final_strategy = await self._combine_strategies(strategies, environmental_data)
            
            # Evaluate strategy effectiveness
            effectiveness_score = self._evaluate_strategy_effectiveness(final_strategy, environmental_data)
            
            results = {
                "region": region,
                "timestamp": datetime.now().isoformat(),
                "primary_strategy": final_strategy,
                "alternative_strategies": strategies,
                "effectiveness_score": effectiveness_score,
                "recommendations": self._generate_action_recommendations(final_strategy),
                "resource_allocation": self._optimize_resource_allocation(final_strategy),
                "priority_actions": self._identify_priority_actions(final_strategy),
                "expected_outcomes": self._predict_strategy_outcomes(final_strategy, environmental_data)
            }
            
            # Store optimization results for learning
            self.optimization_results.append(results)
            
            logger.info(f"Conservation strategy optimization completed for {region}")
            return results
            
        except Exception as e:
            logger.error(f"Strategy optimization failed for {region}: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_environment_state(self, region: str, environmental_data: Dict) -> np.ndarray:
        """Prepare environment state vector for RL agents."""
        # Extract key environmental indicators
        features = []
        
        # Deforestation metrics
        if "deforestation_analysis" in environmental_data:
            deforestation = environmental_data["deforestation_analysis"]
            features.extend([
                deforestation.get("forest_cover", 0) / 100,
                deforestation.get("change_rate", 0),
                1.0 if deforestation.get("risk_level") == "high" else 
                0.5 if deforestation.get("risk_level") == "medium" else 0.0
            ])
        else:
            features.extend([0.5, 0.0, 0.0])  # Default values
        
        # Climate indicators
        if "climate_forecast" in environmental_data:
            climate = environmental_data["climate_forecast"]
            features.extend([
                climate.get("extreme_weather_risk", 0),
                climate.get("confidence_level", 0.5)
            ])
        else:
            features.extend([0.0, 0.5])
        
        # Region-specific features
        region_encoding = self._encode_region(region)
        features.extend(region_encoding)
        
        # Time-based features
        now = datetime.now()
        features.extend([
            now.month / 12.0,  # Seasonal factor
            now.hour / 24.0    # Time of day factor
        ])
        
        # Normalize to appropriate range for RL
        state = np.array(features, dtype=np.float32)
        state = np.clip(state, -1.0, 1.0)
        
        return state
    
    def _encode_region(self, region: str) -> List[float]:
        """Encode region information as numerical features."""
        region_map = {
            "amazon": [1.0, 0.0, 0.0, 0.0],
            "congo_basin": [0.0, 1.0, 0.0, 0.0],
            "boreal_forest": [0.0, 0.0, 1.0, 0.0],
            "great_barrier_reef": [0.0, 0.0, 0.0, 1.0]
        }
        return region_map.get(region.lower(), [0.0, 0.0, 0.0, 0.0])
    
    async def _optimize_with_dqn(self, state: np.ndarray) -> Dict:
        """Optimize using Deep Q-Network agent."""
        try:
            # Get action from DQN agent
            action = self.dqn_agent.select_action(state, epsilon=0.0)  # Greedy selection
            
            # Convert action to conservation strategy
            strategy = self._action_to_strategy(action)
            
            return {
                "method": "Deep Q-Network",
                "action_id": int(action),
                "strategy": strategy,
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.warning(f"DQN optimization failed: {str(e)}")
            return {
                "method": "DQN (fallback)",
                "action_id": 0,
                "strategy": self._get_default_strategy(),
                "confidence": 0.5
            }
    
    async def _optimize_with_policy(self, state: np.ndarray) -> Dict:
        """Optimize using policy gradient agent."""
        try:
            # Get action from policy agent
            action = self.policy_agent.select_action(state)
            
            # Convert action to conservation strategy
            strategy = self._action_to_strategy(action)
            
            return {
                "method": "Policy Gradient",
                "action_id": int(action),
                "strategy": strategy,
                "confidence": 0.80
            }
            
        except Exception as e:
            logger.warning(f"Policy optimization failed: {str(e)}")
            return {
                "method": "Policy (fallback)",
                "action_id": 1,
                "strategy": self._get_default_strategy(),
                "confidence": 0.5
            }
    
    async def _optimize_multi_objective(self, state: np.ndarray, environmental_data: Dict) -> Dict:
        """Optimize using multi-objective approach."""
        try:
            # Define objectives
            objectives = [
                "biodiversity_preservation",
                "carbon_sequestration",
                "cost_effectiveness",
                "implementation_feasibility"
            ]
            
            # Generate Pareto-optimal solutions
            pareto_solutions = self.multi_objective_optimizer.optimize(
                state, objectives, environmental_data
            )
            
            # Select best compromise solution
            best_solution = self._select_best_pareto_solution(pareto_solutions)
            
            return {
                "method": "Multi-Objective Optimization",
                "strategy": best_solution,
                "pareto_solutions": pareto_solutions[:3],  # Top 3 alternatives
                "confidence": 0.90
            }
            
        except Exception as e:
            logger.warning(f"Multi-objective optimization failed: {str(e)}")
            return {
                "method": "Multi-Objective (fallback)",
                "strategy": self._get_default_strategy(),
                "confidence": 0.5
            }
    
    def _action_to_strategy(self, action: int) -> Dict:
        """Convert RL action to conservation strategy."""
        # Define action space mapping
        action_mapping = {
            0: "monitoring_enhancement",
            1: "habitat_restoration",
            2: "protected_area_expansion",
            3: "community_engagement",
            4: "anti_poaching_patrol",
            5: "reforestation_program",
            6: "sustainable_tourism",
            7: "research_initiative",
            8: "emergency_intervention",
            9: "policy_advocacy"
        }
        
        primary_action = action_mapping.get(action % len(action_mapping), "monitoring_enhancement")
        
        # Generate detailed strategy based on action
        strategies = {
            "monitoring_enhancement": {
                "type": "monitoring",
                "priority": "high",
                "resources_required": "medium",
                "timeline": "immediate",
                "description": "Enhance monitoring systems with advanced sensors and AI analysis"
            },
            "habitat_restoration": {
                "type": "restoration",
                "priority": "high",
                "resources_required": "high",
                "timeline": "long_term",
                "description": "Implement comprehensive habitat restoration program"
            },
            "protected_area_expansion": {
                "type": "protection",
                "priority": "medium",
                "resources_required": "very_high",
                "timeline": "long_term",
                "description": "Expand protected area boundaries and improve enforcement"
            },
            "community_engagement": {
                "type": "social",
                "priority": "high",
                "resources_required": "medium",
                "timeline": "medium_term",
                "description": "Engage local communities in conservation efforts"
            },
            "reforestation_program": {
                "type": "restoration",
                "priority": "high",
                "resources_required": "high",
                "timeline": "long_term",
                "description": "Large-scale reforestation with native species"
            }
        }
        
        return strategies.get(primary_action, strategies["monitoring_enhancement"])
    
    def _get_default_strategy(self) -> Dict:
        """Get default conservation strategy."""
        return {
            "type": "monitoring",
            "priority": "medium",
            "resources_required": "low",
            "timeline": "immediate",
            "description": "Implement basic monitoring and assessment protocols"
        }
    
    async def _combine_strategies(self, strategies: Dict, environmental_data: Dict) -> Dict:
        """Combine multiple strategy recommendations into final strategy."""
        # Weight strategies based on confidence and environmental urgency
        weights = {
            "dqn": 0.3,
            "policy": 0.3,
            "pareto": 0.4
        }
        
        # Extract confidence scores
        confidences = {
            name: strategy.get("confidence", 0.5) 
            for name, strategy in strategies.items()
        }
        
        # Adjust weights based on confidence
        total_confidence = sum(confidences.values())
        if total_confidence > 0:
            for name in weights:
                if name in confidences:
                    weights[name] *= confidences[name] / (total_confidence / len(confidences))
        
        # Select highest-weighted strategy as primary
        primary_strategy_name = max(weights, key=weights.get)
        primary_strategy = strategies[primary_strategy_name]["strategy"]
        
        # Enhance with environmental urgency factors
        urgency_factor = self._calculate_urgency_factor(environmental_data)
        if urgency_factor > 0.7:
            primary_strategy["priority"] = "critical"
            primary_strategy["timeline"] = "immediate"
        
        return primary_strategy
    
    def _calculate_urgency_factor(self, environmental_data: Dict) -> float:
        """Calculate environmental urgency factor."""
        urgency_factors = []
        
        # Deforestation urgency
        if "deforestation_analysis" in environmental_data:
            deforestation = environmental_data["deforestation_analysis"]
            change_rate = deforestation.get("change_rate", 0)
            urgency_factors.append(min(1.0, change_rate * 10))
        
        # Climate urgency
        if "climate_forecast" in environmental_data:
            climate = environmental_data["climate_forecast"]
            weather_risk = climate.get("extreme_weather_risk", 0)
            urgency_factors.append(weather_risk)
        
        # Alert urgency
        if "alerts" in environmental_data:
            alert_count = len(environmental_data["alerts"])
            high_severity_alerts = sum(
                1 for alert in environmental_data["alerts"] 
                if alert.get("severity") == "high"
            )
            urgency_factors.append(min(1.0, (alert_count + high_severity_alerts * 2) / 5))
        
        return np.mean(urgency_factors) if urgency_factors else 0.3
    
    def _select_best_pareto_solution(self, pareto_solutions: List[Dict]) -> Dict:
        """Select best solution from Pareto frontier."""
        if not pareto_solutions:
            return self._get_default_strategy()
        
        # For now, select the first solution (most balanced)
        # In practice, this would involve more sophisticated selection criteria
        return pareto_solutions[0]
    
    def _evaluate_strategy_effectiveness(self, strategy: Dict, environmental_data: Dict) -> float:
        """Evaluate expected effectiveness of conservation strategy."""
        # Base effectiveness score
        effectiveness = 0.5
        
        # Strategy type bonuses
        type_bonuses = {
            "restoration": 0.3,
            "protection": 0.25,
            "monitoring": 0.15,
            "social": 0.2
        }
        
        strategy_type = strategy.get("type", "monitoring")
        effectiveness += type_bonuses.get(strategy_type, 0.1)
        
        # Priority adjustment
        priority_multipliers = {
            "critical": 1.2,
            "high": 1.1,
            "medium": 1.0,
            "low": 0.9
        }
        
        priority = strategy.get("priority", "medium")
        effectiveness *= priority_multipliers.get(priority, 1.0)
        
        # Environmental matching bonus
        if self._strategy_matches_environment(strategy, environmental_data):
            effectiveness += 0.15
        
        return min(1.0, max(0.0, effectiveness))
    
    def _strategy_matches_environment(self, strategy: Dict, environmental_data: Dict) -> bool:
        """Check if strategy matches environmental conditions."""
        # Simple matching logic
        if "deforestation_analysis" in environmental_data:
            deforestation = environmental_data["deforestation_analysis"]
            if deforestation.get("risk_level") == "high":
                return strategy.get("type") in ["restoration", "protection"]
        
        if "alerts" in environmental_data:
            if len(environmental_data["alerts"]) > 0:
                return strategy.get("priority") in ["high", "critical"]
        
        return True
    
    def _generate_action_recommendations(self, strategy: Dict) -> List[str]:
        """Generate specific action recommendations."""
        recommendations = []
        
        strategy_type = strategy.get("type", "monitoring")
        
        if strategy_type == "restoration":
            recommendations.extend([
                "Identify degraded areas suitable for restoration",
                "Source native plant species and seeds",
                "Engage local communities in restoration activities",
                "Establish monitoring protocols for restoration success"
            ])
        elif strategy_type == "protection":
            recommendations.extend([
                "Strengthen law enforcement in protected areas",
                "Install additional monitoring equipment",
                "Train local rangers and guards",
                "Implement community-based protection programs"
            ])
        elif strategy_type == "monitoring":
            recommendations.extend([
                "Deploy advanced satellite monitoring systems",
                "Establish ground-based sensor networks",
                "Train personnel in data analysis techniques",
                "Develop real-time alert systems"
            ])
        elif strategy_type == "social":
            recommendations.extend([
                "Conduct community awareness programs",
                "Develop sustainable livelihood alternatives",
                "Establish community conservation groups",
                "Provide conservation education and training"
            ])
        
        return recommendations
    
    def _optimize_resource_allocation(self, strategy: Dict) -> Dict:
        """Optimize resource allocation for the strategy."""
        base_budget = 100000  # Base budget in USD
        
        # Adjust budget based on strategy requirements
        resource_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 1.8,
            "very_high": 2.5
        }
        
        resources_required = strategy.get("resources_required", "medium")
        total_budget = base_budget * resource_multipliers.get(resources_required, 1.0)
        
        # Allocate budget across categories
        allocation = {
            "personnel": int(total_budget * 0.4),
            "equipment": int(total_budget * 0.25),
            "training": int(total_budget * 0.15),
            "community_programs": int(total_budget * 0.15),
            "contingency": int(total_budget * 0.05)
        }
        
        return {
            "total_budget_usd": int(total_budget),
            "allocation": allocation,
            "cost_per_hectare": int(total_budget / 1000),  # Assuming 1000 hectares
            "roi_estimate": self._calculate_roi_estimate(strategy)
        }
    
    def _calculate_roi_estimate(self, strategy: Dict) -> float:
        """Calculate estimated return on investment."""
        # Simple ROI calculation based on strategy type and priority
        base_roi = 2.0
        
        type_multipliers = {
            "restoration": 3.5,
            "protection": 2.8,
            "monitoring": 2.0,
            "social": 2.5
        }
        
        priority_multipliers = {
            "critical": 1.5,
            "high": 1.3,
            "medium": 1.0,
            "low": 0.8
        }
        
        strategy_type = strategy.get("type", "monitoring")
        priority = strategy.get("priority", "medium")
        
        roi = base_roi * type_multipliers.get(strategy_type, 2.0) * priority_multipliers.get(priority, 1.0)
        
        return round(roi, 2)
    
    def _identify_priority_actions(self, strategy: Dict) -> List[Dict]:
        """Identify priority actions for immediate implementation."""
        priority_actions = []
        
        strategy_type = strategy.get("type", "monitoring")
        
        if strategy_type == "restoration":
            priority_actions = [
                {
                    "action": "Site assessment and planning",
                    "timeline": "1-2 weeks",
                    "resources": "Survey team, GIS equipment",
                    "importance": "critical"
                },
                {
                    "action": "Community stakeholder engagement",
                    "timeline": "2-4 weeks",
                    "resources": "Community liaison team",
                    "importance": "high"
                },
                {
                    "action": "Seed and seedling procurement",
                    "timeline": "4-6 weeks",
                    "resources": "Nursery partnerships, transportation",
                    "importance": "high"
                }
            ]
        elif strategy_type == "protection":
            priority_actions = [
                {
                    "action": "Security assessment",
                    "timeline": "1 week",
                    "resources": "Security experts, local rangers",
                    "importance": "critical"
                },
                {
                    "action": "Equipment deployment",
                    "timeline": "2-3 weeks",
                    "resources": "Monitoring equipment, installation team",
                    "importance": "high"
                }
            ]
        elif strategy_type == "monitoring":
            priority_actions = [
                {
                    "action": "System design and procurement",
                    "timeline": "2-3 weeks",
                    "resources": "Technical team, equipment vendors",
                    "importance": "critical"
                },
                {
                    "action": "Installation and calibration",
                    "timeline": "3-4 weeks",
                    "resources": "Installation team, calibration equipment",
                    "importance": "high"
                }
            ]
        
        return priority_actions
    
    def _predict_strategy_outcomes(self, strategy: Dict, environmental_data: Dict) -> Dict:
        """Predict expected outcomes of the conservation strategy."""
        # Simulate outcome prediction based on strategy and environmental conditions
        strategy_type = strategy.get("type", "monitoring")
        
        outcomes = {
            "biodiversity_improvement": 0.0,
            "habitat_recovery": 0.0,
            "carbon_sequestration": 0.0,
            "community_benefit": 0.0,
            "cost_effectiveness": 0.0
        }
        
        # Strategy-specific outcome predictions
        if strategy_type == "restoration":
            outcomes.update({
                "biodiversity_improvement": 0.75,
                "habitat_recovery": 0.85,
                "carbon_sequestration": 0.80,
                "community_benefit": 0.60,
                "cost_effectiveness": 0.70
            })
        elif strategy_type == "protection":
            outcomes.update({
                "biodiversity_improvement": 0.80,
                "habitat_recovery": 0.70,
                "carbon_sequestration": 0.60,
                "community_benefit": 0.50,
                "cost_effectiveness": 0.75
            })
        elif strategy_type == "monitoring":
            outcomes.update({
                "biodiversity_improvement": 0.40,
                "habitat_recovery": 0.30,
                "carbon_sequestration": 0.20,
                "community_benefit": 0.70,
                "cost_effectiveness": 0.90
            })
        elif strategy_type == "social":
            outcomes.update({
                "biodiversity_improvement": 0.60,
                "habitat_recovery": 0.50,
                "carbon_sequestration": 0.40,
                "community_benefit": 0.90,
                "cost_effectiveness": 0.80
            })
        
        # Adjust based on environmental urgency
        urgency_factor = self._calculate_urgency_factor(environmental_data)
        if urgency_factor > 0.7:
            # High urgency may reduce effectiveness initially but improve long-term outcomes
            for key in outcomes:
                outcomes[key] *= (0.9 + urgency_factor * 0.2)
        
        return {
            "predicted_outcomes": outcomes,
            "confidence_interval": 0.15,
            "time_to_measurable_impact": "6-12 months",
            "long_term_sustainability": 0.75
        }
    
    async def train_agent(self):
        """Train the RL agents using reinforcement learning."""
        logger.info("Starting RL agent training")
        
        # Simulate training process
        training_episodes = self.config.get("training_episodes", 1000)
        
        for episode in range(min(10, training_episodes)):  # Limit for demo
            logger.info(f"Training episode {episode + 1}/{min(10, training_episodes)}")
            
            # Simulate episode
            state = self.env.reset()
            total_reward = 0
            
            for step in range(100):  # Max steps per episode
                # Select action
                action = self.dqn_agent.select_action(state, epsilon=self.epsilon)
                
                # Take step in environment
                next_state, reward, done, _ = self.env.step(action)
                
                # Store experience
                experience = Experience(state, action, reward, next_state, done)
                self.memory.push(experience)
                
                # Update agent
                if len(self.memory) > self.batch_size:
                    self.dqn_agent.update(self.memory.sample(self.batch_size))
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Log training progress
            self.training_history.append({
                "episode": episode,
                "total_reward": total_reward,
                "epsilon": self.epsilon
            })
            
            await asyncio.sleep(0.1)  # Simulate training time
        
        logger.info("RL agent training completed successfully")
    
    def get_agent_info(self) -> Dict:
        """Get information about RL agents."""
        return {
            "dqn_agent": {
                "type": "Deep Q-Network",
                "state_dim": getattr(self.env.observation_space, 'shape', [10])[0],
                "action_dim": getattr(self.env.action_space, 'n', 10),
                "epsilon": self.epsilon
            },
            "policy_agent": {
                "type": "Policy Gradient",
                "state_dim": getattr(self.env.observation_space, 'shape', [10])[0],
                "action_dim": getattr(self.env.action_space, 'n', 10)
            },
            "training_episodes": len(self.training_history),
            "optimization_runs": len(self.optimization_results)
        }