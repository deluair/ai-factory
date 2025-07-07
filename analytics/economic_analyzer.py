"""Economic analyzer for the AI Token Factory stack."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from models.base_models import EconomicLayer
from utils.formatters import format_currency, format_percentage


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric_name: str
    current_value: float
    trend_direction: str  # 'up', 'down', 'stable'
    trend_magnitude: float  # percentage change
    confidence_score: float  # 0-1


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    layer: str
    category: str
    description: str
    potential_impact: float
    implementation_difficulty: str  # 'low', 'medium', 'high'
    priority_score: float  # 0-100


class EconomicAnalyzer:
    """Advanced economic analysis for the AI Token Factory."""
    
    def __init__(self):
        self.historical_data = []
        self.benchmarks = self._load_industry_benchmarks()
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load industry benchmarks for comparison."""
        return {
            "neoclouds": {
                "gross_margin_pct": 25.0,
                "utilization_pct": 75.0,
                "efficiency_ratio": 1.8,
                "power_efficiency_pue": 1.3
            },
            "inference_providers": {
                "gross_margin_pct": 60.0,
                "utilization_pct": 80.0,
                "efficiency_ratio": 3.2,
                "cost_per_1k_tokens": 0.015
            },
            "applications": {
                "gross_margin_pct": 75.0,
                "utilization_pct": 45.0,
                "efficiency_ratio": 8.5,
                "revenue_per_user": 25.0
            }
        }
    
    def analyze_layer_performance(self, layer: EconomicLayer, layer_type: str) -> Dict[str, Any]:
        """Analyze performance of a specific layer against benchmarks."""
        summary = layer.economic_summary()
        benchmarks = self.benchmarks.get(layer_type, {})
        
        performance_analysis = {
            "metrics": summary,
            "benchmark_comparison": {},
            "performance_score": 0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }
        
        # Compare against benchmarks
        score_components = []
        
        for metric, benchmark_value in benchmarks.items():
            if metric in summary:
                actual_value = summary[metric]
                
                # Calculate performance ratio
                if benchmark_value != 0:
                    performance_ratio = actual_value / benchmark_value
                    
                    # Determine if higher is better based on metric type
                    higher_is_better = metric in ['gross_margin_pct', 'utilization_pct', 'efficiency_ratio', 'revenue_per_user']
                    
                    if higher_is_better:
                        score = min(performance_ratio * 100, 150)  # Cap at 150%
                        status = "above" if performance_ratio > 1.0 else "below"
                    else:
                        score = min((2 - performance_ratio) * 100, 150) if performance_ratio <= 2 else 0
                        status = "below" if performance_ratio > 1.0 else "above"
                    
                    performance_analysis["benchmark_comparison"][metric] = {
                        "actual": actual_value,
                        "benchmark": benchmark_value,
                        "ratio": performance_ratio,
                        "status": status,
                        "score": score
                    }
                    
                    score_components.append(score)
                    
                    # Categorize as strength or weakness
                    if (higher_is_better and performance_ratio > 1.1) or (not higher_is_better and performance_ratio < 0.9):
                        performance_analysis["strengths"].append(f"{metric}: {status} benchmark by {abs(performance_ratio - 1) * 100:.1f}%")
                    elif (higher_is_better and performance_ratio < 0.9) or (not higher_is_better and performance_ratio > 1.1):
                        performance_analysis["weaknesses"].append(f"{metric}: {status} benchmark by {abs(performance_ratio - 1) * 100:.1f}%")
        
        # Calculate overall performance score
        if score_components:
            performance_analysis["performance_score"] = np.mean(score_components)
        
        # Generate recommendations based on weaknesses
        performance_analysis["recommendations"] = self._generate_layer_recommendations(
            layer_type, performance_analysis["weaknesses"], summary
        )
        
        return performance_analysis
    
    def _generate_layer_recommendations(self, layer_type: str, weaknesses: List[str], metrics: Dict[str, float]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on identified weaknesses."""
        recommendations = []
        
        if layer_type == "neoclouds":
            if any("utilization" in weakness for weakness in weaknesses):
                recommendations.append(OptimizationRecommendation(
                    layer="neoclouds",
                    category="utilization",
                    description="Implement dynamic workload balancing and predictive scaling to improve GPU utilization",
                    potential_impact=metrics.get('total_revenue', 0) * 0.15,
                    implementation_difficulty="medium",
                    priority_score=85
                ))
            
            if any("margin" in weakness for weakness in weaknesses):
                recommendations.append(OptimizationRecommendation(
                    layer="neoclouds",
                    category="cost_optimization",
                    description="Optimize power efficiency and cooling systems to reduce operational costs",
                    potential_impact=metrics.get('total_cost', 0) * 0.12,
                    implementation_difficulty="high",
                    priority_score=75
                ))
        
        elif layer_type == "inference_providers":
            if any("utilization" in weakness for weakness in weaknesses):
                recommendations.append(OptimizationRecommendation(
                    layer="inference_providers",
                    category="capacity_optimization",
                    description="Implement intelligent request routing and model caching to improve compute utilization",
                    potential_impact=metrics.get('total_revenue', 0) * 0.20,
                    implementation_difficulty="medium",
                    priority_score=90
                ))
            
            if any("efficiency" in weakness for weakness in weaknesses):
                recommendations.append(OptimizationRecommendation(
                    layer="inference_providers",
                    category="model_optimization",
                    description="Deploy model quantization and optimization techniques to reduce token generation costs",
                    potential_impact=metrics.get('total_cost', 0) * 0.25,
                    implementation_difficulty="high",
                    priority_score=80
                ))
        
        elif layer_type == "applications":
            if any("utilization" in weakness for weakness in weaknesses):
                recommendations.append(OptimizationRecommendation(
                    layer="applications",
                    category="user_engagement",
                    description="Improve user onboarding and feature discovery to increase daily active usage",
                    potential_impact=metrics.get('total_revenue', 0) * 0.18,
                    implementation_difficulty="low",
                    priority_score=95
                ))
            
            if any("margin" in weakness for weakness in weaknesses):
                recommendations.append(OptimizationRecommendation(
                    layer="applications",
                    category="token_efficiency",
                    description="Optimize prompt engineering and implement response caching to reduce token costs",
                    potential_impact=metrics.get('total_cost', 0) * 0.30,
                    implementation_difficulty="medium",
                    priority_score=85
                ))
        
        return recommendations
    
    def analyze_value_flow(self, neoclouds: List, inference_providers: List, applications: List) -> Dict[str, Any]:
        """Analyze value flow through the entire stack."""
        
        # Calculate value at each stage
        raw_input_cost = sum(nc.cost_structure.total_cost() for nc in neoclouds)
        gpu_cluster_value = sum(nc.revenue_model.total_revenue() for nc in neoclouds)
        
        token_generation_cost = sum(ip.cost_structure.total_cost() for ip in inference_providers)
        token_value = sum(ip.revenue_model.total_revenue() for ip in inference_providers)
        
        application_cost = sum(app.cost_structure.total_cost() for app in applications)
        productivity_value = sum(
            app.productivity_impact_analysis()['estimated_productivity_value_usd']
            for app in applications
        )
        
        # Calculate value multipliers at each stage
        stage1_multiplier = gpu_cluster_value / raw_input_cost if raw_input_cost > 0 else 0
        stage2_multiplier = token_value / gpu_cluster_value if gpu_cluster_value > 0 else 0
        stage3_multiplier = productivity_value / token_value if token_value > 0 else 0
        total_multiplier = productivity_value / raw_input_cost if raw_input_cost > 0 else 0
        
        # Identify bottlenecks
        bottlenecks = []
        if stage1_multiplier < 1.5:
            bottlenecks.append("Neocloud layer: Low value creation from raw materials to GPU clusters")
        if stage2_multiplier < 2.0:
            bottlenecks.append("Inference layer: Inefficient conversion of compute to tokens")
        if stage3_multiplier < 3.0:
            bottlenecks.append("Application layer: Limited productivity value from tokens")
        
        return {
            "value_stages": {
                "raw_materials": raw_input_cost,
                "gpu_clusters": gpu_cluster_value,
                "ai_tokens": token_value,
                "productivity": productivity_value
            },
            "stage_multipliers": {
                "materials_to_clusters": stage1_multiplier,
                "clusters_to_tokens": stage2_multiplier,
                "tokens_to_productivity": stage3_multiplier,
                "total_stack": total_multiplier
            },
            "efficiency_metrics": {
                "cost_efficiency": productivity_value / (raw_input_cost + token_generation_cost + application_cost),
                "value_density": productivity_value / token_value if token_value > 0 else 0,
                "stack_leverage": total_multiplier
            },
            "bottlenecks": bottlenecks,
            "optimization_potential": self._calculate_optimization_potential(
                stage1_multiplier, stage2_multiplier, stage3_multiplier
            )
        }
    
    def _calculate_optimization_potential(self, stage1: float, stage2: float, stage3: float) -> Dict[str, float]:
        """Calculate optimization potential for each stage."""
        # Target multipliers based on industry best practices
        target_multipliers = {"stage1": 2.0, "stage2": 3.0, "stage3": 4.0}
        
        return {
            "neocloud_potential": max(0, (target_multipliers["stage1"] - stage1) / target_multipliers["stage1"] * 100),
            "inference_potential": max(0, (target_multipliers["stage2"] - stage2) / target_multipliers["stage2"] * 100),
            "application_potential": max(0, (target_multipliers["stage3"] - stage3) / target_multipliers["stage3"] * 100)
        }
    
    def calculate_market_position(self, layer_metrics: Dict[str, Any], layer_type: str) -> Dict[str, Any]:
        """Calculate market position and competitive analysis."""
        benchmarks = self.benchmarks.get(layer_type, {})
        
        # Calculate percentile rankings
        percentile_rankings = {}
        for metric, value in layer_metrics.items():
            if metric in benchmarks:
                benchmark = benchmarks[metric]
                # Simplified percentile calculation (in real scenario, use market data)
                if value >= benchmark * 1.2:
                    percentile = 90
                elif value >= benchmark * 1.1:
                    percentile = 75
                elif value >= benchmark * 0.9:
                    percentile = 50
                elif value >= benchmark * 0.8:
                    percentile = 25
                else:
                    percentile = 10
                
                percentile_rankings[metric] = percentile
        
        # Calculate overall market position
        if percentile_rankings:
            overall_percentile = np.mean(list(percentile_rankings.values()))
        else:
            overall_percentile = 50
        
        # Determine market position category
        if overall_percentile >= 80:
            position = "Market Leader"
        elif overall_percentile >= 60:
            position = "Strong Performer"
        elif overall_percentile >= 40:
            position = "Market Average"
        elif overall_percentile >= 20:
            position = "Below Average"
        else:
            position = "Underperformer"
        
        return {
            "overall_percentile": overall_percentile,
            "market_position": position,
            "metric_rankings": percentile_rankings,
            "competitive_advantages": self._identify_competitive_advantages(percentile_rankings),
            "improvement_areas": self._identify_improvement_areas(percentile_rankings)
        }
    
    def _identify_competitive_advantages(self, rankings: Dict[str, float]) -> List[str]:
        """Identify competitive advantages based on high-performing metrics."""
        advantages = []
        for metric, percentile in rankings.items():
            if percentile >= 75:
                advantages.append(f"Strong {metric.replace('_', ' ')}: {percentile}th percentile")
        return advantages
    
    def _identify_improvement_areas(self, rankings: Dict[str, float]) -> List[str]:
        """Identify areas needing improvement based on low-performing metrics."""
        improvements = []
        for metric, percentile in rankings.items():
            if percentile <= 25:
                improvements.append(f"Improve {metric.replace('_', ' ')}: {percentile}th percentile")
        return improvements
    
    def generate_executive_insights(self, full_analysis: Dict[str, Any]) -> List[str]:
        """Generate high-level executive insights from the analysis."""
        insights = []
        
        # Value flow insights
        value_flow = full_analysis.get('value_flow', {})
        if value_flow:
            total_multiplier = value_flow.get('stage_multipliers', {}).get('total_stack', 0)
            if total_multiplier > 10:
                insights.append(f"🚀 Exceptional value creation: {total_multiplier:.1f}x return on raw material investment")
            elif total_multiplier > 5:
                insights.append(f"📈 Strong value creation: {total_multiplier:.1f}x return on raw material investment")
            else:
                insights.append(f"⚠️ Value creation opportunity: Only {total_multiplier:.1f}x return on raw material investment")
        
        # Performance insights
        layer_performances = []
        for layer_type in ['neoclouds', 'inference_providers', 'applications']:
            layer_data = full_analysis.get(layer_type, {})
            if layer_data and 'performance_score' in layer_data:
                layer_performances.append((layer_type, layer_data['performance_score']))
        
        if layer_performances:
            best_layer = max(layer_performances, key=lambda x: x[1])
            worst_layer = min(layer_performances, key=lambda x: x[1])
            
            insights.append(f"🏆 Best performing layer: {best_layer[0].replace('_', ' ').title()} ({best_layer[1]:.0f}/100)")
            if worst_layer[1] < 70:
                insights.append(f"🎯 Focus area: {worst_layer[0].replace('_', ' ').title()} needs improvement ({worst_layer[1]:.0f}/100)")
        
        # Market position insights
        market_leaders = []
        for layer_type in ['neoclouds', 'inference_providers', 'applications']:
            layer_data = full_analysis.get(layer_type, {})
            market_data = layer_data.get('market_position', {})
            if market_data.get('market_position') == 'Market Leader':
                market_leaders.append(layer_type.replace('_', ' ').title())
        
        if market_leaders:
            insights.append(f"🥇 Market leadership in: {', '.join(market_leaders)}")
        
        return insights