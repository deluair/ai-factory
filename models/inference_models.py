"""Inference provider layer models for the AI Token Factory."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from .base_models import EconomicLayer, CostStructure, RevenueModel, CostItem, RevenueStream, CostType, Currency


class ModelSize(Enum):
    """AI model size categories."""
    SMALL = "small"  # 7B parameters
    MEDIUM = "medium"  # 13B-30B parameters
    LARGE = "large"  # 70B+ parameters
    XLARGE = "xlarge"  # 175B+ parameters


@dataclass
class ModelSpec:
    """AI model specifications."""
    name: str
    size: ModelSize
    parameters_billions: float
    tokens_per_second_per_gpu: float
    memory_gb_required: int
    context_length: int


@dataclass
class TokenPricing:
    """Token pricing structure."""
    input_tokens_per_dollar: int
    output_tokens_per_dollar: int
    model_spec: ModelSpec
    
    @property
    def input_cost_per_token(self) -> float:
        """Cost per input token in USD."""
        return 1.0 / self.input_tokens_per_dollar if self.input_tokens_per_dollar > 0 else 0
    
    @property
    def output_cost_per_token(self) -> float:
        """Cost per output token in USD."""
        return 1.0 / self.output_tokens_per_dollar if self.output_tokens_per_dollar > 0 else 0


@dataclass
class ComputeUnit:
    """Compute unit for inference."""
    unit_id: str
    gpu_count: int
    gpu_type: str
    model_spec: ModelSpec
    utilization_rate: float  # 0.0 to 1.0
    hourly_gpu_cost: float  # Cost to rent GPUs from neocloud
    
    @property
    def tokens_per_hour(self) -> float:
        """Calculate tokens generated per hour."""
        return (self.gpu_count * 
                self.model_spec.tokens_per_second_per_gpu * 
                3600 * 
                self.utilization_rate)
    
    @property
    def hourly_compute_cost(self) -> float:
        """Calculate hourly compute cost."""
        return self.gpu_count * self.hourly_gpu_cost
    
    @property
    def cost_per_token(self) -> float:
        """Calculate cost per token generated."""
        tokens_per_hour = self.tokens_per_hour
        if tokens_per_hour == 0:
            return float('inf')
        return self.hourly_compute_cost / tokens_per_hour


@dataclass
class UsageMetrics:
    """Usage metrics for inference provider."""
    monthly_input_tokens: int
    monthly_output_tokens: int
    average_requests_per_second: float
    peak_requests_per_second: float
    average_response_time_ms: float


class InferenceProvider(EconomicLayer):
    """Inference provider economic model."""
    
    def __init__(self, name: str, compute_units: List[ComputeUnit], 
                 token_pricing: List[TokenPricing], usage_metrics: UsageMetrics):
        self.compute_units = compute_units
        self.token_pricing = token_pricing
        self.usage_metrics = usage_metrics
        
        # Calculate cost structure and revenue
        cost_structure = self._calculate_costs()
        revenue_model = self._calculate_revenue()
        
        super().__init__(name, cost_structure, revenue_model)
    
    def _calculate_costs(self) -> CostStructure:
        """Calculate comprehensive cost structure."""
        costs = []
        
        # Compute costs (GPU rental from neoclouds)
        total_monthly_compute_cost = 0
        for unit in self.compute_units:
            monthly_cost = unit.hourly_compute_cost * 24 * 30
            total_monthly_compute_cost += monthly_cost
            
            costs.append(CostItem(
                f"Compute - {unit.unit_id}",
                monthly_cost,
                Currency.USD_PER_MONTH,
                CostType.VARIABLE,
                f"{unit.gpu_count} x {unit.gpu_type} for {unit.model_spec.name}"
            ))
        
        # Infrastructure and platform costs (estimated as 15% of compute)
        platform_cost = total_monthly_compute_cost * 0.15
        costs.append(CostItem(
            "Platform Infrastructure",
            platform_cost,
            Currency.USD_PER_MONTH,
            CostType.OPERATIONAL,
            "API infrastructure, load balancers, monitoring"
        ))
        
        # Engineering and operations (estimated as 25% of compute)
        engineering_cost = total_monthly_compute_cost * 0.25
        costs.append(CostItem(
            "Engineering & Operations",
            engineering_cost,
            Currency.USD_PER_MONTH,
            CostType.OPERATIONAL,
            "ML engineers, DevOps, customer support"
        ))
        
        # Model licensing and R&D (estimated as 10% of compute)
        rd_cost = total_monthly_compute_cost * 0.10
        costs.append(CostItem(
            "R&D & Model Licensing",
            rd_cost,
            Currency.USD_PER_MONTH,
            CostType.OPERATIONAL,
            "Model development, licensing, research"
        ))
        
        return CostStructure(costs)
    
    def _calculate_revenue(self) -> RevenueModel:
        """Calculate revenue model based on token usage."""
        streams = []
        
        # Calculate revenue for each model/pricing tier
        for pricing in self.token_pricing:
            # Estimate token distribution (simplified)
            input_tokens = self.usage_metrics.monthly_input_tokens // len(self.token_pricing)
            output_tokens = self.usage_metrics.monthly_output_tokens // len(self.token_pricing)
            
            input_revenue = input_tokens * pricing.input_cost_per_token
            output_revenue = output_tokens * pricing.output_cost_per_token
            total_revenue = input_revenue + output_revenue
            
            streams.append(RevenueStream(
                f"Tokens - {pricing.model_spec.name}",
                total_revenue,
                Currency.USD_PER_MONTH,
                f"{input_tokens:,} input + {output_tokens:,} output tokens"
            ))
        
        return RevenueModel(streams)
    
    def calculate_utilization(self) -> float:
        """Calculate average utilization across all compute units."""
        if not self.compute_units:
            return 0
        
        total_gpus = sum(unit.gpu_count for unit in self.compute_units)
        weighted_utilization = sum(unit.gpu_count * unit.utilization_rate for unit in self.compute_units)
        
        return (weighted_utilization / total_gpus) * 100 if total_gpus > 0 else 0
    
    def optimize_costs(self) -> Dict[str, float]:
        """Return cost optimization recommendations."""
        recommendations = {}
        
        # Utilization optimization
        avg_utilization = self.calculate_utilization() / 100
        if avg_utilization < 0.85:
            # Calculate potential cost savings from better utilization
            underutilized_cost = sum(
                unit.hourly_compute_cost * 24 * 30 * (0.85 - unit.utilization_rate)
                for unit in self.compute_units 
                if unit.utilization_rate < 0.85
            )
            recommendations['utilization_improvement'] = underutilized_cost
        
        # Model efficiency optimization
        inefficient_units = [
            unit for unit in self.compute_units 
            if unit.cost_per_token > self._calculate_average_cost_per_token() * 1.2
        ]
        
        if inefficient_units:
            potential_savings = sum(
                unit.hourly_compute_cost * 24 * 30 * 0.2  # Assume 20% savings possible
                for unit in inefficient_units
            )
            recommendations['model_efficiency'] = potential_savings
        
        return recommendations
    
    def _calculate_average_cost_per_token(self) -> float:
        """Calculate average cost per token across all units."""
        if not self.compute_units:
            return 0
        
        total_cost = sum(unit.hourly_compute_cost for unit in self.compute_units)
        total_tokens = sum(unit.tokens_per_hour for unit in self.compute_units)
        
        return total_cost / total_tokens if total_tokens > 0 else 0
    
    def token_economics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get token economics summary."""
        summary = {}
        
        for pricing in self.token_pricing:
            model_name = pricing.model_spec.name
            
            # Find corresponding compute unit
            compute_unit = next(
                (unit for unit in self.compute_units if unit.model_spec.name == model_name),
                None
            )
            
            if compute_unit:
                summary[model_name] = {
                    'input_cost_per_token': pricing.input_cost_per_token * 1000000,  # per million tokens
                    'output_cost_per_token': pricing.output_cost_per_token * 1000000,  # per million tokens
                    'compute_cost_per_token': compute_unit.cost_per_token * 1000000,  # per million tokens
                    'tokens_per_second': compute_unit.model_spec.tokens_per_second_per_gpu * compute_unit.gpu_count,
                    'utilization_pct': compute_unit.utilization_rate * 100,
                    'gross_margin_pct': ((pricing.output_cost_per_token - compute_unit.cost_per_token) / 
                                       pricing.output_cost_per_token * 100) if pricing.output_cost_per_token > 0 else 0
                }
        
        return summary
    
    def capacity_analysis(self) -> Dict[str, float]:
        """Analyze capacity and performance metrics."""
        total_tokens_per_hour = sum(unit.tokens_per_hour for unit in self.compute_units)
        total_tokens_per_month = total_tokens_per_hour * 24 * 30
        
        actual_monthly_tokens = self.usage_metrics.monthly_input_tokens + self.usage_metrics.monthly_output_tokens
        capacity_utilization = (actual_monthly_tokens / total_tokens_per_month * 100) if total_tokens_per_month > 0 else 0
        
        return {
            'total_capacity_tokens_per_month': total_tokens_per_month,
            'actual_tokens_per_month': actual_monthly_tokens,
            'capacity_utilization_pct': capacity_utilization,
            'average_requests_per_second': self.usage_metrics.average_requests_per_second,
            'peak_requests_per_second': self.usage_metrics.peak_requests_per_second,
            'average_response_time_ms': self.usage_metrics.average_response_time_ms
        }