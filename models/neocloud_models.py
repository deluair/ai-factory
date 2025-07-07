"""Neocloud layer models for the AI Token Factory."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from .base_models import EconomicLayer, CostStructure, RevenueModel, CostItem, RevenueStream, CostType, Currency


@dataclass
class GPUSpec:
    """GPU specifications."""
    model: str
    memory_gb: int
    compute_units: int
    power_watts: int
    cost_usd: float


@dataclass
class DataCenterSpec:
    """Data center specifications."""
    location: str
    power_capacity_mw: float
    cooling_efficiency_pue: float  # Power Usage Effectiveness
    electricity_cost_per_kwh: float
    water_cost_per_gallon: float
    real_estate_cost_per_sqft: float
    sqft_total: int


@dataclass
class GPUCluster:
    """GPU cluster configuration."""
    cluster_id: str
    gpu_spec: GPUSpec
    gpu_count: int
    utilization_rate: float  # 0.0 to 1.0
    rental_price_per_gpu_hour: float
    
    @property
    def total_power_consumption_kw(self) -> float:
        """Calculate total power consumption in kW."""
        return (self.gpu_count * self.gpu_spec.power_watts * self.utilization_rate) / 1000
    
    @property
    def monthly_revenue_potential(self) -> float:
        """Calculate maximum monthly revenue if fully utilized."""
        hours_per_month = 24 * 30
        return self.gpu_count * self.rental_price_per_gpu_hour * hours_per_month
    
    @property
    def actual_monthly_revenue(self) -> float:
        """Calculate actual monthly revenue based on utilization."""
        return self.monthly_revenue_potential * self.utilization_rate


class NeocloudProvider(EconomicLayer):
    """Neocloud provider economic model."""
    
    def __init__(self, name: str, datacenter: DataCenterSpec, clusters: List[GPUCluster]):
        self.datacenter = datacenter
        self.clusters = clusters
        
        # Calculate cost structure
        cost_structure = self._calculate_costs()
        revenue_model = self._calculate_revenue()
        
        super().__init__(name, cost_structure, revenue_model)
    
    def _calculate_costs(self) -> CostStructure:
        """Calculate comprehensive cost structure."""
        costs = []
        
        # Capital costs
        total_gpu_cost = sum(cluster.gpu_count * cluster.gpu_spec.cost_usd for cluster in self.clusters)
        costs.append(CostItem(
            "GPU Hardware", total_gpu_cost, Currency.USD, CostType.CAPITAL,
            "Initial GPU hardware investment"
        ))
        
        # Data center infrastructure (estimated as 2x GPU cost)
        datacenter_infra_cost = total_gpu_cost * 2
        costs.append(CostItem(
            "Datacenter Infrastructure", datacenter_infra_cost, Currency.USD, CostType.CAPITAL,
            "Servers, networking, cooling, power infrastructure"
        ))
        
        # Operational costs
        # Electricity
        total_power_kw = sum(cluster.total_power_consumption_kw for cluster in self.clusters)
        # Include PUE for cooling overhead
        total_power_with_pue = total_power_kw * self.datacenter.cooling_efficiency_pue
        monthly_electricity_cost = total_power_with_pue * 24 * 30 * self.datacenter.electricity_cost_per_kwh
        
        costs.append(CostItem(
            "Electricity", monthly_electricity_cost, Currency.USD_PER_MONTH, CostType.OPERATIONAL,
            f"Power consumption: {total_power_with_pue:.1f} kW with PUE {self.datacenter.cooling_efficiency_pue}"
        ))
        
        # Cooling water (estimated)
        monthly_water_cost = total_power_kw * 0.5 * 24 * 30 * self.datacenter.water_cost_per_gallon  # 0.5 gal/kWh estimate
        costs.append(CostItem(
            "Cooling Water", monthly_water_cost, Currency.USD_PER_MONTH, CostType.OPERATIONAL,
            "Water for cooling systems"
        ))
        
        # Real estate
        monthly_real_estate = self.datacenter.sqft_total * self.datacenter.real_estate_cost_per_sqft / 12
        costs.append(CostItem(
            "Real Estate", monthly_real_estate, Currency.USD_PER_MONTH, CostType.FIXED,
            "Data center facility costs"
        ))
        
        # Staff and maintenance (estimated as 5% of capital costs monthly)
        monthly_staff_maintenance = (total_gpu_cost + datacenter_infra_cost) * 0.05 / 12
        costs.append(CostItem(
            "Staff & Maintenance", monthly_staff_maintenance, Currency.USD_PER_MONTH, CostType.OPERATIONAL,
            "Technical staff, maintenance, support"
        ))
        
        return CostStructure(costs)
    
    def _calculate_revenue(self) -> RevenueModel:
        """Calculate revenue model."""
        streams = []
        
        for cluster in self.clusters:
            streams.append(RevenueStream(
                f"GPU Rental - {cluster.cluster_id}",
                cluster.actual_monthly_revenue,
                Currency.USD_PER_MONTH,
                f"{cluster.gpu_count} x {cluster.gpu_spec.model} at {cluster.utilization_rate*100:.1f}% utilization"
            ))
        
        return RevenueModel(streams)
    
    def calculate_utilization(self) -> float:
        """Calculate average utilization across all clusters."""
        if not self.clusters:
            return 0
        
        total_gpus = sum(cluster.gpu_count for cluster in self.clusters)
        weighted_utilization = sum(cluster.gpu_count * cluster.utilization_rate for cluster in self.clusters)
        
        return (weighted_utilization / total_gpus) * 100 if total_gpus > 0 else 0
    
    def optimize_costs(self) -> Dict[str, float]:
        """Return cost optimization recommendations."""
        recommendations = {}
        
        # Power efficiency recommendations
        current_pue = self.datacenter.cooling_efficiency_pue
        if current_pue > 1.3:
            potential_savings = self._calculate_power_savings(1.2)
            recommendations['cooling_optimization'] = potential_savings
        
        # Utilization improvements
        avg_utilization = self.calculate_utilization() / 100
        if avg_utilization < 0.8:
            potential_revenue_increase = sum(cluster.monthly_revenue_potential * (0.8 - cluster.utilization_rate) 
                                           for cluster in self.clusters if cluster.utilization_rate < 0.8)
            recommendations['utilization_improvement'] = potential_revenue_increase
        
        return recommendations
    
    def _calculate_power_savings(self, target_pue: float) -> float:
        """Calculate potential power cost savings from PUE improvement."""
        current_pue = self.datacenter.cooling_efficiency_pue
        if target_pue >= current_pue:
            return 0
        
        base_power = sum(cluster.total_power_consumption_kw for cluster in self.clusters)
        current_total_power = base_power * current_pue
        target_total_power = base_power * target_pue
        
        power_savings_kw = current_total_power - target_total_power
        monthly_savings = power_savings_kw * 24 * 30 * self.datacenter.electricity_cost_per_kwh
        
        return monthly_savings
    
    def cluster_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for each cluster."""
        summary = {}
        
        for cluster in self.clusters:
            summary[cluster.cluster_id] = {
                'gpu_count': cluster.gpu_count,
                'utilization_pct': cluster.utilization_rate * 100,
                'monthly_revenue': cluster.actual_monthly_revenue,
                'revenue_potential': cluster.monthly_revenue_potential,
                'power_consumption_kw': cluster.total_power_consumption_kw,
                'revenue_per_gpu': cluster.actual_monthly_revenue / cluster.gpu_count if cluster.gpu_count > 0 else 0
            }
        
        return summary