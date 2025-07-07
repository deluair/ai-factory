"""Base economic models for the AI Token Factory."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from enum import Enum


class CostType(Enum):
    """Types of costs in the economic model."""
    CAPITAL = "capital"
    OPERATIONAL = "operational"
    VARIABLE = "variable"
    FIXED = "fixed"


class Currency(Enum):
    """Supported currencies."""
    USD = "USD"
    USD_PER_HOUR = "USD/hr"
    USD_PER_MONTH = "USD/month"
    USD_PER_TOKEN = "USD/token"
    USD_PER_GPU = "USD/GPU"
    USD_PER_WATT = "USD/Wh"


@dataclass
class CostItem:
    """Individual cost item."""
    name: str
    amount: float
    currency: Currency
    cost_type: CostType
    description: Optional[str] = None

    @property
    def monthly_cost(self) -> float:
        """Convert cost to monthly USD."""
        if self.currency == Currency.USD_PER_HOUR:
            return self.amount * 24 * 30  # Approximate month
        elif self.currency == Currency.USD_PER_MONTH:
            return self.amount
        elif self.currency == Currency.USD:
            return self.amount / 12  # Assume annual cost
        return self.amount


@dataclass
class CostStructure:
    """Cost structure for an economic layer."""
    costs: List[CostItem]
    
    def total_cost(self, cost_type: Optional[CostType] = None) -> float:
        """Calculate total cost, optionally filtered by type."""
        filtered_costs = self.costs
        if cost_type:
            filtered_costs = [c for c in self.costs if c.cost_type == cost_type]
        return sum(cost.monthly_cost for cost in filtered_costs)
    
    def cost_breakdown(self) -> Dict[CostType, float]:
        """Get cost breakdown by type."""
        breakdown = {}
        for cost_type in CostType:
            breakdown[cost_type] = self.total_cost(cost_type)
        return breakdown


@dataclass
class RevenueStream:
    """Individual revenue stream."""
    name: str
    amount: float
    currency: Currency
    description: Optional[str] = None

    @property
    def monthly_revenue(self) -> float:
        """Convert revenue to monthly USD."""
        if self.currency == Currency.USD_PER_HOUR:
            return self.amount * 24 * 30
        elif self.currency == Currency.USD_PER_MONTH:
            return self.amount
        elif self.currency == Currency.USD:
            return self.amount / 12
        return self.amount


@dataclass
class RevenueModel:
    """Revenue model for an economic layer."""
    streams: List[RevenueStream]
    
    def total_revenue(self) -> float:
        """Calculate total monthly revenue."""
        return sum(stream.monthly_revenue for stream in self.streams)
    
    def revenue_breakdown(self) -> Dict[str, float]:
        """Get revenue breakdown by stream."""
        return {stream.name: stream.monthly_revenue for stream in self.streams}


class EconomicLayer(ABC):
    """Abstract base class for economic layers."""
    
    def __init__(self, name: str, cost_structure: CostStructure, revenue_model: RevenueModel):
        self.name = name
        self.cost_structure = cost_structure
        self.revenue_model = revenue_model
    
    @property
    def gross_margin(self) -> float:
        """Calculate gross margin percentage."""
        revenue = self.revenue_model.total_revenue()
        if revenue == 0:
            return 0
        return ((revenue - self.cost_structure.total_cost()) / revenue) * 100
    
    @property
    def profit(self) -> float:
        """Calculate monthly profit."""
        return self.revenue_model.total_revenue() - self.cost_structure.total_cost()
    
    @property
    def efficiency_ratio(self) -> float:
        """Calculate efficiency as revenue per dollar of cost."""
        total_cost = self.cost_structure.total_cost()
        if total_cost == 0:
            return 0
        return self.revenue_model.total_revenue() / total_cost
    
    @abstractmethod
    def calculate_utilization(self) -> float:
        """Calculate utilization rate specific to the layer."""
        pass
    
    @abstractmethod
    def optimize_costs(self) -> Dict[str, float]:
        """Return cost optimization recommendations."""
        pass
    
    def economic_summary(self) -> Dict[str, float]:
        """Get comprehensive economic summary."""
        return {
            'total_revenue': self.revenue_model.total_revenue(),
            'total_cost': self.cost_structure.total_cost(),
            'profit': self.profit,
            'gross_margin_pct': self.gross_margin,
            'efficiency_ratio': self.efficiency_ratio,
            'utilization_pct': self.calculate_utilization()
        }