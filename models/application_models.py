"""Application layer models for the AI Token Factory."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from .base_models import EconomicLayer, CostStructure, RevenueModel, CostItem, RevenueStream, CostType, Currency


class ApplicationType(Enum):
    """Types of AI applications."""
    CODE_EDITOR = "code_editor"
    CHATBOT = "chatbot"
    CONTENT_GENERATOR = "content_generator"
    DATA_ANALYST = "data_analyst"
    DESIGN_TOOL = "design_tool"
    PRODUCTIVITY_SUITE = "productivity_suite"


class PricingModel(Enum):
    """Application pricing models."""
    FREEMIUM = "freemium"
    SUBSCRIPTION = "subscription"
    PAY_PER_USE = "pay_per_use"
    ENTERPRISE = "enterprise"


@dataclass
class UserSegment:
    """User segment definition."""
    name: str
    user_count: int
    avg_monthly_revenue_per_user: float
    avg_tokens_per_user_per_month: int
    churn_rate_monthly: float  # 0.0 to 1.0
    acquisition_cost: float


@dataclass
class ProductivityMetrics:
    """Productivity metrics for applications."""
    time_saved_hours_per_user_per_month: float
    tasks_completed_per_user_per_month: int
    user_satisfaction_score: float  # 1.0 to 10.0
    feature_adoption_rate: float  # 0.0 to 1.0
    daily_active_users_pct: float  # 0.0 to 1.0





@dataclass
class UserMetrics:
    """User engagement and behavior metrics."""
    monthly_active_users: int
    daily_active_users: int
    average_session_duration_minutes: float
    sessions_per_user_per_month: int
    retention_rate_30_day: float  # 0.0 to 1.0
    net_promoter_score: float  # -100 to 100


@dataclass
class TokenUsagePattern:
    """Token usage patterns for the application."""
    avg_tokens_per_request: int
    requests_per_user_per_day: int
    peak_usage_multiplier: float  # Peak vs average usage
    token_cost_per_1000: float  # Cost from inference provider


@dataclass
class TokenUsagePattern:
    """Token usage patterns for the application."""
    avg_tokens_per_request: int
    requests_per_user_per_day: int
    peak_usage_multiplier: float  # Peak vs average usage
    token_cost_per_1000: float  # Cost from inference provider

class Application(EconomicLayer):
    """Application layer economic model."""
    
    def __init__(self, name: str, app_type: ApplicationType, pricing_model: PricingModel,
                 user_segments: List[UserSegment], user_metrics: UserMetrics,
                 productivity_metrics: ProductivityMetrics, token_usage: TokenUsagePattern):
        self.app_type = app_type
        self.pricing_model = pricing_model
        self.user_segments = user_segments
        self.user_metrics = user_metrics
        self.productivity_metrics = productivity_metrics
        self.token_usage = token_usage
        
        # Calculate cost structure and revenue
        cost_structure = self._calculate_costs()
        revenue_model = self._calculate_revenue()
        
        super().__init__(name, cost_structure, revenue_model)
    
    def _calculate_costs(self) -> CostStructure:
        """Calculate comprehensive cost structure."""
        costs = []
        
        # Token costs (primary variable cost)
        total_users = sum(segment.user_count for segment in self.user_segments)
        monthly_tokens = total_users * self.token_usage.requests_per_user_per_day * 30 * self.token_usage.avg_tokens_per_request
        monthly_token_cost = (monthly_tokens / 1000) * self.token_usage.token_cost_per_1000
        
        costs.append(CostItem(
            "AI Token Costs",
            monthly_token_cost,
            Currency.USD_PER_MONTH,
            CostType.VARIABLE,
            f"{monthly_tokens:,} tokens at ${self.token_usage.token_cost_per_1000}/1K"
        ))
        
        # Platform and infrastructure (estimated as 20% of token costs)
        platform_cost = monthly_token_cost * 0.20
        costs.append(CostItem(
            "Platform Infrastructure",
            platform_cost,
            Currency.USD_PER_MONTH,
            CostType.OPERATIONAL,
            "Hosting, CDN, databases, monitoring"
        ))
        
        # Product development (estimated based on user base)
        dev_cost = total_users * 0.50  # $0.50 per user for development
        costs.append(CostItem(
            "Product Development",
            dev_cost,
            Currency.USD_PER_MONTH,
            CostType.OPERATIONAL,
            "Engineering, design, product management"
        ))
        
        # Customer acquisition costs
        total_acquisition_cost = sum(
            segment.user_count * segment.churn_rate_monthly * segment.acquisition_cost
            for segment in self.user_segments
        )
        costs.append(CostItem(
            "Customer Acquisition",
            total_acquisition_cost,
            Currency.USD_PER_MONTH,
            CostType.VARIABLE,
            "Marketing, sales, user acquisition"
        ))
        
        # Customer support (estimated as 5% of revenue)
        total_revenue = sum(
            segment.user_count * segment.avg_monthly_revenue_per_user
            for segment in self.user_segments
        )
        support_cost = total_revenue * 0.05
        costs.append(CostItem(
            "Customer Support",
            support_cost,
            Currency.USD_PER_MONTH,
            CostType.OPERATIONAL,
            "Support staff, documentation, training"
        ))
        
        # General & Administrative (estimated as 15% of revenue)
        admin_cost = total_revenue * 0.15
        costs.append(CostItem(
            "General & Administrative",
            admin_cost,
            Currency.USD_PER_MONTH,
            CostType.FIXED,
            "Legal, accounting, management, office"
        ))
        
        return CostStructure(costs)
    
    def _calculate_revenue(self) -> RevenueModel:
        """Calculate revenue model based on user segments."""
        streams = []
        
        for segment in self.user_segments:
            monthly_revenue = segment.user_count * segment.avg_monthly_revenue_per_user
            
            streams.append(RevenueStream(
                f"Revenue - {segment.name}",
                monthly_revenue,
                Currency.USD_PER_MONTH,
                f"{segment.user_count:,} users at ${segment.avg_monthly_revenue_per_user:.2f}/month"
            ))
        
        return RevenueModel(streams)
    
    def calculate_utilization(self) -> float:
        """Calculate user engagement utilization."""
        if self.user_metrics.monthly_active_users == 0:
            return 0
        
        # Use DAU/MAU ratio as a proxy for utilization
        dau_mau_ratio = (self.user_metrics.daily_active_users / self.user_metrics.monthly_active_users) * 100
        
        # Adjust by feature adoption rate
        utilization = dau_mau_ratio * self.productivity_metrics.feature_adoption_rate
        
        return min(utilization, 100)  # Cap at 100%
    
    def optimize_costs(self) -> Dict[str, float]:
        """Return cost optimization recommendations."""
        recommendations = {}
        
        # Token usage optimization
        current_token_efficiency = self._calculate_token_efficiency()
        if current_token_efficiency < 0.7:  # Below 70% efficiency
            potential_savings = self._calculate_token_optimization_savings()
            recommendations['token_optimization'] = potential_savings
        
        # User acquisition optimization
        high_churn_segments = [
            segment for segment in self.user_segments 
            if segment.churn_rate_monthly > 0.05  # Above 5% monthly churn
        ]
        
        if high_churn_segments:
            potential_savings = sum(
                segment.user_count * segment.churn_rate_monthly * segment.acquisition_cost * 0.5
                for segment in high_churn_segments
            )
            recommendations['churn_reduction'] = potential_savings
        
        # Feature adoption improvements
        if self.productivity_metrics.feature_adoption_rate < 0.6:
            potential_revenue_increase = self._calculate_adoption_revenue_impact()
            recommendations['feature_adoption'] = potential_revenue_increase
        
        return recommendations
    
    def _calculate_token_efficiency(self) -> float:
        """Calculate token usage efficiency."""
        # Simplified efficiency metric based on productivity vs token usage
        productivity_score = (
            self.productivity_metrics.time_saved_hours_per_user_per_month * 10 +  # Value time highly
            self.productivity_metrics.tasks_completed_per_user_per_month +
            self.productivity_metrics.user_satisfaction_score
        )
        
        tokens_per_user = self.token_usage.requests_per_user_per_day * 30 * self.token_usage.avg_tokens_per_request
        
        # Normalize to 0-1 scale
        efficiency = min(productivity_score / (tokens_per_user / 1000), 1.0)
        return efficiency
    
    def _calculate_token_optimization_savings(self) -> float:
        """Calculate potential savings from token optimization."""
        total_users = sum(segment.user_count for segment in self.user_segments)
        current_monthly_tokens = total_users * self.token_usage.requests_per_user_per_day * 30 * self.token_usage.avg_tokens_per_request
        current_token_cost = (current_monthly_tokens / 1000) * self.token_usage.token_cost_per_1000
        
        # Assume 20% reduction possible through optimization
        return current_token_cost * 0.20
    
    def _calculate_adoption_revenue_impact(self) -> float:
        """Calculate potential revenue increase from better feature adoption."""
        total_revenue = sum(
            segment.user_count * segment.avg_monthly_revenue_per_user
            for segment in self.user_segments
        )
        
        # Assume 15% revenue increase possible with better adoption
        return total_revenue * 0.15
    
    def user_economics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get user economics summary by segment."""
        summary = {}
        
        for segment in self.user_segments:
            monthly_tokens_per_user = self.token_usage.requests_per_user_per_day * 30 * self.token_usage.avg_tokens_per_request
            token_cost_per_user = (monthly_tokens_per_user / 1000) * self.token_usage.token_cost_per_1000
            
            summary[segment.name] = {
                'user_count': segment.user_count,
                'monthly_revenue_per_user': segment.avg_monthly_revenue_per_user,
                'token_cost_per_user': token_cost_per_user,
                'gross_margin_per_user': segment.avg_monthly_revenue_per_user - token_cost_per_user,
                'churn_rate_monthly_pct': segment.churn_rate_monthly * 100,
                'acquisition_cost': segment.acquisition_cost,
                'ltv_cac_ratio': self._calculate_ltv_cac_ratio(segment),
                'tokens_per_user_per_month': monthly_tokens_per_user
            }
        
        return summary
    
    def _calculate_ltv_cac_ratio(self, segment: UserSegment) -> float:
        """Calculate Lifetime Value to Customer Acquisition Cost ratio."""
        if segment.churn_rate_monthly == 0 or segment.acquisition_cost == 0:
            return 0
        
        # Simple LTV calculation: monthly revenue / churn rate
        ltv = segment.avg_monthly_revenue_per_user / segment.churn_rate_monthly
        return ltv / segment.acquisition_cost
    
    def productivity_impact_analysis(self) -> Dict[str, float]:
        """Analyze productivity impact and value creation."""
        total_users = sum(segment.user_count for segment in self.user_segments)
        
        return {
            'total_hours_saved_per_month': total_users * self.productivity_metrics.time_saved_hours_per_user_per_month,
            'total_tasks_completed_per_month': total_users * self.productivity_metrics.tasks_completed_per_user_per_month,
            'average_user_satisfaction': self.productivity_metrics.user_satisfaction_score,
            'feature_adoption_rate_pct': self.productivity_metrics.feature_adoption_rate * 100,
            'daily_active_users_pct': self.productivity_metrics.daily_active_users_pct * 100,
            'estimated_productivity_value_usd': total_users * self.productivity_metrics.time_saved_hours_per_user_per_month * 50,  # $50/hour value
            'roi_for_users': self._calculate_user_roi()
        }
    
    def _calculate_user_roi(self) -> float:
        """Calculate ROI for users using the application."""
        avg_revenue_per_user = sum(
            segment.user_count * segment.avg_monthly_revenue_per_user
            for segment in self.user_segments
        ) / sum(segment.user_count for segment in self.user_segments)
        
        # Estimate value created (time saved * hourly rate)
        value_created = self.productivity_metrics.time_saved_hours_per_user_per_month * 50  # $50/hour
        
        if avg_revenue_per_user == 0:
            return 0
        
        return (value_created - avg_revenue_per_user) / avg_revenue_per_user * 100