"""Realistic simulation data for the AI Token Factory Economics Stack."""

from typing import List
from models.neocloud_models import NeocloudProvider, GPUCluster, GPUSpec, DataCenterSpec
from models.inference_models import InferenceProvider, ComputeUnit, ModelSpec, TokenPricing, UsageMetrics, ModelSize
from models.application_models import Application, UserSegment, UserMetrics, ProductivityMetrics, TokenUsagePattern, ApplicationType, PricingModel


def create_sample_neoclouds() -> List[NeocloudProvider]:
    """Create sample neocloud providers with realistic data."""
    
    # GPU specifications based on real market data
    b100_spec = GPUSpec(
        model="NVIDIA B100",
        memory_gb=192,
        compute_units=24000, # Placeholder, actual core counts are complex
        power_watts=700,
        cost_usd=32500
    )

    gb200_spec = GPUSpec(
        model="NVIDIA GB200 Superchip",
        memory_gb=384, # 2x B200 GPUs
        compute_units=48000, # Placeholder
        power_watts=1200,
        cost_usd=65000
    )
    
    h100_spec = GPUSpec(
        model="NVIDIA H100",
        memory_gb=80,
        compute_units=16896,
        power_watts=700,
        cost_usd=30000
    )
    
    a100_spec = GPUSpec(
        model="NVIDIA A100",
        memory_gb=80, # 80GB is more common now
        compute_units=6912,
        power_watts=400,
        cost_usd=12000 # Price has come down
    )
    
    v100_spec = GPUSpec(
        model="NVIDIA V100",
        memory_gb=32,
        compute_units=5120,
        power_watts=300,
        cost_usd=8000
    )
    
    # Data center specifications
    aws_datacenter = DataCenterSpec(
        location="US-East-1 (Virginia)",
        power_capacity_mw=100,
        cooling_efficiency_pue=1.2,
        electricity_cost_per_kwh=0.087,
        water_cost_per_gallon=0.004,
        real_estate_cost_per_sqft=25,
        sqft_total=50000
    )
    
    azure_datacenter = DataCenterSpec(
        location="US-West-2 (Washington)",
        power_capacity_mw=80,
        cooling_efficiency_pue=1.15,
        electricity_cost_per_kwh=0.095,
        water_cost_per_gallon=0.003,
        real_estate_cost_per_sqft=30,
        sqft_total=40000
    )
    
    gcp_datacenter = DataCenterSpec(
        location="US-Central-1 (Iowa)",
        power_capacity_mw=120,
        cooling_efficiency_pue=1.1,
        electricity_cost_per_kwh=0.082,
        water_cost_per_gallon=0.0035,
        real_estate_cost_per_sqft=20,
        sqft_total=60000
    )
    
    # Create GPU clusters
    aws_clusters = [
        GPUCluster(
            cluster_id="aws-gb200-cluster-1",
            gpu_spec=gb200_spec,
            gpu_count=500, # These are deployed in racks
            utilization_rate=0.90,
            rental_price_per_gpu_hour=10.50 # Premium for new tech
        ),
        GPUCluster(
            cluster_id="aws-h100-cluster-1",
            gpu_spec=h100_spec,
            gpu_count=2000,
            utilization_rate=0.85,
            rental_price_per_gpu_hour=4.10 # slight price adjustment
        ),
        GPUCluster(
            cluster_id="aws-a100-cluster-1",
            gpu_spec=a100_spec,
            gpu_count=3000,
            utilization_rate=0.78,
            rental_price_per_gpu_hour=1.80
        )
    ]
    
    azure_clusters = [
        GPUCluster(
            cluster_id="azure-gb200-cluster-1",
            gpu_spec=gb200_spec,
            gpu_count=400,
            utilization_rate=0.88,
            rental_price_per_gpu_hour=10.20
        ),
        GPUCluster(
            cluster_id="azure-b100-cluster-1",
            gpu_spec=b100_spec,
            gpu_count=1000,
            utilization_rate=0.85,
            rental_price_per_gpu_hour=5.50
        ),
        GPUCluster(
            cluster_id="azure-h100-cluster-1",
            gpu_spec=h100_spec,
            gpu_count=1500,
            utilization_rate=0.82,
            rental_price_per_gpu_hour=4.00
        )
    ]
    
    gcp_clusters = [
        GPUCluster(
            cluster_id="gcp-b100-cluster-1",
            gpu_spec=b100_spec,
            gpu_count=1500,
            utilization_rate=0.92,
            rental_price_per_gpu_hour=5.80
        ),
        GPUCluster(
            cluster_id="gcp-h100-cluster-1",
            gpu_spec=h100_spec,
            gpu_count=2500,
            utilization_rate=0.88,
            rental_price_per_gpu_hour=3.90
        ),
        GPUCluster(
            cluster_id="gcp-a100-cluster-1",
            gpu_spec=a100_spec,
            gpu_count=4000,
            utilization_rate=0.80,
            rental_price_per_gpu_hour=1.75
        )
    ]
    
    # Create neocloud providers
    neoclouds = [
        NeocloudProvider("AWS Neocloud", aws_datacenter, aws_clusters),
        NeocloudProvider("Azure Neocloud", azure_datacenter, azure_clusters),
        NeocloudProvider("GCP Neocloud", gcp_datacenter, gcp_clusters)
    ]
    
    return neoclouds


def create_sample_inference_providers() -> List[InferenceProvider]:
    """Create sample inference providers with realistic data."""
    
    # Model specifications
    gpt5_spec = ModelSpec(
        name="GPT-5",
        size=ModelSize.XLARGE,
        parameters_billions=1800, # Trillion-parameter model
        tokens_per_second_per_gpu=150, # Higher throughput on new hardware
        memory_gb_required=300,
        context_length=128000
    )

    gpt4_spec = ModelSpec(
        name="GPT-4 Turbo",
        size=ModelSize.LARGE,
        parameters_billions=175,
        tokens_per_second_per_gpu=80,
        memory_gb_required=60,
        context_length=128000
    )
    
    claude_spec = ModelSpec(
        name="Claude-3",
        size=ModelSize.LARGE,
        parameters_billions=70,
        tokens_per_second_per_gpu=80,
        memory_gb_required=30,
        context_length=100000
    )
    
    llama_spec = ModelSpec(
        name="Llama-2-70B",
        size=ModelSize.LARGE,
        parameters_billions=70,
        tokens_per_second_per_gpu=75,
        memory_gb_required=35,
        context_length=4096
    )
    
    # OpenAI-like provider
    openai_compute_units = [
        ComputeUnit(
            unit_id="openai-gpt5-gb200-cluster",
            gpu_count=1000,
            gpu_type="GB200",
            model_spec=gpt5_spec,
            utilization_rate=0.95,
            hourly_gpu_cost=10.00
        ),
        ComputeUnit(
            unit_id="openai-gpt4-b100-cluster",
            gpu_count=2000,
            gpu_type="B100",
            model_spec=gpt4_spec,
            utilization_rate=0.92,
            hourly_gpu_cost=5.25
        )
    ]
    
    openai_pricing = [
        TokenPricing(
            input_tokens_per_dollar=100000,  # $0.01 per 1K tokens for GPT-4 Turbo
            output_tokens_per_dollar=33333,   # $0.03 per 1K tokens
            model_spec=gpt4_spec
        ),
        TokenPricing(
            input_tokens_per_dollar=50000, # $0.02 per 1K for GPT-5
            output_tokens_per_dollar=16667, # $0.06 per 1K
            model_spec=gpt5_spec
        )
    ]
    
    openai_usage = UsageMetrics(
        monthly_input_tokens=15000000000,  # 15B tokens
        monthly_output_tokens=5000000000,   # 5B tokens
        average_requests_per_second=2500,
        peak_requests_per_second=8000,
        average_response_time_ms=1200
    )
    
    # Anthropic-like provider
    anthropic_compute_units = [
        ComputeUnit(
            unit_id="anthropic-claude-b100-cluster",
            gpu_count=800,
            gpu_type="B100",
            model_spec=claude_spec,
            utilization_rate=0.90,
            hourly_gpu_cost=5.00
        )
    ]
    
    anthropic_pricing = [
        TokenPricing(
            input_tokens_per_dollar=125000,  # $0.008 per 1K tokens
            output_tokens_per_dollar=41667,   # $0.024 per 1K tokens
            model_spec=claude_spec
        )
    ]
    
    anthropic_usage = UsageMetrics(
        monthly_input_tokens=8000000000,   # 8B tokens
        monthly_output_tokens=2500000000,  # 2.5B tokens
        average_requests_per_second=1200,
        peak_requests_per_second=4000,
        average_response_time_ms=900
    )
    
    # Meta-like provider (open source)
    meta_compute_units = [
        ComputeUnit(
            unit_id="meta-llama3-b100-cluster",
            gpu_count=1200,
            gpu_type="B100",
            model_spec=llama_spec, # Assuming Llama 3 runs on B100s
            utilization_rate=0.88,
            hourly_gpu_cost=5.10
        )
    ]
    
    meta_pricing = [
        TokenPricing(
            input_tokens_per_dollar=200000,  # $0.005 per 1K tokens
            output_tokens_per_dollar=100000,  # $0.01 per 1K tokens
            model_spec=llama_spec
        )
    ]
    
    meta_usage = UsageMetrics(
        monthly_input_tokens=12000000000,  # 12B tokens
        monthly_output_tokens=4000000000,  # 4B tokens
        average_requests_per_second=1800,
        peak_requests_per_second=6000,
        average_response_time_ms=800
    )
    
    # Create inference providers
    providers = [
        InferenceProvider("OpenAI", openai_compute_units, openai_pricing, openai_usage),
        InferenceProvider("Anthropic", anthropic_compute_units, anthropic_pricing, anthropic_usage),
        InferenceProvider("Meta AI", meta_compute_units, meta_pricing, meta_usage)
    ]
    
    return providers


def create_sample_applications() -> List[Application]:
    """Create sample applications with realistic data."""
    
    # GitHub Copilot-like code editor
    copilot_segments = [
        UserSegment(
            name="Individual Developers",
            user_count=1500000,
            avg_monthly_revenue_per_user=10.0,
            avg_tokens_per_user_per_month=50000,
            churn_rate_monthly=0.03,
            acquisition_cost=25.0
        ),
        UserSegment(
            name="Enterprise Teams",
            user_count=200000,
            avg_monthly_revenue_per_user=39.0,
            avg_tokens_per_user_per_month=150000,
            churn_rate_monthly=0.015,
            acquisition_cost=150.0
        )
    ]
    
    copilot_user_metrics = UserMetrics(
        monthly_active_users=1700000,
        daily_active_users=850000,
        average_session_duration_minutes=45,
        sessions_per_user_per_month=22,
        retention_rate_30_day=0.85,
        net_promoter_score=65
    )
    
    copilot_productivity = ProductivityMetrics(
        time_saved_hours_per_user_per_month=8.5,
        tasks_completed_per_user_per_month=120,
        user_satisfaction_score=8.2,
        feature_adoption_rate=0.75,
        daily_active_users_pct=0.50
    )
    
    copilot_token_usage = TokenUsagePattern(
        avg_tokens_per_request=150,
        requests_per_user_per_day=25,
        peak_usage_multiplier=2.5,
        token_cost_per_1000=0.015
    )
    
    # ChatGPT-like chatbot
    chatgpt_segments = [
        UserSegment(
            name="Free Users",
            user_count=100000000,
            avg_monthly_revenue_per_user=0.0,
            avg_tokens_per_user_per_month=25000,
            churn_rate_monthly=0.15,
            acquisition_cost=2.0
        ),
        UserSegment(
            name="Plus Subscribers",
            user_count=5000000,
            avg_monthly_revenue_per_user=20.0,
            avg_tokens_per_user_per_month=100000,
            churn_rate_monthly=0.05,
            acquisition_cost=15.0
        ),
        UserSegment(
            name="Enterprise",
            user_count=50000,
            avg_monthly_revenue_per_user=500.0,
            avg_tokens_per_user_per_month=500000,
            churn_rate_monthly=0.02,
            acquisition_cost=2000.0
        )
    ]
    
    chatgpt_user_metrics = UserMetrics(
        monthly_active_users=105050000,
        daily_active_users=25000000,
        average_session_duration_minutes=12,
        sessions_per_user_per_month=15,
        retention_rate_30_day=0.65,
        net_promoter_score=55
    )
    
    chatgpt_productivity = ProductivityMetrics(
        time_saved_hours_per_user_per_month=3.2,
        tasks_completed_per_user_per_month=45,
        user_satisfaction_score=7.8,
        feature_adoption_rate=0.60,
        daily_active_users_pct=0.24
    )
    
    chatgpt_token_usage = TokenUsagePattern(
        avg_tokens_per_request=200,
        requests_per_user_per_day=8,
        peak_usage_multiplier=3.0,
        token_cost_per_1000=0.025
    )
    
    # Jasper-like content generator
    jasper_segments = [
        UserSegment(
            name="Freelancers",
            user_count=150000,
            avg_monthly_revenue_per_user=49.0,
            avg_tokens_per_user_per_month=80000,
            churn_rate_monthly=0.08,
            acquisition_cost=75.0
        ),
        UserSegment(
            name="Small Business",
            user_count=80000,
            avg_monthly_revenue_per_user=99.0,
            avg_tokens_per_user_per_month=150000,
            churn_rate_monthly=0.06,
            acquisition_cost=200.0
        ),
        UserSegment(
            name="Enterprise",
            user_count=15000,
            avg_monthly_revenue_per_user=499.0,
            avg_tokens_per_user_per_month=400000,
            churn_rate_monthly=0.03,
            acquisition_cost=1500.0
        )
    ]
    
    jasper_user_metrics = UserMetrics(
        monthly_active_users=245000,
        daily_active_users=98000,
        average_session_duration_minutes=25,
        sessions_per_user_per_month=18,
        retention_rate_30_day=0.78,
        net_promoter_score=72
    )
    
    jasper_productivity = ProductivityMetrics(
        time_saved_hours_per_user_per_month=12.0,
        tasks_completed_per_user_per_month=85,
        user_satisfaction_score=8.5,
        feature_adoption_rate=0.82,
        daily_active_users_pct=0.40
    )
    
    jasper_token_usage = TokenUsagePattern(
        avg_tokens_per_request=300,
        requests_per_user_per_day=15,
        peak_usage_multiplier=2.0,
        token_cost_per_1000=0.020
    )
    
    # Create applications
    applications = [
        Application(
            "AI Code Assistant",
            ApplicationType.CODE_EDITOR,
            PricingModel.SUBSCRIPTION,
            copilot_segments,
            copilot_user_metrics,
            copilot_productivity,
            copilot_token_usage
        ),
        Application(
            "AI Chatbot Platform",
            ApplicationType.CHATBOT,
            PricingModel.FREEMIUM,
            chatgpt_segments,
            chatgpt_user_metrics,
            chatgpt_productivity,
            chatgpt_token_usage
        ),
        Application(
            "AI Content Generator",
            ApplicationType.CONTENT_GENERATOR,
            PricingModel.SUBSCRIPTION,
            jasper_segments,
            jasper_user_metrics,
            jasper_productivity,
            jasper_token_usage
        )
    ]
    
    return applications


def load_market_data() -> dict:
    """Load market data and industry benchmarks."""
    return {
        "gpu_market": {
            "h100_availability": 0.65,  # Supply constraint
            "a100_availability": 0.85,
            "price_trend_monthly_pct": -2.5,  # Prices declining
            "demand_growth_yoy_pct": 180
        },
        "inference_market": {
            "total_tokens_per_month_billions": 500,
            "average_price_per_1k_tokens": 0.018,
            "growth_rate_monthly_pct": 15,
            "competition_intensity": 0.85
        },
        "application_market": {
            "total_ai_app_users_millions": 250,
            "average_revenue_per_user": 45,
            "market_penetration_pct": 12,
            "user_growth_monthly_pct": 8.5
        },
        "economic_indicators": {
            "electricity_price_trend_pct": 3.2,
            "data_center_capacity_utilization": 0.78,
            "venture_funding_ai_billions": 25.2,
            "enterprise_ai_adoption_pct": 35
        }
    }