"""Configuration settings for AI Token Factory Economics Stack."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
import yaml
from enum import Enum


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class NeocloudConfig:
    """Configuration for neocloud providers."""
    # Default GPU specifications
    default_gpu_models: List[str] = field(default_factory=lambda: [
        "NVIDIA H100", "NVIDIA A100", "NVIDIA V100", "NVIDIA RTX 4090"
    ])
    
    # Power and cooling settings
    power_cost_per_kwh: float = 0.12  # USD per kWh
    cooling_efficiency_ratio: float = 0.3  # Cooling power as ratio of compute power
    power_usage_effectiveness: float = 1.4  # PUE ratio
    
    # Data center settings
    datacenter_locations: List[str] = field(default_factory=lambda: [
        "US-East", "US-West", "EU-Central", "Asia-Pacific"
    ])
    
    # Economic parameters
    gpu_rental_markup: float = 2.0  # Markup over cost
    utilization_target: float = 85.0  # Target utilization percentage
    depreciation_years: int = 4  # GPU depreciation period
    
    # Operational costs
    staff_cost_per_gpu_per_month: float = 15.0
    maintenance_cost_percentage: float = 5.0  # Percentage of hardware cost
    insurance_cost_percentage: float = 2.0
    
    # Capacity planning
    min_cluster_size: int = 8
    max_cluster_size: int = 1024
    cluster_growth_rate: float = 0.15  # Monthly growth rate


@dataclass
class InferenceConfig:
    """Configuration for inference providers."""
    # Model specifications
    supported_model_sizes: List[str] = field(default_factory=lambda: [
        "small", "medium", "large", "xlarge"
    ])
    
    # Pricing strategy
    base_input_price_per_1k: float = 0.0015  # Base price for input tokens
    base_output_price_per_1k: float = 0.002   # Base price for output tokens
    price_scaling_factor: Dict[str, float] = field(default_factory=lambda: {
        "small": 1.0,
        "medium": 2.5,
        "large": 5.0,
        "xlarge": 10.0
    })
    
    # Performance targets
    target_response_time_ms: float = 500.0
    target_throughput_tokens_per_second: float = 1000.0
    target_uptime_percentage: float = 99.9
    
    # Cost structure
    gpu_cost_percentage: float = 70.0  # Percentage of revenue going to GPU costs
    infrastructure_cost_percentage: float = 15.0
    operational_cost_percentage: float = 10.0
    profit_margin_target: float = 5.0
    
    # Scaling parameters
    auto_scaling_enabled: bool = True
    scale_up_threshold: float = 80.0  # Utilization percentage
    scale_down_threshold: float = 40.0
    min_instances: int = 2
    max_instances: int = 100
    
    # Token economics
    context_length_multiplier: Dict[int, float] = field(default_factory=lambda: {
        4096: 1.0,
        8192: 1.2,
        16384: 1.5,
        32768: 2.0,
        65536: 3.0
    })


@dataclass
class ApplicationConfig:
    """Configuration for applications."""
    # Application types
    supported_app_types: List[str] = field(default_factory=lambda: [
        "code_editor", "chatbot", "content_generator", "data_analyst", "design_tool"
    ])
    
    # User segments
    user_segments: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "individual": {
            "monthly_fee": 20.0,
            "token_allowance": 100000,
            "overage_rate": 0.01
        },
        "team": {
            "monthly_fee": 100.0,
            "token_allowance": 1000000,
            "overage_rate": 0.008
        },
        "enterprise": {
            "monthly_fee": 500.0,
            "token_allowance": 10000000,
            "overage_rate": 0.005
        }
    })
    
    # Productivity metrics
    productivity_targets: Dict[str, float] = field(default_factory=lambda: {
        "time_saved_hours_per_month": 20.0,
        "productivity_increase_percentage": 25.0,
        "task_completion_rate": 85.0,
        "user_satisfaction_score": 8.5
    })
    
    # Growth parameters
    user_acquisition_cost: float = 50.0
    monthly_churn_rate: float = 5.0  # Percentage
    viral_coefficient: float = 0.3  # New users per existing user
    
    # Feature usage
    feature_adoption_rates: Dict[str, float] = field(default_factory=lambda: {
        "basic_features": 95.0,
        "advanced_features": 60.0,
        "premium_features": 25.0
    })
    
    # Token usage patterns
    usage_patterns: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "daily_active": {"tokens_per_session": 1000, "sessions_per_day": 3},
        "weekly_active": {"tokens_per_session": 2000, "sessions_per_week": 5},
        "monthly_active": {"tokens_per_session": 5000, "sessions_per_month": 8}
    })


@dataclass
class MarketConfig:
    """Configuration for market data and benchmarks."""
    # Market size and growth
    total_addressable_market: float = 150_000_000_000  # $150B
    serviceable_addressable_market: float = 45_000_000_000  # $45B
    annual_growth_rate: float = 35.0  # Percentage
    
    # Industry benchmarks
    industry_benchmarks: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "neocloud": {
            "average_gpu_utilization": 75.0,
            "average_gross_margin": 18.0,
            "average_power_efficiency": 1.3
        },
        "inference": {
            "average_response_time_ms": 800.0,
            "average_gross_margin": 22.0,
            "average_uptime": 99.5
        },
        "application": {
            "average_monthly_churn": 8.0,
            "average_gross_margin": 75.0,
            "average_user_satisfaction": 7.8
        }
    })
    
    # Competitive landscape
    market_leaders: Dict[str, List[str]] = field(default_factory=lambda: {
        "neocloud": ["AWS", "Google Cloud", "Microsoft Azure", "Lambda Labs"],
        "inference": ["OpenAI", "Anthropic", "Cohere", "Hugging Face"],
        "application": ["GitHub Copilot", "ChatGPT", "Jasper", "Copy.ai"]
    })
    
    # Economic indicators
    inflation_rate: float = 3.2
    interest_rate: float = 5.25
    currency_exchange_rates: Dict[str, float] = field(default_factory=lambda: {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0
    })
    
    # Technology trends
    technology_adoption_curves: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "transformer_models": {"current_adoption": 85.0, "growth_rate": 15.0},
        "edge_inference": {"current_adoption": 25.0, "growth_rate": 45.0},
        "multimodal_ai": {"current_adoption": 40.0, "growth_rate": 60.0}
    })


@dataclass
class SimulationConfig:
    """Main configuration for the AI Token Factory simulation."""
    # Simulation parameters
    simulation_duration_months: int = 12
    time_step_days: int = 1
    monte_carlo_iterations: int = 1000
    random_seed: Optional[int] = 42
    
    # Output settings
    output_directory: str = "output"
    save_intermediate_results: bool = True
    generate_charts: bool = True
    chart_format: str = "png"  # png, svg, pdf
    
    # Logging configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_file_path: str = "simulation.log"
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: Optional[int] = None  # None = auto-detect
    memory_limit_gb: Optional[float] = None
    
    # Layer configurations
    neocloud: NeocloudConfig = field(default_factory=NeocloudConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    
    # Validation settings
    validate_inputs: bool = True
    strict_validation: bool = False
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    include_raw_data: bool = False
    compress_output: bool = True


def load_config(file_path: Union[str, Path], 
               config_format: Optional[ConfigFormat] = None) -> SimulationConfig:
    """Load configuration from file.
    
    Args:
        file_path: Path to configuration file
        config_format: Format of the configuration file (auto-detected if None)
    
    Returns:
        SimulationConfig object
    
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        ValueError: If configuration format is unsupported
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    # Auto-detect format if not specified
    if config_format is None:
        suffix = path.suffix.lower()
        if suffix == '.json':
            config_format = ConfigFormat.JSON
        elif suffix in ['.yaml', '.yml']:
            config_format = ConfigFormat.YAML
        elif suffix == '.toml':
            config_format = ConfigFormat.TOML
        else:
            raise ValueError(f"Cannot auto-detect configuration format for file: {file_path}")
    
    # Load configuration data
    with open(path, 'r', encoding='utf-8') as f:
        if config_format == ConfigFormat.JSON:
            config_data = json.load(f)
        elif config_format == ConfigFormat.YAML:
            config_data = yaml.safe_load(f)
        elif config_format == ConfigFormat.TOML:
            import toml
            config_data = toml.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_format}")
    
    # Convert to SimulationConfig object
    return _dict_to_config(config_data)


def save_config(config: SimulationConfig, 
               file_path: Union[str, Path],
               config_format: ConfigFormat = ConfigFormat.JSON) -> None:
    """Save configuration to file.
    
    Args:
        config: SimulationConfig object to save
        file_path: Path where to save the configuration
        config_format: Format for the configuration file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary
    config_data = _config_to_dict(config)
    
    # Save configuration data
    with open(path, 'w', encoding='utf-8') as f:
        if config_format == ConfigFormat.JSON:
            json.dump(config_data, f, indent=2, default=str)
        elif config_format == ConfigFormat.YAML:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif config_format == ConfigFormat.TOML:
            import toml
            toml.dump(config_data, f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_format}")


def get_default_config() -> SimulationConfig:
    """Get default configuration.
    
    Returns:
        SimulationConfig with default values
    """
    return SimulationConfig()


def validate_config(config: SimulationConfig) -> Dict[str, List[str]]:
    """Validate configuration settings.
    
    Args:
        config: Configuration to validate
    
    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    # Validate simulation parameters
    if config.simulation_duration_months <= 0:
        errors.append("Simulation duration must be positive")
    
    if config.time_step_days <= 0:
        errors.append("Time step must be positive")
    
    if config.monte_carlo_iterations <= 0:
        errors.append("Monte Carlo iterations must be positive")
    
    # Validate neocloud configuration
    if config.neocloud.power_cost_per_kwh <= 0:
        errors.append("Power cost per kWh must be positive")
    
    if config.neocloud.utilization_target <= 0 or config.neocloud.utilization_target > 100:
        errors.append("Utilization target must be between 0 and 100")
    
    if config.neocloud.gpu_rental_markup <= 1:
        warnings.append("GPU rental markup is very low, may not be profitable")
    
    # Validate inference configuration
    if config.inference.target_response_time_ms <= 0:
        errors.append("Target response time must be positive")
    
    if config.inference.target_uptime_percentage <= 0 or config.inference.target_uptime_percentage > 100:
        errors.append("Target uptime must be between 0 and 100")
    
    total_cost_percentage = (
        config.inference.gpu_cost_percentage +
        config.inference.infrastructure_cost_percentage +
        config.inference.operational_cost_percentage +
        config.inference.profit_margin_target
    )
    
    if abs(total_cost_percentage - 100.0) > 1.0:
        warnings.append(f"Inference cost percentages sum to {total_cost_percentage}%, not 100%")
    
    # Validate application configuration
    if config.application.user_acquisition_cost <= 0:
        errors.append("User acquisition cost must be positive")
    
    if config.application.monthly_churn_rate < 0 or config.application.monthly_churn_rate > 100:
        errors.append("Monthly churn rate must be between 0 and 100")
    
    # Validate market configuration
    if config.market.total_addressable_market <= 0:
        errors.append("Total addressable market must be positive")
    
    if config.market.serviceable_addressable_market > config.market.total_addressable_market:
        warnings.append("Serviceable addressable market is larger than total addressable market")
    
    return {
        'errors': errors,
        'warnings': warnings
    }


def _dict_to_config(config_data: Dict[str, Any]) -> SimulationConfig:
    """Convert dictionary to SimulationConfig object."""
    # Extract nested configurations
    neocloud_data = config_data.get('neocloud', {})
    inference_data = config_data.get('inference', {})
    application_data = config_data.get('application', {})
    market_data = config_data.get('market', {})
    
    # Create nested config objects
    neocloud_config = NeocloudConfig(**neocloud_data)
    inference_config = InferenceConfig(**inference_data)
    application_config = ApplicationConfig(**application_data)
    market_config = MarketConfig(**market_data)
    
    # Create main config object
    main_config_data = {k: v for k, v in config_data.items() 
                       if k not in ['neocloud', 'inference', 'application', 'market']}
    
    return SimulationConfig(
        neocloud=neocloud_config,
        inference=inference_config,
        application=application_config,
        market=market_config,
        **main_config_data
    )


def _config_to_dict(config: SimulationConfig) -> Dict[str, Any]:
    """Convert SimulationConfig object to dictionary."""
    from dataclasses import asdict
    return asdict(config)


# Configuration presets
DEVELOPMENT_CONFIG = SimulationConfig(
    simulation_duration_months=3,
    monte_carlo_iterations=100,
    log_level="DEBUG",
    generate_charts=True
)

PRODUCTION_CONFIG = SimulationConfig(
    simulation_duration_months=24,
    monte_carlo_iterations=10000,
    log_level="INFO",
    parallel_processing=True,
    compress_output=True
)

TESTING_CONFIG = SimulationConfig(
    simulation_duration_months=1,
    monte_carlo_iterations=10,
    log_level="WARNING",
    save_intermediate_results=False,
    generate_charts=False
)