"""Report generation for AI Token Factory Economics Stack."""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jinja2 import Template

from models.base_models import EconomicLayer
from models.neocloud_models import NeocloudProvider
from models.inference_models import InferenceProvider
from models.application_models import Application
from utils.formatters import (
    format_currency, format_percentage, format_number, 
    format_ratio, format_duration, format_bytes
)
from analytics.economic_analyzer import EconomicAnalyzer, TrendAnalysis, OptimizationRecommendation


class ReportGenerator:
    """Generate comprehensive reports for AI Token Factory analysis."""
    
    def __init__(self, output_dir: Union[str, Path] = "reports"):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_executive_report(self,
                                neoclouds: List[NeocloudProvider],
                                inference_providers: List[InferenceProvider],
                                applications: List[Application],
                                market_data: Dict[str, Any],
                                output_format: str = "html") -> Path:
        """Generate executive summary report.
        
        Args:
            neoclouds: List of neocloud providers
            inference_providers: List of inference providers
            applications: List of applications
            market_data: Market data and benchmarks
            output_format: Output format ('html', 'pdf', 'json')
        
        Returns:
            Path to generated report
        """
        analyzer = EconomicAnalyzer()
        
        # Calculate key metrics
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_executive_summary(neoclouds, inference_providers, applications),
            'layer_analysis': {
                'neoclouds': self._analyze_layer(neoclouds, 'Neoclouds'),
                'inference_providers': self._analyze_layer(inference_providers, 'Inference Providers'),
                'applications': self._analyze_layer(applications, 'Applications')
            },
            'stack_analysis': self._analyze_full_stack(neoclouds, inference_providers, applications),
            'market_position': analyzer.calculate_market_position(neoclouds + inference_providers + applications, market_data),
            'recommendations': [asdict(rec) for rec in analyzer._generate_layer_recommendations(
                neoclouds + inference_providers + applications
            )[:10]],
            'trends': self._analyze_trends(neoclouds, inference_providers, applications)
        }
        
        # Generate report based on format
        if output_format.lower() == 'html':
            return self._generate_html_report(report_data, 'executive_report')
        elif output_format.lower() == 'json':
            return self._generate_json_report(report_data, 'executive_report')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_layer_report(self,
                            layer_data: List[EconomicLayer],
                            layer_name: str,
                            market_data: Dict[str, Any],
                            output_format: str = "html") -> Path:
        """Generate detailed layer analysis report.
        
        Args:
            layer_data: List of economic layers to analyze
            layer_name: Name of the layer
            market_data: Market data and benchmarks
            output_format: Output format ('html', 'json')
        
        Returns:
            Path to generated report
        """
        analyzer = EconomicAnalyzer()
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'layer_name': layer_name,
            'overview': self._analyze_layer(layer_data, layer_name),
            'detailed_analysis': self._generate_detailed_layer_analysis(layer_data),
            'benchmarks': analyzer.analyze_layer_performance(layer_data, market_data),
            'optimization': [asdict(rec) for rec in analyzer._generate_layer_recommendations(layer_data)],
            'charts': self._generate_layer_charts(layer_data, layer_name)
        }
        
        if output_format.lower() == 'html':
            return self._generate_html_report(report_data, f'{layer_name.lower().replace(" ", "_")}_report')
        elif output_format.lower() == 'json':
            return self._generate_json_report(report_data, f'{layer_name.lower().replace(" ", "_")}_report')
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def generate_financial_report(self,
                                neoclouds: List[NeocloudProvider],
                                inference_providers: List[InferenceProvider],
                                applications: List[Application],
                                output_format: str = "csv") -> Path:
        """Generate financial analysis report.
        
        Args:
            neoclouds: List of neocloud providers
            inference_providers: List of inference providers
            applications: List of applications
            output_format: Output format ('csv', 'excel', 'json')
        
        Returns:
            Path to generated report
        """
        financial_data = []
        
        # Collect financial data from all layers
        for layer_list, layer_type in [(neoclouds, 'Neocloud'), 
                                      (inference_providers, 'Inference'), 
                                      (applications, 'Application')]:
            for item in layer_list:
                total_costs = item.total_cost()
                revenue = item.total_revenue()
                cost_structure = item.cost_structure
                
                financial_data.append({
                    'Layer': layer_type,
                    'Provider/App': item.name,
                    'Revenue': revenue,
                    'Total_Costs': total_costs,
                    'Gross_Margin_Pct': item.gross_margin(),
                    'Profit': item.profit(),
                    'Efficiency': item.efficiency_ratio(),
                    'Utilization_Pct': item.calculate_utilization(),
                    'Fixed_Costs': sum(cost.amount for cost in cost_structure.costs if cost.cost_type.value == 'fixed'),
                    'Variable_Costs': sum(cost.amount for cost in cost_structure.costs if cost.cost_type.value == 'variable'),
                    'Operational_Costs': sum(cost.amount for cost in cost_structure.costs if cost.cost_type.value == 'operational')
                })
        
        df = pd.DataFrame(financial_data)
        
        if output_format.lower() == 'csv':
            output_path = self.output_dir / f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_path, index=False)
        elif output_format.lower() == 'excel':
            output_path = self.output_dir / f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(output_path, index=False)
        elif output_format.lower() == 'json':
            output_path = self.output_dir / f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return output_path
    
    def generate_charts(self,
                       neoclouds: List[NeocloudProvider],
                       inference_providers: List[InferenceProvider],
                       applications: List[Application]) -> Dict[str, Path]:
        """Generate visualization charts.
        
        Args:
            neoclouds: List of neocloud providers
            inference_providers: List of inference providers
            applications: List of applications
        
        Returns:
            Dictionary mapping chart names to file paths
        """
        charts = {}
        
        # Revenue comparison chart
        charts['revenue_comparison'] = self._create_revenue_comparison_chart(
            neoclouds, inference_providers, applications
        )
        
        # Margin analysis chart
        charts['margin_analysis'] = self._create_margin_analysis_chart(
            neoclouds, inference_providers, applications
        )
        
        # Efficiency distribution chart
        charts['efficiency_distribution'] = self._create_efficiency_distribution_chart(
            neoclouds, inference_providers, applications
        )
        
        # Cost structure chart
        charts['cost_structure'] = self._create_cost_structure_chart(
            neoclouds, inference_providers, applications
        )
        
        return charts
    
    def _generate_executive_summary(self,
                                  neoclouds: List[NeocloudProvider],
                                  inference_providers: List[InferenceProvider],
                                  applications: List[Application]) -> Dict[str, Any]:
        """Generate executive summary data."""
        total_revenue = (
            sum(nc.total_revenue() for nc in neoclouds) +
            sum(ip.total_revenue() for ip in inference_providers) +
            sum(app.total_revenue() for app in applications)
        )
        
        total_costs = (
            sum(nc.total_cost() for nc in neoclouds) +
            sum(ip.total_cost() for ip in inference_providers) +
            sum(app.total_cost() for app in applications)
        )
        
        return {
            'total_revenue': total_revenue,
            'total_costs': total_costs,
            'total_profit': total_revenue - total_costs,
            'overall_margin': ((total_revenue - total_costs) / total_revenue * 100) if total_revenue > 0 else 0,
            'layer_count': {
                'neoclouds': len(neoclouds),
                'inference_providers': len(inference_providers),
                'applications': len(applications)
            },
            'top_performers': {
                'highest_revenue': self._find_top_performer(neoclouds + inference_providers + applications, 'revenue'),
                'highest_margin': self._find_top_performer(neoclouds + inference_providers + applications, 'margin'),
                'highest_efficiency': self._find_top_performer(neoclouds + inference_providers + applications, 'efficiency_ratio')
            }
        }
    
    def _analyze_layer(self, layer_data: List[EconomicLayer], layer_name: str) -> Dict[str, Any]:
        """Analyze a specific layer."""
        if not layer_data:
            return {'error': f'No data available for {layer_name}'}
        
        revenues = [item.total_revenue() for item in layer_data]
        margins = [item.gross_margin() for item in layer_data]
        efficiencies = [item.efficiency_ratio() for item in layer_data]
        utilizations = [item.calculate_utilization() for item in layer_data]
        
        return {
            'count': len(layer_data),
            'total_revenue': sum(revenues),
            'average_revenue': sum(revenues) / len(revenues),
            'average_margin': sum(margins) / len(margins),
            'average_efficiency': sum(efficiencies) / len(efficiencies),
            'average_utilization': sum(utilizations) / len(utilizations),
            'revenue_range': {'min': min(revenues), 'max': max(revenues)},
            'margin_range': {'min': min(margins), 'max': max(margins)},
            'top_performer': max(layer_data, key=lambda x: x.total_revenue()).name,
            'most_efficient': max(layer_data, key=lambda x: x.efficiency_ratio()).name
        }
    
    def _analyze_full_stack(self,
                          neoclouds: List[NeocloudProvider],
                          inference_providers: List[InferenceProvider],
                          applications: List[Application]) -> Dict[str, Any]:
        """Analyze the full stack."""
        all_layers = neoclouds + inference_providers + applications
        
        if not all_layers:
            return {'error': 'No data available for stack analysis'}
        
        # Calculate value flow
        neocloud_revenue = sum(nc.total_revenue() for nc in neoclouds)
        inference_revenue = sum(ip.total_revenue() for ip in inference_providers)
        app_revenue = sum(app.total_revenue() for app in applications)
        
        # Calculate efficiency multipliers
        stack_efficiency = 1.0
        if neoclouds:
            stack_efficiency *= sum(nc.efficiency_ratio() for nc in neoclouds) / len(neoclouds)
        if inference_providers:
            stack_efficiency *= sum(ip.efficiency_ratio() for ip in inference_providers) / len(inference_providers)
        if applications:
            stack_efficiency *= sum(app.efficiency_ratio() for app in applications) / len(applications)
        
        return {
            'value_flow': {
                'neocloud_revenue': neocloud_revenue,
                'inference_revenue': inference_revenue,
                'application_revenue': app_revenue,
                'value_multiplier': app_revenue / neocloud_revenue if neocloud_revenue > 0 else 0
            },
            'stack_efficiency': stack_efficiency,
            'bottlenecks': self._identify_bottlenecks(neoclouds, inference_providers, applications),
            'optimization_potential': self._calculate_optimization_potential(all_layers)
        }
    
    def _analyze_trends(self,
                       neoclouds: List[NeocloudProvider],
                       inference_providers: List[InferenceProvider],
                       applications: List[Application]) -> Dict[str, Any]:
        """Analyze trends (simulated for demo)."""
        # In a real implementation, this would analyze historical data
        return {
            'revenue_growth': {
                'neoclouds': 15.2,
                'inference_providers': 45.7,
                'applications': 67.3
            },
            'margin_trends': {
                'neoclouds': -2.1,
                'inference_providers': 3.4,
                'applications': 8.9
            },
            'efficiency_improvements': {
                'neoclouds': 5.6,
                'inference_providers': 12.3,
                'applications': 18.7
            }
        }
    
    def _generate_detailed_layer_analysis(self, layer_data: List[EconomicLayer]) -> List[Dict[str, Any]]:
        """Generate detailed analysis for each item in a layer."""
        detailed_analysis = []
        
        for item in layer_data:
            costs = item.total_cost()
            
            analysis = {
                'name': item.name,
                'revenue': item.total_revenue(),
                'costs': costs,
                'profit': item.profit(),
                'margin': item.gross_margin(),
                'efficiency': item.efficiency_ratio(),
                'utilization': item.calculate_utilization(),
                'cost_breakdown': {
                    cost_item.name: cost_item.amount for cost_item in costs.items
                },
                'optimization_score': self._calculate_optimization_score(item)
            }
            
            detailed_analysis.append(analysis)
        
        return detailed_analysis
    
    def _generate_layer_charts(self, layer_data: List[EconomicLayer], layer_name: str) -> Dict[str, str]:
        """Generate charts for a specific layer."""
        charts = {}
        
        if not layer_data:
            return charts
        
        # Revenue distribution chart
        revenues = [item.total_revenue() for item in layer_data]
        names = [item.name for item in layer_data]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, revenues)
        plt.title(f'{layer_name} - Revenue Distribution')
        plt.xlabel('Provider/Application')
        plt.ylabel('Revenue ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = self.output_dir / f'{layer_name.lower()}_revenue_chart.png'
        plt.savefig(chart_path)
        plt.close()
        
        charts['revenue_distribution'] = str(chart_path)
        
        return charts
    
    def _create_revenue_comparison_chart(self,
                                       neoclouds: List[NeocloudProvider],
                                       inference_providers: List[InferenceProvider],
                                       applications: List[Application]) -> Path:
        """Create revenue comparison chart."""
        layers = ['Neoclouds', 'Inference Providers', 'Applications']
        revenues = [
            sum(nc.total_revenue() for nc in neoclouds),
            sum(ip.total_revenue() for ip in inference_providers),
            sum(app.total_revenue() for app in applications)
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(layers, revenues, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Revenue Comparison by Layer')
        plt.xlabel('Layer')
        plt.ylabel('Revenue ($)')
        
        # Add value labels on bars
        for bar, revenue in zip(bars, revenues):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + revenue*0.01,
                    format_currency(revenue), ha='center', va='bottom')
        
        plt.tight_layout()
        
        chart_path = self.output_dir / 'revenue_comparison.png'
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    
    def _create_margin_analysis_chart(self,
                                    neoclouds: List[NeocloudProvider],
                                    inference_providers: List[InferenceProvider],
                                    applications: List[Application]) -> Path:
        """Create margin analysis chart."""
        all_items = neoclouds + inference_providers + applications
        names = [item.name for item in all_items]
        margins = [item.gross_margin() for item in all_items]
        colors = (['#1f77b4'] * len(neoclouds) + 
                 ['#ff7f0e'] * len(inference_providers) + 
                 ['#2ca02c'] * len(applications))
        
        plt.figure(figsize=(12, 6))
        plt.bar(names, margins, color=colors)
        plt.title('Gross Margin Analysis')
        plt.xlabel('Provider/Application')
        plt.ylabel('Gross Margin (%)')
        plt.xticks(rotation=45)
        plt.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Target Margin (20%)')
        plt.legend()
        plt.tight_layout()
        
        chart_path = self.output_dir / 'margin_analysis.png'
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    
    def _create_efficiency_distribution_chart(self,
                                            neoclouds: List[NeocloudProvider],
                                            inference_providers: List[InferenceProvider],
                                            applications: List[Application]) -> Path:
        """Create efficiency distribution chart."""
        all_items = neoclouds + inference_providers + applications
        efficiencies = [item.efficiency_ratio() for item in all_items]
        
        plt.figure(figsize=(10, 6))
        plt.hist(efficiencies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Efficiency Distribution Across All Layers')
        plt.xlabel('Efficiency Score')
        plt.ylabel('Frequency')
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Baseline Efficiency (1.0)')
        plt.legend()
        plt.tight_layout()
        
        chart_path = self.output_dir / 'efficiency_distribution.png'
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    
    def _create_cost_structure_chart(self,
                                   neoclouds: List[NeocloudProvider],
                                   inference_providers: List[InferenceProvider],
                                   applications: List[Application]) -> Path:
        """Create cost structure chart."""
        cost_categories = {'Fixed': 0, 'Variable': 0, 'Operational': 0}
        
        for layer in [neoclouds, inference_providers, applications]:
            for item in layer:
                costs = item.total_cost()
                for cost_item in costs.items:
                    if cost_item.cost_type.value == 'fixed':
                        cost_categories['Fixed'] += cost_item.amount
                    elif cost_item.cost_type.value == 'variable':
                        cost_categories['Variable'] += cost_item.amount
                    elif cost_item.cost_type.value == 'operational':
                        cost_categories['Operational'] += cost_item.amount
        
        plt.figure(figsize=(8, 8))
        plt.pie(cost_categories.values(), labels=cost_categories.keys(), autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff', '#99ff99'])
        plt.title('Overall Cost Structure')
        plt.tight_layout()
        
        chart_path = self.output_dir / 'cost_structure.png'
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    
    def _generate_html_report(self, report_data: Dict[str, Any], report_name: str) -> Path:
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Token Factory - {{ report_data.get('layer_name', 'Executive') }} Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f8ff; padding: 20px; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .section { margin: 20px 0; }
        .recommendation { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 AI Token Factory Economics Stack</h1>
        <h2>{{ report_data.get('layer_name', 'Executive') }} Report</h2>
        <p>Generated: {{ report_data.timestamp }}</p>
    </div>
    
    {% if report_data.summary %}
    <div class="section">
        <h3>Executive Summary</h3>
        <div class="metric">
            <strong>Total Revenue:</strong> ${{ "{:,.2f}".format(report_data.summary.total_revenue) }}
        </div>
        <div class="metric">
            <strong>Total Costs:</strong> ${{ "{:,.2f}".format(report_data.summary.total_costs) }}
        </div>
        <div class="metric">
            <strong>Overall Margin:</strong> {{ "{:.1f}%".format(report_data.summary.overall_margin) }}
        </div>
    </div>
    {% endif %}
    
    {% if report_data.recommendations %}
    <div class="section">
        <h3>Key Recommendations</h3>
        {% for rec in report_data.recommendations[:5] %}
        <div class="recommendation">
            <strong>{{ rec.title }}</strong> ({{ rec.priority }})<br>
            {{ rec.description }}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div class="section">
        <p><em>This report was generated automatically by the AI Token Factory Economics Stack analysis system.</em></p>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        html_content = template.render(report_data=report_data)
        
        output_path = self.output_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_json_report(self, report_data: Dict[str, Any], report_name: str) -> Path:
        """Generate JSON report."""
        output_path = self.output_dir / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return output_path
    
    def _find_top_performer(self, items: List[EconomicLayer], metric: str) -> Dict[str, Any]:
        """Find top performer by specified metric."""
        if not items:
            return {'name': 'N/A', 'value': 0}
        
        if metric == 'revenue':
            top_item = max(items, key=lambda x: x.total_revenue())
            return {'name': top_item.name, 'value': top_item.total_revenue()}
        elif metric == 'margin':
            top_item = max(items, key=lambda x: x.gross_margin())
            return {'name': top_item.name, 'value': top_item.gross_margin()}
        elif metric == 'efficiency_ratio':
            top_item = max(items, key=lambda x: x.efficiency_ratio())
            return {'name': top_item.name, 'value': top_item.efficiency_ratio()}
        else:
            return {'name': 'N/A', 'value': 0}
    
    def _identify_bottlenecks(self,
                            neoclouds: List[NeocloudProvider],
                            inference_providers: List[InferenceProvider],
                            applications: List[Application]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in the stack."""
        bottlenecks = []
        
        # Check utilization rates
        for nc in neoclouds:
            if nc.calculate_utilization() < 70:
                bottlenecks.append({
                    'type': 'Low Utilization',
                    'layer': 'Neocloud',
                    'item': nc.name,
                    'value': nc.calculate_utilization(),
                    'impact': 'High'
                })
        
        # Check efficiency scores
        all_items = neoclouds + inference_providers + applications
        for item in all_items:
            if item.efficiency_ratio() < 0.8:
                layer_type = 'Neocloud' if item in neoclouds else 'Inference' if item in inference_providers else 'Application'
                bottlenecks.append({
                    'type': 'Low Efficiency',
                    'layer': layer_type,
                    'item': item.name,
                    'value': item.efficiency_ratio(),
                    'impact': 'Medium'
                })
        
        return bottlenecks
    
    def _calculate_optimization_potential(self, items: List[EconomicLayer]) -> Dict[str, float]:
        """Calculate optimization potential."""
        if not items:
            return {'revenue_potential': 0, 'cost_reduction_potential': 0, 'efficiency_gain_potential': 0}
        
        current_efficiency = sum(item.efficiency_ratio() for item in items) / len(items)
        target_efficiency = 1.2  # 20% above baseline
        
        current_revenue = sum(item.total_revenue() for item in items)
        potential_revenue_gain = current_revenue * (target_efficiency - current_efficiency)
        
        return {
            'revenue_potential': potential_revenue_gain,
            'cost_reduction_potential': current_revenue * 0.1,  # Assume 10% cost reduction potential
            'efficiency_gain_potential': (target_efficiency - current_efficiency) * 100
        }
    
    def _calculate_optimization_score(self, item: EconomicLayer) -> float:
        """Calculate optimization score for an item."""
        # Weighted score based on multiple factors
        margin_score = min(item.gross_margin() / 30, 1.0)  # Target 30% margin
        efficiency_score = min(item.efficiency_ratio() / 1.5, 1.0)  # Target 1.5 efficiency
        utilization_score = min(item.calculate_utilization() / 85, 1.0)  # Target 85% utilization
        
        return (margin_score * 0.4 + efficiency_score * 0.4 + utilization_score * 0.2) * 100


def generate_report(neoclouds: List[NeocloudProvider],
                   inference_providers: List[InferenceProvider],
                   applications: List[Application],
                   market_data: Dict[str, Any],
                   report_type: str = "executive",
                   output_format: str = "html") -> Path:
    """Generate report using ReportGenerator.
    
    Args:
        neoclouds: List of neocloud providers
        inference_providers: List of inference providers
        applications: List of applications
        market_data: Market data and benchmarks
        report_type: Type of report ('executive', 'financial', 'layer')
        output_format: Output format ('html', 'json', 'csv')
    
    Returns:
        Path to generated report
    """
    generator = ReportGenerator()
    
    if report_type == "executive":
        return generator.generate_executive_report(
            neoclouds, inference_providers, applications, market_data, output_format
        )
    elif report_type == "financial":
        return generator.generate_financial_report(
            neoclouds, inference_providers, applications, output_format
        )
    else:
        raise ValueError(f"Unsupported report type: {report_type}")