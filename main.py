#!/usr/bin/env python3
"""Main simulation runner for the AI Token Factory Economics Stack."""

import json
from typing import Dict, List, Any
from data.simulation_data import (
    create_sample_neoclouds,
    create_sample_inference_providers,
    create_sample_applications,
    load_market_data
)
from models.base_models import EconomicLayer
from analytics.economic_analyzer import EconomicAnalyzer
from utils.formatters import format_currency, format_percentage


class AITokenFactorySimulation:
    """Main simulation class for the AI Token Factory Economics Stack."""
    
    def __init__(self):
        self.neoclouds = create_sample_neoclouds()
        self.inference_providers = create_sample_inference_providers()
        self.applications = create_sample_applications()
        self.market_data = load_market_data()
        self.analyzer = EconomicAnalyzer()
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete economic simulation."""
        print("🏭 AI Token Factory Economics Stack Simulation")
        print("=" * 50)
        
        results = {
            "neoclouds": self._analyze_neoclouds(),
            "inference_providers": self._analyze_inference_providers(),
            "applications": self._analyze_applications(),
            "stack_summary": self._analyze_full_stack(),
            "market_data": self.market_data
        }
        
        self._print_summary(results)
        return results
    
    def _analyze_neoclouds(self) -> Dict[str, Any]:
        """Analyze neocloud layer economics."""
        print("\n🌩️  NEOCLOUD LAYER ANALYSIS")
        print("-" * 30)
        
        neocloud_results = {}
        total_revenue = 0
        total_cost = 0
        total_gpus = 0
        
        for neocloud in self.neoclouds:
            summary = neocloud.economic_summary()
            cluster_perf = neocloud.cluster_performance_summary()
            optimizations = neocloud.optimize_costs()
            
            total_revenue += summary['total_revenue']
            total_cost += summary['total_cost']
            total_gpus += sum(cluster['gpu_count'] for cluster in cluster_perf.values())
            
            print(f"\n{neocloud.name}:")
            print(f"  Revenue: {format_currency(summary['total_revenue'])}/month")
            print(f"  Costs: {format_currency(summary['total_cost'])}/month")
            print(f"  Profit: {format_currency(summary['profit'])}/month")
            print(f"  Gross Margin: {format_percentage(summary['gross_margin_pct'])}")
            print(f"  Utilization: {format_percentage(summary['utilization_pct'])}")
            
            if optimizations:
                print(f"  💡 Optimization Opportunities:")
                for opt_type, savings in optimizations.items():
                    print(f"    {opt_type}: {format_currency(savings)}/month")
            
            neocloud_results[neocloud.name] = {
                "summary": summary,
                "cluster_performance": cluster_perf,
                "optimizations": optimizations
            }
        
        # Layer totals
        layer_summary = {
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "total_profit": total_revenue - total_cost,
            "total_gpus": total_gpus,
            "average_margin": ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0
        }
        
        print(f"\n📊 Neocloud Layer Totals:")
        print(f"  Total Revenue: {format_currency(layer_summary['total_revenue'])}/month")
        print(f"  Total Profit: {format_currency(layer_summary['total_profit'])}/month")
        print(f"  Total GPUs: {layer_summary['total_gpus']:,}")
        print(f"  Average Margin: {format_percentage(layer_summary['average_margin'])}")
        
        neocloud_results["layer_summary"] = layer_summary
        return neocloud_results
    
    def _analyze_inference_providers(self) -> Dict[str, Any]:
        """Analyze inference provider layer economics."""
        print("\n🧠 INFERENCE PROVIDER LAYER ANALYSIS")
        print("-" * 35)
        
        provider_results = {}
        total_revenue = 0
        total_cost = 0
        total_tokens = 0
        
        for provider in self.inference_providers:
            summary = provider.economic_summary()
            token_economics = provider.token_economics_summary()
            capacity_analysis = provider.capacity_analysis()
            optimizations = provider.optimize_costs()
            
            total_revenue += summary['total_revenue']
            total_cost += summary['total_cost']
            total_tokens += capacity_analysis['actual_tokens_per_month']
            
            print(f"\n{provider.name}:")
            print(f"  Revenue: {format_currency(summary['total_revenue'])}/month")
            print(f"  Costs: {format_currency(summary['total_cost'])}/month")
            print(f"  Profit: {format_currency(summary['profit'])}/month")
            print(f"  Gross Margin: {format_percentage(summary['gross_margin_pct'])}")
            print(f"  Tokens/Month: {capacity_analysis['actual_tokens_per_month']:,.0f}")
            print(f"  Capacity Utilization: {format_percentage(capacity_analysis['capacity_utilization_pct'])}")
            
            if optimizations:
                print(f"  💡 Optimization Opportunities:")
                for opt_type, savings in optimizations.items():
                    print(f"    {opt_type}: {format_currency(savings)}/month")
            
            provider_results[provider.name] = {
                "summary": summary,
                "token_economics": token_economics,
                "capacity_analysis": capacity_analysis,
                "optimizations": optimizations
            }
        
        # Layer totals
        layer_summary = {
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "total_profit": total_revenue - total_cost,
            "total_tokens_per_month": total_tokens,
            "average_margin": ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0,
            "average_cost_per_1k_tokens": (total_cost / (total_tokens / 1000)) if total_tokens > 0 else 0
        }
        
        print(f"\n📊 Inference Layer Totals:")
        print(f"  Total Revenue: {format_currency(layer_summary['total_revenue'])}/month")
        print(f"  Total Profit: {format_currency(layer_summary['total_profit'])}/month")
        print(f"  Total Tokens: {layer_summary['total_tokens_per_month']:,.0f}/month")
        print(f"  Avg Cost/1K Tokens: {format_currency(layer_summary['average_cost_per_1k_tokens'])}")
        print(f"  Average Margin: {format_percentage(layer_summary['average_margin'])}")
        
        provider_results["layer_summary"] = layer_summary
        return provider_results
    
    def _analyze_applications(self) -> Dict[str, Any]:
        """Analyze application layer economics."""
        print("\n📱 APPLICATION LAYER ANALYSIS")
        print("-" * 30)
        
        app_results = {}
        total_revenue = 0
        total_cost = 0
        total_users = 0
        total_productivity_value = 0
        
        for app in self.applications:
            summary = app.economic_summary()
            user_economics = app.user_economics_summary()
            productivity_impact = app.productivity_impact_analysis()
            optimizations = app.optimize_costs()
            
            total_revenue += summary['total_revenue']
            total_cost += summary['total_cost']
            total_users += sum(segment['user_count'] for segment in user_economics.values())
            total_productivity_value += productivity_impact['estimated_productivity_value_usd']
            
            print(f"\n{app.name}:")
            print(f"  Revenue: {format_currency(summary['total_revenue'])}/month")
            print(f"  Costs: {format_currency(summary['total_cost'])}/month")
            print(f"  Profit: {format_currency(summary['profit'])}/month")
            print(f"  Gross Margin: {format_percentage(summary['gross_margin_pct'])}")
            print(f"  Total Users: {total_users:,}")
            print(f"  Productivity Value: {format_currency(productivity_impact['estimated_productivity_value_usd'])}/month")
            print(f"  User ROI: {format_percentage(productivity_impact['roi_for_users'])}")
            
            if optimizations:
                print(f"  💡 Optimization Opportunities:")
                for opt_type, value in optimizations.items():
                    print(f"    {opt_type}: {format_currency(value)}/month")
            
            app_results[app.name] = {
                "summary": summary,
                "user_economics": user_economics,
                "productivity_impact": productivity_impact,
                "optimizations": optimizations
            }
        
        # Layer totals
        layer_summary = {
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "total_profit": total_revenue - total_cost,
            "total_users": total_users,
            "total_productivity_value": total_productivity_value,
            "average_margin": ((total_revenue - total_cost) / total_revenue * 100) if total_revenue > 0 else 0,
            "average_revenue_per_user": total_revenue / total_users if total_users > 0 else 0
        }
        
        print(f"\n📊 Application Layer Totals:")
        print(f"  Total Revenue: {format_currency(layer_summary['total_revenue'])}/month")
        print(f"  Total Profit: {format_currency(layer_summary['total_profit'])}/month")
        print(f"  Total Users: {layer_summary['total_users']:,}")
        print(f"  Productivity Value: {format_currency(layer_summary['total_productivity_value'])}/month")
        print(f"  Avg Revenue/User: {format_currency(layer_summary['average_revenue_per_user'])}/month")
        print(f"  Average Margin: {format_percentage(layer_summary['average_margin'])}")
        
        app_results["layer_summary"] = layer_summary
        return app_results
    
    def _analyze_full_stack(self) -> Dict[str, Any]:
        """Analyze the complete AI Token Factory stack."""
        print("\n🏭 FULL STACK ANALYSIS")
        print("-" * 25)
        
        # Calculate stack totals
        neocloud_revenue = sum(nc.revenue_model.total_revenue() for nc in self.neoclouds)
        neocloud_cost = sum(nc.cost_structure.total_cost() for nc in self.neoclouds)
        
        inference_revenue = sum(ip.revenue_model.total_revenue() for ip in self.inference_providers)
        inference_cost = sum(ip.cost_structure.total_cost() for ip in self.inference_providers)
        
        app_revenue = sum(app.revenue_model.total_revenue() for app in self.applications)
        app_cost = sum(app.cost_structure.total_cost() for app in self.applications)
        
        # Stack economics
        total_stack_revenue = neocloud_revenue + inference_revenue + app_revenue
        total_stack_cost = neocloud_cost + inference_cost + app_cost
        total_stack_profit = total_stack_revenue - total_stack_cost
        
        # Value flow analysis
        raw_materials_cost = sum(
            nc.cost_structure.total_cost()
            for nc in self.neoclouds
        )
        
        intelligence_value = sum(
            app.productivity_impact_analysis()['estimated_productivity_value_usd']
            for app in self.applications
        )
        
        value_multiplier = intelligence_value / raw_materials_cost if raw_materials_cost > 0 else 0
        
        stack_summary = {
            "layer_breakdown": {
                "neoclouds": {
                    "revenue": neocloud_revenue,
                    "cost": neocloud_cost,
                    "profit": neocloud_revenue - neocloud_cost,
                    "margin_pct": ((neocloud_revenue - neocloud_cost) / neocloud_revenue * 100) if neocloud_revenue > 0 else 0
                },
                "inference_providers": {
                    "revenue": inference_revenue,
                    "cost": inference_cost,
                    "profit": inference_revenue - inference_cost,
                    "margin_pct": ((inference_revenue - inference_cost) / inference_revenue * 100) if inference_revenue > 0 else 0
                },
                "applications": {
                    "revenue": app_revenue,
                    "cost": app_cost,
                    "profit": app_revenue - app_cost,
                    "margin_pct": ((app_revenue - app_cost) / app_revenue * 100) if app_revenue > 0 else 0
                }
            },
            "stack_totals": {
                "total_revenue": total_stack_revenue,
                "total_cost": total_stack_cost,
                "total_profit": total_stack_profit,
                "overall_margin_pct": (total_stack_profit / total_stack_revenue * 100) if total_stack_revenue > 0 else 0
            },
            "value_transformation": {
                "raw_materials_cost": raw_materials_cost,
                "intelligence_value_created": intelligence_value,
                "value_multiplier": value_multiplier,
                "efficiency_ratio": intelligence_value / total_stack_cost if total_stack_cost > 0 else 0
            }
        }
        
        print(f"\n💰 Stack Economics:")
        for layer, metrics in stack_summary["layer_breakdown"].items():
            print(f"  {layer.replace('_', ' ').title()}:")
            print(f"    Revenue: {format_currency(metrics['revenue'])}/month")
            print(f"    Profit: {format_currency(metrics['profit'])}/month")
            print(f"    Margin: {format_percentage(metrics['margin_pct'])}")
        
        print(f"\n🏭 Value Transformation:")
        vt = stack_summary["value_transformation"]
        print(f"  Raw Materials → Intelligence")
        print(f"  Input Cost: {format_currency(vt['raw_materials_cost'])}/month")
        print(f"  Output Value: {format_currency(vt['intelligence_value_created'])}/month")
        print(f"  Value Multiplier: {vt['value_multiplier']:.1f}x")
        print(f"  Stack Efficiency: {vt['efficiency_ratio']:.2f}")
        
        return stack_summary
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print executive summary."""
        print("\n" + "=" * 50)
        print("📈 EXECUTIVE SUMMARY")
        print("=" * 50)
        
        stack = results["stack_summary"]
        totals = stack["stack_totals"]
        value_transform = stack["value_transformation"]
        
        print(f"\n🎯 Key Metrics:")
        print(f"  Total Stack Revenue: {format_currency(totals['total_revenue'])}/month")
        print(f"  Total Stack Profit: {format_currency(totals['total_profit'])}/month")
        print(f"  Overall Margin: {format_percentage(totals['overall_margin_pct'])}")
        print(f"  Value Multiplier: {value_transform['value_multiplier']:.1f}x")
        
        print(f"\n🏆 Best Performing Layer:")
        best_layer = max(stack["layer_breakdown"].items(), key=lambda x: x[1]['margin_pct'])
        print(f"  {best_layer[0].replace('_', ' ').title()}: {format_percentage(best_layer[1]['margin_pct'])} margin")
        
        print(f"\n💡 Key Insights:")
        print(f"  • The AI Token Factory transforms ${value_transform['raw_materials_cost']:,.0f} of raw materials")
        print(f"    into ${value_transform['intelligence_value_created']:,.0f} of intelligence value monthly")
        print(f"  • Each dollar invested generates {value_transform['efficiency_ratio']:.2f} dollars of productivity value")
        print(f"  • The stack operates at {totals['overall_margin_pct']:.1f}% overall gross margin")
        
    def save_results(self, results: Dict[str, Any], filename: str = "simulation_results.json"):
        """Save simulation results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filename}")


def main():
    """Main entry point."""
    simulation = AITokenFactorySimulation()
    results = simulation.run_simulation()
    simulation.save_results(results)


if __name__ == "__main__":
    main()